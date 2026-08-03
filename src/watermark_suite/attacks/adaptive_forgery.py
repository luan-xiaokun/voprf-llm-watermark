import inspect
import math
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Protocol

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from voprf_py import (
    BlindedElement,
    EvaluationElement,
    Proof,
    PublicKey,
    finalize_batch_blind_results,
    prepare_batch_blind_inputs,
)


class ColorOracle(Protocol):
    """Interface available to a forger: the color of one context-token pair."""

    def query(self, context: tuple[int, ...], token_id: int) -> bool:
        """Return True when ``(context, token_id)`` is green."""


@dataclass
class ColorOracleStats:
    """Cumulative cost of public color-oracle queries."""

    query_count: int = 0
    server_round_count: int = 0
    blinded_elements_bytes: int = 0
    evaluation_elements_bytes: int = 0
    proof_bytes: int = 0
    total_communication_bytes: int = 0
    preparation_time_seconds: float = 0.0
    evaluation_time_seconds: float = 0.0
    finalization_time_seconds: float = 0.0
    total_time_seconds: float = 0.0

    def difference(self, previous: "ColorOracleStats") -> "ColorOracleStats":
        return ColorOracleStats(
            query_count=self.query_count - previous.query_count,
            server_round_count=(
                self.server_round_count - previous.server_round_count
            ),
            blinded_elements_bytes=(
                self.blinded_elements_bytes - previous.blinded_elements_bytes
            ),
            evaluation_elements_bytes=(
                self.evaluation_elements_bytes
                - previous.evaluation_elements_bytes
            ),
            proof_bytes=self.proof_bytes - previous.proof_bytes,
            total_communication_bytes=(
                self.total_communication_bytes
                - previous.total_communication_bytes
            ),
            preparation_time_seconds=(
                self.preparation_time_seconds
                - previous.preparation_time_seconds
            ),
            evaluation_time_seconds=(
                self.evaluation_time_seconds - previous.evaluation_time_seconds
            ),
            finalization_time_seconds=(
                self.finalization_time_seconds
                - previous.finalization_time_seconds
            ),
            total_time_seconds=(
                self.total_time_seconds - previous.total_time_seconds
            ),
        )


class VOPRFColorOracle:
    """A color oracle backed by the public VOPRF detection interface.

    This client owns no watermark key. It blinds the submitted pair, sends the
    blinded element through ``server_interface``, verifies/finalizes the result,
    and locally maps the pseudorandom output to a color.
    """

    def __init__(
        self,
        server_public_key: PublicKey,
        server_interface: Callable[
            [list[BlindedElement]], tuple[list[EvaluationElement], Proof]
        ],
        gamma: float,
    ) -> None:
        if not 0 < gamma < 1:
            raise ValueError("gamma must be in the open interval (0, 1)")

        self.server_public_key = server_public_key
        self.server_interface = server_interface
        self.gamma = gamma
        self._stats = ColorOracleStats()

    @property
    def query_count(self) -> int:
        return self._stats.query_count

    @property
    def stats(self) -> ColorOracleStats:
        return ColorOracleStats(**asdict(self._stats))

    def query(self, context: tuple[int, ...], token_id: int) -> bool:
        token_ids = (*context, token_id)
        if not token_ids:
            raise ValueError("a color query must contain at least one token")
        if any(token < 0 or token >= 2**32 for token in token_ids):
            raise ValueError("token IDs must be unsigned 32-bit integers")

        total_start = time.perf_counter()
        preparation_start = total_start
        message = struct.pack(f">{len(token_ids)}I", *token_ids)
        states, blinded_elements = prepare_batch_blind_inputs([message])
        blinded_elements_bytes = sum(
            len(element.to_binary()) for element in blinded_elements
        )
        preparation_end = time.perf_counter()

        evaluation_start = preparation_end
        evaluation_elements, proof = self.server_interface(blinded_elements)
        evaluation_elements_bytes = sum(
            len(element.to_binary()) for element in evaluation_elements
        )
        proof_bytes = len(proof.to_binary())
        evaluation_end = time.perf_counter()

        finalization_start = evaluation_end
        outputs = finalize_batch_blind_results(
            [message],
            states,
            evaluation_elements,
            proof,
            self.server_public_key,
        )
        if len(outputs) != 1:
            raise RuntimeError(
                f"the VOPRF interface returned {len(outputs)} outputs for one query"
            )

        finalization_end = time.perf_counter()
        total_communication_bytes = (
            blinded_elements_bytes + evaluation_elements_bytes + proof_bytes
        )
        self._stats.query_count += 1
        self._stats.server_round_count += 1
        self._stats.blinded_elements_bytes += blinded_elements_bytes
        self._stats.evaluation_elements_bytes += evaluation_elements_bytes
        self._stats.proof_bytes += proof_bytes
        self._stats.total_communication_bytes += total_communication_bytes
        self._stats.preparation_time_seconds += (
            preparation_end - preparation_start
        )
        self._stats.evaluation_time_seconds += (
            evaluation_end - evaluation_start
        )
        self._stats.finalization_time_seconds += (
            finalization_end - finalization_start
        )
        self._stats.total_time_seconds += finalization_end - total_start

        output = outputs[0]
        threshold = int(self.gamma * (1 << (8 * len(output))))
        return int.from_bytes(output, "big") < threshold


class LocalColorOracle:
    """A counted color-only interface around a scheme's detector rule.

    The adaptive forger receives this object rather than the detector or its
    keying material.  Each cache miss is one submitted context-token pair.
    """

    def __init__(
        self,
        color_interface: Callable[[tuple[int, ...], int], bool],
    ) -> None:
        self.color_interface = color_interface
        self._stats = ColorOracleStats()

    @property
    def query_count(self) -> int:
        return self._stats.query_count

    @property
    def stats(self) -> ColorOracleStats:
        return ColorOracleStats(**asdict(self._stats))

    def query(self, context: tuple[int, ...], token_id: int) -> bool:
        start = time.perf_counter()
        result = bool(self.color_interface(context, token_id))
        elapsed = time.perf_counter() - start
        self._stats.query_count += 1
        self._stats.server_round_count += 1
        self._stats.evaluation_time_seconds += elapsed
        self._stats.total_time_seconds += elapsed
        return result


@dataclass(frozen=True)
class AdaptiveForgeryStep:
    """Decision made for one generated position."""

    position: int
    context: tuple[int, ...] | None
    candidate_token_ids: list[int]
    checked_candidate_token_ids: list[int]
    selected_token_id: int
    selected_rank: int
    selected_green: bool | None
    fallback: bool
    oracle_query_count: int
    cache_hit_count: int
    oracle_time_seconds: float
    selected_log_probability: float | None = None
    top_candidate_log_probability: float | None = None
    log_probability_gap: float | None = None


@dataclass
class AdaptiveForgeryResult:
    """Text and query accounting produced by one adaptive forgery run."""

    text: str
    token_ids: list[int]
    steps: list[AdaptiveForgeryStep]
    generated_token_num: int
    oracle_query_count: int
    color_check_count: int
    cache_hit_count: int
    scored_position_num: int
    scored_token_num: int
    selected_green_token_num: int
    selected_green_ratio: float
    fallback_count: int
    unique_selected_pair_num: int
    duplicate_selected_pair_num: int
    elapsed_seconds: float
    oracle_time_seconds: float
    non_oracle_time_seconds: float
    tokens_per_second: float
    queries_per_generated_token: float
    queries_per_scored_token: float
    color_checks_per_scored_token: float
    fallback_ratio: float
    mean_green_candidate_rank: float | None
    mean_selected_negative_log_likelihood: float | None
    local_model_perplexity: float | None
    mean_log_probability_gap: float | None
    tokenization_preserved: bool
    oracle_protocol_stats: ColorOracleStats | None

    def to_dict(self, trace_level: str = "compact") -> dict:
        if trace_level not in {"none", "compact", "full"}:
            raise ValueError("trace_level must be one of: none, compact, full")

        result = asdict(self)
        result.pop("steps")
        result.pop("token_ids")

        if trace_level == "full":
            result["token_ids"] = self.token_ids
            result["steps"] = [asdict(step) for step in self.steps]
        elif trace_level == "compact":
            cumulative_queries = 0
            trace = []
            for step in self.steps:
                cumulative_queries += step.oracle_query_count
                trace.append(
                    {
                        "position": step.position,
                        "selected_token_id": step.selected_token_id,
                        "selected_rank": step.selected_rank,
                        "selected_green": step.selected_green,
                        "fallback": step.fallback,
                        "color_check_count": len(
                            step.checked_candidate_token_ids
                        ),
                        "oracle_query_count": step.oracle_query_count,
                        "cache_hit_count": step.cache_hit_count,
                        "cumulative_oracle_query_count": cumulative_queries,
                        "oracle_time_seconds": step.oracle_time_seconds,
                        "selected_log_probability": (
                            step.selected_log_probability
                        ),
                        "log_probability_gap": step.log_probability_gap,
                    }
                )
            result["trace"] = trace
        return result


class AdaptiveCandidateSelector:
    """Apply the left-to-right first-green selection rule.

    The cache is scoped to one forged text. Repeated pairs receive their
    previously learned deterministic color without another oracle submission.
    """

    def __init__(
        self,
        oracle: ColorOracle,
        window_size: int,
        max_candidates: int,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")

        self.oracle = oracle
        self.window_size = window_size
        self.max_candidates = max_candidates
        self._color_cache: dict[tuple[tuple[int, ...], int], bool] = {}

    def select(
        self,
        position: int,
        audited_token_ids: Sequence[int],
        candidate_token_ids: Sequence[int],
    ) -> AdaptiveForgeryStep:
        candidates = list(candidate_token_ids[: self.max_candidates])
        if not candidates:
            raise ValueError("candidate_token_ids must not be empty")

        # The detector cannot score a token until a complete color context exists.
        if len(audited_token_ids) < self.window_size:
            return AdaptiveForgeryStep(
                position=position,
                context=None,
                candidate_token_ids=candidates,
                checked_candidate_token_ids=[],
                selected_token_id=candidates[0],
                selected_rank=1,
                selected_green=None,
                fallback=False,
                oracle_query_count=0,
                cache_hit_count=0,
                oracle_time_seconds=0.0,
            )

        context = tuple(audited_token_ids[-self.window_size :])
        checked_candidates = []
        oracle_query_count = 0
        cache_hit_count = 0
        oracle_time_seconds = 0.0

        for rank, token_id in enumerate(candidates, start=1):
            checked_candidates.append(token_id)
            pair = (context, token_id)
            if pair in self._color_cache:
                is_green = self._color_cache[pair]
                cache_hit_count += 1
            else:
                query_start = time.perf_counter()
                is_green = bool(self.oracle.query(context, token_id))
                oracle_time_seconds += time.perf_counter() - query_start
                self._color_cache[pair] = is_green
                oracle_query_count += 1

            if is_green:
                return AdaptiveForgeryStep(
                    position=position,
                    context=context,
                    candidate_token_ids=candidates,
                    checked_candidate_token_ids=checked_candidates,
                    selected_token_id=token_id,
                    selected_rank=rank,
                    selected_green=True,
                    fallback=False,
                    oracle_query_count=oracle_query_count,
                    cache_hit_count=cache_hit_count,
                    oracle_time_seconds=oracle_time_seconds,
                )

        # All k candidates were red. The first (highest-probability) one is kept.
        return AdaptiveForgeryStep(
            position=position,
            context=context,
            candidate_token_ids=candidates,
            checked_candidate_token_ids=checked_candidates,
            selected_token_id=candidates[0],
            selected_rank=1,
            selected_green=False,
            fallback=True,
            oracle_query_count=oracle_query_count,
            cache_hit_count=cache_hit_count,
            oracle_time_seconds=oracle_time_seconds,
        )


class AdaptiveWatermarkForger:
    """Forge a watermark left-to-right using a local causal language model."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        oracle: ColorOracle,
        window_size: int,
        max_candidates: int,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")

        self.model = model
        self.tokenizer = tokenizer
        self.oracle = oracle
        self.window_size = window_size
        self.max_candidates = max_candidates
        forward_parameters = inspect.signature(model.forward).parameters
        self._supports_attention_mask = (
            "attention_mask" in forward_parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in forward_parameters.values()
            )
        )

    @torch.inference_mode()
    def forge(
        self,
        prompt: str,
        max_new_tokens: int,
        target_scored_pairs: int | None = None,
        suppress_token_ids: Sequence[int] | None = None,
        stop_on_eos: bool = True,
    ) -> AdaptiveForgeryResult:
        """Generate a forged continuation.

        Language-model probabilities are conditioned on ``prompt`` plus the
        generated prefix. Color contexts contain generated tokens only, matching
        this repository's detector, which audits ``generated_text`` separately
        from its prompt.
        """

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if target_scored_pairs is not None and target_scored_pairs <= 0:
            raise ValueError("target_scored_pairs must be positive or None")

        start = time.perf_counter()
        oracle_stats_before = self._oracle_stats()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("forge currently supports exactly one prompt at a time")
        if input_ids.shape[1] == 0:
            bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_token_id is None:
                raise ValueError("the prompt produced no tokens and no BOS token is set")
            input_ids = torch.tensor([[bos_token_id]], dtype=torch.long)

        device = self._model_device()
        input_ids = input_ids.to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device)

        model_inputs = {
            "input_ids": input_ids,
            "use_cache": True,
            "return_dict": True,
        }
        if self._supports_attention_mask:
            model_inputs["attention_mask"] = attention_mask

        outputs = self.model(**model_inputs)
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = getattr(outputs, "past_key_values", None)

        selector = AdaptiveCandidateSelector(
            oracle=self.oracle,
            window_size=self.window_size,
            max_candidates=self.max_candidates,
        )
        suppressed = set(suppress_token_ids or [])
        generated_token_ids: list[int] = []
        steps: list[AdaptiveForgeryStep] = []
        selected_pairs: set[tuple[tuple[int, ...], int]] = set()

        for position in range(max_new_tokens):
            scores = next_token_logits[0].detach().float().clone()
            valid_suppressed = [
                token_id
                for token_id in suppressed
                if 0 <= token_id < scores.shape[0]
            ]
            if valid_suppressed:
                scores[valid_suppressed] = -torch.inf

            finite_candidate_num = int(torch.isfinite(scores).sum().item())
            candidate_num = min(self.max_candidates, finite_candidate_num)
            if candidate_num == 0:
                raise RuntimeError("the local model has no finite candidate logits")
            candidate_token_ids = torch.topk(
                scores, k=candidate_num, largest=True, sorted=True
            ).indices.tolist()
            log_probabilities = torch.log_softmax(scores, dim=0)

            step = selector.select(
                position=position,
                audited_token_ids=generated_token_ids,
                candidate_token_ids=candidate_token_ids,
            )
            selected_log_probability = float(
                log_probabilities[step.selected_token_id].item()
            )
            top_candidate_log_probability = float(
                log_probabilities[candidate_token_ids[0]].item()
            )
            step = replace(
                step,
                selected_log_probability=selected_log_probability,
                top_candidate_log_probability=(
                    top_candidate_log_probability
                ),
                log_probability_gap=(
                    top_candidate_log_probability
                    - selected_log_probability
                ),
            )
            steps.append(step)
            generated_token_ids.append(step.selected_token_id)
            if step.context is not None:
                selected_pairs.add((step.context, step.selected_token_id))

            if (
                target_scored_pairs is not None
                and len(selected_pairs) >= target_scored_pairs
            ):
                break

            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if (
                stop_on_eos
                and eos_token_id is not None
                and step.selected_token_id == eos_token_id
            ):
                break
            if position + 1 == max_new_tokens:
                break

            selected = torch.tensor(
                [[step.selected_token_id]], dtype=torch.long, device=device
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )

            if past_key_values is None:
                next_input_ids = torch.cat(
                    [
                        input_ids,
                        torch.tensor(
                            [generated_token_ids],
                            dtype=torch.long,
                            device=device,
                        ),
                    ],
                    dim=1,
                )
            else:
                next_input_ids = selected

            model_inputs = {
                "input_ids": next_input_ids,
                "use_cache": True,
                "return_dict": True,
            }
            if past_key_values is not None:
                model_inputs["past_key_values"] = past_key_values
            if self._supports_attention_mask:
                model_inputs["attention_mask"] = attention_mask

            outputs = self.model(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = getattr(outputs, "past_key_values", None)

        if (
            target_scored_pairs is not None
            and len(selected_pairs) < target_scored_pairs
        ):
            raise RuntimeError(
                "generation cap was reached before the requested number of "
                f"unique scored pairs: requested {target_scored_pairs}, "
                f"observed {len(selected_pairs)} after "
                f"{len(generated_token_ids)} tokens"
            )

        elapsed_seconds = time.perf_counter() - start
        text = self.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        roundtrip_token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        scored_steps = [step for step in steps if step.selected_green is not None]
        selected_pair_colors: dict[
            tuple[tuple[int, ...], int], bool
        ] = {}
        for step in scored_steps:
            if step.context is not None:
                selected_pair_colors.setdefault(
                    (step.context, step.selected_token_id),
                    bool(step.selected_green),
                )
        selected_green_token_num = sum(selected_pair_colors.values())
        scored_position_num = len(scored_steps)
        scored_token_num = len(selected_pair_colors)
        green_candidate_ranks = [
            step.selected_rank
            for step in scored_steps
            if step.selected_green is True
        ]
        selected_log_probabilities = [
            step.selected_log_probability
            for step in steps
            if step.selected_log_probability is not None
        ]
        log_probability_gaps = [
            step.log_probability_gap
            for step in steps
            if step.log_probability_gap is not None
        ]
        oracle_query_count = sum(
            step.oracle_query_count for step in steps
        )
        color_check_count = sum(
            len(step.checked_candidate_token_ids) for step in steps
        )
        fallback_count = sum(step.fallback for step in steps)
        oracle_time_seconds = sum(
            step.oracle_time_seconds for step in steps
        )
        generated_token_num = len(generated_token_ids)
        mean_selected_nll = (
            -sum(selected_log_probabilities)
            / len(selected_log_probabilities)
            if selected_log_probabilities
            else None
        )
        if mean_selected_nll is None:
            local_model_perplexity = None
        else:
            try:
                local_model_perplexity = math.exp(mean_selected_nll)
            except OverflowError:
                local_model_perplexity = float("inf")
        oracle_stats_after = self._oracle_stats()
        oracle_protocol_stats = None
        if oracle_stats_before is not None and oracle_stats_after is not None:
            oracle_protocol_stats = oracle_stats_after.difference(
                oracle_stats_before
            )

        return AdaptiveForgeryResult(
            text=text,
            token_ids=generated_token_ids,
            steps=steps,
            generated_token_num=generated_token_num,
            oracle_query_count=oracle_query_count,
            color_check_count=color_check_count,
            cache_hit_count=sum(step.cache_hit_count for step in steps),
            scored_position_num=scored_position_num,
            scored_token_num=scored_token_num,
            selected_green_token_num=selected_green_token_num,
            selected_green_ratio=(
                selected_green_token_num / scored_token_num
                if scored_token_num
                else 0.0
            ),
            fallback_count=fallback_count,
            unique_selected_pair_num=len(selected_pair_colors),
            duplicate_selected_pair_num=(
                scored_position_num - len(selected_pair_colors)
            ),
            elapsed_seconds=elapsed_seconds,
            oracle_time_seconds=oracle_time_seconds,
            non_oracle_time_seconds=max(
                0.0, elapsed_seconds - oracle_time_seconds
            ),
            tokens_per_second=(
                generated_token_num / elapsed_seconds
                if elapsed_seconds
                else 0.0
            ),
            queries_per_generated_token=(
                oracle_query_count / generated_token_num
                if generated_token_num
                else 0.0
            ),
            queries_per_scored_token=(
                oracle_query_count / scored_token_num
                if scored_token_num
                else 0.0
            ),
            color_checks_per_scored_token=(
                color_check_count / scored_token_num
                if scored_token_num
                else 0.0
            ),
            fallback_ratio=(
                fallback_count / scored_position_num
                if scored_position_num
                else 0.0
            ),
            mean_green_candidate_rank=(
                sum(green_candidate_ranks) / len(green_candidate_ranks)
                if green_candidate_ranks
                else None
            ),
            mean_selected_negative_log_likelihood=mean_selected_nll,
            local_model_perplexity=local_model_perplexity,
            mean_log_probability_gap=(
                sum(log_probability_gaps) / len(log_probability_gaps)
                if log_probability_gaps
                else None
            ),
            tokenization_preserved=roundtrip_token_ids == generated_token_ids,
            oracle_protocol_stats=oracle_protocol_stats,
        )

    def _model_device(self) -> torch.device:
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            return torch.device(model_device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _oracle_stats(self) -> ColorOracleStats | None:
        stats = getattr(self.oracle, "stats", None)
        return stats if isinstance(stats, ColorOracleStats) else None


def theoretical_green_probability(gamma: float, max_candidates: int) -> float:
    """Probability that at least one of k independent candidates is green."""

    if not 0 < gamma < 1:
        raise ValueError("gamma must be in the open interval (0, 1)")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    return 1 - (1 - gamma) ** max_candidates


def theoretical_queries_per_scored_token(
    gamma: float, max_candidates: int
) -> float:
    """Expected queries for a geometric search truncated after k candidates."""

    return theoretical_green_probability(gamma, max_candidates) / gamma
