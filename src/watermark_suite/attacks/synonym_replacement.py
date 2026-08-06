from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import nltk
from nltk.tokenize import TreebankWordTokenizer


_CONTENT_POS_PREFIXES = {"N", "V", "J", "R"}


@dataclass(frozen=True)
class SynonymReplacementResult:
    text: str
    eligible_word_count: int
    selected_word_count: int
    replaced_word_count: int
    failed_replacement_count: int
    target_replacement_rate: float
    realized_replacement_rate: float
    mean_selected_candidate_rank: float | None

    def provenance(self) -> dict[str, int | float | None]:
        return {
            "eligible_word_count": self.eligible_word_count,
            "selected_word_count": self.selected_word_count,
            "replaced_word_count": self.replaced_word_count,
            "failed_replacement_count": self.failed_replacement_count,
            "target_replacement_rate": self.target_replacement_rate,
            "realized_replacement_rate": self.realized_replacement_rate,
            "mean_selected_candidate_rank": (
                self.mean_selected_candidate_rank
            ),
        }


@dataclass
class _PreparedText:
    text: str
    tokens: list[str]
    spans: list[tuple[int, int]]
    tags: list[str]
    eligible_indices: list[int]
    selected_indices: list[int]
    rng: random.Random
    replacements: dict[int, str]
    selected_ranks: list[int]


@dataclass(frozen=True)
class _MaskedRequest:
    text_index: int
    token_index: int
    original_word: str
    original_pos: str
    masked_context: str


class SynonymReplacer:
    """Contextual masked-LM replacement with reproducible sampling.

    The experiment runtime owns model loading.  This class only implements
    the attack policy, which keeps model provenance and GPU residency under
    the Experiment Run architecture.
    """

    def __init__(
        self,
        tokenizer: Any,
        unmasker: Callable[..., Any],
        *,
        pos_tagger: Callable[[list[str]], list[tuple[str, str]]] | None = None,
        model_max_length: int | None = None,
    ) -> None:
        if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
            raise ValueError("masked-LM tokenizer must define a mask token")
        self.tokenizer = tokenizer
        self.unmasker = unmasker
        self.pos_tagger = pos_tagger or nltk.pos_tag
        self.word_tokenizer = TreebankWordTokenizer()
        self.model_max_length = self._resolve_model_max_length(
            model_max_length
        )

    def _resolve_model_max_length(self, explicit: int | None) -> int:
        candidates = [explicit]
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        candidates.append(tokenizer_limit)
        model = getattr(self.unmasker, "model", None)
        config = getattr(model, "config", None)
        candidates.append(getattr(config, "max_position_embeddings", None))
        valid = [
            value
            for value in candidates
            if isinstance(value, int) and 2 < value < 100_000
        ]
        return min(valid) if valid else 512

    @staticmethod
    def _coarse_pos(tag: str) -> str | None:
        prefix = tag[:1]
        return prefix if prefix in _CONTENT_POS_PREFIXES else None

    def _tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        try:
            return self.pos_tagger(tokens)
        except LookupError as error:
            raise RuntimeError(
                "NLTK POS tagger data is unavailable. Install it before the "
                "run with `python -m nltk.downloader "
                "averaged_perceptron_tagger_eng`."
            ) from error

    def _masked_context(self, text: str, span: tuple[int, int]) -> str:
        prefix_ids = self.tokenizer.encode(
            text[: span[0]], add_special_tokens=False
        )
        suffix_ids = self.tokenizer.encode(
            text[span[1] :], add_special_tokens=False
        )
        special_token_num = self.tokenizer.num_special_tokens_to_add(
            pair=False
        )
        context_budget = max(self.model_max_length - special_token_num - 1, 0)
        left_num = min(len(prefix_ids), context_budget // 2)
        right_num = min(len(suffix_ids), context_budget - left_num)
        left_num = min(
            len(prefix_ids), context_budget - right_num
        )
        while True:
            selected_prefix = prefix_ids[-left_num:] if left_num else []
            token_ids = [
                *selected_prefix,
                self.tokenizer.mask_token_id,
                *suffix_ids[:right_num],
            ]
            masked_context = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            round_trip_ids = self.tokenizer.encode(
                masked_context,
                add_special_tokens=False,
            )
            mask_count = round_trip_ids.count(
                self.tokenizer.mask_token_id
            )
            if mask_count != 1:
                raise RuntimeError(
                    "masked context must contain exactly one mask token "
                    f"after decode/re-tokenize; found {mask_count}"
                )
            encoded_length = len(round_trip_ids) + special_token_num
            if encoded_length <= self.model_max_length:
                return masked_context

            retained_context_num = left_num + right_num
            if retained_context_num == 0:
                raise RuntimeError(
                    "masked token alone exceeds the masked-LM position "
                    f"limit of {self.model_max_length}"
                )
            trim_num = min(
                encoded_length - self.model_max_length,
                retained_context_num,
            )
            for _ in range(trim_num):
                if left_num >= right_num and left_num:
                    left_num -= 1
                else:
                    right_num -= 1

    def _prepare_text(
        self,
        text: str,
        *,
        replacement_rate: float,
        seed: int,
    ) -> _PreparedText:
        spans = list(self.word_tokenizer.span_tokenize(text))
        tokens = [text[start:end] for start, end in spans]
        tagged = self._tag(tokens)
        tags = [tag for _, tag in tagged]
        eligible_indices = [
            index
            for index, (word, tag) in enumerate(zip(tokens, tags))
            if len(word) >= 3
            and word.isalpha()
            and self._coarse_pos(tag) is not None
        ]
        rng = random.Random(seed)
        selected_num = int(len(eligible_indices) * replacement_rate)
        selected_indices = sorted(
            rng.sample(eligible_indices, selected_num)
        )
        return _PreparedText(
            text=text,
            tokens=tokens,
            spans=spans,
            tags=tags,
            eligible_indices=eligible_indices,
            selected_indices=selected_indices,
            rng=rng,
            replacements={},
            selected_ranks=[],
        )

    @staticmethod
    def _preserve_case(candidate: str, original: str) -> str:
        if original.isupper():
            return candidate.upper()
        if original.istitle():
            return candidate.capitalize()
        return candidate.lower() if original.islower() else candidate

    def _candidate_matches_pos(
        self,
        prepared: _PreparedText,
        request: _MaskedRequest,
        candidate: str,
    ) -> bool:
        start = max(request.token_index - 4, 0)
        end = min(request.token_index + 5, len(prepared.tokens))
        context = prepared.tokens[start:end]
        relative_index = request.token_index - start
        context[relative_index] = candidate
        candidate_tag = self._tag(context)[relative_index][1]
        return self._coarse_pos(candidate_tag) == self._coarse_pos(
            request.original_pos
        )

    def _valid_candidates(
        self,
        prepared: _PreparedText,
        request: _MaskedRequest,
        predictions: Sequence[dict[str, Any]],
    ) -> list[tuple[str, float, int]]:
        valid = []
        for rank, prediction in enumerate(predictions, start=1):
            raw = str(prediction.get("token_str", ""))
            candidate = raw.strip()
            score = prediction.get("score", 0.0)
            if (
                not candidate
                or raw.lstrip().startswith("##")
                or not candidate.isalpha()
                or candidate.casefold() == request.original_word.casefold()
                or not isinstance(score, (int, float))
                or score < 0
                or not self._candidate_matches_pos(
                    prepared, request, candidate
                )
            ):
                continue
            valid.append((candidate, float(score), rank))
        return valid

    @staticmethod
    def _prediction_batches(
        predictions: Any,
        request_num: int,
    ) -> list[list[dict[str, Any]]]:
        if request_num == 1 and predictions and isinstance(predictions[0], dict):
            return [predictions]
        if not isinstance(predictions, list) or len(predictions) != request_num:
            raise RuntimeError(
                "fill-mask pipeline returned an unexpected number of results"
            )
        if not all(isinstance(value, list) for value in predictions):
            raise RuntimeError("fill-mask pipeline returned malformed results")
        return predictions

    def replace_many(
        self,
        texts: Sequence[str],
        *,
        seeds: Sequence[int],
        replacement_rate: float,
        top_k: int = 15,
        candidate_sampling: str = "score-weighted",
        batch_size: int = 16,
    ) -> list[SynonymReplacementResult]:
        if len(texts) != len(seeds):
            raise ValueError("texts and seeds must have the same length")
        if not 0 < replacement_rate < 1:
            raise ValueError("replacement_rate must be between zero and one")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if candidate_sampling != "score-weighted":
            raise ValueError(
                "candidate_sampling must be score-weighted"
            )
        prepared = [
            self._prepare_text(
                text,
                replacement_rate=replacement_rate,
                seed=seed,
            )
            for text, seed in zip(texts, seeds)
        ]
        requests = [
            _MaskedRequest(
                text_index=text_index,
                token_index=token_index,
                original_word=value.tokens[token_index],
                original_pos=value.tags[token_index],
                masked_context=self._masked_context(
                    value.text, value.spans[token_index]
                ),
            )
            for text_index, value in enumerate(prepared)
            for token_index in value.selected_indices
        ]
        if requests:
            raw_predictions = self.unmasker(
                [request.masked_context for request in requests],
                top_k=top_k,
                batch_size=batch_size,
            )
            prediction_batches = self._prediction_batches(
                raw_predictions, len(requests)
            )
            for request, predictions in zip(requests, prediction_batches):
                value = prepared[request.text_index]
                candidates = self._valid_candidates(
                    value, request, predictions
                )
                if not candidates:
                    continue
                weights = [candidate[1] for candidate in candidates]
                if not any(weights):
                    weights = None
                candidate, _, rank = value.rng.choices(
                    candidates, weights=weights, k=1
                )[0]
                value.replacements[request.token_index] = self._preserve_case(
                    candidate, request.original_word
                )
                value.selected_ranks.append(rank)

        results = []
        for value in prepared:
            transformed = value.text
            for token_index, replacement in sorted(
                value.replacements.items(), reverse=True
            ):
                start, end = value.spans[token_index]
                transformed = transformed[:start] + replacement + transformed[end:]
            eligible_num = len(value.eligible_indices)
            selected_num = len(value.selected_indices)
            replaced_num = len(value.replacements)
            results.append(
                SynonymReplacementResult(
                    text=transformed,
                    eligible_word_count=eligible_num,
                    selected_word_count=selected_num,
                    replaced_word_count=replaced_num,
                    failed_replacement_count=selected_num - replaced_num,
                    target_replacement_rate=replacement_rate,
                    realized_replacement_rate=(
                        replaced_num / eligible_num if eligible_num else 0.0
                    ),
                    mean_selected_candidate_rank=(
                        sum(value.selected_ranks) / len(value.selected_ranks)
                        if value.selected_ranks
                        else None
                    ),
                )
            )
        return results
