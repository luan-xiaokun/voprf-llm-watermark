import inspect
import struct
from typing import Callable

import torch
from transformers.generation import (
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import set_seed
from voprf_py import BlindedElement, EvaluationElement, Proof, PublicKey, VoprfServer

from ..adapter import GpuTimingLogitsProcessor, WatermarkAdapter
from .rejection_sampling import (
    recover_sampling_mixin,
    replace_with_rejection_sampling_mixin,
)
from .top_k_adaptive_sampling import TopkAdaptiveLogitsProcessor


class NaiveLogitsProcessor(LogitsProcessor):
    def __init__(
        self, window_size: int, delta: float, gamma: float, voprf_server: VoprfServer
    ):
        self.window_size = window_size
        self.delta = delta
        self.gamma = gamma
        self.voprf_server = voprf_server

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            logits = logits.unsqueeze(0)
        elif logits.dim() != 2:
            raise ValueError(f"Unsupported scores dimension: {logits.dim()}")

        device = input_ids.device
        _, vocab_size = logits.shape

        context_bytes_list = [
            struct.pack(f">{len(ctx)}I", *ctx)
            for ctx in input_ids[:, -self.window_size :].tolist()
        ]
        vocab_bytes = [struct.pack(">I", token_id) for token_id in range(vocab_size)]

        allowed_tokens_batch = []
        batch_indices = []
        for i, context_bytes in enumerate(context_bytes_list):
            msg_inputs = [context_bytes + token_bytes for token_bytes in vocab_bytes]
            msg_hashes = self.voprf_server.batch_evaluate(msg_inputs)
            allowed_tokens = [
                j
                for j, h in enumerate(msg_hashes)
                if int.from_bytes(h, "big") / (1 << 8 * len(h)) < self.gamma * (2**256)
            ]
            allowed_tokens = torch.tensor(allowed_tokens, dtype=torch.long)
            allowed_tokens_batch.append(allowed_tokens)
            batch_indices.append(torch.full_like(allowed_tokens, i))

        allowed_tokens_concat = torch.cat(allowed_tokens_batch).to(device)
        batch_indices_concat = torch.cat(batch_indices).to(device)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[batch_indices_concat, allowed_tokens_concat] = False
        logits[mask] += self.delta

        return logits


class VOWAdapter(WatermarkAdapter):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        window_size: int,
        delta: float,
        gamma: float,
        seed: bytes,
        naive_baseline: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.delta = delta
        self.gamma = gamma
        self.voprf_server = VoprfServer(seed)
        self.seed = int.from_bytes(seed, "big")
        self.naive_baseline = naive_baseline

        assert window_size > 0, "Window size must be greater than 0"
        assert delta > 0, "Delta must be greater than 0"
        assert 0 < gamma < 1, "Gamma must be in the range (0, 1)"

        self.original_sample = None

    def get_public_key(self) -> PublicKey:
        return self.voprf_server.get_public_key()

    def get_server_interface(
        self,
    ) -> Callable[[list[BlindedElement]], tuple[list[EvaluationElement], Proof]]:
        def server_interface(
            blinded_elements: list[BlindedElement],
        ) -> tuple[list[EvaluationElement], Proof]:
            return self.voprf_server.batch_blind_evaluate(blinded_elements)

        return server_interface

    def get_logits_processor(self, do_sample, num_beams, top_k):
        if self.naive_baseline:
            return LogitsProcessorList(
                [
                    NaiveLogitsProcessor(
                        window_size=self.window_size,
                        delta=self.delta,
                        gamma=self.gamma,
                        voprf_server=self.voprf_server,
                    )
                ]
            )

        if not do_sample:
            top_k = 1
            do_sample = True

        if top_k is not None:
            return LogitsProcessorList(
                [
                    TopkAdaptiveLogitsProcessor(
                        window_size=self.window_size,
                        delta=self.delta,
                        gamma=self.gamma,
                        top_k=top_k,
                        voprf_server=self.voprf_server,
                    )
                ]
            )

    def get_generation_config(
        self,
        max_new_tokens: int | None,
        stop_strings: str | list[str] | None,
        do_sample: bool,
        num_beams: int,
        top_p: float | None,
        top_k: int | None,
        temperature: float | None,
        suppress_tokens: list[int] | None,
        no_watermark: bool,
    ) -> GenerationConfig:
        # several decoding strategies:
        # 1. greedy decoding, equivalent to top-1 sampling
        # 2. no top-k multinomial sampling
        # 3. top-k multinomial sampling
        # the top-k effect is already achieved by the logits processor
        # so we set `top_k` to None in the generation config
        if not no_watermark:
            do_sample = True
            top_k = None

        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            suppress_tokens=suppress_tokens,
        )

    def _pre_generation(
        self,
        do_sample: bool,
        num_beams: int,
        top_p: float | None,
        top_k: float | None,
        no_watermark: bool,
    ) -> None:
        if num_beams > 1:
            raise NotImplementedError("Beam search is not supported in this adapter.")

        if self.naive_baseline:
            return

        if not do_sample:
            top_k = 1
            do_sample = True

        if top_k is None and not no_watermark:
            # here, the _sample method of self.model is replaced
            self.original_sample = replace_with_rejection_sampling_mixin(
                self.model,
                self.window_size,
                self.delta,
                self.gamma,
                self.voprf_server,
            )

    def _post_generation(self):
        if self.original_sample is not None:
            recover_sampling_mixin(self.model, self.original_sample)
            self.original_sample = None

    def __call__(
        self,
        prompts: str,
        num_samples: int = 1,
        max_new_tokens: int | None = None,
        stop_strings: str | list[str] | None = None,
        seed: int | None = None,
        do_sample: bool = False,
        num_beams: int = 1,
        top_p: float | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        suppress_tokens: list[int] | None = None,
        no_watermark: bool = False,
        **model_specific_params,
    ) -> str | list[str]:
        prompts_ = prompts
        if isinstance(prompts, str):
            prompts_ = [prompts]

        batch_encoding = self.tokenizer(prompts_, return_tensors="pt", padding=True)
        input_ids: torch.LongTensor = batch_encoding["input_ids"]
        attention_mask: torch.LongTensor = batch_encoding["attention_mask"]
        inputs = {
            "input_ids": input_ids.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
        }
        if (
            "attention_mask"
            not in inspect.signature(self.model.forward).parameters.keys()
        ):
            del inputs["attention_mask"]

        if seed is not None:
            set_seed(seed)

        # # currently we do not support top-p sampling for watermarking
        # if top_p is not None and not no_watermark:
        #     raise NotImplementedError(
        #         "Top-p sampling is not supported in this adapter."
        #     )

        # beam search is also not supported, yet
        if num_beams > 1:
            raise NotImplementedError("Beam search is not supported in this adapter.")

        # greedy decoding is equivalent to top-1 sampling
        if not do_sample:
            top_k = 1
            do_sample = True

        # multinomial sampling
        original_sample = None
        if top_k is None and not no_watermark and not self.naive_baseline:
            # here, the _sample method of self.model is replaced
            original_sample = replace_with_rejection_sampling_mixin(
                self.model,
                self.window_size,
                self.delta,
                self.gamma,
                self.voprf_server,
            )

        # prepare for top-k adaptive sampling
        logits_processor = None
        if do_sample and top_k is not None and not no_watermark:
            logits_processor = TopkAdaptiveLogitsProcessor(
                window_size=self.window_size,
                delta=self.delta,
                gamma=self.gamma,
                top_k=top_k,
                voprf_server=self.voprf_server,
            )
            top_k = None  # reset top_k to None for the generation config
        if self.naive_baseline:
            logits_processor = NaiveLogitsProcessor(
                window_size=self.window_size,
                delta=self.delta,
                gamma=self.gamma,
                voprf_server=self.voprf_server,
            )
        timing_logits_processor = GpuTimingLogitsProcessor(logits_processor)
        logits_processor = LogitsProcessorList([timing_logits_processor])

        generation_config = GenerationConfig(
            num_return_sequences=num_samples,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            suppress_tokens=suppress_tokens,
        )

        output_ids = self.model.generate(
            **inputs,
            generation_config=generation_config,
            tokenizer=self.tokenizer,
            logits_processor=logits_processor,
            **model_specific_params,
        )

        if self.timing and isinstance(
            timing_logits_processor, GpuTimingLogitsProcessor
        ):
            print(timing_logits_processor.get_results())

        # recover the original _sample method if it was replaced
        if original_sample is not None:
            recover_sampling_mixin(self.model, original_sample)

        if self.model.config.is_encoder_decoder:
            generated_ids = output_ids
        else:
            generated_ids = output_ids[:, input_ids.shape[1] :]

        if num_samples > 1 and isinstance(prompts, list):
            generated_ids = generated_ids.view(input_ids.size(0), num_samples, -1)

        if isinstance(prompts, str):
            generated_ids = generated_ids.squeeze(0)

        return self._decode_generation(generated_ids)

    def _decode_generation(self, generated_ids: torch.LongTensor) -> str | list[str]:
        if generated_ids.dim() == 1:
            return self.tokenizer.decode(
                generated_ids.tolist(), skip_special_tokens=True
            )
        if generated_ids.dim() == 2:
            return [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in generated_ids.tolist()
            ]
        if generated_ids.dim() == 3:
            return [
                self.tokenizer.decode(
                    generated_ids[i].tolist(), skip_special_tokens=True
                )
                for i in range(len(generated_ids))
            ]
        raise TypeError(
            f"Generated outputs aren't 1D, 2D or 3D, but instead are {generated_ids.shape}"
        )

    def __getattr__(self, name: str):
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
