import inspect
import secrets
from typing import Callable

import torch
from transformers.generation import GenerationConfig, LogitsProcessorList
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import set_seed
from voprf_py import BlindedElement, EvaluationElement, Proof, PublicKey, VoprfServer

from rejection_sampling import (
    recover_sampling_mixin,
    replace_with_rejection_sampling_mixin,
)
from top_k_adaptive_sampling import TopkAdaptiveLogitsProcessor


class WatermarkAdapter:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        window_size: int,
        delta: float,
        gamma: float,
        seed: bytes,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.delta = delta
        self.gamma = gamma
        self.voprf_server = VoprfServer(seed)

        assert window_size > 0, "Window size must be greater than 0"
        assert delta > 0, "Delta must be greater than 0"
        assert 0 < gamma < 1, "Gamma must be in the range (0, 1)"

    def get_public_key(self) -> PublicKey:
        return self.voprf_server.get_public_key()

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
        if top_k is None and not no_watermark:
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
            logits_processor = LogitsProcessorList(
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
            top_k = None  # reset top_k to None for the generation config

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

    def get_server_interface(
        self,
    ) -> Callable[[list[BlindedElement]], tuple[list[EvaluationElement], Proof]]:
        def server_interface(
            blinded_elements: list[BlindedElement],
        ) -> tuple[list[EvaluationElement], Proof]:
            return self.voprf_server.batch_blind_evaluate(blinded_elements)

        return server_interface


def generate_watermark_master_seed() -> bytes:
    return secrets.token_bytes(32)
