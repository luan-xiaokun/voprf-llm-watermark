import hashlib
import random

import numpy as np
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..adapter import WatermarkAdapter
from .generate import generate_text_asymmetric

DEFAULT_PK_PATH = "data/pdw/pk"
DEFAULT_SK_PATH = "data/pdw/sk"
DEFAULT_PARAMS_PATH = "data/pdw/params"

_RETRYABLE_GENERATION_ERRORS = (
    "tried to plant another error but already at max_planted_errors",
    "signature segment",
    "sample_token took too long to sample a valid token",
)


def _attempt_seed(
    base_seed: int,
    prompt_index: int,
    attempt_index: int,
) -> int:
    payload = f"pdw:{base_seed}:{prompt_index}:{attempt_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _seed_generators(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _is_retryable_generation_error(error: ValueError) -> bool:
    message = str(error)
    return any(fragment in message for fragment in _RETRYABLE_GENERATION_ERRORS)


class PDWAdapter(WatermarkAdapter):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        sk_path: str | None = DEFAULT_SK_PATH,
        pk_path: str | None = DEFAULT_PK_PATH,
        params_path: str | None = DEFAULT_PARAMS_PATH,
        signature_segment_length: int = 16,
        bit_size: int = 2,
        message_length: int = 8,
        max_planted_errors: int = 2,
        max_generation_attempts: int = 3,
        seed: int = 0,
        timing: bool = False,
    ):
        if max_generation_attempts <= 0:
            raise ValueError("max_generation_attempts must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.sk_path = sk_path
        self.pk_path = pk_path
        self.params_path = params_path
        self.signature_segment_length = signature_segment_length
        self.bit_size = bit_size
        self.message_length = message_length
        self.max_planted_errors = max_planted_errors
        self.max_generation_attempts = max_generation_attempts
        self.seed = seed
        self.timing = timing

    # this is only for downstream task evaluation
    def __call__(
        self,
        prompts: list[str],
        max_new_tokens: int,
        do_sample: bool = False,
        pad_token_id: int | None = None,
        stop_strings: list[str] | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[str]:
        del do_sample, pad_token_id, stop_strings, kwargs, max_new_tokens
        sample_type = "multinomial"
        texts = []
        base_seed = self.seed if seed is None else seed
        for prompt_index, prompt in enumerate(prompts):
            for attempt_index in range(self.max_generation_attempts):
                attempt_seed = _attempt_seed(
                    base_seed,
                    prompt_index,
                    attempt_index,
                )
                _seed_generators(attempt_seed)
                try:
                    generated_text, *_ = generate_text_asymmetric(
                        prompt,
                        self.model,
                        self.tokenizer,
                        sample_type,
                        self.message_length,
                        self.signature_segment_length,
                        self.bit_size,
                        self.max_planted_errors,
                        self.sk_path,
                        self.pk_path,
                        self.params_path,
                        continue_until_stop_token=False,
                        print_timing=self.timing,
                    )
                except ValueError as error:
                    if not _is_retryable_generation_error(error):
                        raise
                    if attempt_index + 1 == self.max_generation_attempts:
                        raise RuntimeError(
                            "PDW generation failed for prompt index "
                            f"{prompt_index} after "
                            f"{self.max_generation_attempts} attempts: "
                            f"{error}"
                        ) from error
                    continue
                break
            texts.append(generated_text)
        return texts
