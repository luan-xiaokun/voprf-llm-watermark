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
        seed: int = 0,
        timing: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sk_path = sk_path
        self.pk_path = pk_path
        self.params_path = params_path
        self.signature_segment_length = signature_segment_length
        self.bit_size = bit_size
        self.message_length = message_length
        self.max_planted_errors = max_planted_errors
        self.timing = timing

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # this is only for downstream task evaluation
    def __call__(
        self,
        prompts: list[str],
        max_new_tokens: int,
        do_sample: bool = False,
        pad_token_id: int | None = None,
        stop_strings: list[str] | None = None,
        **kwargs,
    ) -> list[str]:
        sample_type = "multinomial" if do_sample else "argmax"
        texts = []
        for prompt in prompts:
            (generated_text, _, pk, params, _, _) = generate_text_asymmetric(
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
            texts.append(generated_text)
        return texts
