import re

import numpy as np
import pandas as pd
import torch
from transformers.generation import LogitsProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from watermark_suite.schemes import WatermarkAdapter


class GreenCache:
    def __init__(self, green_cache_path: str) -> None:
        self.green_cache = pd.read_parquet(green_cache_path, engine="pyarrow")

    def query(self, context: tuple[int, int, int, int]) -> np.ndarray | None:
        try:
            return self.green_cache.loc[context]["tokens"]
        except KeyError:
            return None

    def batch_query(
        self, contexts: list[tuple[int, int, int, int]]
    ) -> list[np.ndarray | None]:
        result = self.green_cache.reindex(contexts)["tokens"]
        return [None if np.any(pd.isna(value)) else value for value in result]


class LearningLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        window_size: int,
        delta: float,
        green_cache: GreenCache,
    ) -> None:
        self.window_size = window_size
        self.delta = delta
        self.green_cache = green_cache
        self.cache_hit = 0

    @torch.no_grad()
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if scores.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            scores = scores.unsqueeze(0)
        elif scores.dim() != 2:
            raise ValueError(f"Unsupported scores dimension: {scores.dim()}")

        batch_size, _ = scores.shape

        contexts = [tuple(c) for c in input_ids[:, -self.window_size :].tolist()]
        cached_green_tokens = self.green_cache.batch_query(contexts)

        for i in range(batch_size):
            if cached_green_tokens[i] is not None:
                self.cache_hit += 1
                green_token_index = torch.from_numpy(cached_green_tokens[i])
                scores[i][green_token_index] += self.delta

        return scores


class LearningAdapter(WatermarkAdapter):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        window_size: int,
        delta: float,
        green_cache: GreenCache,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.delta = delta
        self.green_cache = green_cache
        self.logits_processor = LearningLogitsProcessor(
            window_size=window_size, delta=delta, green_cache=green_cache
        )

        assert window_size > 0, "Window size must be greater than 0"
        assert delta > 0, "Delta must be greater than 0"

    def get_logits_processor(self, do_sample, num_beams, top_k):
        return self.logits_processor

    def _post_generation(self):
        print(f"Cache hit: {self.logits_processor.cache_hit}")
