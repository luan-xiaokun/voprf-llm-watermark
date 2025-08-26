from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class DetectionError(Exception):
    pass


@dataclass
class DetectionResult:
    total_token_num: int
    p_value: float


@dataclass
class DetectionCost:
    token_num: int
    total_time: float


class WatermarkDetector:
    tokenizer: PreTrainedTokenizerBase
    seed: int

    def detect(
        self, text: str, token_num: int | None = None, *args, **kwargs
    ) -> DetectionResult:
        pass

    def batch_detect(
        self, texts: list[str], token_num: int | None = None, *args, **kwargs
    ) -> tuple[list[DetectionResult], list[DetectionCost]]:
        pass
