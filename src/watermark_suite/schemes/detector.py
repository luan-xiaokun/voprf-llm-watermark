from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class DetectionError(Exception):
    pass


@dataclass(kw_only=True)
class DetectionResult:
    """
    Standardized result for watermark detection.

    Attributes:
        total_token_num: Total number of tokens processed.
        p_value: The final p-value of the detection test, or ``None`` when
            the detector exposes a non-probabilistic decision score.
        step_size: If incremental detection was requested, the step size used.
        milestones: List of token counts corresponding to incremental steps.
        step_p_values: List of p-values at each milestone.
        step_scores: List of raw scores (e.g., z-score, confidence) at each milestone.
    """

    total_token_num: int
    p_value: float | None
    step_size: int | None = None
    milestones: list[int] | None = None
    step_p_values: list[float] | None = None
    step_scores: list[float] | None = None


@dataclass
class DetectionCost:
    token_num: int
    total_time: float


class WatermarkDetector:
    tokenizer: PreTrainedTokenizerBase
    seed: int

    def detect(
        self,
        text: str,
        token_num: int | None = None,
        step_size: int | None = None,
        *args,
        **kwargs,
    ) -> DetectionResult:
        pass

    def batch_detect(
        self,
        texts: list[str],
        token_num: int | None = None,
        step_size: int | None = None,
        *args,
        **kwargs,
    ) -> tuple[list[DetectionResult], list[DetectionCost]]:
        pass
