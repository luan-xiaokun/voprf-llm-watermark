from .adaptive_forgery import AdaptiveForgeryStageAdapter
from .detection import DetectionStageAdapter
from .downstream import DownstreamStageAdapter
from .generation import GenerationStageAdapter
from .null_detection import NullDetectionStageAdapter
from .perplexity import PerplexityStageAdapter
from .result_aggregation import ResultAggregationStageAdapter
from .robustness import RobustnessStageAdapter
from .text_evaluation import TextEvaluationStageAdapter
from .token_window_corpus import TokenWindowCorpusStageAdapter


def default_stage_adapters():
    adapters = (
        GenerationStageAdapter(),
        NullDetectionStageAdapter(),
        AdaptiveForgeryStageAdapter(),
        DetectionStageAdapter(),
        PerplexityStageAdapter(),
        DownstreamStageAdapter(),
        RobustnessStageAdapter(),
        TextEvaluationStageAdapter(),
        TokenWindowCorpusStageAdapter(),
        ResultAggregationStageAdapter(),
    )
    return {adapter.kind: adapter for adapter in adapters}


__all__ = [
    "AdaptiveForgeryStageAdapter",
    "DetectionStageAdapter",
    "DownstreamStageAdapter",
    "GenerationStageAdapter",
    "NullDetectionStageAdapter",
    "PerplexityStageAdapter",
    "ResultAggregationStageAdapter",
    "RobustnessStageAdapter",
    "TextEvaluationStageAdapter",
    "TokenWindowCorpusStageAdapter",
    "default_stage_adapters",
]
