from .adaptive_forgery import AdaptiveForgeryStageAdapter
from .detection import DetectionStageAdapter
from .downstream import DownstreamStageAdapter
from .generation import GenerationStageAdapter
from .perplexity import PerplexityStageAdapter
from .result_aggregation import ResultAggregationStageAdapter
from .robustness import RobustnessStageAdapter
from .text_evaluation import TextEvaluationStageAdapter


def default_stage_adapters():
    adapters = (
        GenerationStageAdapter(),
        AdaptiveForgeryStageAdapter(),
        DetectionStageAdapter(),
        PerplexityStageAdapter(),
        DownstreamStageAdapter(),
        RobustnessStageAdapter(),
        TextEvaluationStageAdapter(),
        ResultAggregationStageAdapter(),
    )
    return {adapter.kind: adapter for adapter in adapters}


__all__ = [
    "AdaptiveForgeryStageAdapter",
    "DetectionStageAdapter",
    "DownstreamStageAdapter",
    "GenerationStageAdapter",
    "PerplexityStageAdapter",
    "ResultAggregationStageAdapter",
    "RobustnessStageAdapter",
    "TextEvaluationStageAdapter",
    "default_stage_adapters",
]
