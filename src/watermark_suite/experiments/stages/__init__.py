from .adaptive_forgery import AdaptiveForgeryStageAdapter
from .detection import DetectionStageAdapter
from .generation import GenerationStageAdapter
from .perplexity import PerplexityStageAdapter


def default_stage_adapters():
    adapters = (
        GenerationStageAdapter(),
        AdaptiveForgeryStageAdapter(),
        DetectionStageAdapter(),
        PerplexityStageAdapter(),
    )
    return {adapter.kind: adapter for adapter in adapters}


__all__ = [
    "AdaptiveForgeryStageAdapter",
    "DetectionStageAdapter",
    "GenerationStageAdapter",
    "PerplexityStageAdapter",
    "default_stage_adapters",
]
