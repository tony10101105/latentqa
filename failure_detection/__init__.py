"""
Failure detection module for identifying model failures and spurious feature usage.
"""

from .benchmarks import BENCHMARKS, load_benchmark, get_benchmark_by_name
from .model_evaluator import ModelEvaluator
from .judge_evaluator import JudgeEvaluator
from .failure_collector import FailureCollector
from .pipeline import FailureDetectionPipeline

__all__ = [
    "BENCHMARKS",
    "load_benchmark",
    "get_benchmark_by_name",
    "ModelEvaluator",
    "JudgeEvaluator",
    "FailureCollector",
    "FailureDetectionPipeline",
]
