"""DeepEval LLM-as-a-judge benchmarking suite for Gurukul offline assistant."""

from __future__ import annotations

from gurukul.benchmarks.rubrics import get_deepeval_metrics
from gurukul.benchmarks.run_benchmark import run, run as run_benchmark, execute_queries
from gurukul.benchmarks.settings import settings

__all__ = [
    "run",
    "run_benchmark",
    "execute_queries",
    "get_deepeval_metrics",
    "settings",
]
