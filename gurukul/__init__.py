"""Gurukul (गुरुकुल) — Zero-LLM Multilingual Offline FactBot & Benchmarking Library.

Provides 100% offline factual question answering, multi-turn dialogue management,
and DeepEval benchmark evaluation over custom JSON fact cards and school curricula.
"""

from __future__ import annotations

from gurukul.engine import OfflineFactBot
from gurukul.index import BM25Index, EmbeddingGemma2ONNX, FactCardIndex
from gurukul.models import (
    BatchDebugResponse,
    ConversationSession,
    DebugQueryResult,
    FactCard,
    FactSource,
    FollowUpChip,
    FollowUpIntent,
    OfflineChatResponse,
    OfflineQueryLog,
)
from gurukul.normalizer import (
    clean_text,
    detect_script,
    infer_preferred_language,
)


def run_benchmark(*args, **kwargs):
    """Convenience wrapper to run the DeepEval benchmark suite."""
    from gurukul.benchmarks import run_benchmark as _run

    return _run(*args, **kwargs)


__version__ = "0.2.0"

__all__ = [
    "OfflineFactBot",
    "FactCardIndex",
    "EmbeddingGemma2ONNX",
    "BM25Index",
    "FactCard",
    "FactSource",
    "FollowUpChip",
    "FollowUpIntent",
    "OfflineChatResponse",
    "ConversationSession",
    "OfflineQueryLog",
    "BatchDebugResponse",
    "DebugQueryResult",
    "clean_text",
    "detect_script",
    "infer_preferred_language",
    "run_benchmark",
    "__version__",
]
