from __future__ import annotations ## This keeps the type hints clean and easy to maintain.

from enum import Enum
from typing import Any 

from pydantic import BaseModel, Field

# adding the document chunks schema
class DocumentChunks(BaseModel):
    chunk_id: str
    document_id: str
    source_name: str
    text: str
    page_number: int | None=None
    metadata: dict[str, Any] = Field(default_factory=dict)

class RetrivedChunk(DocumentChunks):
    score: float
    rank: int

class RetrivalTrace(BaseModel):
    query: str
    tok_k: int
    chunks: list[RetrivedChunk]

class LatencyMetrics(BaseModel):
    embedding_ms: float = 0.0 
    retrieval_ms: float = 0.0
    prompt_construction_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

class RagResponse(BaseModel):
    answer: str
    citation: list[str]
    retrival: RetrivalTrace
    latency: LatencyMetrics

class IngestionResult(BaseModel):
    document_count: int
    chunk_count: int
    sources: list[str]

class FailureType(str, Enum):
    VOCABULARY_MISMATCH = 'vocabulary_mismatch'
    EXACT_IDENTIFIER = 'exact_indentifier'
    MULTI_HOP = "multi_hop"
    UNSUPPORTED_ANSWER = 'unsupported_answer'

class BenchmarkCase(BaseModel):
    id: str
    category: str
    question: str
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_source: str | None=None
    expected_sources: list[str] = Field(default_factory=list)
    must_abstain: bool = False

class BenchmarkResult(BaseModel):
    test_case_id: str
    category: str
    question: str
    actual_answer: str
    expected_source_retrieved: bool | None
    answer_contains_expected_text: bool | None
    citations_present: bool
    correct_source_cited: bool | None
    abstention_expected: bool
    abstention_observed: bool
    retrieved_chunks_ids: list[str]
    latency_ms: float
    passed: bool
    failure_reason: str | None=None