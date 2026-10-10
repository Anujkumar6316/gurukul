from __future__ import annotations
from typing import  Protocol

from app.shared.models import DocumentChunks, RetrivedChunk

class EmbeddingProvider(Protocol):
    dimension: int
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...

class VectorStore(Protocol):
    def upsert(self, chunks: list[DocumentChunks], vectors: list[list[float]])-> None: ...
    def search(self, vector_query: list[float], tok_k:int) -> list[list[RetrivedChunk]]: ...

class GenerationProvider(Protocol):
    def generate(self, prompt: str) -> str: ...

