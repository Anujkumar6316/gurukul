from __future__ import annotations
from path import Path
from pydantic.settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root/"data"/"sample_data"
    artifacts_dir: Path = project_root/"artifacts"

    collection_name: str = "baseline_rag"
    embeddings_model: str = "sentence-transformers/all-mini"
    embedding_backend: str = "hash"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.3"

    generation_backend: str = "fixture"

    chunk_size: int=500
    chunk_overlap: int = 50
    tok_k: int = 3

settings = Settings()