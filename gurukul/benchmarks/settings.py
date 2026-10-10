"""Global configuration settings for Gurukul DeepEval Benchmark Suite.

Centralizes paths, default judge models, OpenAI-compatible endpoint configurations,
metric scoring thresholds, and artifact filenames.
Automatically loads environment variables from .env files.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field


def _load_env_file(env_path: Path) -> None:
    """Lightweight zero-dependency .env loader."""
    if not env_path.is_file():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val


_benchmark_dir = Path(__file__).resolve().parent
_project_root = _benchmark_dir.parent.parent

_load_env_file(_benchmark_dir / ".env")
_load_env_file(_project_root / ".env")


class BenchmarkSettings(BaseModel):
    """Global configuration settings for the benchmark package."""

    project_root: Path = Field(default_factory=lambda: _project_root)
    artifact_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "GURUKUL_ARTIFACT_DIR",
                "/home/mindmap/.gemini/antigravity/brain/9c1fac78-044a-407d-ac16-cbf96b100902",
            )
        )
    )
    local_artifact_dir: Path = Field(
        default_factory=lambda: _project_root / "artifacts"
    )

    # OpenAI-Compatible Provider Configuration
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_base_url: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL")
    )
    openai_model_id: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL_ID")
    )

    # Google Gemini Configuration
    gemini_api_key: str | None = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )

    # Judge Model Defaults
    default_judge: str = Field(
        default_factory=lambda: os.getenv("BENCHMARK_JUDGE")
        or os.getenv("OPENAI_MODEL_ID")
        or "gemini-1.5-flash"
    )
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # Concurrency: parallel worker count for offline inference and judge evaluation
    benchmark_workers: int = Field(
        default_factory=lambda: int(
            os.getenv("BENCHMARK_WORKERS") or min(8, os.cpu_count() or 4)
        )
    )
    benchmark_workers: int = Field(
        default_factory=lambda: int(os.getenv("BENCHMARK_WORKERS", "4"))
    )

    @property
    def effective_judge(self) -> str:
        """Active judge model, preferring openai_model_id if explicitly set."""
        return self.openai_model_id or self.default_judge

    # Scoring Thresholds
    language_consistency_threshold: float = Field(
        default_factory=lambda: float(os.getenv("BENCHMARK_THRESHOLD_LANG", "0.8"))
    )
    follow_up_offer_threshold: float = Field(
        default_factory=lambda: float(os.getenv("BENCHMARK_THRESHOLD_OFFER", "0.8"))
    )
    curriculum_grounding_threshold: float = Field(
        default_factory=lambda: float(os.getenv("BENCHMARK_THRESHOLD_GROUNDING", "0.7"))
    )
    abstention_accuracy_threshold: float = Field(
        default_factory=lambda: float(os.getenv("BENCHMARK_THRESHOLD_ABSTENTION", "0.7"))
    )

    # Artifact Filenames
    default_batch_input_name: str = "batch_debug_results.json"
    default_report_name: str = "benchmark_report.md"
    default_html_report_name: str = "benchmark_report.html"
    default_scorecard_name: str = "benchmark_scorecard.json"

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "ignore",
    }

    def apply_environment(self) -> None:
        """Ensures active configuration is synced into os.environ for client SDKs."""
        if self.openai_base_url:
            os.environ["OPENAI_BASE_URL"] = self.openai_base_url
        if self.openai_api_key and "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.gemini_api_key and "GEMINI_API_KEY" not in os.environ:
            os.environ["GEMINI_API_KEY"] = self.gemini_api_key


settings = BenchmarkSettings()
settings.apply_environment()
