"""NLU Aspect and Modality Classification for multilingual educational queries.

Zero-shot prototype centroid embeddings and statistical subword classification.
Zero brittle regex pattern arrays or hardcoded keyword heuristics.
"""
from __future__ import annotations

import json
import pickle
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from gurukul.normalizer import clean_text


class QueryAspect(str, Enum):
    CAUSE = "cause"
    REMEDY = "remedy"
    DEFINITION = "definition"
    SYMPTOM = "symptom"
    SCHEDULE = "schedule"
    GENERAL = "general"


class QueryModality(str, Enum):
    ONLINE = "online"
    READING = "reading"
    EXAM = "exam"
    LONG_SESSION = "long_session"
    GENERAL = "general"


class QueryAnalysis(BaseModel):
    query: str
    aspect: QueryAspect = QueryAspect.GENERAL
    modality: QueryModality = QueryModality.GENERAL
    aspect_confidence: float = 1.0
    modality_confidence: float = 1.0
    detected_aspect_tokens: list[str] = Field(default_factory=list)
    detected_modality_tokens: list[str] = Field(default_factory=list)


def _load_prototype_data() -> dict[str, dict[str, list[str]]]:
    """Locates and loads data/nlu_aspect_prototypes.json."""
    candidates = [
        Path(__file__).resolve().parent / "data" / "nlu_aspect_prototypes.json",
        Path(__file__).resolve().parents[1] / "data" / "nlu_aspect_prototypes.json",
        Path.cwd() / "data" / "nlu_aspect_prototypes.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("Could not find data/nlu_aspect_prototypes.json")


class SemanticAspectClassifier:
    """Zero-shot centroid embedding classifier with margin-gated thresholding.
    
    Computes normalized prototype centroid vectors once from data/nlu_aspect_prototypes.json,
    and caches them in .cache/nlu_aspect_centroids.pkl for O(1) startup.
    """

    _instances: dict[int, SemanticAspectClassifier] = {}

    def __init__(self, encoder: Any, cache_dir: Path | str | None = None) -> None:
        self.encoder = encoder
        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir) / "nlu_aspect_centroids.pkl"
        else:
            base_cache = Path(__file__).resolve().parents[1] / "data" / ".cache"
            cache_path = base_cache / "nlu_aspect_centroids.pkl"

        self.aspect_vectors: dict[str, np.ndarray] = {}
        self.modality_vectors: dict[str, np.ndarray] = {}

        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                    self.aspect_vectors = data["aspect_vectors"]
                    self.modality_vectors = data["modality_vectors"]
                    return
            except Exception:
                pass

        proto_data = _load_prototype_data()

        for aspect, texts in proto_data.get("aspects", {}).items():
            vecs = np.array(list(encoder.embed(texts)))
            mean_vec = np.mean(vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec /= norm
            self.aspect_vectors[aspect] = mean_vec

        for modality, texts in proto_data.get("modalities", {}).items():
            vecs = np.array(list(encoder.embed(texts)))
            mean_vec = np.mean(vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec /= norm
            self.modality_vectors[modality] = mean_vec

        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {
                            "aspect_vectors": self.aspect_vectors,
                            "modality_vectors": self.modality_vectors,
                        },
                        f,
                    )
            except Exception:
                pass

    @classmethod
    def get_or_create(cls, encoder: Any, cache_dir: Path | str | None = None) -> SemanticAspectClassifier:
        key = id(encoder)
        if key not in cls._instances:
            cls._instances[key] = cls(encoder, cache_dir=cache_dir)
        return cls._instances[key]

    def classify(
        self,
        query: str,
        q_vec: np.ndarray | None = None,
        aspect_tau: float = 0.76,
        aspect_margin: float = 0.005,
        modality_tau: float = 0.81,
        modality_margin: float = 0.010,
    ) -> QueryAnalysis:
        if q_vec is None:
            q_clean = clean_text(query)
            q_vec = np.array(list(self.encoder.embed([q_clean]))[0])
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec /= norm

        # 1. Aspect classification
        a_scores = sorted(
            [(a, float(np.dot(vec, q_vec))) for a, vec in self.aspect_vectors.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        top_a, top_a_score = a_scores[0]
        sec_a, sec_a_score = a_scores[1]

        final_aspect = QueryAspect.GENERAL
        if top_a != "general" and top_a_score >= aspect_tau and (top_a_score - sec_a_score) >= aspect_margin:
            final_aspect = QueryAspect(top_a)

        # 2. Modality classification
        m_scores = sorted(
            [(m, float(np.dot(vec, q_vec))) for m, vec in self.modality_vectors.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        top_m, top_m_score = m_scores[0]
        sec_m, sec_m_score = m_scores[1]

        final_modality = QueryModality.GENERAL
        if top_m != "general" and top_m_score >= modality_tau and (top_m_score - sec_m_score) >= modality_margin:
            final_modality = QueryModality(top_m)

        return QueryAnalysis(
            query=query,
            aspect=final_aspect,
            modality=final_modality,
            aspect_confidence=round(top_a_score, 3),
            modality_confidence=round(top_m_score, 3),
            detected_aspect_tokens=[f"centroid:{top_a}:{top_a_score:.2f}"] if final_aspect != QueryAspect.GENERAL else [],
            detected_modality_tokens=[f"centroid:{top_m}:{top_m_score:.2f}"] if final_modality != QueryModality.GENERAL else [],
        )


class StatisticalAspectClassifier:
    """Fallback subword TF-IDF + SGDClassifier when ONNX neural encoder is not provided."""

    _instance: StatisticalAspectClassifier | None = None

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.aspect_pipeline: Any = None
        self.modality_pipeline: Any = None
        self._train()

    def _train(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline

        proto_data = _load_prototype_data()

        # Aspect training data
        a_X, a_y = [], []
        for aspect, texts in proto_data.get("aspects", {}).items():
            for t in texts:
                a_X.append(clean_text(t))
                a_y.append(aspect)

        # Modality training data
        m_X, m_y = [], []
        for modality, texts in proto_data.get("modalities", {}).items():
            for t in texts:
                m_X.append(clean_text(t))
                m_y.append(modality)

        a_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
            ("clf", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, random_state=42)),
        ])
        a_pipe.fit(a_X, a_y)
        self.aspect_pipeline = a_pipe

        m_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
            ("clf", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, random_state=42)),
        ])
        m_pipe.fit(m_X, m_y)
        self.modality_pipeline = m_pipe

    @classmethod
    def get_instance(cls) -> StatisticalAspectClassifier:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def classify(self, query: str) -> QueryAnalysis:
        cleaned = clean_text(query)
        if not cleaned:
            return QueryAnalysis(query=query)

        a_probs = self.aspect_pipeline.predict_proba([cleaned])[0]
        a_classes = self.aspect_pipeline.classes_
        a_top_idx = int(np.argmax(a_probs))
        top_a = str(a_classes[a_top_idx])
        top_a_p = float(a_probs[a_top_idx])

        m_probs = self.modality_pipeline.predict_proba([cleaned])[0]
        m_classes = self.modality_pipeline.classes_
        m_top_idx = int(np.argmax(m_probs))
        top_m = str(m_classes[m_top_idx])
        top_m_p = float(m_probs[m_top_idx])

        final_aspect = QueryAspect(top_a) if top_a != "general" and top_a_p >= 0.50 else QueryAspect.GENERAL
        final_modality = QueryModality(top_m) if top_m != "general" and top_m_p >= 0.50 else QueryModality.GENERAL

        return QueryAnalysis(
            query=query,
            aspect=final_aspect,
            modality=final_modality,
            aspect_confidence=round(top_a_p, 3),
            modality_confidence=round(top_m_p, 3),
        )


def classify_query(
    query: str,
    encoder: Any = None,
    q_vec: np.ndarray | None = None,
    cache_dir: Path | str | None = None,
) -> QueryAnalysis:
    """Classifies aspect and modality of a student query using zero-shot prototype embeddings.

    Reuses pre-computed query embedding vector when provided.
    Zero hardcoded regexes or heuristic keyword lists.
    """
    if encoder is not None or q_vec is not None:
        try:
            if encoder is not None:
                classifier = SemanticAspectClassifier.get_or_create(encoder, cache_dir=cache_dir)
                return classifier.classify(query, q_vec=q_vec)
        except Exception:
            pass

    return StatisticalAspectClassifier.get_instance().classify(query)
