from __future__ import annotations

import json
import os
import pickle
import unicodedata
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier


def clean_text(text: str) -> str:
    """Universal Unicode normalization and punctuation stripping.

    Uses Unicode character categories (no fragile word lists or hardcoded regexes).
    """
    if not text:
        return ""

    # Canonical Decomposition followed by Canonical Composition (NFC)
    text = unicodedata.normalize("NFC", text)

    # Filter out zero-width characters and punctuation using Unicode categories
    chars: list[str] = []
    for c in text:
        # P = Punctuation, S = Symbol, C = Other/Control
        cat = unicodedata.category(c)
        if cat.startswith("P") or cat.startswith("S"):
            chars.append(" ")
        elif cat == "Cf":  # Format characters like ZWJ / ZWNJ
            continue
        else:
            chars.append(c.lower())

    # Collapse multiple whitespace
    cleaned = " ".join("".join(chars).split())
    return cleaned


def detect_script(text: str) -> str:
    """Identifies the dominant script using Unicode codepoint ranges.

    Zero hardcoded vocabulary lists.
    """
    if not text:
        return "latin"

    devanagari_count = 0
    latin_count = 0

    for char in text:
        cp = ord(char)
        # Devanagari block: U+0900 to U+097F
        if 0x0900 <= cp <= 0x097F:
            devanagari_count += 1
        elif 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A:
            latin_count += 1

    total_indicative = devanagari_count + latin_count
    if total_indicative == 0:
        return "devanagari" if devanagari_count > 0 else "latin"

    # Robust majority threshold: at least 40% of alphabet characters must be Devanagari
    if devanagari_count / total_indicative >= 0.40:
        return "devanagari"
    return "latin"


class StatisticalLanguageIdentifier:
    """Fast, offline statistical subword character n-gram language classifier.

    Uses dual-model subword character n-grams (2-4 char bounds) trained via TF-IDF
    and balanced Logistic Regression on authentic curriculum corpora:
    - Devanagari Pipeline: Classifies authentic Marathi vs Hindi textbook text.
    - Latin Pipeline: Differentiates English from Romanized Indic (MarEnglish / Hinglish).

    Zero fragile hardcoded word lists or runtime regexes.
    """

    _instance: StatisticalLanguageIdentifier | None = None

    def __init__(self, cache_path: Path | str | None = None) -> None:
        if cache_path:
            self.cache_path = Path(cache_path)
        else:
            pkg_cache = Path(__file__).resolve().parent / "data" / ".cache" / "lang_detector_v4.pkl"
            repo_cache = Path(__file__).resolve().parents[1] / "data" / ".cache" / "lang_detector_v4.pkl"
            cwd_cache = Path.cwd() / "data" / ".cache" / "lang_detector_v4.pkl"
            if pkg_cache.exists():
                self.cache_path = pkg_cache
            elif repo_cache.exists():
                self.cache_path = repo_cache
            elif cwd_cache.exists():
                self.cache_path = cwd_cache
            else:
                self.cache_path = repo_cache

        self.dev_model: Any = None
        self.latin_model: Any = None
        self._load_or_train()

    @classmethod
    def get_instance(cls) -> StatisticalLanguageIdentifier:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_or_train(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and "dev_model" in data and "latin_model" in data:
                        self.dev_model = data["dev_model"]
                        self.latin_model = data["latin_model"]
                        return
            except Exception:
                pass

        # Try to locate external language corpora asset
        corpora_file = self.cache_path.parent.parent / "language_corpora.json"
        if not corpora_file.exists():
            repo_corpora = Path(__file__).resolve().parents[1] / "data" / "language_corpora.json"
            if repo_corpora.exists():
                corpora_file = repo_corpora

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline

            if corpora_file.exists():
                with open(corpora_file, "r", encoding="utf-8") as f:
                    corpora = json.load(f)

                dev_texts = [clean_text(t) for t in corpora.get("devanagari_mr", [])] + [clean_text(t) for t in corpora.get("devanagari_hi", [])]
                dev_labels = ["mr"] * len(corpora.get("devanagari_mr", [])) + ["hi"] * len(corpora.get("devanagari_hi", []))

                latin_texts = [clean_text(t) for t in corpora.get("latin_en", [])] + [clean_text(t) for t in corpora.get("latin_mr", [])] + [clean_text(t) for t in corpora.get("latin_hi", [])]
                latin_labels = ["en"] * len(corpora.get("latin_en", [])) + ["mr"] * len(corpora.get("latin_mr", [])) + ["hi"] * len(corpora.get("latin_hi", []))
            else:
                return

            dev_pipe = make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True),
                LogisticRegression(class_weight="balanced", C=5.0, random_state=42),
            )
            dev_pipe.fit(dev_texts, dev_labels)

            latin_pipe = make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True),
                LogisticRegression(class_weight="balanced", C=5.0, random_state=42),
            )
            latin_pipe.fit(latin_texts, latin_labels)

            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({"dev_model": dev_pipe, "latin_model": latin_pipe}, f)

            self.dev_model = dev_pipe
            self.latin_model = latin_pipe
        except Exception:
            pass

    def predict(self, text: str, default: str = "mr") -> str:
        """Predicts query language preference statistically using learned subword models.

        Zero manual word lists or heuristic word matching.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return default

        script = detect_script(cleaned)

        try:
            if script == "devanagari":
                if self.dev_model is not None:
                    return str(self.dev_model.predict([cleaned])[0])
                return "mr"
            else:
                if self.latin_model is not None:
                    return str(self.latin_model.predict([cleaned])[0])
                return "en"
        except Exception:
            return "mr" if script == "devanagari" else "en"


def infer_preferred_language(text: str, default: str = "mr") -> str:
    """Infers query language preference ('mr', 'hi', 'en') statistically.

    - Devanagari script is evaluated via subword character n-grams for 'mr' vs 'hi'.
    - Latin script is evaluated via subword character n-grams for English vs Romanized Marathi/Hindi.
    """
    detector = StatisticalLanguageIdentifier.get_instance()
    return detector.predict(text, default=default)
