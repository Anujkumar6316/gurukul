from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from gurukul.models import (
    ConversationSession,
    FactCard,
    FollowUpChip,
    FollowUpIntent,
)
from gurukul.normalizer import clean_text


class NLUResult(BaseModel):
    """Result of conversational follow-up natural language understanding."""

    intent: FollowUpIntent
    target_card_id: str | None = None
    target_chip: FollowUpChip | None = None
    confidence: float = 1.0
    explanation: str | None = None


class StatisticalFollowUpIntentClassifier:
    """Statistical character/subword n-gram intent classifier.

    Predicts whether a conversational turn is an affirmation (AFFIRM), negation (DENY),
    or curriculum/other (OTHER) without relying on brittle regex or exact word lists.
    """

    def __init__(self, cache_path: Path | str | None = None) -> None:
        if cache_path:
            self.cache_path = Path(cache_path)
        else:
            pkg_cache = Path(__file__).resolve().parent / "data" / ".cache" / "follow_up_nlu.pkl"
            repo_cache = Path(__file__).resolve().parents[1] / "data" / ".cache" / "follow_up_nlu.pkl"
            self.cache_path = pkg_cache if pkg_cache.exists() else repo_cache

        self.pipeline: Pipeline | None = None
        self._load_or_train()

    def _load_or_train(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self.pipeline = pickle.load(f)
                return
            except Exception:
                pass

        self._train_and_cache()

    def _train_and_cache(self) -> None:
        import json

        # Load versioned dialogue training asset
        corpus_path: Path | None = None
        for c in [
            Path(__file__).resolve().parent / "data" / "dialogue_acts_train.json",
            Path(__file__).resolve().parents[1] / "data" / "dialogue_acts_train.json",
            Path.cwd() / "data" / "dialogue_acts_train.json",
        ]:
            if c.exists():
                corpus_path = c
                break

        if not corpus_path:
            raise FileNotFoundError("Dialogue acts training asset 'dialogue_acts_train.json' not found.")

        with open(corpus_path, "r", encoding="utf-8") as f:
            d_data = json.load(f)

        aff = d_data.get("affirmations", [])
        neg = d_data.get("negations", [])
        curr = d_data.get("curriculum_queries", [])

        X = aff + neg + curr
        y = (
            ["affirm"] * len(aff)
            + ["deny"] * len(neg)
            + ["other"] * len(curr)
        )

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
            ("clf", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, random_state=42)),
        ])
        pipeline.fit(X, y)
        self.pipeline = pipeline

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.pipeline, f)
        except Exception:
            pass

    def predict(self, text: str) -> tuple[str, float]:
        """Returns (predicted_class, probability). Classes: 'affirm', 'deny', 'other'."""
        cleaned = clean_text(text)
        if not cleaned or not self.pipeline:
            return "other", 0.0

        probs = self.pipeline.predict_proba([cleaned])[0]
        classes = self.pipeline.classes_
        top_idx = int(np.argmax(probs))
        return str(classes[top_idx]), float(probs[top_idx])


ORDINAL_MAP: dict[str, int] = {
    # Digits
    "1": 0, "2": 1, "3": 2, "4": 3,
    "#1": 0, "#2": 1, "#3": 2, "#4": 3,
    "१": 0, "२": 1, "३": 2, "४": 3,
    # English
    "first": 0, "second": 1, "third": 2, "fourth": 3,
    "1st": 0, "2nd": 1, "3rd": 2, "4th": 3,
    "one": 0, "two": 1, "three": 2, "four": 3,
    "option 1": 0, "option 2": 1, "option 3": 2, "option 4": 3,
    "chip 1": 0, "chip 2": 1, "chip 3": 2, "chip 4": 3,
    # Marathi
    "पहिला": 0, "दुसरा": 1, "तिसरा": 2, "चौथा": 3,
    "पहिला पर्याय": 0, "दुसरा पर्याय": 1, "तिसरा पर्याय": 2, "चौथा पर्याय": 3,
    "एक": 0, "दोन": 1, "तीन": 2, "चार": 3,
    # Hindi
    "पहला": 0, "दूसरा": 1, "तीसरा": 2, "चौथा": 3,
    "पहला विकल्प": 0, "दूसरा विकल्प": 1, "तीसरा विकल्प": 2, "चौथा विकल्प": 3,
}


def extract_ordinal(query: str) -> int | None:
    """Extracts ordinal chip selection index (0-based) from an utterance."""
    cleaned = clean_text(query).lower()
    if cleaned in ORDINAL_MAP:
        return ORDINAL_MAP[cleaned]

    words = cleaned.split()
    for w in words:
        if w in ORDINAL_MAP:
            return ORDINAL_MAP[w]

    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram in ORDINAL_MAP:
            return ORDINAL_MAP[bigram]

    return None


class ConversationalFollowUpNLU:
    """Conversational Follow-Up Natural Language Understanding Engine.

    Interprets student dialogue turns in the context of the active conversation session,
    resolving affirmative interests in pending offers, polite rejections, chip selections,
    or new curriculum questions.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.classifier = StatisticalFollowUpIntentClassifier(
            cache_path=(Path(cache_dir) / "follow_up_nlu.pkl") if cache_dir else None
        )

    def parse_turn(
        self,
        query: str,
        session: ConversationSession | None,
        cards_by_id: dict[str, FactCard],
        encoder: Any = None,
    ) -> NLUResult:
        """Parses a user utterance into an actionable dialogue intent."""
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return NLUResult(intent=FollowUpIntent.NEW_QUERY, confidence=1.0)

        # If no prior session or no prior turn context, treat as fresh query
        if not session or (not session.last_offered_card_id and not session.last_chips):
            return NLUResult(intent=FollowUpIntent.NEW_QUERY, confidence=1.0)

        # 1. Direct Ordinal / Chip Number Selection (whole string or extracted token)
        ord_idx = extract_ordinal(cleaned_query)
        if ord_idx is not None and 0 <= ord_idx < len(session.last_chips):
            chip = session.last_chips[ord_idx]
            return NLUResult(
                intent=FollowUpIntent.SELECT_OPTION,
                target_card_id=chip.card_id,
                target_chip=chip,
                confidence=1.0,
                explanation=f"Selected chip #{ord_idx + 1} by ordinal identifier",
            )

        # 2. Statistical Intent Classification (Affirmation vs Negation)
        intent_class, prob = self.classifier.predict(cleaned_query)

        if intent_class == "affirm" and prob >= 0.50:
            if session.last_offered_card_id and session.last_offered_card_id in cards_by_id:
                return NLUResult(
                    intent=FollowUpIntent.AFFIRM,
                    target_card_id=session.last_offered_card_id,
                    confidence=prob,
                    explanation=f"Statistical affirmation ({prob:.2f}) for offered card",
                )

        if intent_class == "deny" and prob >= 0.50:
            return NLUResult(
                intent=FollowUpIntent.DENY,
                target_card_id=None,
                confidence=prob,
                explanation=f"Statistical negation ({prob:.2f})",
            )

        # 3. Fuzzy & Lexical Alignment against Active Chips
        if session.last_chips:
            best_chip: FollowUpChip | None = None
            best_ratio = 0.0
            for chip in session.last_chips:
                chip_title_clean = clean_text(chip.title)
                ratio = fuzz.token_set_ratio(cleaned_query, chip_title_clean)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_chip = chip

            if best_ratio >= 75.0 and best_chip:
                return NLUResult(
                    intent=FollowUpIntent.SELECT_OPTION,
                    target_card_id=best_chip.card_id,
                    target_chip=best_chip,
                    confidence=round(best_ratio / 100.0, 3),
                    explanation=f"Fuzzy lexical match ({best_ratio}%) with active chip",
                )

        # 4. Contextual Semantic Question Alignment to Pending Offered Card
        if session.last_offered_card_id and encoder and session.last_offered_card_id in cards_by_id:
            offered_card = cards_by_id[session.last_offered_card_id]
            candidates_to_compare = [offered_card.get_offer_question(session.language)]
            candidates_to_compare.extend(offered_card.question_aliases[:3])

            try:
                q_vec = np.array(list(encoder.embed([cleaned_query]))[0])
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec /= q_norm

                cand_vecs = np.array(list(encoder.embed(candidates_to_compare)))
                cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
                cand_norms[cand_norms == 0] = 1.0
                cand_vecs /= cand_norms

                sims = np.dot(cand_vecs, q_vec)
                max_sim = float(np.max(sims))

                threshold = 0.78 if "embeddinggemma" in getattr(encoder, "model_name", "").lower() else 0.65
                if max_sim >= threshold:
                    return NLUResult(
                        intent=FollowUpIntent.AFFIRM,
                        target_card_id=session.last_offered_card_id,
                        confidence=round(max_sim, 3),
                        explanation=f"Contextual semantic alignment ({max_sim:.2f}) with offered card",
                    )
            except Exception:
                pass

        # 5. Default Fallback -> Treat as New Information Query
        return NLUResult(
            intent=FollowUpIntent.NEW_QUERY,
            confidence=1.0,
            explanation="Uncorrelated with prior turn; routing as new query",
        )
