from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator


class FollowUpIntent(str, Enum):
    AFFIRM = "affirm"
    DENY = "deny"
    SELECT_OPTION = "select_option"
    NEW_QUERY = "new_query"


class FactSource(BaseModel):
    book_id: str = "curriculum_knowledgebase"
    page: int | str | None = None

    def citation_str(self) -> str:
        if self.page is not None:
            return f"{self.book_id}, p. {self.page}"
        return self.book_id


class FactCard(BaseModel):
    """Flexible factual knowledge card for offline neural search and Q&A."""

    id: str = Field(default_factory=lambda: f"card_{uuid.uuid4().hex[:8]}")
    topic: str | None = None
    category: str | None = None
    subject: str | None = None
    grade: int | None = None
    grades: list[int] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    question_aliases: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    answer_templates: dict[str, str] = Field(default_factory=dict)
    source: FactSource | None = None
    follow_up_cards: list[str] = Field(default_factory=list)
    offer_questions: dict[str, str] = Field(default_factory=dict)
    aspect: str | None = None
    modality: str | None = None
    is_curriculum: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_aspect(self) -> str:
        if self.aspect:
            return self.aspect
        if self.metadata and "aspect" in self.metadata:
            return str(self.metadata["aspect"])
        return "general"

    def get_modality(self) -> str:
        if self.modality:
            return self.modality
        if self.metadata and "modality" in self.metadata:
            return str(self.metadata["modality"])
        return "general"

    @model_validator(mode="before")
    @classmethod
    def normalize_flexible_inputs(cls, data: Any) -> Any:
        """Allows users to supply simplified schemas (e.g. `answer`, `answers`, `question`)."""
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # 1. Flexible ID generation
        if not d.get("id"):
            base = d.get("topic") or d.get("question") or (d.get("question_aliases") and d["question_aliases"][0]) or uuid.uuid4().hex[:8]
            clean_base = "".join(c if c.isalnum() else "_" for c in str(base).lower())[:32].strip("_")
            d["id"] = clean_base or f"card_{uuid.uuid4().hex[:8]}"

        # 2. Flexible Question Aliases
        if not d.get("question_aliases"):
            aliases: list[str] = []
            if d.get("question"):
                aliases.append(str(d["question"]).strip())
            elif d.get("questions") and isinstance(d["questions"], list):
                aliases.extend([str(q).strip() for q in d["questions"] if str(q).strip()])
            elif d.get("topic"):
                aliases.append(str(d["topic"]).strip())
            d["question_aliases"] = aliases

        # 3. Flexible Answer Templates
        if not d.get("answer_templates"):
            if d.get("answers") and isinstance(d["answers"], dict):
                d["answer_templates"] = {str(k): str(v) for k, v in d["answers"].items()}
            elif d.get("answer"):
                ans_val = str(d["answer"]).strip()
                from gurukul.normalizer import detect_script
                script = detect_script(ans_val)
                lang = "mr" if script == "devanagari" else "en"
                d["answer_templates"] = {lang: ans_val}

        # 4. Populate languages if missing
        if not d.get("languages") and d.get("answer_templates"):
            d["languages"] = list(d["answer_templates"].keys())

        # 5. Determine is_curriculum explicitly at ingestion
        if "is_curriculum" not in d:
            if d.get("metadata", {}).get("is_curriculum") is not None:
                d["is_curriculum"] = bool(d["metadata"]["is_curriculum"])
            else:
                book_id = str(d.get("source", {}).get("book_id", "") if d.get("source") else "")
                if book_id in ("custom_knowledgebase", "user_ingested") or book_id.startswith("knowledgebase"):
                    d["is_curriculum"] = False
                elif str(d.get("id", "")).startswith("kb_") and book_id != "curriculum_knowledgebase":
                    d["is_curriculum"] = False
                elif not d.get("source") and d.get("id") and not str(d.get("id")).startswith("concept_"):
                    d["is_curriculum"] = False
                else:
                    d["is_curriculum"] = True

        return d

    def get_answer(self, lang: str = "mr") -> str:
        """Returns the answer in the requested language, falling back to mr -> en -> hi -> first available."""
        if lang in self.answer_templates:
            return self.answer_templates[lang]
        for fallback_lang in ["mr", "en", "hi"]:
            if fallback_lang in self.answer_templates:
                return self.answer_templates[fallback_lang]
        if self.answer_templates:
            return next(iter(self.answer_templates.values()))
        return ""

    def get_offer_question(self, lang: str = "mr") -> str:
        """Generates a natural offer question for this card."""
        if self.offer_questions and lang in self.offer_questions:
            return self.offer_questions[lang]

        from gurukul.normalizer import detect_script, infer_preferred_language

        aliases = self.question_aliases or []
        target_script = "devanagari" if lang in ("mr", "hi") else "latin"

        matching = [
            a.strip("?|।॥ ").strip()
            for a in aliases
            if detect_script(a) == target_script and infer_preferred_language(a, default=lang) == lang
        ]
        if not matching:
            matching = [a.strip("?|।॥ ").strip() for a in aliases if detect_script(a) == target_script]

        raw = matching[0] if matching else (self.topic or self.id).replace("_", " ")

        if lang == "en":
            raw_l = raw.lower()
            if any(raw_l.startswith(w + " ") for w in ["what", "which", "who", "where", "when", "why", "how"]):
                return f"Would you like to know {raw}?"
            return f"Would you like to know more about {raw}?"
        elif lang == "hi":
            if any(w in raw for w in ["कहाँ", "कब", "कैसे", "किसने", "कौन", "कितना", "क्या"]):
                return f"क्या आप जानना चाहेंगे कि {raw}?"
            return f"क्या आप {raw} के बारे में अधिक जानना चाहेंगे?"
        else:  # mr
            if any(w in raw for w in ["कधी", "कुठे", "कोठे", "कोणी", "कसे", "किती", "कोण", "कोणती", "कोणता", "काय", "का"]):
                return f"{raw}, हे तुम्हाला जाणून घ्यायला आवडेल का?"
            return f"{raw} याविषयी अधिक जाणून घ्यायला आवडेल का?"

    def get_title(self, lang: str = "mr") -> str:
        """Return a natural offer question title for the card."""
        return self.get_offer_question(lang)


def format_offer_prompt(question: str, lang: str = "mr") -> str:
    """Returns the natural conversational offer prompt."""
    return question


class FollowUpChip(BaseModel):
    card_id: str
    title: str
    topic: str | None = None
    language: str = "mr"


class OfflineChatResponse(BaseModel):
    query: str
    card_id: str | None = None
    topic: str | None = None
    language: str = "mr"
    answer: str
    citation: str | None = None
    confidence: float
    is_abstention: bool = False
    follow_up_offer: str | None = None
    offered_card_id: str | None = None
    follow_up_chips: list[FollowUpChip] = Field(default_factory=list)
    suggested_topics: list[str] = Field(default_factory=list)
    intent: str | None = None
    session_id: str | None = None
    latency_ms: float = 0.0


class ConversationSession(BaseModel):
    session_id: str
    last_card_id: str | None = None
    last_offered_card_id: str | None = None
    last_offer_question: str | None = None
    last_chips: list[FollowUpChip] = Field(default_factory=list)
    language: str = "mr"
    grade: int | None = None
    history: list[dict[str, str]] = Field(default_factory=list)


class OfflineQueryLog(BaseModel):
    query: str
    language: str
    grade: int | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_abstention: bool = True
    top_matched_card_id: str | None = None
    top_score: float = 0.0


class DebugQueryResult(BaseModel):
    line_number: int
    query: str
    language: str
    intent: str | None = None
    card_id: str | None = None
    card_topic: str | None = None
    confidence: float
    is_abstention: bool = False
    answer_preview: str
    citation: str | None = None
    follow_up_offer: str | None = None
    offered_card_id: str | None = None
    follow_up_chips: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class BatchDebugResponse(BaseModel):
    total_queries: int
    matched_count: int
    abstention_count: int
    match_rate_pct: float
    avg_latency_ms: float
    language_breakdown: dict[str, int] = Field(default_factory=dict)
    intent_breakdown: dict[str, int] = Field(default_factory=dict)
    results: list[DebugQueryResult] = Field(default_factory=list)
