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
    metadata: dict[str, Any] = Field(default_factory=dict)

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
        """Generates a natural 'Would you like to know...' style offer question for this card."""
        from gurukul.normalizer import detect_script, infer_preferred_language

        aliases = self.question_aliases or []

        if lang == "en":
            cands = [
                a.strip("?|।॥ ").strip()
                for a in aliases
                if detect_script(a) == "latin" and infer_preferred_language(a, default="en") == "en"
            ]
            raw = cands[0] if cands else (self.topic or self.id).replace("_", " ")
            raw_l = raw.lower()

            # Sanitize leading imperative/question prefixes if present
            for prefix in ("tell me about ", "tell me ", "explain ", "describe "):
                if raw_l.startswith(prefix):
                    raw = raw[len(prefix):].strip()
                    raw_l = raw.lower()
                    break

            if " vs " in raw_l:
                parts = raw.split(" vs ")
                return f"Would you like to know the difference between {parts[0].strip()} and {parts[1].strip()}?"
            elif raw_l.startswith("difference between "):
                return f"Would you like to know the {raw_l}?"
            elif "where was " in raw_l and " born" in raw_l:
                subj = raw_l.replace("where was ", "").replace(" born", "").strip()
                return f"Would you like to know where {subj.title()} was born?"
            elif raw_l.startswith("where was "):
                subj = raw_l.replace("where was ", "").strip()
                return f"Would you like to know where {subj.title()} was?"
            elif raw_l.startswith(("who was ", "who is ")):
                prefix = "who was " if raw_l.startswith("who was ") else "who is "
                person = raw[len(prefix):].strip()
                return f"Would you like to know who {person.title()} was?"
            elif raw_l.startswith("who wrote "):
                work = raw[len("who wrote "):].strip()
                return f"Would you like to know who wrote {work.title()}?"
            elif raw_l.startswith("why did "):
                rest = raw[len("why did "):].strip()
                return f"Would you like to know why {rest}?"
            elif raw_l.startswith("when was "):
                rest = raw[len("when was "):].strip()
                return f"Would you like to know when {rest}?"
            elif raw_l.startswith(("how do ", "how does ")):
                prefix = "how do " if raw_l.startswith("how do ") else "how does "
                rest = raw[len(prefix):].strip()
                return f"Would you like to know how {rest}?"
            elif raw_l.startswith("what is a "):
                term = raw[len("what is a "):].strip()
                return f"Would you like to know what a {term} is?"
            elif raw_l.startswith("what is an "):
                term = raw[len("what is an "):].strip()
                return f"Would you like to know what an {term} is?"
            elif raw_l.startswith("what is the "):
                term = raw[len("what is the "):].strip()
                return f"Would you like to know what the {term} is?"
            elif raw_l.startswith("what was the "):
                term = raw[len("what was the "):].strip()
                return f"Would you like to know what the {term} was?"
            elif raw_l.startswith("what is "):
                term = raw[len("what is "):].strip()
                return f"Would you like to know what {term} is?"
            elif raw_l.startswith("what was "):
                term = raw[len("what was "):].strip()
                return f"Would you like to know what {term} was?"
            elif raw_l.startswith("what are the "):
                term = raw[len("what are the "):].strip()
                return f"Would you like to know what the {term} are?"
            elif raw_l.startswith("what were the "):
                term = raw[len("what were the "):].strip()
                return f"Would you like to know what the {term} were?"
            elif raw_l.startswith("what are "):
                term = raw[len("what are "):].strip()
                return f"Would you like to know what {term} are?"
            elif raw_l.startswith("what were "):
                term = raw[len("what were "):].strip()
                return f"Would you like to know what {term} were?"
            elif raw_l.startswith(("speed of ", "rate of ")):
                return f"Would you like to know the {raw_l}?"
            else:
                clean_name = raw.strip()
                if clean_name.lower().startswith(("what ", "which ", "who ", "where ", "when ", "why ", "how ")):
                    return f"Would you like to know {clean_name}?"
                return f"Would you like to know more about {clean_name}?"

        elif lang == "hi":
            hi_cands = [a.strip("?|।॥ ").strip() for a in aliases if infer_preferred_language(a, default="mr") == "hi"]
            if not hi_cands:
                hi_cands = [a.strip("?|।॥ ").strip() for a in aliases if detect_script(a) == "devanagari"]
            raw = hi_cands[0] if hi_cands else (self.topic or self.id).replace("_", " ")

            for p in ["मुझे बताइए कि ", "बताइए कि ", "बताओ कि ", "बताइए ", "बताओ "]:
                if raw.startswith(p):
                    raw = raw[len(p):].strip()
                    break

            if "अंतर" in raw:
                return f"क्या आप {raw} जानना चाहेंगे?"
            elif "किसे कहते हैं" in raw:
                term = raw.replace("किसे कहते हैं", "").strip()
                return f"क्या आप जानना चाहेंगे कि {term} किसे कहते हैं?"
            elif "क्या है" in raw:
                term = raw.replace("क्या है", "").strip()
                return f"क्या आप जानना चाहेंगे कि {term} क्या है?"
            elif any(qw in raw for qw in ["कहाँ", "कब", "कैसे", "किसने", "कौन", "कितना"]):
                return f"क्या आप जानना चाहेंगे कि {raw}?"
            else:
                return f"क्या आप {raw} के बारे में जानना चाहेंगे?"

        else:  # mr
            mr_cands = [a.strip("?|।॥ ").strip() for a in aliases if infer_preferred_language(a, default="mr") == "mr"]
            if not mr_cands:
                mr_cands = [a.strip("?|।॥ ").strip() for a in aliases if detect_script(a) == "devanagari"]
            raw = mr_cands[0] if mr_cands else (self.topic or self.id).replace("_", " ")

            for p in ["मला सांगा ", "सांगा की ", "सांगा ", "माहिती सांगा "]:
                if raw.startswith(p):
                    raw = raw[len(p):].strip()
                    break

            if "फरक" in raw:
                return f"{raw} जाणून घ्यायला आवडेल का?"
            elif "म्हणजे काय" in raw:
                term = raw.replace("म्हणजे काय", "").strip()
                return f"{term} म्हणजे काय, हे जाणून घ्यायला आवडेल का?"
            elif any(qw in raw for qw in ["कधी", "कुठे", "कोठे", "कोणी", "कसे", "किती", "कोण", "कोणती", "कोणता"]):
                return f"{raw}, हे तुम्हाला जाणून घ्यायला आवडेल का?"
            elif "काय आहे" in raw:
                term = raw.replace("काय आहे", "").strip()
                return f"{term} काय आहे, हे जाणून घ्यायला आवडेल का?"
            else:
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
