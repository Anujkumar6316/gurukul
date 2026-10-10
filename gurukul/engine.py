from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from gurukul.index import FactCardIndex
from gurukul.models import (
    BatchDebugResponse,
    ConversationSession,
    DebugQueryResult,
    FactCard,
    FollowUpChip,
    FollowUpIntent,
    OfflineChatResponse,
    OfflineQueryLog,
)
from gurukul.nlu import ConversationalFollowUpNLU
from gurukul.normalizer import infer_preferred_language


class OfflineFactBot:
    """Offline curriculum and custom knowledgebase chatbot engine.

    Executes 100% locally on device without network access or cloud LLM dependencies.
    Uses dense multilingual sentence embeddings for semantic retrieval,
    BM25 lexical anchoring, RapidFuzz reranking, and statistical NLU for multi-turn dialogue.
    """

    CONFIDENCE_THRESHOLD = 0.50

    REFUSAL_MESSAGES = {
        "mr": "माफ करा, हा प्रश्न सध्याच्या उपलब्ध ज्ञान कार्ड्समध्ये उपलब्ध नाही. तुम्ही इतर विषयांवर प्रश्न विचारू शकता.",
        "hi": "क्षमा करें, यह प्रश्न वर्तमान उपलब्ध ज्ञान कार्ड्स में उपलब्ध नहीं है। आप अन्य विषयों पर प्रश्न पूछ सकते हैं।",
        "en": "Sorry, this question is outside the currently loaded fact cards. Feel free to ask about other available topics.",
    }

    AFFIRM_PREFACES = {
        "mr": "नक्कीच! याबद्दल उपलब्ध माहिती पुढीलप्रमाणे:",
        "hi": "ज़रूर! इसके बारे में जानकारी निम्नलिखित है:",
        "en": "Certainly! Here is what the documentation explains:",
    }

    DENIAL_MESSAGES = {
        "mr": "काही हरकत नाही! तुम्हाला दुसरा कोणताही प्रश्न विचारायचा असल्यास नक्की विचारू शकता, किंवा खालील विषयांपैकी निवडा:",
        "hi": "कोई बात नहीं! यदि आपका कोई अन्य प्रश्न है तो बेझिझक पूछें, या नीचे दिए गए विषयों में से चुनें:",
        "en": "No problem at all! Feel free to ask any other question, or explore these topics:",
    }

    OPTION_PREFACES = {
        "mr": "या विषयावरील माहिती पुढीलप्रमाणे:",
        "hi": "इस विषय पर जानकारी निम्नलिखित है:",
        "en": "Here is the information on that topic:",
    }

    def __init__(
        self,
        cards: Path | str | list[dict] | list[FactCard] | None = None,
        data_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
        model_name: str | None = None,
        confidence_threshold: float = 0.50,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.index = FactCardIndex(
            cards=cards,
            data_path=data_path,
            cache_dir=cache_dir,
            model_name=model_name,
        )
        self.nlu = ConversationalFollowUpNLU(cache_dir=self.index.cache_dir)
        self.sessions: dict[str, ConversationSession] = {}
        self.sync_queue: list[OfflineQueryLog] = []

    def get_or_create_session(
        self,
        session_id: str,
        lang: str = "mr",
        grade: int | None = None,
    ) -> ConversationSession:
        """Retrieves or creates a multi-turn conversation session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                language=lang,
                grade=grade,
            )
        return self.sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        """Resets dialogue state for a specific session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def resolve_query(
        self,
        query: str,
        session_id: str | None = None,
        lang: str | None = None,
        grade: int | None = None,
    ) -> OfflineChatResponse:
        """Resolves a query against the offline fact cards with conversational context."""
        start_time = time.perf_counter()

        session: ConversationSession | None = None
        if session_id:
            session = self.get_or_create_session(session_id, lang=lang or "mr", grade=grade)
            if lang and lang.lower() != "auto":
                effective_lang = lang
                session.language = lang
            else:
                effective_lang = session.language
            effective_grade = grade if grade is not None else session.grade
        else:
            effective_lang = (
                lang if (lang and lang.lower() != "auto") else infer_preferred_language(query, default="mr")
            )
            effective_grade = grade

        # Multi-turn conversational NLU evaluation
        if session and (session.last_offered_card_id or session.last_chips):
            nlu_res = self.nlu.parse_turn(
                query=query,
                session=session,
                cards_by_id=self.index.cards_by_id,
                encoder=self.index.encoder,
            )

            # 1. Affirmation: accepted follow-up offer
            if nlu_res.intent == FollowUpIntent.AFFIRM and nlu_res.target_card_id:
                target_card = self.index.get_card(nlu_res.target_card_id)
                if target_card:
                    raw_answer = target_card.get_answer(effective_lang)
                    preface = self.AFFIRM_PREFACES.get(effective_lang, self.AFFIRM_PREFACES["mr"])
                    answer = f"{preface}\n\n{raw_answer}"
                    citation = target_card.source.citation_str() if target_card.source else None

                    offer_prompt, natural_q, featured_id = self.index.get_random_follow_up_offer(
                        target_card.id, lang=effective_lang
                    )
                    chips = self.index.get_follow_up_chips(
                        target_card.id, lang=effective_lang, max_chips=4, featured_id=featured_id
                    )

                    session.last_card_id = target_card.id
                    session.last_offered_card_id = featured_id
                    session.last_offer_question = natural_q
                    session.last_chips = chips

                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    return OfflineChatResponse(
                        query=query,
                        card_id=target_card.id,
                        topic=target_card.topic,
                        language=effective_lang,
                        answer=answer,
                        citation=citation,
                        confidence=round(nlu_res.confidence, 3),
                        is_abstention=False,
                        follow_up_offer=offer_prompt,
                        offered_card_id=featured_id,
                        follow_up_chips=chips,
                        suggested_topics=[],
                        intent=FollowUpIntent.AFFIRM.value,
                        session_id=session_id,
                        latency_ms=round(elapsed_ms, 2),
                    )

            # 2. Negation: politely declined offer
            if nlu_res.intent == FollowUpIntent.DENY:
                ack_msg = self.DENIAL_MESSAGES.get(effective_lang, self.DENIAL_MESSAGES["mr"])
                suggested_topics = self.index.get_suggested_topics(lang=effective_lang, count=4)
                chips = session.last_chips or []

                session.last_offered_card_id = None
                session.last_offer_question = None

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return OfflineChatResponse(
                    query=query,
                    card_id=None,
                    topic=None,
                    language=effective_lang,
                    answer=ack_msg,
                    citation=None,
                    confidence=round(nlu_res.confidence, 3),
                    is_abstention=False,
                    follow_up_offer=None,
                    offered_card_id=None,
                    follow_up_chips=chips,
                    suggested_topics=suggested_topics,
                    intent=FollowUpIntent.DENY.value,
                    session_id=session_id,
                    latency_ms=round(elapsed_ms, 2),
                )

            # 3. Selection of active chip option
            if nlu_res.intent == FollowUpIntent.SELECT_OPTION and nlu_res.target_card_id:
                target_card = self.index.get_card(nlu_res.target_card_id)
                if target_card:
                    raw_answer = target_card.get_answer(effective_lang)
                    preface = self.OPTION_PREFACES.get(effective_lang, self.OPTION_PREFACES["mr"])
                    answer = f"{preface}\n\n{raw_answer}"
                    citation = target_card.source.citation_str() if target_card.source else None

                    offer_prompt, natural_q, featured_id = self.index.get_random_follow_up_offer(
                        target_card.id, lang=effective_lang
                    )
                    chips = self.index.get_follow_up_chips(
                        target_card.id, lang=effective_lang, max_chips=4, featured_id=featured_id
                    )

                    session.last_card_id = target_card.id
                    session.last_offered_card_id = featured_id
                    session.last_offer_question = natural_q
                    session.last_chips = chips

                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    return OfflineChatResponse(
                        query=query,
                        card_id=target_card.id,
                        topic=target_card.topic,
                        language=effective_lang,
                        answer=answer,
                        citation=citation,
                        confidence=round(nlu_res.confidence, 3),
                        is_abstention=False,
                        follow_up_offer=offer_prompt,
                        offered_card_id=featured_id,
                        follow_up_chips=chips,
                        suggested_topics=[],
                        intent=FollowUpIntent.SELECT_OPTION.value,
                        session_id=session_id,
                        latency_ms=round(elapsed_ms, 2),
                    )

        # For new information queries with auto/unspecified language, infer the query's language
        if session and (not lang or lang.lower() == "auto"):
            inferred_lang = infer_preferred_language(query, default=session.language)
            effective_lang = inferred_lang
            session.language = effective_lang

        # Standard neural semantic search
        results = self.index.search(query=query, grade=effective_grade, top_k=3)

        if results and results[0][1] >= self.confidence_threshold:
            top_card, score = results[0]
            answer = top_card.get_answer(effective_lang)
            citation = top_card.source.citation_str() if top_card.source else None

            offer_prompt, natural_q, featured_id = self.index.get_random_follow_up_offer(
                top_card.id, lang=effective_lang
            )
            chips = self.index.get_follow_up_chips(
                top_card.id, lang=effective_lang, max_chips=4, featured_id=featured_id
            )

            if session:
                session.last_card_id = top_card.id
                session.last_offered_card_id = featured_id
                session.last_offer_question = natural_q
                session.last_chips = chips

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return OfflineChatResponse(
                query=query,
                card_id=top_card.id,
                topic=top_card.topic,
                language=effective_lang,
                answer=answer,
                citation=citation,
                confidence=round(score, 3),
                is_abstention=False,
                follow_up_offer=offer_prompt,
                offered_card_id=featured_id,
                follow_up_chips=chips,
                suggested_topics=[],
                intent=FollowUpIntent.NEW_QUERY.value,
                session_id=session_id,
                latency_ms=round(elapsed_ms, 2),
            )

        # Query falls below confidence threshold -> Calibrated Abstention
        top_card_id = results[0][0].id if results else None
        top_score = results[0][1] if results else 0.0

        refusal_msg = self.REFUSAL_MESSAGES.get(effective_lang, self.REFUSAL_MESSAGES["mr"])
        suggested_topics = self.index.get_suggested_topics(lang=effective_lang, count=4)

        exploration_chips: list[FollowUpChip] = []
        if results:
            for card, _ in results[:3]:
                exploration_chips.append(
                    FollowUpChip(
                        card_id=card.id,
                        title=card.get_title(effective_lang),
                        topic=card.topic,
                        language=effective_lang,
                    )
                )

        if session:
            session.last_offered_card_id = None
            session.last_offer_question = None
            session.last_chips = exploration_chips

        log_entry = OfflineQueryLog(
            query=query,
            language=effective_lang,
            grade=effective_grade,
            is_abstention=True,
            top_matched_card_id=top_card_id,
            top_score=round(top_score, 3),
        )
        self.sync_queue.append(log_entry)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return OfflineChatResponse(
            query=query,
            card_id=None,
            topic=None,
            language=effective_lang,
            answer=refusal_msg,
            citation=None,
            confidence=round(top_score, 3),
            is_abstention=True,
            follow_up_offer=None,
            offered_card_id=None,
            follow_up_chips=exploration_chips,
            suggested_topics=suggested_topics,
            intent=FollowUpIntent.NEW_QUERY.value,
            session_id=session_id,
            latency_ms=round(elapsed_ms, 2),
        )

    def get_card_by_id(
        self,
        card_id: str,
        lang: str = "mr",
        session_id: str | None = None,
    ) -> OfflineChatResponse | None:
        """Directly retrieves a card by ID when a user selects a chip."""
        start_time = time.perf_counter()
        card = self.index.get_card(card_id)
        if not card:
            return None

        answer = card.get_answer(lang)
        citation = card.source.citation_str() if card.source else None
        offer_prompt, natural_q, featured_id = self.index.get_random_follow_up_offer(
            card_id, lang=lang
        )
        chips = self.index.get_follow_up_chips(
            card_id, lang=lang, max_chips=4, featured_id=featured_id
        )

        if session_id:
            session = self.get_or_create_session(session_id, lang=lang)
            session.last_card_id = card.id
            session.last_offered_card_id = featured_id
            session.last_offer_question = natural_q
            session.last_chips = chips
            session.language = lang

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return OfflineChatResponse(
            query=f"chip_click:{card_id}",
            card_id=card.id,
            topic=card.topic,
            language=lang,
            answer=answer,
            citation=citation,
            confidence=1.0,
            is_abstention=False,
            follow_up_offer=offer_prompt,
            offered_card_id=featured_id,
            follow_up_chips=chips,
            suggested_topics=[],
            intent=FollowUpIntent.SELECT_OPTION.value,
            session_id=session_id,
            latency_ms=round(elapsed_ms, 2),
        )

    def get_sync_queue(self) -> list[OfflineQueryLog]:
        """Returns the list of queued offline queries."""
        return list(self.sync_queue)

    def clear_sync_queue(self) -> None:
        """Clears the sync queue after successful synchronization."""
        self.sync_queue.clear()

    def export_sync_queue(self, filepath: Path | str) -> int:
        """Exports queued queries to a JSON file on disk."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = [entry.model_dump() for entry in self.sync_queue]
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return len(data)

    def run_batch_debug(
        self,
        queries: list[str],
        lang: str | None = None,
        grade: int | None = None,
        conversational: bool = True,
    ) -> BatchDebugResponse:
        """Executes batch debugging over a list of queries.

        Supports chained conversational simulation or independent per-query evaluation.
        """
        import uuid

        results: list[DebugQueryResult] = []
        lang_counts: dict[str, int] = {}
        intent_counts: dict[str, int] = {}
        total_latency = 0.0
        matched_count = 0
        abstention_count = 0

        chained_session_id = f"batch-debug-{uuid.uuid4().hex[:8]}" if conversational else None

        for idx, raw_query in enumerate(queries, 1):
            clean_q = raw_query.strip().strip("\",'{[]}:").strip()
            if not clean_q or clean_q in {"{", "}", "[", "]", "question_aliases"}:
                continue

            session_id = chained_session_id if conversational else f"query-{idx}-{uuid.uuid4().hex[:6]}"
            resp = self.resolve_query(clean_q, session_id=session_id, lang=lang, grade=grade)

            is_match = not resp.is_abstention
            if is_match:
                matched_count += 1
            else:
                abstention_count += 1

            lang_counts[resp.language] = lang_counts.get(resp.language, 0) + 1
            intent_key = resp.intent or ("abstention" if resp.is_abstention else "new_query")
            intent_counts[intent_key] = intent_counts.get(intent_key, 0) + 1
            total_latency += resp.latency_ms

            chip_titles = [c.title for c in resp.follow_up_chips]

            results.append(
                DebugQueryResult(
                    line_number=idx,
                    query=clean_q,
                    language=resp.language,
                    intent=resp.intent,
                    card_id=resp.card_id,
                    card_topic=resp.topic,
                    confidence=resp.confidence,
                    is_abstention=resp.is_abstention,
                    answer_preview=resp.answer[:140] + ("..." if len(resp.answer) > 140 else ""),
                    citation=resp.citation,
                    follow_up_offer=resp.follow_up_offer,
                    offered_card_id=resp.offered_card_id,
                    follow_up_chips=chip_titles,
                    latency_ms=resp.latency_ms,
                )
            )

        total_q = len(results)
        avg_lat = round(total_latency / total_q, 2) if total_q > 0 else 0.0
        rate = round((matched_count / total_q) * 100.0, 1) if total_q > 0 else 0.0

        return BatchDebugResponse(
            total_queries=total_q,
            matched_count=matched_count,
            abstention_count=abstention_count,
            match_rate_pct=rate,
            avg_latency_ms=avg_lat,
            language_breakdown=lang_counts,
            intent_breakdown=intent_counts,
            results=results,
        )
