from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import json
import html
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.offline.engine import OfflineFactBot
from app.offline.models import (
    BatchDebugResponse,
    DebugQueryResult,
    OfflineChatResponse,
    OfflineQueryLog,
)

BANNER_ART = r"""
========================================================================================
  ██████╗ ██╗   ██╗██████╗ ██╗   ██╗██╗  ██╗██╗   ██╗██╗     
 ██╔════╝ ██║   ██║██╔══██╗██║   ██║██║ ██╔╝██║   ██║██║     
 ██║  ███╗██║   ██║██████╔╝██║   ██║█████╔╝ ██║   ██║██║     
 ██║   ██║██║   ██║██╔══██╗██║   ██║██╔═██╗ ██║   ██║██║     
 ╚██████╔╝╚██████╔╝██║  ██║╚██████╔╝██║  ██╗╚██████╔╝███████╗
  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
========================================================================================
  GURUKUL (गुरुकुल) — Offline Educational Assistant for Maharashtra Schools
  Curriculum Grounded · Zero Hallucination · Neural Semantic Search (ONNX Runtime)
========================================================================================
"""


class ChatRequest(BaseModel):
    query: str = Field(..., description="The student query to resolve")
    session_id: str | None = Field(None, description="Optional conversation session ID for multi-turn dialogue")
    lang: str | None = Field(None, description="Preferred language: mr, hi, or en")
    grade: int | None = Field(None, description="Optional grade filter (1-8)")


class BatchDebugJsonRequest(BaseModel):
    queries: list[str] = Field(..., description="List of queries to evaluate")
    lang: str | None = Field(None, description="Language override: mr, hi, en, or auto")
    grade: int | None = Field(None, description="Optional grade filter (1-8)")
    conversational: bool = Field(True, description="Whether to simulate conversational flow")


class HealthResponse(BaseModel):
    status: str
    service: str
    indexed_cards: int
    vector_dimension: int
    embeddings_cache_ready: bool
    offline_mode: bool
    uptime_seconds: float


def create_api_app(bot: OfflineFactBot | None = None) -> FastAPI:
    """Creates the FastAPI REST application."""
    start_time = time.time()
    if bot is None:
        bot = OfflineFactBot()

    app = FastAPI(
        title="Gurukul AI — Educational REST API",
        description="Offline-first curriculum chatbot API powered by Balbharati fact cards and ONNX semantic search.",
        version="1.0.0",
    )

    api_app_routes(app, bot, start_time)
    return app


def create_unified_app(bot: OfflineFactBot | None = None) -> FastAPI:
    """Creates the unified FastAPI app with Gradio Studio mounted at /studio and root."""
    start_time = time.time()
    if bot is None:
        bot = OfflineFactBot()

    app = FastAPI(
        title="Gurukul AI — Educational Engine & Studio",
        description="Unified educational service: FastAPI REST endpoints + Gradio Studio.",
        version="1.0.0",
    )

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/studio")

    api_app_routes(app, bot, start_time)

    # Mount Gradio Studio (Pure native components, zero custom HTML)
    studio = build_studio(bot)
    app = gr.mount_gradio_app(app, studio, path="/studio")

    return app


def api_app_routes(app: FastAPI, bot: OfflineFactBot, start_time: float) -> None:
    """Registers standard REST endpoints on the FastAPI app."""

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        has_vectors = bot.index.vectors is not None
        dim = int(bot.index.vectors.shape[1]) if has_vectors else 0
        return HealthResponse(
            status="ok",
            service="gurukul-offline-engine",
            indexed_cards=len(bot.index.cards_by_id),
            vector_dimension=dim,
            embeddings_cache_ready=bot.index.cache_file.exists(),
            offline_mode=True,
            uptime_seconds=round(time.time() - start_time, 2),
        )

    @app.post("/api/chat", response_model=OfflineChatResponse)
    async def chat_endpoint(request: ChatRequest):
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        return bot.resolve_query(
            query=request.query,
            session_id=request.session_id,
            lang=request.lang,
            grade=request.grade,
        )

    @app.post("/api/sessions/{session_id}/reset")
    async def reset_session_endpoint(session_id: str):
        bot.reset_session(session_id)
        return {"status": "reset", "session_id": session_id}

    @app.get("/api/cards/search")
    async def search_cards_endpoint(q: str = ""):
        if not q.strip():
            return {"query": q, "total_matches": 0, "cards": []}
        q_clean = q.strip().lower()
        matches = []
        for cid, card in bot.index.cards_by_id.items():
            if q_clean in cid.lower() or any(q_clean in a.lower() for a in card.question_aliases):
                matches.append({
                    "id": card.id,
                    "title": card.get_offer_question("en"),
                    "subject": card.subject,
                    "grades": card.grades,
                    "follow_up_cards": card.follow_up_cards,
                })
        return {"query": q, "total_matches": len(matches), "cards": matches[:50]}

    @app.get("/api/cards/{card_id}")
    async def get_card_endpoint(card_id: str, lang: str = "mr"):
        card = bot.index.get_card(card_id)
        if not card:
            raise HTTPException(status_code=404, detail=f"Card '{card_id}' not found.")
        return {
            "id": card.id,
            "title": card.get_title(lang),
            "topic": card.topic,
            "grades": card.grades,
            "subject": card.subject,
            "answer": card.get_answer(lang),
            "citation": card.source.citation_str() if card.source else None,
            "follow_up_cards": card.follow_up_cards,
            "question_aliases": card.question_aliases,
        }

    @app.get("/api/curriculum")
    async def curriculum_tree_endpoint():
        return bot.index.get_curriculum_tree()

    @app.get("/api/sync-queue", response_model=list[OfflineQueryLog])
    async def get_sync_queue_endpoint():
        return bot.get_sync_queue()

    @app.delete("/api/sync-queue")
    async def clear_sync_queue_endpoint():
        count = len(bot.get_sync_queue())
        bot.clear_sync_queue()
        return {"status": "cleared", "records_cleared": count}

    @app.post("/api/debug/batch", response_model=BatchDebugResponse)
    async def batch_debug_endpoint(
        file: UploadFile | None = File(None, description="Uploaded .txt file with one query per line"),
        queries_text: str | None = Form(None, description="Multiline text of queries (one per line)"),
        lang: str | None = Form(None, description="Language override: mr, hi, en, or auto"),
        grade: int | None = Form(None, description="Optional grade filter (1-8)"),
        conversational: bool = Form(True, description="Whether to simulate conversational chain across turns"),
    ):
        queries: list[str] = []
        if file and file.filename:
            content_bytes = await file.read()
            content_str = content_bytes.decode("utf-8", errors="replace")
            queries.extend([line.strip() for line in content_str.splitlines() if line.strip()])
        elif queries_text:
            queries.extend([line.strip() for line in queries_text.splitlines() if line.strip()])

        if not queries:
            raise HTTPException(
                status_code=400,
                detail="No queries provided. Please upload a .txt file or provide queries_text.",
            )

        effective_lang = None if (not lang or lang.lower() == "auto") else lang
        return bot.run_batch_debug(
            queries=queries,
            lang=effective_lang,
            grade=grade,
            conversational=conversational,
        )

    @app.post("/api/debug/batch-json", response_model=BatchDebugResponse)
    async def batch_debug_json_endpoint(request: BatchDebugJsonRequest):
        if not request.queries:
            raise HTTPException(status_code=400, detail="Queries list cannot be empty.")
        effective_lang = None if (not request.lang or request.lang.lower() == "auto") else request.lang
        return bot.run_batch_debug(
            queries=request.queries,
            lang=effective_lang,
            grade=request.grade,
            conversational=request.conversational,
        )


def get_available_benchmark_artifacts() -> list[tuple[str, str]]:
    """Discovers available benchmark input artifact files (.json or .txt)."""
    options = [("Auto-discover latest artifact", "")]
    seen = set()
    for search_dir in [Path("artifacts"), Path.cwd()]:
        if search_dir.exists():
            for p in sorted(search_dir.glob("*.json")) + sorted(search_dir.glob("*.txt")):
                if (
                    p.name.startswith("benchmark_")
                    or p.name.endswith(".metadata.json")
                    or p.name in {"pyproject.toml", "uv.lock"}
                ):
                    continue
                if p.name not in seen:
                    seen.add(p.name)
                    options.append((f"{p.name} ({p.parent.name}/)", str(p.resolve())))
    return options


def load_existing_benchmark_report() -> tuple[str, str, str, dict[str, Any], str | None]:
    """Loads existing benchmark report files if available."""
    html_file = None
    md_file = None
    json_file = None

    for candidate_dir in [Path("artifacts"), Path.cwd()]:
        h = candidate_dir / "benchmark_report.html"
        m = candidate_dir / "benchmark_report.md"
        j = candidate_dir / "benchmark_scorecard.json"
        if h.exists() and html_file is None:
            html_file = h
        if m.exists() and md_file is None:
            md_file = m
        if j.exists() and json_file is None:
            json_file = j

    status_text = (
        "### ℹ️ No Previous Benchmark Report Found\n"
        "Run a benchmark above or choose an artifact to generate evaluation results."
    )
    iframe_html = (
        "<div style='padding: 32px; text-align: center; color: #64748b; font-size: 0.95rem;'>"
        "No benchmark results available yet. Run a benchmark above or select an artifact to view results."
        "</div>"
    )
    md_text = "*No report loaded yet.*"
    scorecard: dict[str, Any] = {}
    download_path = None

    if html_file and html_file.exists():
        html_content = html_file.read_text(encoding="utf-8")
        escaped_html = html.escape(html_content)
        iframe_html = (
            f'<iframe srcdoc="{escaped_html}" '
            f'style="width: 100%; height: 750px; border: 1px solid #cbd5e1; border-radius: 8px; background: #fff;" '
            f'frameborder="0"></iframe>'
        )
        download_path = str(html_file.resolve())

    if md_file and md_file.exists():
        md_text = md_file.read_text(encoding="utf-8")

    if json_file and json_file.exists():
        try:
            scorecard = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            scorecard = {}

    if html_file:
        total_q = scorecard.get("total_queries", "N/A")
        judge = scorecard.get("judge_model", "N/A")
        ts = scorecard.get("timestamp", "N/A")
        status_text = (
            f"### ✅ Benchmark Report Loaded\n"
            f"- **Report Source:** `{html_file.name}`\n"
            f"- **Queries Evaluated:** `{total_q}`\n"
            f"- **Judge Model:** `{judge}`\n"
            f"- **Generated At:** `{ts}`"
        )

    return status_text, iframe_html, md_text, scorecard, download_path


def build_studio(bot: OfflineFactBot) -> gr.Blocks:
    """Builds the Gradio Studio interface using purely native Gradio components (NO custom HTML)."""

    with gr.Blocks(title="Gurukul — Studio") as demo:
        gr.Markdown("# 🎓 Gurukul — Offline Curriculum Studio")
        gr.Markdown(
            "Digital teacher based on Maharashtra State Curriculum (Balbharati). "
            "Local on-device neural search, zero internet dependency, and 100% textbook verification."
        )

        session_state = gr.State(lambda: str(uuid.uuid4()))

        with gr.Tabs():
            # Tab 1: Chat Assistant
            with gr.Tab("💬 Chat Assistant"):
                with gr.Row():
                    with gr.Column(scale=8):
                        chatbot = gr.Chatbot(
                            label="Classroom Dialogue",
                            height=480,
                        )
                    with gr.Column(scale=4):
                        gr.Markdown("### ⚙️ Settings")
                        lang_select = gr.Dropdown(
                            label="Language",
                            choices=[
                                ("English", "en"),
                                ("Auto Detect", "auto"),
                                ("Marathi", "mr"),
                                ("Hindi", "hi"),
                            ],
                            value="en",
                        )
                        grade_select = gr.Dropdown(
                            label="Grade Filter",
                            choices=[
                                ("All Grades", None),
                                ("Grade 1", 1),
                                ("Grade 2", 2),
                                ("Grade 3", 3),
                                ("Grade 4", 4),
                                ("Grade 5", 5),
                                ("Grade 6", 6),
                                ("Grade 7", 7),
                                ("Grade 8", 8),
                            ],
                            value=None,
                        )
                        metrics_box = gr.Markdown(
                            "**System Status:**\n- Mode: 100% Offline\n- Model: FastEmbed MiniLM-L12\n- Accuracy: Textbook Citation Verified"
                        )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Ask Question",
                        placeholder="e.g. What is habitat? / Who was Shivaji Maharaj? / What is the speed of light?...",
                        lines=1,
                        scale=9,
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

                gr.Markdown("#### 🔍 Suggested Follow-up Questions:")
                with gr.Row():
                    chip_btn_1 = gr.Button("Who was Shivaji Maharaj", visible=True)
                    chip_btn_2 = gr.Button("What is habitat", visible=True)
                    chip_btn_3 = gr.Button("What is the speed of light", visible=True)
                    chip_btn_4 = gr.Button("Who composed our national anthem", visible=True)

                gr.Examples(
                    examples=[
                        ["Who was Shivaji Maharaj", "en", None],
                        ["What is habitat", "en", 6],
                        ["What is the speed of light", "en", 8],
                        ["Who composed our national anthem", "en", None],
                        ["Where do animals live and what are their shelters", "en", 6],
                        ["छत्रपती शिवाजी महाराज कोण होते", "mr", None],
                        ["भारत का राष्ट्रगान किसने लिखा", "hi", None],
                        ["crypto bitcoin wallet trading", "en", None],
                    ],
                    inputs=[msg_input, lang_select, grade_select],
                    label="Sample Questions:",
                )

                def user_submit(
                    user_msg: str,
                    history: list[dict[str, str]],
                    lang: str,
                    grade: int | None,
                    session_id: Any,
                ):
                    sid = session_id() if callable(session_id) else (session_id or str(uuid.uuid4()))
                    if not user_msg or not user_msg.strip():
                        return history, "", metrics_box, gr.update(), gr.update(), gr.update(), gr.update(), sid

                    clean_query = user_msg.strip()
                    chosen_lang = None if (not lang or lang == "auto") else lang
                    resp = bot.resolve_query(clean_query, session_id=sid, lang=chosen_lang, grade=grade)

                    history = history or []
                    history.append({"role": "user", "content": clean_query})

                    if resp.is_abstention:
                        topic_header = {
                            "en": "**Suggested Topics:**",
                            "mr": "**सुचवलेले विषय:**",
                            "hi": "**सुझाए गए विषय:**",
                        }.get(resp.language, "**Suggested Topics:**")
                        reply_text = f"⚠️ **{resp.answer}**\n\n"
                        if resp.suggested_topics:
                            reply_text += f"{topic_header}\n" + "\n".join([f"- {t}" for t in resp.suggested_topics])
                    else:
                        citation_header = {
                            "en": "📖 **Citation (Textbook):**",
                            "mr": "📖 **संदर्भ (Textbook Citation):**",
                            "hi": "📖 **संदर्भ (Textbook Citation):**",
                        }.get(resp.language, "📖 **Citation (Textbook):**")
                        latency_label = {
                            "en": "⚡ **Latency:**",
                            "mr": "⚡ **प्रतिसाद वेळ:**",
                            "hi": "⚡ **प्रतिक्रिया समय:**",
                        }.get(resp.language, "⚡ **Latency:**")
                        conf_label = {
                            "en": "**Confidence:**",
                            "mr": "**विश्वासार्हता:**",
                            "hi": "**सटीकता:**",
                        }.get(resp.language, "**Confidence:**")
                        reply_text = f"{resp.answer}\n\n"
                        if resp.citation:
                            reply_text += f"{citation_header} {resp.citation}\n\n"
                        if resp.follow_up_offer:
                            reply_text += f"👉 *{resp.follow_up_offer}*\n\n"
                        reply_text += f"{latency_label} {resp.latency_ms} ms | {conf_label} {resp.confidence}"

                    history.append({"role": "assistant", "content": reply_text})

                    lang_names = {"en": "English (en)", "mr": "Marathi (mr)", "hi": "Hindi (hi)"}
                    status_markdown = (
                        f"**Last Response:**\n"
                        f"- Card: `{resp.card_id or 'Abstained'}`\n"
                        f"- Intent: `{resp.intent or 'new_query'}`\n"
                        f"- Language: `{lang_names.get(resp.language, resp.language)}`\n"
                        f"- Latency: `{resp.latency_ms} ms`\n"
                        f"- Confidence: `{resp.confidence}`"
                    )

                    chips = resp.follow_up_chips or []
                    default_chips = {
                        "en": [
                            "Who was Shivaji Maharaj",
                            "What is habitat",
                            "What is the speed of light",
                            "Who wrote the national anthem",
                        ],
                        "mr": [
                            "छत्रपती शिवाजी महाराज कोण होते",
                            "अधिवास म्हणजे काय",
                            "प्रकाशाचा वेग किती आहे",
                            "राष्ट्रगीत कोणी लिहिले",
                        ],
                        "hi": [
                            "छत्रपति शिवाजी महाराज कौन थे",
                            "अधिवास क्या है",
                            "प्रकाश का वेग कितना है",
                            "भारत का राष्ट्रगान किसने लिखा",
                        ],
                    }
                    def_list = default_chips.get(resp.language, default_chips["en"])
                    c1_text = chips[0].title if len(chips) > 0 else def_list[0]
                    c2_text = chips[1].title if len(chips) > 1 else def_list[1]
                    c3_text = chips[2].title if len(chips) > 2 else def_list[2]
                    c4_text = chips[3].title if len(chips) > 3 else def_list[3]

                    return (
                        history,
                        "",
                        status_markdown,
                        gr.update(value=c1_text, visible=True),
                        gr.update(value=c2_text, visible=True),
                        gr.update(value=c3_text, visible=True),
                        gr.update(value=c4_text, visible=True),
                        sid,
                    )

                submit_btn.click(
                    fn=user_submit,
                    inputs=[msg_input, chatbot, lang_select, grade_select, session_state],
                    outputs=[chatbot, msg_input, metrics_box, chip_btn_1, chip_btn_2, chip_btn_3, chip_btn_4, session_state],
                )
                msg_input.submit(
                    fn=user_submit,
                    inputs=[msg_input, chatbot, lang_select, grade_select, session_state],
                    outputs=[chatbot, msg_input, metrics_box, chip_btn_1, chip_btn_2, chip_btn_3, chip_btn_4, session_state],
                )

                def clear_chat(session_id: Any):
                    sid = session_id() if callable(session_id) else session_id
                    if sid:
                        bot.reset_session(sid)
                    new_sid = str(uuid.uuid4())
                    return [], "", "**Info:** Conversation cleared.", new_sid

                clear_btn.click(
                    fn=clear_chat,
                    inputs=[session_state],
                    outputs=[chatbot, msg_input, metrics_box, session_state],
                )

                def click_chip(chip_text, history, lang, grade, session_id):
                    return user_submit(chip_text, history, lang, grade, session_id)

                chip_btn_1.click(
                    fn=click_chip,
                    inputs=[chip_btn_1, chatbot, lang_select, grade_select, session_state],
                    outputs=[chatbot, msg_input, metrics_box, chip_btn_1, chip_btn_2, chip_btn_3, chip_btn_4, session_state],
                )
                chip_btn_2.click(
                    fn=click_chip,
                    inputs=[chip_btn_2, chatbot, lang_select, grade_select, session_state],
                    outputs=[chatbot, msg_input, metrics_box, chip_btn_1, chip_btn_2, chip_btn_3, chip_btn_4, session_state],
                )
                chip_btn_3.click(
                    fn=click_chip,
                    inputs=[chip_btn_3, chatbot, lang_select, grade_select, session_state],
                    outputs=[chatbot, msg_input, metrics_box, chip_btn_1, chip_btn_2, chip_btn_3, chip_btn_4, session_state],
                )
                chip_btn_4.click(
                    fn=click_chip,
                    inputs=[chip_btn_4, chatbot, lang_select, grade_select, session_state],
                    outputs=[chatbot, msg_input, metrics_box, chip_btn_1, chip_btn_2, chip_btn_3, chip_btn_4, session_state],
                )

            # Tab 2: Curriculum Browser & Card ID Search
            with gr.Tab("📚 Curriculum Tree"):
                gr.Markdown("### 🔍 Search & Browse Curriculum Fact Cards")
                gr.Markdown(
                    "Search fact cards directly by **Card ID** (e.g., `person_shivaji_maharaj`, `kb_25d82af7`, `science_speed_of_light`) "
                    "or concept keyword, and inspect textbook answers, citations, aliases, and follow-up connectivity."
                )

                with gr.Row():
                    card_search_input = gr.Textbox(
                        label="Search by Card ID or Concept Keyword",
                        placeholder="e.g. person_shivaji_maharaj, kb_25d82af7, national_anthem, habitat, speed of light...",
                        lines=1,
                        scale=9,
                    )
                    card_search_btn = gr.Button("🔍 Search Card ID", variant="primary", scale=2)
                    card_reset_btn = gr.Button("Reset", variant="secondary", scale=1)

                card_search_status = gr.Markdown(value="*Enter a Card ID or keyword above to inspect card details.*")
                matching_cards_dropdown = gr.Dropdown(
                    label="Multiple Matches Found (Select Card to View)",
                    choices=[],
                    visible=False,
                )
                card_details_json = gr.JSON(
                    label="Card Inspector (Grounded Content & Metadata)",
                    value={},
                )

                def search_card_ui(query_text: str):
                    if not query_text or not query_text.strip():
                        return (
                            "*Please enter a Card ID or keyword to search.*",
                            gr.update(visible=False, choices=[]),
                            {},
                        )
                    q = query_text.strip()

                    # 1. Exact ID match
                    card = bot.index.get_card(q)
                    if card:
                        status_md = (
                            f"### ✅ Found Card: `{card.id}`\n"
                            f"- **Title / Offer Question:** {card.get_offer_question('en')}\n"
                            f"- **Subject:** `{card.subject}` | **Grades:** `{card.grades}`\n"
                            f"- **Citation:** `{card.source.citation_str() if card.source else 'N/A'}`\n"
                            f"- **Follow-up Connected Cards:** `{card.follow_up_cards}`"
                        )
                        card_data = {
                            "id": card.id,
                            "title_en": card.get_offer_question("en"),
                            "title_mr": card.get_offer_question("mr"),
                            "title_hi": card.get_offer_question("hi"),
                            "subject": card.subject,
                            "grades": card.grades,
                            "citation": card.source.citation_str() if card.source else None,
                            "answers": {
                                "en": card.get_answer("en"),
                                "mr": card.get_answer("mr"),
                                "hi": card.get_answer("hi"),
                            },
                            "follow_up_cards": card.follow_up_cards,
                            "question_aliases": card.question_aliases,
                            "entities": card.entities,
                        }
                        return status_md, gr.update(visible=False, choices=[]), card_data

                    # 2. Substring / Alias search across all 952 cards
                    q_lower = q.lower()
                    matches = []
                    for cid, c in bot.index.cards_by_id.items():
                        if q_lower in cid.lower():
                            matches.append((f"{cid} ({c.get_offer_question('en')[:50]})", cid))
                        elif any(q_lower in a.lower() for a in c.question_aliases):
                            matches.append((f"{cid} ({c.get_offer_question('en')[:50]})", cid))

                    if matches:
                        first_cid = matches[0][1]
                        c = bot.index.get_card(first_cid)
                        status_md = (
                            f"### 🔍 Found {len(matches)} matching cards for `{q}`\n"
                            f"- Currently viewing: `{first_cid}`\n"
                            f"- Pick other matches from the dropdown below:"
                        )
                        card_data = {
                            "id": c.id,
                            "title_en": c.get_offer_question("en"),
                            "title_mr": c.get_offer_question("mr"),
                            "title_hi": c.get_offer_question("hi"),
                            "subject": c.subject,
                            "grades": c.grades,
                            "citation": c.source.citation_str() if c.source else None,
                            "answers": {
                                "en": c.get_answer("en"),
                                "mr": c.get_answer("mr"),
                                "hi": c.get_answer("hi"),
                            },
                            "follow_up_cards": c.follow_up_cards,
                            "question_aliases": c.question_aliases,
                            "entities": c.entities,
                        }
                        return (
                            status_md,
                            gr.update(visible=True, choices=matches, value=first_cid),
                            card_data,
                        )

                    return (
                        f"⚠️ No card found matching Card ID or keyword `{q}`.",
                        gr.update(visible=False, choices=[]),
                        {},
                    )

                def on_select_match(selected_cid: str):
                    if not selected_cid:
                        return "", {}
                    c = bot.index.get_card(selected_cid)
                    if not c:
                        return "", {}
                    status_md = (
                        f"### ✅ Inspecting Card: `{c.id}`\n"
                        f"- **Title / Offer Question:** {c.get_offer_question('en')}\n"
                        f"- **Subject:** `{c.subject}` | **Grades:** `{c.grades}`\n"
                        f"- **Citation:** `{c.source.citation_str() if c.source else 'N/A'}`\n"
                        f"- **Follow-up Connected Cards:** `{c.follow_up_cards}`"
                    )
                    card_data = {
                        "id": c.id,
                        "title_en": c.get_offer_question("en"),
                        "title_mr": c.get_offer_question("mr"),
                        "title_hi": c.get_offer_question("hi"),
                        "subject": c.subject,
                        "grades": c.grades,
                        "citation": c.source.citation_str() if c.source else None,
                        "answers": {
                            "en": c.get_answer("en"),
                            "mr": c.get_answer("mr"),
                            "hi": c.get_answer("hi"),
                        },
                        "follow_up_cards": c.follow_up_cards,
                        "question_aliases": c.question_aliases,
                        "entities": c.entities,
                    }
                    return status_md, card_data

                def on_reset_search():
                    return "*Enter a Card ID or keyword above to inspect card details.*", gr.update(visible=False, choices=[]), {}, ""

                card_search_btn.click(
                    fn=search_card_ui,
                    inputs=[card_search_input],
                    outputs=[card_search_status, matching_cards_dropdown, card_details_json],
                )
                card_search_input.submit(
                    fn=search_card_ui,
                    inputs=[card_search_input],
                    outputs=[card_search_status, matching_cards_dropdown, card_details_json],
                )
                matching_cards_dropdown.change(
                    fn=on_select_match,
                    inputs=[matching_cards_dropdown],
                    outputs=[card_search_status, card_details_json],
                )
                card_reset_btn.click(
                    fn=on_reset_search,
                    outputs=[card_search_status, matching_cards_dropdown, card_details_json, card_search_input],
                )

                with gr.Accordion("🌳 Full Hierarchical Curriculum Tree (Grades & Subjects)", open=False):
                    gr.JSON(
                        value=bot.index.get_curriculum_tree(),
                        label="Curriculum Hierarchy",
                    )

            # Tab 3: Offline Sync Queue
            with gr.Tab("📋 Sync Queue"):
                gr.Markdown("### Unanswered Queries Queue")
                gr.Markdown(
                    "When students ask questions that are not available in current fact cards, "
                    "those queries are logged locally here and synced for teacher review once internet connectivity is available."
                )

                queue_view = gr.JSON(
                    value=[item.model_dump() for item in bot.get_sync_queue()],
                    label="Local Records",
                )
                with gr.Row():
                    refresh_queue_btn = gr.Button("Refresh Queue")
                    clear_queue_btn = gr.Button("Clear Queue", variant="stop")

                def refresh_queue():
                    return [item.model_dump() for item in bot.get_sync_queue()]

                def clear_queue_action():
                    bot.clear_sync_queue()
                    return []

                refresh_queue_btn.click(fn=refresh_queue, outputs=[queue_view])
                clear_queue_btn.click(fn=clear_queue_action, outputs=[queue_view])

            # Tab 4: Batch Evaluation & Flow Debugger
            with gr.Tab("🔬 Batch Debugger"):
                gr.Markdown("### 🧪 Batch Query Testing & Flow Debugger")
                gr.Markdown(
                    "Upload a `.txt` file containing one query per line or paste queries directly to test and inspect "
                    "the neural semantic search, NLU dialogue intent recognition, follow-up offers, and response accuracy."
                )

                with gr.Row():
                    with gr.Column(scale=6):
                        dbg_file_upload = gr.File(
                            label="Upload .txt File (One query per line)",
                            file_types=[".txt"],
                        )
                        dbg_pasted_queries = gr.Textbox(
                            label="Or Paste Queries (One query per line)",
                            lines=6,
                            placeholder="who was shivaji maharaj\nyes\nwhat is photosynthesis\nराष्ट्रगीत कोणी लिहिले\nहो सांगा\nभारत का राष्ट्रगान किसने लिखा\nनहीं",
                        )
                    with gr.Column(scale=4):
                        gr.Markdown("#### ⚙️ Debugger Settings")
                        dbg_lang_select = gr.Dropdown(
                            label="Language Mode",
                            choices=[
                                ("Auto Detect", "auto"),
                                ("English (en)", "en"),
                                ("Marathi (mr)", "mr"),
                                ("Hindi (hi)", "hi"),
                            ],
                            value="auto",
                        )
                        dbg_grade_select = gr.Dropdown(
                            label="Grade Filter",
                            choices=[
                                ("All Grades", None),
                                ("Grade 1", 1),
                                ("Grade 2", 2),
                                ("Grade 3", 3),
                                ("Grade 4", 4),
                                ("Grade 5", 5),
                                ("Grade 6", 6),
                                ("Grade 7", 7),
                                ("Grade 8", 8),
                            ],
                            value=None,
                        )
                        dbg_flow_mode = gr.Radio(
                            label="Execution Mode",
                            choices=[
                                ("Simulate Dialogue Flow (Chained Conversation)", True),
                                ("Independent Query Evaluations", False),
                            ],
                            value=True,
                        )
                        run_batch_btn = gr.Button("🚀 Run Batch Debugging", variant="primary")

                gr.Markdown("### 📊 Evaluation Metrics")
                dbg_summary_md = gr.Markdown(value="*Upload a file or paste queries above, then click 'Run Batch Debugging' to view results.*")

                dbg_results_df = gr.DataFrame(
                    headers=["#", "Query", "Lang", "Intent", "Card ID", "Conf", "Match?", "Follow-up Offer", "Latency (ms)"],
                    datatype=["number", "str", "str", "str", "str", "number", "str", "str", "number"],
                    label="Query Results Grid",
                    interactive=False,
                )

                with gr.Accordion("🔍 Structured JSON Report & Export", open=False):
                    dbg_json_view = gr.JSON(label="Full Batch Debug Response")
                    dbg_download_file = gr.File(label="Download Report (.json)")

                def run_batch_ui(file_obj, text_input, lang, grade, is_conversational):
                    queries: list[str] = []
                    if file_obj is not None:
                        try:
                            file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
                            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                                for line in f:
                                    clean_l = line.strip()
                                    if clean_l:
                                        queries.append(clean_l)
                        except Exception as e:
                            return f"⚠️ Error reading file: {e}", [], {}, None
                    elif text_input and text_input.strip():
                        queries = [l.strip() for l in text_input.splitlines() if l.strip()]

                    if not queries:
                        return "⚠️ Please upload a .txt file or paste at least one query.", [], {}, None

                    chosen_lang = None if (not lang or lang == "auto") else lang
                    resp = bot.run_batch_debug(
                        queries=queries,
                        lang=chosen_lang,
                        grade=grade,
                        conversational=is_conversational,
                    )

                    summary = (
                        f"**Batch Evaluation Summary:**\n"
                        f"- **Total Queries Processed:** `{resp.total_queries}`\n"
                        f"- **Matched Curriculum Cards:** `{resp.matched_count}` ({resp.match_rate_pct}%)\n"
                        f"- **Abstentions (Out of syllabus):** `{resp.abstention_count}`\n"
                        f"- **Average Latency:** `{resp.avg_latency_ms} ms`\n"
                        f"- **Language Distribution:** `{dict(resp.language_breakdown)}`\n"
                        f"- **Intent Distribution:** `{dict(resp.intent_breakdown)}`"
                    )

                    table_rows = []
                    for r in resp.results:
                        table_rows.append([
                            r.line_number,
                            r.query,
                            r.language,
                            r.intent or "-",
                            r.card_id or "Abstained",
                            r.confidence,
                            "❌ Abstained" if r.is_abstention else "✅ Matched",
                            r.follow_up_offer or "-",
                            r.latency_ms,
                        ])

                    temp_f = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
                    json.dump(resp.model_dump(), temp_f, ensure_ascii=False, indent=2)
                    temp_f.close()

                    return summary, table_rows, resp.model_dump(), temp_f.name

                run_batch_btn.click(
                    fn=run_batch_ui,
                    inputs=[dbg_file_upload, dbg_pasted_queries, dbg_lang_select, dbg_grade_select, dbg_flow_mode],
                    outputs=[dbg_summary_md, dbg_results_df, dbg_json_view, dbg_download_file],
                )

            # Tab 5: Benchmark Suite
            with gr.Tab("🏆 Benchmark Suite"):
                gr.Markdown("### 🏆 DeepEval LLM-as-a-Judge Benchmark Suite")
                gr.Markdown(
                    "Evaluate offline model responses using DeepEval and 4 custom G-Eval rubrics: "
                    "**Curriculum Grounding**, **Language Consistency**, **Follow-up Offer Quality**, and **Abstention Accuracy**. "
                    "Run benchmarks or inspect interactive reports directly in the studio."
                )

                init_status, init_iframe, init_md, init_scorecard, init_dl = load_existing_benchmark_report()

                with gr.Row():
                    with gr.Column(scale=6):
                        bench_source_mode = gr.Radio(
                            label="Input Source",
                            choices=[
                                "Existing Artifact File",
                                "Upload File (.json / .txt)",
                                "Paste Queries Directly",
                            ],
                            value="Existing Artifact File",
                        )
                        art_choices = get_available_benchmark_artifacts()
                        bench_artifact_dropdown = gr.Dropdown(
                            label="Select Existing Artifact",
                            choices=art_choices,
                            value=art_choices[0][1] if art_choices else "",
                            visible=True,
                        )
                        bench_file_upload = gr.File(
                            label="Upload .json or .txt File",
                            file_types=[".json", ".txt"],
                            visible=False,
                        )
                        bench_query_text = gr.Textbox(
                            label="Paste Queries (One query per line)",
                            lines=4,
                            placeholder="who was shivaji maharaj\nप्रकाशसंश्लेषण म्हणजे काय\nभारत का राष्ट्रगान किसने लिखा\nhow to build a rocket to mars",
                            visible=False,
                        )
                    with gr.Column(scale=4):
                        gr.Markdown("#### ⚙️ Benchmark Settings")
                        from gurukul.benchmarks.settings import settings as benchmark_settings

                        env_judge = benchmark_settings.effective_judge
                        bench_judge_model = gr.Dropdown(
                            label="Judge Model",
                            choices=[
                                ("Default (.env configuration)", "default"),
                                ("gemini-1.5-flash", "gemini-1.5-flash"),
                                ("gpt-4o-mini", "gpt-4o-mini"),
                                ("gemma4:31b-cloud", "gemma4:31b-cloud"),
                                ("ollama/llama3.1:8b", "ollama/llama3.1:8b"),
                            ],
                            value="default",
                            info=f"Default uses .env configuration (currently: {env_judge})",
                            allow_custom_value=True,
                        )
                        with gr.Row():
                            bench_dry_run = gr.Checkbox(
                                label="Dry Run (Fast offline validation without LLM API)",
                                value=True,
                            )
                            bench_rerun = gr.Checkbox(
                                label="Re-run queries through bot first",
                                value=False,
                            )
                        bench_workers = gr.Slider(
                            minimum=1,
                            maximum=8,
                            step=1,
                            value=4,
                            label="Parallel Workers",
                        )
                        with gr.Row():
                            bench_run_btn = gr.Button("🚀 Run Benchmark", variant="primary", scale=2)
                            bench_load_btn = gr.Button("📂 Load Latest Report", variant="secondary", scale=1)

                gr.Markdown("### 📊 Benchmark Results & Scorecards")
                bench_status_md = gr.Markdown(value=init_status)

                with gr.Tabs():
                    with gr.Tab("🌐 Interactive HTML Dashboard"):
                        bench_html_view = gr.HTML(value=init_iframe, label="Interactive Report")
                    with gr.Tab("📝 Markdown Scorecard"):
                        bench_md_view = gr.Markdown(value=init_md)
                    with gr.Tab("🔍 Scorecard JSON"):
                        bench_json_view = gr.JSON(value=init_scorecard, label="Scorecard Raw JSON")
                    with gr.Tab("💾 Download Artifacts"):
                        bench_download_file = gr.File(value=init_dl, label="Download Interactive Report (.html)")

                def on_source_mode_change(mode: str):
                    if mode == "Existing Artifact File":
                        choices = get_available_benchmark_artifacts()
                        default_val = choices[0][1] if choices else ""
                        return (
                            gr.update(visible=True, choices=choices, value=default_val),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )
                    elif mode == "Upload File (.json / .txt)":
                        return (
                            gr.update(visible=False),
                            gr.update(visible=True),
                            gr.update(visible=False),
                        )
                    else:  # "Paste Queries Directly"
                        return (
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=True),
                        )

                bench_source_mode.change(
                    fn=on_source_mode_change,
                    inputs=[bench_source_mode],
                    outputs=[bench_artifact_dropdown, bench_file_upload, bench_query_text],
                )

                def run_benchmark_ui(
                    source_mode: str,
                    selected_artifact: str,
                    uploaded_file: Any,
                    pasted_queries: str,
                    judge_model: str,
                    is_dry_run: bool,
                    is_rerun: bool,
                    worker_count: int,
                    progress=gr.Progress(track_tqdm=True),
                ):
                    progress(0.02, desc="Preparing benchmark input artifact...")
                    input_path = None
                    if source_mode == "Upload File (.json / .txt)":
                        if uploaded_file is None:
                            return (
                                "### ⚠️ Error: Please upload a .json or .txt file.",
                                gr.update(),
                                gr.update(),
                                {},
                                None,
                            )
                        input_path = uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
                    elif source_mode == "Paste Queries Directly":
                        if not pasted_queries or not pasted_queries.strip():
                            return (
                                "### ⚠️ Error: Please paste at least one query.",
                                gr.update(),
                                gr.update(),
                                {},
                                None,
                            )
                        progress(0.04, desc="Saving pasted queries to temporary artifact...")
                        temp_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
                        temp_txt.write(pasted_queries.strip())
                        temp_txt.close()
                        input_path = temp_txt.name
                    else:  # Existing Artifact File
                        input_path = selected_artifact.strip() if selected_artifact and selected_artifact.strip() else None

                    target_out_dir = Path("artifacts")
                    target_out_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        from gurukul.benchmarks.run_benchmark import run as run_benchmark_runner
                        from gurukul.benchmarks.settings import settings as benchmark_settings

                        judge_str = (judge_model or "").strip()
                        if not judge_str or judge_str.lower() == "default" or judge_str.lower().startswith("default"):
                            effective_judge_arg = None
                            display_judge = f"{benchmark_settings.effective_judge} (.env default)"
                        else:
                            effective_judge_arg = judge_str
                            display_judge = judge_str

                        def update_progress(ratio: float, desc: str):
                            try:
                                progress(ratio, desc=desc)
                            except Exception:
                                pass

                        html_file = run_benchmark_runner(
                            input_path=input_path,
                            judge_model=effective_judge_arg,
                            output_dir=target_out_dir,
                            dry_run=is_dry_run,
                            rerun=is_rerun,
                            workers=int(worker_count) if worker_count else 4,
                            bot=bot,
                            progress_callback=update_progress,
                        )

                        progress(0.90, desc="Rendering interactive HTML dashboard and reports...")
                        html_content = html_file.read_text(encoding="utf-8")
                        escaped_html = html.escape(html_content)
                        iframe_html = (
                            f'<iframe srcdoc="{escaped_html}" '
                            f'style="width: 100%; height: 750px; border: 1px solid #cbd5e1; border-radius: 8px; background: #fff;" '
                            f'frameborder="0"></iframe>'
                        )

                        md_file = target_out_dir / "benchmark_report.md"
                        md_text = md_file.read_text(encoding="utf-8") if md_file.exists() else "*Report generated.*"

                        json_file = target_out_dir / "benchmark_scorecard.json"
                        scorecard = {}
                        if json_file.exists():
                            try:
                                scorecard = json.loads(json_file.read_text(encoding="utf-8"))
                            except Exception:
                                scorecard = {}
                        else:
                            scorecard = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "input_artifact": str(input_path or "auto-discovered"),
                                "judge_model": f"{display_judge} ({'dry-run' if is_dry_run else 'evaluated'})",
                                "dry_run": is_dry_run,
                            }

                        mode_label = "Dry-Run Validation" if is_dry_run else "Live DeepEval Evaluation"
                        status_text = (
                            f"### ✅ Benchmark Run Completed ({mode_label})\n"
                            f"- **Artifact Generated:** `{html_file.name}`\n"
                            f"- **Judge Model:** `{display_judge}`\n"
                            f"- **Mode:** `{'Dry Run (No API calls)' if is_dry_run else 'Live LLM-as-a-Judge'}`\n"
                            f"- **Parallel Workers:** `{worker_count}`"
                        )

                        progress(1.0, desc="Benchmark complete! Loading dashboard...")
                        return status_text, iframe_html, md_text, scorecard, str(html_file.resolve())

                    except Exception as exc:
                        error_text = (
                            f"### ⚠️ Benchmark Error\n"
                            f"- **Exception:** `{exc}`\n\n"
                            f"*Tip: If you do not have external API keys configured, check **Dry Run** to validate without external LLM calls.*"
                        )
                        return error_text, gr.update(), gr.update(), {}, None

                bench_run_btn.click(
                    fn=run_benchmark_ui,
                    inputs=[
                        bench_source_mode,
                        bench_artifact_dropdown,
                        bench_file_upload,
                        bench_query_text,
                        bench_judge_model,
                        bench_dry_run,
                        bench_rerun,
                        bench_workers,
                    ],
                    outputs=[
                        bench_status_md,
                        bench_html_view,
                        bench_md_view,
                        bench_json_view,
                        bench_download_file,
                    ],
                    show_progress="full",
                )

                def on_load_report():
                    return load_existing_benchmark_report()

                bench_load_btn.click(
                    fn=on_load_report,
                    outputs=[
                        bench_status_md,
                        bench_html_view,
                        bench_md_view,
                        bench_json_view,
                        bench_download_file,
                    ],
                )

    return demo


def print_banner_and_commands() -> None:
    """Prints the project banner and command guide without starting any services."""
    print(BANNER_ART)

    # Check readiness
    cache_path = Path("data/.cache/fact_embeddings.npz")
    cache_ready = cache_path.exists()
    cloudflared_ready = bool(subprocess.run(["which", "cloudflared"], capture_output=True).stdout)

    print("  [System Readiness]")
    print(f"  • Curriculum Cards:     952 cards indexed (Balbharati)")
    print(f"  • Neural Embeddings:    {'Ready (5.2 MB cache)' if cache_ready else 'Building on first run'}")
    print(f"  • ONNX Runtime Engine:  Active (paraphrase-multilingual-MiniLM-L12-v2)")
    print(f"  • Cloudflare Tunnel:    {'Installed (/usr/local/bin/cloudflared)' if cloudflared_ready else 'Not found'}")
    print("\n" + "-" * 88)
    print("  [Run With Poe the Poet (Recommended)]")
    print("  • poe api           -> Start FastAPI REST API server (http://0.0.0.0:8000/docs)")
    print("  • poe studio        -> Start Gradio Studio Web UI (http://0.0.0.0:7860) [No custom HTML]")
    print("  • poe api-share     -> Start unified service & share publicly via Cloudflare Tunnel")
    print("  • poe cli           -> Start interactive terminal chatbot REPL")
    print("  • poe test          -> Run all automated unit tests")
    print("\n  [Direct Python Commands]")
    print("  • python main.py --service api [--host 0.0.0.0] [--port 8000]")
    print("  • python main.py --service studio [--host 0.0.0.0] [--port 7860]")
    print("  • python main.py --service api-share [--port 8000]")
    print("=" * 88 + "\n")


def run_api_service(host: str = "0.0.0.0", port: int = 8000, cards: str | None = None) -> None:
    """Starts the FastAPI REST API service."""
    bot = OfflineFactBot(cards=cards)
    app = create_api_app(bot)
    print(f"\n🚀 Starting FastAPI REST API on http://{host}:{port}")
    print(f"📖 OpenAPI Docs available at http://{host}:{port}/docs\n")
    uvicorn.run(app, host=host, port=port)


def run_studio_service(host: str = "0.0.0.0", port: int = 7860, cards: str | None = None) -> None:
    """Starts the native Gradio Studio service."""
    bot = OfflineFactBot(cards=cards)
    studio = build_studio(bot)
    print(f"\n🎨 Starting Gradio Studio UI on http://{host}:{port}\n")
    studio.launch(server_name=host, server_port=port, share=False)


def run_api_share_service(port: int = 8000, cards: str | None = None) -> None:
    """Starts the unified API + Studio server and creates a Cloudflare tunnel."""
    bot = OfflineFactBot(cards=cards)
    app = create_unified_app(bot)

    # 1. Start uvicorn server in a separate thread
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    print(f"\n🚀 Local service started at http://127.0.0.1:{port}")
    print("🌐 Initiating Cloudflare Tunnel (cloudflared)...")

    # 2. Start cloudflared tunnel
    cf_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tunnel_url: str | None = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    try:
        # Read cloudflared output line-by-line to extract the public URL
        assert cf_proc.stdout is not None
        for line in cf_proc.stdout:
            match = url_pattern.search(line)
            if match and not tunnel_url:
                tunnel_url = match.group(0)
                print("\n" + "=" * 88)
                print("  🎉 Cloudflare Tunnel Successfully Established!")
                print(f"  • Public Studio UI:   {tunnel_url}/studio")
                print(f"  • Public API Docs:    {tunnel_url}/docs")
                print(f"  • Public Health:      {tunnel_url}/health")
                print(f"  • Local Endpoint:     http://127.0.0.1:{port}")
                print("=" * 88 + "\n")
                print("Press Ctrl+C to stop the tunnel and server.\n")

            # Echo errors if any
            if "ERR" in line:
                print(f"  [cloudflared] {line.strip()}")

    except KeyboardInterrupt:
        print("\n🛑 Shutting down Cloudflare tunnel and server...")
    finally:
        cf_proc.terminate()
        cf_proc.wait()
        server.should_exit = True
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gurukul — Offline Educational Assistant",
        add_help=True,
    )
    parser.add_argument(
        "--service",
        choices=["api", "studio", "api-share"],
        default=None,
        help="Target service to run. When omitted, main.py only displays the banner and commands.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind")
    parser.add_argument("--cards", type=str, default=None, help="Path to custom fact cards JSON file")

    args = parser.parse_args()

    # Default action: Don't start any service, only show banner and command guide
    if args.service is None:
        print_banner_and_commands()
        return

    # Specific service execution
    if args.service == "api":
        port = args.port or 8000
        run_api_service(host=args.host, port=port, cards=args.cards)
    elif args.service == "studio":
        port = args.port or 7860
        run_studio_service(host=args.host, port=port, cards=args.cards)
    elif args.service == "api-share":
        port = args.port or 8000
        run_api_share_service(port=port, cards=args.cards)


if __name__ == "__main__":
    main()
