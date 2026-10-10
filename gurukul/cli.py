from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from gurukul.engine import OfflineFactBot
from gurukul.index import FactCardIndex

BANNER_ART = r"""
========================================================================================
  ██████╗ ██╗   ██╗██████╗ ██╗   ██╗██╗  ██╗██╗   ██╗██╗     
 ██╔════╝ ██║   ██║██╔══██╗██║   ██║██║ ██╔╝██║   ██║██║     
 ██║  ███╗██║   ██║██████╔╝██║   ██║█████╔╝ ██║   ██║██║     
 ██║   ██║██║   ██║██╔══██╗██║   ██║██╔═██╗ ██║   ██║██║     
 ╚██████╔╝╚██████╔╝██║  ██║╚██████╔╝██║  ██╗╚██████╔╝███████╗
  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
========================================================================================
  GURUKUL — Zero-LLM Multilingual Offline FactBot & Benchmark Engine
  Curriculum & Custom Fact Cards · 100% Offline · ONNX Multilingual Search
========================================================================================
"""


def run_chat_cli(
    cards: str | None = None,
    lang: str | None = None,
    grade: int | None = None,
    model: str | None = None,
) -> None:
    print(BANNER_ART)
    print("Commands:")
    print("  /lang <mr|hi|en|auto>  - Switch language (Default: auto)")
    print("  /grade <1-8>           - Set grade filter (Default: All)")
    print("  /tree                  - View knowledge tree summary")
    print("  /sync                  - View pending offline queries log")
    print("  /reset                 - Reset active conversation context")
    print("  /quit                  - Exit")
    print("=" * 88)

    print(f"Loading fact cards ({cards or 'default curriculum'})...")
    bot = OfflineFactBot(cards=cards, model_name=model)
    print(f"Loaded {len(bot.index.cards_by_id)} cards into offline semantic index.\n")

    session_id = "cli-session"
    current_lang = lang if lang and lang != "auto" else None
    current_grade = grade
    last_chips: list[Any] = []

    while True:
        try:
            lang_label = current_lang.upper() if current_lang else "AUTO"
            prompt_str = f"[{lang_label}" + (f" | Gr.{current_grade}" if current_grade else "") + "] प्रश्न > "
            user_input = input(prompt_str).strip()
            if not user_input:
                continue

            if user_input.lower() in ("/quit", "exit", "quit", "q"):
                print("धन्यवाद! (Goodbye!)")
                break

            if user_input.startswith(("/reset", "/clear")):
                bot.reset_session(session_id)
                last_chips = []
                print("संवाद रीसेट केला गेला (Conversation session reset).")
                continue

            if user_input.startswith("/lang"):
                parts = user_input.split()
                if len(parts) > 1:
                    choice = parts[1].lower()
                    if choice in ("mr", "hi", "en"):
                        current_lang = choice
                        print(f"Language set to: {current_lang}")
                    elif choice == "auto":
                        current_lang = None
                        print("Language set to: Auto-detect")
                    else:
                        print("Usage: /lang mr | hi | en | auto")
                else:
                    current_lang = None
                    print("Language reset to Auto-detect.")
                continue

            if user_input.startswith("/grade"):
                parts = user_input.split()
                if len(parts) > 1:
                    try:
                        g = int(parts[1])
                        current_grade = g if 1 <= g <= 10 else None
                        print(f"Grade filter set to: {current_grade or 'All'}")
                    except ValueError:
                        print("Invalid grade. Usage: /grade 1-8")
                else:
                    current_grade = None
                    print("Grade filter cleared.")
                continue

            if user_input.startswith("/tree"):
                tree = bot.index.get_curriculum_tree()
                print("\n--- Knowledgebase Overview ---")
                for gr, subjects in sorted(tree.items()):
                    print(f"\n{gr}:")
                    for subj, cards_list in subjects.items():
                        print(f"  • {subj}: {len(cards_list)} entries")
                print("------------------------------\n")
                continue

            if user_input.startswith("/sync"):
                queue = bot.get_sync_queue()
                print(f"\n--- Local Offline Sync Queue ({len(queue)} pending) ---")
                for i, item in enumerate(queue, 1):
                    print(f"{i}. [{item.language}] \"{item.query}\" (Top match: {item.top_matched_card_id or 'None'}, score: {item.top_score})")
                print("---------------------------------------------------\n")
                continue

            if user_input.isdigit() and last_chips:
                idx = int(user_input) - 1
                if 0 <= idx < len(last_chips):
                    target_chip = last_chips[idx]
                    print(f"\n[Selected: {target_chip.title}]")
                    resp = bot.get_card_by_id(
                        target_chip.card_id,
                        lang=current_lang or target_chip.language,
                        session_id=session_id,
                    )
                    if resp:
                        _print_response(resp)
                        last_chips = resp.follow_up_chips
                        continue

            resp = bot.resolve_query(
                query=user_input,
                session_id=session_id,
                lang=current_lang,
                grade=current_grade,
            )
            _print_response(resp)
            last_chips = resp.follow_up_chips

        except (KeyboardInterrupt, EOFError):
            print("\nधन्यवाद! (Goodbye!)")
            break


def _print_response(resp: Any) -> None:
    print(f"\n{'=' * 65}")
    status = "⚠️ ABSTENTION (Out of syllabus/cards)" if resp.is_abstention else f"✅ MATCH [{resp.card_id or 'Card'}]"
    print(f"{status} | Conf: {resp.confidence:.2f} | Latency: {resp.latency_ms:.1f}ms | Lang: {resp.language}")
    if resp.topic:
        print(f"Topic: {resp.topic}")
    if resp.citation:
        print(f"Source: {resp.citation}")
    print(f"\n{resp.answer}\n")

    if resp.follow_up_offer:
        print(f"💡 {resp.follow_up_offer}\n")

    if resp.follow_up_chips:
        print("Related Topics (type number to view):")
        for i, chip in enumerate(resp.follow_up_chips, 1):
            print(f"  [{i}] {chip.title}")
        print()

    if resp.suggested_topics:
        print("Suggested Topics:")
        for t in resp.suggested_topics:
            print(f"  • {t}")
        print()
    print("=" * 65)


def run_serve(args: argparse.Namespace) -> None:
    """Launches FastAPI REST API or Gradio Studio UI."""
    try:
        from main import build_studio, create_api_app, create_unified_app
        import uvicorn
    except ImportError as e:
        print(f"API/Studio server requires optional dependencies: pip install 'gurukul[api,ui]' ({e})")
        sys.exit(1)

    bot = OfflineFactBot(cards=args.cards, model_name=args.model)

    if args.studio:
        demo = build_studio(bot)
        print(f"Starting Gradio Studio on http://{args.host}:{args.port}")
        demo.launch(server_name=args.host, server_port=args.port)
    else:
        app = create_unified_app(bot)
        print(f"Starting unified REST API & Studio on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)


def run_benchmark_cli(args: argparse.Namespace) -> None:
    """Runs DeepEval benchmark suite."""
    from gurukul.benchmarks.run_benchmark import run
    out_dir = Path(args.output_dir) if args.output_dir else None
    run(
        input_path=args.input,
        judge_model=args.judge,
        output_dir=out_dir,
        dry_run=args.dry_run,
        rerun=args.rerun,
        cards=args.cards,
        workers=args.workers,
    )


def run_download_model_cli(args: argparse.Namespace) -> None:
    """Pre-downloads ONNX embedding model weights to local cache for 100% offline usage."""
    model_name = args.model or "google/embeddinggemma-2"
    cache_dir = args.cache_dir
    print(f"📦 Downloading/verifying ONNX model weights for '{model_name}'...")
    if "embeddinggemma-2" in model_name.lower() or "embeddinggemma_2" in model_name.lower():
        from gurukul.index import EmbeddingGemma2ONNX
        encoder = EmbeddingGemma2ONNX(model_name=model_name, cache_dir=cache_dir)
        _ = list(encoder.embed(["gurukul offline warmup"]))
        model_dir = getattr(encoder, "model_path", cache_dir or "Hugging Face Cache")
    else:
        from fastembed import TextEmbedding
        encoder = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        _ = list(encoder.embed(["gurukul offline warmup"]))
        model_dir = getattr(encoder.model, "_model_dir", encoder.cache_dir)
    print(f"✅ Model '{model_name}' is cached and verified for 100% offline usage!")
    print(f"📍 Local ONNX cache location: {model_dir}\n")


def run_index_cli(args: argparse.Namespace) -> None:
    """Precomputes vector embeddings cache for instant reload."""
    print(f"Precomputing embeddings for: {args.cards or 'default curriculum'}...")
    index = FactCardIndex(cards=args.cards, model_name=args.model)
    print(f"Successfully indexed {len(index.cards_by_id)} cards.")
    print(f"Vectors cache saved to: {index.cache_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gurukul",
        description="Gurukul: Offline Zero-LLM FactBot & Benchmarking Suite",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. Chat
    chat_p = subparsers.add_parser("chat", help="Start interactive terminal chatbot")
    chat_p.add_argument("--cards", type=str, default=None, help="Path to custom fact cards JSON file")
    chat_p.add_argument("--lang", type=str, default="auto", choices=["auto", "mr", "hi", "en"], help="Default language")
    chat_p.add_argument("--grade", type=int, default=None, help="Grade filter (1-8)")
    chat_p.add_argument("--model", type=str, default=None, help="ONNX embedding model name")

    # 2. Serve
    serve_p = subparsers.add_parser("serve", help="Launch FastAPI REST API / Gradio Studio")
    serve_p.add_argument("--cards", type=str, default=None, help="Path to custom fact cards JSON file")
    serve_p.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_p.add_argument("--studio", action="store_true", help="Launch standalone Gradio Studio instead of REST API")
    serve_p.add_argument("--model", type=str, default=None, help="ONNX embedding model name")

    # 3. Benchmark
    bench_p = subparsers.add_parser("benchmark", help="Run DeepEval LLM-as-a-judge benchmark")
    bench_p.add_argument("--cards", type=str, default=None, help="Path to custom fact cards JSON file")
    bench_p.add_argument("--input", type=str, default=None, help="Input artifact (queries.txt or batch_results.json)")
    bench_p.add_argument("--judge", type=str, default=None, help="Judge model name")
    bench_p.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")
    bench_p.add_argument("--dry-run", action="store_true", help="Run dry run without invoking LLM judge")
    bench_p.add_argument("--rerun", action="store_true", help="Re-execute queries through engine")
    bench_p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count for offline inference and judge evaluation (1 = sequential)",
    )

    # 4. Index
    idx_p = subparsers.add_parser("index", help="Build/precompute embeddings cache for cards")
    idx_p.add_argument("--cards", type=str, default=None, help="Path to custom fact cards JSON file")
    idx_p.add_argument("--model", type=str, default=None, help="ONNX embedding model name")

    # 5. Download Model
    dl_p = subparsers.add_parser("download-model", help="Download and warm up ONNX embedding model for offline use")
    dl_p.add_argument("--model", type=str, default="google/embeddinggemma-2", help="Model name (default: google/embeddinggemma-2)")
    dl_p.add_argument("--cache-dir", type=str, default=None, help="Custom directory for ONNX model cache")

    if len(sys.argv) == 1:
        print(BANNER_ART)
        parser.print_help()
        return

    args = parser.parse_args()

    if args.subcommand == "chat":
        run_chat_cli(cards=args.cards, lang=args.lang, grade=args.grade, model=args.model)
    elif args.subcommand == "serve":
        run_serve(args)
    elif args.subcommand == "benchmark":
        run_benchmark_cli(args)
    elif args.subcommand == "index":
        run_index_cli(args)
    elif args.subcommand == "download-model":
        run_download_model_cli(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
