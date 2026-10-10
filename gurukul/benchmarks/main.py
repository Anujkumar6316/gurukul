"""Banner Entry Point for the Gurukul DeepEval Benchmark Suite.

Running `python -m benchmarks.main` with no arguments prints the benchmark banner
and command guide. Passing flags runs the DeepEval LLM-as-a-judge suite over
either a raw `.txt` query list (executed through `OfflineFactBot` first) or a
`batch_debug_results.json` export from the Gradio Batch Debugger / REST API.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from gurukul.benchmarks.run_benchmark import run
from gurukul.benchmarks.settings import settings


BANNER_ART = r"""
========================================================================================
  ██████╗ ██╗   ██╗██████╗ ██╗   ██╗██╗  ██╗██╗   ██╗██╗
 ██╔════╝ ██║   ██║██╔══██╗██║   ██║██║ ██╔╝██║   ██║██║
 ██║  ███╗██║   ██║██████╔╝██║   ██║█████╔╝ ██║   ██║██║
 ██║   ██║██║   ██║██╔══██╗██║   ██║██╔═██╗ ██║   ██║██║
 ╚██████╔╝╚██████╔╝██║  ██║╚██████╔╝██║ ╚██████╔╝███████╗
  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═════╝ ╚══════╝
========================================================================================
  BENCHMARK SUITE — DeepEval LLM-as-a-Judge (G-Eval Rubrics)
  Language Consistency · Follow-up Offer Quality · Grounding · Abstention
========================================================================================
"""


def _deepeval_installed() -> bool:
    """Returns True if the optional DeepEval dependency is importable."""
    return importlib.util.find_spec("deepeval") is not None


def _next_auto_input() -> Path | None:
    """Returns the input artifact the runner would auto-discover, if any."""
    try:
        from gurukul.benchmarks.run_benchmark import resolve_input_artifact

        return resolve_input_artifact(None)
    except FileNotFoundError:
        return None


def print_banner_and_commands() -> None:
    """Prints the benchmark banner and command guide without evaluating anything."""
    print(BANNER_ART)

    print("  [Judge Configuration]")
    print(f"  • Default Judge:        {settings.effective_judge}")
    print(f"  • Gemini API Key:       {'Loaded' if settings.gemini_api_key else 'Missing (set GEMINI_API_KEY)'}")
    print(f"  • OpenAI API Key:       {'Loaded' if settings.openai_api_key else 'Missing (set OPENAI_API_KEY)'}")
    print(f"  • DeepEval Installed:   {'Yes' if _deepeval_installed() else 'No (run: uv sync --extra benchmark)'}")
    print(
        "  • Thresholds:           "
        f"lang={settings.language_consistency_threshold} "
        f"offer={settings.follow_up_offer_threshold} "
        f"grounding={settings.curriculum_grounding_threshold} "
        f"abstention={settings.abstention_accuracy_threshold}"
    )
    print(f"  • Parallel Workers:     {settings.benchmark_workers} (offline inference + judge calls)")

    print("\n  [Input Artifacts]")
    print(f"  • Artifact Directory:   {settings.artifact_dir}")
    print(f"  • Local Artifact Dir:   {settings.local_artifact_dir} ({'found' if settings.local_artifact_dir.exists() else 'not created yet'})")
    print("  • Accepted Formats:     .json (batch debugger export) or .txt (one raw query per line)")
    auto_input = _next_auto_input()
    print(f"  • Next Auto-Discovered: {auto_input if auto_input else 'none found (pass --input)'}")

    print("\n" + "-" * 88)
    print("  [Run With Poe the Poet (Recommended)]")
    print("  • poe benchmark --dry-run                                     -> Validate pipeline, no API calls")
    print("  • poe benchmark --input batch_debug_results.json              -> Judge a batch debugger JSON export")
    print("  • poe benchmark --input queries.txt --judge gemini-1.5-flash  -> Judge raw queries (runs bot first)")
    print(f"  • poe benchmark --input queries.txt --workers 12              -> Raise concurrency (default: {settings.benchmark_workers})")

    print("\n  [Direct Python Commands]")
    print("  • python -m benchmarks.main --input <file.json|file.txt> [--judge <model>] [--workers N] [--output-dir <dir>] [--dry-run]")

    print("\n  [Workflow 1 — Raw Queries (.txt)]")
    print("  1. Write one query per line to queries.txt (Marathi, Hindi, or English).")
    print("  2. python -m benchmarks.main --input queries.txt --dry-run")
    print("     (Executes queries through OfflineFactBot locally; no judge API calls.)")
    print("  3. python -m benchmarks.main --input queries.txt --judge gemini-1.5-flash")

    print("\n  [Workflow 2 — Batch Debugger JSON (.json)]")
    print("  1. Export batch_debug_results.json from Gradio Studio Tab 4 (🔬 Batch Debugger)")
    print("     or POST /api/debug/batch, and place it in the artifact directory.")
    print("  2. python -m benchmarks.main --input batch_debug_results.json --dry-run")
    print("     (Answers are already computed — zero re-inference overhead.)")
    print("  3. python -m benchmarks.main --input batch_debug_results.json --judge gpt-4o-mini")
    print("  4. Optional 100% offline judge: --judge ollama/llama3.1:8b")

    print("\n  [Output Artifacts — written to the output directory]")
    print(f"  • {settings.default_html_report_name}  (interactive HTML report)")
    print(f"  • {settings.default_report_name}       (markdown scorecard)")
    print(f"  • {settings.default_scorecard_name} (machine-readable metrics)")
    print("=" * 88 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gurukul Standalone DeepEval Benchmark Runner",
        add_help=True,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path or filename of the input artifact (batch_debug_results.json or queries.txt)",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=settings.effective_judge,
        help=f"Judge model identifier (default: {settings.effective_judge})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel worker count for offline inference and judge evaluation "
            f"(default: {settings.benchmark_workers}, 1 = sequential)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for report artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate artifact resolution and test case generation without invoking the LLM judge",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-execute queries through the current OfflineFactBot engine even if input is a JSON result artifact",
    )
    args = parser.parse_args()

    # Default action: display the banner and command guide only
    if len(sys.argv) == 1:
        print_banner_and_commands()
        return

    print(BANNER_ART)

    out_dir = Path(args.output_dir) if args.output_dir else None
    run(
        input_path=args.input,
        judge_model=args.judge,
        output_dir=out_dir,
        dry_run=args.dry_run,
        rerun=args.rerun,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
