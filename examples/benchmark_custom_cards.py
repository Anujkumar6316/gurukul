"""Benchmark Custom Fact Cards with Gurukul DeepEval LLM-as-a-Judge Suite.

Demonstrates:
1. Loading custom fact cards (examples/customer_support_kb.json).
2. Running test queries through OfflineFactBot.
3. Evaluating Language Consistency, Follow-up Offers, Grounding, and Abstention.
4. Generating interactive HTML and Markdown report scorecards.
"""

import sys
from pathlib import Path
from gurukul.benchmarks import run_benchmark

SAMPLE_QUERIES = [
    "What is your return policy?",
    "सामान वापस करने की क्या प्रक्रिया है?",
    "माझे पैसे कधी परत मिळतील?",
    "Do you accept UPI or cash on delivery?",
    "Who is the current prime minister of Japan?",  # Out of domain query -> should abstain
]

def main() -> None:
    kb_path = Path(__file__).resolve().parent / "customer_support_kb.json"
    queries_file = Path(__file__).resolve().parent / "test_queries.txt"
    queries_file.write_text("\n".join(SAMPLE_QUERIES), encoding="utf-8")

    is_dry_run = "--dry-run" in sys.argv or "--live" not in sys.argv

    print("=" * 70)
    print("🔬 Running DeepEval Benchmark on Custom Knowledge Base")
    print(f"Cards: {kb_path}")
    print(f"Queries: {queries_file}")
    print(f"Mode: {'Dry-run (Pipeline validation)' if is_dry_run else 'Live LLM-as-a-Judge'}")
    print("=" * 70)

    report_html = run_benchmark(
        input_path=queries_file,
        cards=kb_path,
        dry_run=is_dry_run,
    )

    print("\n✅ Benchmark execution complete!")
    print(f"Report available at: {report_html}")


if __name__ == "__main__":
    main()
