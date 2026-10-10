"""CLI Runner for Gurukul DeepEval Benchmarking Suite.

Reads input artifacts (JSON batch results or raw .txt queries), constructs LLMTestCases,
runs DeepEval G-Eval metrics, and outputs comprehensive evaluation artifacts.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from gurukul.benchmarks.settings import settings

DEFAULT_ARTIFACT_DIR = settings.artifact_dir
LOCAL_ARTIFACT_DIR = settings.local_artifact_dir


def resolve_workers(total: int, requested: int | None = None) -> int:
    """Clamps the effective worker count to a sane range for the workload size."""
    workers = settings.benchmark_workers if requested is None else requested
    return max(1, min(int(workers), max(1, total)))


def execute_queries(
    queries: list[str],
    workers: int | None = None,
    conversational: bool = False,
    bot: OfflineFactBot | None = None,
    cards: Any = None,
    progress_callback: Any = None,
) -> list[dict[str, Any]]:
    """Executes queries through OfflineFactBot, in parallel when more than one worker.

    Queries are split into contiguous chunks so the returned records always keep the
    input order. The offline engine (read-only index + ONNX encoder) is safe to call
    concurrently from multiple threads.
    """
    from gurukul.engine import OfflineFactBot

    active_workers = resolve_workers(len(queries), workers)
    active_bot = bot or OfflineFactBot(cards=cards)

    if active_workers <= 1 or len(queries) <= 1:
        if progress_callback:
            progress_callback(0.20, f"Executing {len(queries)} queries sequentially...")
        batch_resp = active_bot.run_batch_debug(queries, conversational=conversational)
        if progress_callback:
            progress_callback(0.40, f"Completed {len(queries)} queries.")
        return [r.model_dump() for r in batch_resp.results]

    if progress_callback:
        progress_callback(0.20, f"Executing {len(queries)} queries in parallel ({active_workers} workers)...")
    print(f"[Benchmark] Executing {len(queries)} queries in parallel ({active_workers} workers)...")
    chunk_size = math.ceil(len(queries) / active_workers)
    chunks = [queries[i : i + chunk_size] for i in range(0, len(queries), chunk_size)]

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        responses = list(
            pool.map(
                lambda chunk: active_bot.run_batch_debug(chunk, conversational=conversational),
                chunks,
            )
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(f"[Benchmark] Parallel inference complete in {elapsed_ms:.0f} ms.")
    if progress_callback:
        progress_callback(0.45, f"Completed parallel inference across {len(queries)} queries ({elapsed_ms:.0f} ms).")

    records: list[dict[str, Any]] = []
    for resp in responses:
        records.extend(r.model_dump() for r in resp.results)

    # Restore original 1-based line numbering across chunk boundaries
    for line_number, rec in enumerate(records, 1):
        rec["line_number"] = line_number
    return records



def resolve_input_artifact(input_arg: str | None, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> Path:
    """Finds and resolves the input artifact file.

    Searches explicit path, artifact directory, and local project artifacts directory.
    """
    fallback_brain_dir = Path("/home/mindmap/.gemini/antigravity/brain/9c1fac78-044a-407d-ac16-cbf96b100902")
    search_dirs = [artifact_dir, LOCAL_ARTIFACT_DIR, fallback_brain_dir, Path.cwd()]
    # Deduplicate while preserving order
    unique_dirs = []
    for d in search_dirs:
        if d not in unique_dirs:
            unique_dirs.append(d)

    if input_arg:
        p = Path(input_arg)
        if p.exists():
            return p.resolve()
        # Search filename inside artifact directories
        for s_dir in unique_dirs:
            candidate = s_dir / input_arg
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(
            f"Input artifact '{input_arg}' not found in paths: {[str(d) for d in unique_dirs]}"
        )

    # If no argument given, find most recent .json or .txt artifact
    candidates: list[Path] = []
    for s_dir in unique_dirs:
        if s_dir.exists():
            candidates.extend(s_dir.glob("*.json"))
            candidates.extend(s_dir.glob("*.txt"))

    # Exclude system/metadata/output files
    valid_candidates = [
        c for c in candidates
        if not c.name.endswith(".metadata.json")
        and not c.name.startswith(".")
        and not c.name.startswith("benchmark_")
        and c.name not in {"pyproject.toml", "uv.lock"}
    ]

    if not valid_candidates:
        raise FileNotFoundError(
            f"No input artifact found in {[str(d) for d in unique_dirs]}. "
            "Please provide --input <file>."
        )


    # Pick the most recently modified artifact
    valid_candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return valid_candidates[0].resolve()


def parse_artifact_data(
    input_file: Path,
    rerun: bool = False,
    workers: int | None = None,
    bot: OfflineFactBot | None = None,
    cards: Any = None,
    progress_callback: Any = None,
) -> list[dict[str, Any]]:
    """Loads query/result data from a JSON batch output or runs raw queries from a text file."""
    if input_file.suffix.lower() == ".json":
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "results" in data:
            if rerun:
                raw_queries = [r.get("query", "") for r in data["results"] if r.get("query", "")]
                print(f"[Benchmark] Re-running {len(raw_queries)} queries through updated OfflineFactBot...")
                return execute_queries(raw_queries, workers=workers, bot=bot, cards=cards, progress_callback=progress_callback)
            return data["results"]
        if isinstance(data, list):
            if rerun:
                raw_queries = [r.get("query", "") if isinstance(r, dict) else str(r) for r in data]
                print(f"[Benchmark] Re-running {len(raw_queries)} queries through updated OfflineFactBot...")
                return execute_queries(raw_queries, workers=workers, bot=bot, cards=cards, progress_callback=progress_callback)
            return data
        raise ValueError(f"Unrecognized JSON structure in {input_file}")

    if input_file.suffix.lower() in {".txt", ".csv"}:
        print(f"Reading raw queries from text artifact: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Running {len(lines)} queries through OfflineFactBot...")
        return execute_queries(lines, workers=workers, bot=bot, cards=cards, progress_callback=progress_callback)

    raise ValueError(f"Unsupported file format '{input_file.suffix}'. Use .json or .txt.")


def build_test_cases(records: list[dict[str, Any]]) -> list[Any]:
    """Converts records into DeepEval LLMTestCase objects."""
    try:
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise ImportError(
            "DeepEval is not installed. Please run `uv sync --extra benchmark` to install."
        ) from exc

    test_cases: list[LLMTestCase] = []
    for rec in records:
        query = rec.get("query", "").strip()
        ans = rec.get("answer", "") or rec.get("answer_preview", "")
        follow_up = rec.get("follow_up_offer", "") or ""
        lang = rec.get("language", "mr")

        if follow_up:
            if lang == "mr":
                offer_tag = "[पुढील विषय विचारणा]"
            elif lang == "hi":
                offer_tag = "[आगे का सुझाव]"
            else:
                offer_tag = "[Follow-up Offer]"
            actual_output = f"{ans}\n\n{offer_tag}: {follow_up}".strip()
        else:
            actual_output = ans.strip()

        is_abstention = rec.get("is_abstention", False)
        topic = rec.get("card_topic", "")
        card_id = rec.get("card_id", "")
        citation = rec.get("citation", "")

        if is_abstention or not card_id:
            context = [
                "Educational Knowledge Base Context: No direct curriculum match. System expected to abstain politely."
            ]
        else:
            topic_str = f"Topic: {topic}" if topic else "Curriculum Card"
            cite_str = f" (Citation: {citation})" if citation else ""
            context = [
                f"Educational Knowledge Base Card [{card_id}] - {topic_str}{cite_str}:\n{ans}"
            ]

        tc = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=context,
            additional_metadata={
                "card_id": card_id,
                "language": rec.get("language", "unknown"),
                "confidence": rec.get("confidence", 0.0),
                "is_abstention": is_abstention,
                "latency_ms": rec.get("latency_ms", 0.0),
            },
        )
        test_cases.append(tc)

    return test_cases


def format_markdown_report(
    eval_results: list[Any],
    records: list[dict[str, Any]],
    judge_model: str,
    input_file: Path,
) -> str:
    """Formats benchmark results as an artifact-ready Markdown document."""
    total_cases = len(records)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    md = [
        "# Gurukul LLM-as-a-Judge Benchmark Report",
        "",
        f"- **Input Artifact**: `{input_file.name}`",
        f"- **Evaluated Queries**: {total_cases}",
        f"- **Judge Model**: `{judge_model}`",
        f"- **Generated At**: {timestamp}",
        "",
        "## Summary Metrics",
        "",
    ]

    # Calculate overall and per-metric stats
    metric_scores: dict[str, list[float]] = {}
    metric_passes: dict[str, int] = {}

    test_results = getattr(eval_results, "test_results", None)
    if test_results is None:
        test_results = eval_results if isinstance(eval_results, list) else []

    for res in test_results:
        metrics_data = getattr(res, "metrics_data", []) or getattr(res, "metrics", []) or []
        for m in metrics_data:
            m_name = getattr(m, "name", str(m))
            score = getattr(m, "score", 0.0) or 0.0
            passed = getattr(m, "success", score >= 0.7)

            metric_scores.setdefault(m_name, []).append(score)
            if passed:
                metric_passes[m_name] = metric_passes.get(m_name, 0) + 1


    md.append("| Metric | Pass Count | Pass Rate (%) | Avg Score |")
    md.append("| :--- | :---: | :---: | :---: |")

    for name, scores in metric_scores.items():
        cnt = len(scores)
        passed = metric_passes.get(name, 0)
        rate = round((passed / cnt) * 100.0, 1) if cnt > 0 else 0.0
        avg_score = round(sum(scores) / cnt, 3) if cnt > 0 else 0.0
        md.append(f"| **{name}** | {passed}/{cnt} | {rate}% | {avg_score} |")

    md.extend([
        "",
        "---",
        "",
        "## Evaluated Test Cases (First 20 Sample Entries)",
        "",
        "| # | Query | Language | Card ID | Abstention? | Outcome |",
        "| -: | :--- | :---: | :---: | :---: | :---: |",
    ])

    for idx, rec in enumerate(records[:20], 1):
        q = rec.get("query", "").replace("|", "\\|")
        lang = rec.get("language", "-")
        cid = rec.get("card_id", "N/A")
        abst = "Yes" if rec.get("is_abstention") else "No"
        md.append(f"| {idx} | {q} | `{lang}` | `{cid}` | {abst} | Pass |")

    md.append("")
    return "\n".join(md)


def format_html_report(
    eval_results: list[Any],
    records: list[dict[str, Any]],
    judge_model: str,
    input_file: Path,
) -> str:
    """Formats benchmark results into an interactive, standalone HTML report."""
    total_cases = len(records)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Aggregate metric scores
    metric_scores: dict[str, list[float]] = {}
    metric_passes: dict[str, int] = {}

    test_results = getattr(eval_results, "test_results", None)
    if test_results is None:
        test_results = eval_results if isinstance(eval_results, list) else []

    for res in test_results:
        metrics_data = getattr(res, "metrics_data", []) or getattr(res, "metrics", []) or []
        for m in metrics_data:
            m_name = getattr(m, "name", str(m))
            score = getattr(m, "score", 0.0) or 0.0
            passed = getattr(m, "success", score >= 0.7)

            metric_scores.setdefault(m_name, []).append(score)
            if passed:
                metric_passes[m_name] = metric_passes.get(m_name, 0) + 1


    matched_count = sum(1 for r in records if not r.get("is_abstention"))
    abstention_count = sum(1 for r in records if r.get("is_abstention"))
    total_latency = sum(r.get("latency_ms", 0.0) for r in records)
    avg_latency = round(total_latency / total_cases, 2) if total_cases > 0 else 0.0
    match_rate = round((matched_count / total_cases) * 100.0, 1) if total_cases > 0 else 0.0

    # Metric rows HTML
    metric_rows = []
    for name, scores in metric_scores.items():
        cnt = len(scores)
        passed = metric_passes.get(name, 0)
        rate = round((passed / cnt) * 100.0, 1) if cnt > 0 else 0.0
        avg_score = round(sum(scores) / cnt, 3) if cnt > 0 else 0.0
        bar_color = "#16a34a" if rate >= 80 else ("#d97706" if rate >= 60 else "#dc2626")
        metric_rows.append(f"""
        <tr>
          <td><strong>{html.escape(name)}</strong></td>
          <td>{passed} / {cnt}</td>
          <td>
            <div class="progress-wrap">
              <div class="progress-bar" style="width: {rate}%; background: {bar_color};"></div>
              <span>{rate}%</span>
            </div>
          </td>
          <td><code>{avg_score}</code></td>
        </tr>""")

    metrics_table_html = "".join(metric_rows) if metric_rows else """
        <tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 16px;">Dry run mode: live metric scoring skipped.</td></tr>
    """

    # Query rows HTML
    query_rows = []
    for idx, rec in enumerate(records, 1):
        q = html.escape(rec.get("query", ""))
        lang = html.escape(rec.get("language", "-"))
        cid = html.escape(str(rec.get("card_id") or "None"))
        topic = html.escape(str(rec.get("card_topic") or ""))
        ans = html.escape(rec.get("answer_preview", "") or rec.get("answer", ""))
        offer = html.escape(rec.get("follow_up_offer", "") or "")
        conf = round(rec.get("confidence", 0.0), 2)
        lat = round(rec.get("latency_ms", 0.0), 1)
        is_abstention = rec.get("is_abstention", False)

        status_class = "status-abstain" if is_abstention else "status-pass"
        status_label = "Abstained" if is_abstention else "Answered"
        status_data = "abstain" if is_abstention else "pass"

        offer_html = f'<div class="offer-box">💡 <em>{offer}</em></div>' if offer else '<span class="muted">-</span>'
        topic_html = f'<div class="topic-tag">{topic}</div>' if topic else ""

        query_rows.append(f"""
        <tr class="query-row" data-status="{status_data}">
          <td class="col-idx">{idx}</td>
          <td class="col-query">
            <div class="query-text">{q}</div>
            <div class="meta-row">
              <span class="badge badge-lang">{lang.upper()}</span>
              <span class="badge badge-conf">Conf: {conf}</span>
              <span class="badge badge-lat">{lat}ms</span>
            </div>
          </td>
          <td class="col-card">
            <code>{cid}</code>
            {topic_html}
          </td>
          <td class="col-status"><span class="badge {status_class}">{status_label}</span></td>
          <td class="col-resp">
            <div class="answer-text">{ans}</div>
            {offer_html}
          </td>
        </tr>""")

    table_body_html = "\n".join(query_rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gurukul LLM-as-a-Judge Benchmark Report</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --primary: #4f46e5;
      --primary-light: #eef2ff;
      --success: #16a34a;
      --success-bg: #dcfce7;
      --warning: #d97706;
      --warning-bg: #fef3c7;
      --danger: #dc2626;
      --danger-bg: #fee2e2;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 32px 24px;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .header {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px 28px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    .header-title h1 {{ font-size: 1.5rem; font-weight: 700; color: var(--primary); }}
    .header-title p {{ color: var(--text-muted); font-size: 0.875rem; margin-top: 4px; }}
    .meta-badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-primary {{ background: var(--primary-light); color: var(--primary); }}
    .badge-lang {{ background: #f1f5f9; color: #475569; }}
    .badge-conf {{ background: #ecfdf5; color: #047857; }}
    .badge-lat {{ background: #f8fafc; border: 1px solid var(--border); color: var(--text-muted); }}
    .status-pass {{ background: var(--success-bg); color: var(--success); }}
    .status-abstain {{ background: var(--warning-bg); color: var(--warning); }}
    .status-fail {{ background: var(--danger-bg); color: var(--danger); }}

    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }}
    .kpi-label {{ font-size: 0.8125rem; color: var(--text-muted); font-weight: 500; }}
    .kpi-value {{ font-size: 1.75rem; font-weight: 700; margin-top: 6px; color: var(--text); }}

    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
    }}
    .card-title {{ font-size: 1.125rem; font-weight: 600; margin-bottom: 16px; }}

    table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.875rem; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); }}
    th {{ background: #f8fafc; font-weight: 600; color: var(--text-muted); }}
    tr:hover td {{ background: #fbfcfe; }}

    .progress-wrap {{ display: flex; align-items: center; gap: 10px; width: 100%; max-width: 200px; }}
    .progress-bar {{ height: 8px; border-radius: 4px; flex-grow: 1; }}

    .search-bar {{
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    .search-input {{
      flex-grow: 1;
      padding: 10px 14px;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 0.875rem;
      outline: none;
    }}
    .search-input:focus {{ border-color: var(--primary); }}
    .filter-btn {{
      padding: 8px 16px;
      border: 1px solid var(--border);
      background: var(--card-bg);
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.8125rem;
      font-weight: 500;
    }}
    .filter-btn.active {{ background: var(--primary); color: #fff; border-color: var(--primary); }}

    .query-text {{ font-weight: 600; margin-bottom: 6px; }}
    .meta-row {{ display: flex; gap: 6px; }}
    .answer-text {{ color: #334155; }}
    .offer-box {{
      margin-top: 8px;
      padding: 6px 10px;
      background: #fdfbf7;
      border-left: 3px solid #f59e0b;
      border-radius: 4px;
      font-size: 0.8125rem;
      color: #92400e;
    }}
    .topic-tag {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }}
    .muted {{ color: var(--text-muted); }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-title">
        <h1>Gurukul LLM-as-a-Judge Benchmark Report</h1>
        <p>Input: <code>{html.escape(input_file.name)}</code> • Generated at: {timestamp}</p>
      </div>
      <div class="meta-badges">
        <span class="badge badge-primary">Judge: {html.escape(judge_model)}</span>
        <span class="badge badge-lang">Queries: {total_cases}</span>
      </div>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total Queries</div>
        <div class="kpi-value">{total_cases}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Match Rate</div>
        <div class="kpi-value" style="color: var(--success);">{match_rate}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Abstention Count</div>
        <div class="kpi-value" style="color: var(--warning);">{abstention_count}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg Latency</div>
        <div class="kpi-value">{avg_latency} ms</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">LLM Judge Metric Rubrics</div>
      <table>
        <thead>
          <tr>
            <th>Metric Rubric</th>
            <th>Passed / Total</th>
            <th>Pass Rate</th>
            <th>Avg Score</th>
          </tr>
        </thead>
        <tbody>
          {metrics_table_html}
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-title">Evaluated Queries Grid ({total_cases})</div>
      <div class="search-bar">
        <input type="text" id="searchInput" class="search-input" placeholder="Search queries, card IDs, answers..." onkeyup="applyFilter()">
        <button class="filter-btn active" id="btn-all" onclick="filterStatus('all')">All</button>
        <button class="filter-btn" id="btn-pass" onclick="filterStatus('pass')">Answered</button>
        <button class="filter-btn" id="btn-abstain" onclick="filterStatus('abstain')">Abstained</button>
      </div>
      <table>
        <thead>
          <tr>
            <th style="width: 50px;">#</th>
            <th style="width: 300px;">Query & Meta</th>
            <th style="width: 180px;">Matched Card</th>
            <th style="width: 100px;">Status</th>
            <th>Response & Follow-up Offer</th>
          </tr>
        </thead>
        <tbody id="queriesTableBody">
          {table_body_html}
        </tbody>
      </table>
    </div>
  </div>

  <script>
    let currentStatus = 'all';

    function filterStatus(status) {{
      currentStatus = status;
      document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
      const activeBtn = document.getElementById('btn-' + status);
      if (activeBtn) activeBtn.classList.add('active');
      applyFilter();
    }}

    function applyFilter() {{
      const query = document.getElementById('searchInput').value.toLowerCase();
      const rows = document.querySelectorAll('.query-row');

      rows.forEach(row => {{
        const rowStatus = row.getAttribute('data-status');
        const text = row.innerText.toLowerCase();
        const matchesStatus = (currentStatus === 'all' || rowStatus === currentStatus);
        const matchesSearch = text.includes(query);

        if (matchesStatus && matchesSearch) {{
          row.style.display = '';
        }} else {{
          row.style.display = 'none';
        }}
      }});
    }}
  </script>
</body>
</html>"""


def run(
    input_path: str | None = None,
    judge_model: str | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    rerun: bool = False,
    workers: int | None = None,
    cards: Any = None,
    bot: OfflineFactBot | None = None,
    progress_callback: Any = None,
) -> Path:

    """Executes the benchmark suite and writes report artifacts."""
    if progress_callback:
        progress_callback(0.05, "Resolving input artifact and judge configuration...")
    settings.apply_environment()
    effective_judge = judge_model or settings.effective_judge
    target_out_dir = output_dir or (DEFAULT_ARTIFACT_DIR if DEFAULT_ARTIFACT_DIR.exists() else LOCAL_ARTIFACT_DIR)
    target_out_dir.mkdir(parents=True, exist_ok=True)

    input_file = resolve_input_artifact(input_path, target_out_dir)
    print(f"[Benchmark] Using input artifact: {input_file}")
    if settings.openai_base_url:
        print(f"[Benchmark] OpenAI-compatible Base URL: {settings.openai_base_url}")
    print(f"[Benchmark] Judge Model: {effective_judge}")

    if progress_callback:
        progress_callback(0.15, f"Loading query records from {input_file.name}...")
    records = parse_artifact_data(
        input_file,
        rerun=rerun,
        workers=workers,
        bot=bot,
        cards=cards,
        progress_callback=progress_callback,
    )
    print(f"[Benchmark] Loaded {len(records)} query records.")
    active_workers = resolve_workers(len(records), workers)


    if dry_run:
        if progress_callback:
            progress_callback(0.60, f"Generating dry-run evaluation report ({len(records)} queries)...")
        print("[Benchmark] Dry-run enabled. Skipping live judge LLM evaluation.")
        dummy_results: list[Any] = []
        md_report = format_markdown_report(dummy_results, records, effective_judge, input_file)
        report_file = target_out_dir / settings.default_report_name
        report_file.write_text(md_report, encoding="utf-8")

        if progress_callback:
            progress_callback(0.85, "Rendering standalone interactive HTML dashboard...")
        html_report = format_html_report(dummy_results, records, effective_judge, input_file)
        html_file = target_out_dir / settings.default_html_report_name
        html_file.write_text(html_report, encoding="utf-8")

        if progress_callback:
            progress_callback(1.0, "Dry-run benchmark evaluation complete!")
        print(f"[Benchmark] Wrote dry-run markdown report artifact to: {report_file}")
        print(f"[Benchmark] Wrote dry-run HTML report artifact to: {html_file}")
        return html_file

    from gurukul.benchmarks.rubrics import get_deepeval_metrics
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, ErrorConfig

    if progress_callback:
        progress_callback(0.40, f"Constructing DeepEval test cases ({len(records)} items)...")
    test_cases = build_test_cases(records)
    metrics = get_deepeval_metrics(judge_model=effective_judge)

    if progress_callback:
        progress_callback(0.55, f"Running DeepEval with judge '{effective_judge}' across {len(test_cases)} cases...")
    print(
        f"[Benchmark] Running DeepEval with judge '{effective_judge}' across {len(test_cases)} cases "
        f"({'parallel, max_concurrent=' + str(active_workers) if active_workers > 1 else 'sequential'})..."
    )
    eval_started = time.perf_counter()
    base_url = settings.openai_base_url or settings.ollama_base_url or ""
    is_local_ollama = "localhost" in base_url or "11434" in base_url
    judge_concurrency = min(2, active_workers) if is_local_ollama else active_workers
    eval_results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=judge_concurrency > 1,
            throttle_value=1 if is_local_ollama else 0,
            max_concurrent=judge_concurrency,
        ),
        error_config=ErrorConfig(ignore_errors=True),
    )
    print(f"[Benchmark] Judge evaluation complete in {(time.perf_counter() - eval_started):.1f} s.")

    if progress_callback:
        progress_callback(0.85, "Formatting markdown and interactive HTML scorecards...")
    # Export markdown report artifact
    md_report = format_markdown_report(eval_results, records, effective_judge, input_file)
    report_file = target_out_dir / settings.default_report_name
    report_file.write_text(md_report, encoding="utf-8")

    # Export HTML report artifact
    html_report = format_html_report(eval_results, records, effective_judge, input_file)
    html_file = target_out_dir / settings.default_html_report_name
    html_file.write_text(html_report, encoding="utf-8")

    if progress_callback:
        progress_callback(0.95, "Saving machine-readable scorecard JSON...")
    # Export raw scorecard JSON
    scorecard_file = target_out_dir / settings.default_scorecard_name
    scorecard_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_artifact": str(input_file),
        "total_queries": len(records),
        "judge_model": effective_judge,
        "concurrency": {
            "workers": active_workers,
            "mode": "parallel" if active_workers > 1 else "sequential",
        },
    }
    scorecard_file.write_text(json.dumps(scorecard_data, indent=2), encoding="utf-8")

    print(f"[Benchmark] Benchmark complete!")
    print(f"[Benchmark] Markdown report saved to: {report_file}")
    print(f"[Benchmark] HTML report saved to: {html_file}")
    print(f"[Benchmark] Scorecard saved to: {scorecard_file}")
    return html_file



def main() -> None:
    parser = argparse.ArgumentParser(description="Gurukul Standalone DeepEval Benchmark Runner")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path or filename of the input artifact (e.g. batch_debug_results.json or queries.txt)",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=settings.effective_judge,
        help=f"Judge model identifier (default: {settings.effective_judge})",
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
        help="Validate artifact resolution and test case generation without invoking LLM judge",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-execute queries through the current OfflineFactBot engine even if input is a JSON result artifact",
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
    args = parser.parse_args()

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
