"""Backward-compatible proxy re-exporting from gurukul.benchmarks.run_benchmark."""

from gurukul.benchmarks.run_benchmark import *
from gurukul.benchmarks.run_benchmark import (
    DEFAULT_ARTIFACT_DIR,
    LOCAL_ARTIFACT_DIR,
    build_test_cases,
    execute_queries,
    format_html_report,
    format_markdown_report,
    main,
    parse_artifact_data,
    resolve_input_artifact,
    resolve_workers,
    run,
)

if __name__ == "__main__":
    main()
