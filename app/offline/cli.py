"""Backward-compatible proxy re-exporting from top-level gurukul package."""

from gurukul.cli import run_chat_cli


def run_cli() -> None:
    run_chat_cli()


if __name__ == "__main__":
    run_cli()
