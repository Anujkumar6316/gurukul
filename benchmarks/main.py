"""Backward-compatible proxy re-exporting from gurukul.benchmarks.main."""

from gurukul.benchmarks.main import *
from gurukul.benchmarks.main import BANNER_ART, main, print_banner_and_commands

if __name__ == "__main__":
    main()
