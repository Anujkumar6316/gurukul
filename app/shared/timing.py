from __future__ import annotations
from collections.abc import Callable
from time import perf_counter
from typing import TypeVar

T = TypeVar("T")

def measure_ms(operation: Callable[[], T]) -> tuple[T, float]:
    started = perf_counter()
    result = operation()
    return result, (perf_counter()-started)*1000