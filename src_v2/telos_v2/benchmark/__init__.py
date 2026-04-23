"""
Multi-dataset benchmark package: matrix YAML generation, orchestration, metrics, reporting.

Public entrypoints: ``run_benchmark``, ``run_benchmark_matrix`` (loaded lazily to avoid importing
heavy dependencies when only submodule utilities are needed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["run_benchmark", "run_benchmark_matrix"]

if TYPE_CHECKING:
    from collections.abc import Callable

    run_benchmark: Callable[..., int]
    run_benchmark_matrix: Callable[..., int]


def __getattr__(name: str) -> Any:
    if name == "run_benchmark":
        from telos_v2.benchmark.orchestrator import run_benchmark as rb

        return rb
    if name == "run_benchmark_matrix":
        from telos_v2.benchmark.matrix import run_benchmark_matrix as rbm

        return rbm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
