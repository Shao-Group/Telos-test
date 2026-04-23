"""
``benchmark-matrix`` CLI handler: re-exports :func:`~telos_v2.benchmark.matrix.run_benchmark_matrix`.

Logic stays in :mod:`telos_v2.benchmark.matrix` so tests import the generator without pulling in CLI.
"""

from __future__ import annotations

from telos_v2.benchmark.matrix import run_benchmark_matrix

__all__ = ["run_benchmark_matrix"]
