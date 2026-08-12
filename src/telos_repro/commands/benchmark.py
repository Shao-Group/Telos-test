"""
``benchmark`` CLI handler: re-exports :func:`~telos_repro.benchmark.orchestrator.run_benchmark`.
"""

from __future__ import annotations

from telos_repro.benchmark.orchestrator import run_benchmark

__all__ = ["run_benchmark"]
