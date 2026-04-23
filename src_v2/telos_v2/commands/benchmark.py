"""
``benchmark`` CLI handler: re-exports :func:`~telos_v2.benchmark.orchestrator.run_benchmark`.
"""

from __future__ import annotations

from telos_v2.benchmark.orchestrator import run_benchmark

__all__ = ["run_benchmark"]
