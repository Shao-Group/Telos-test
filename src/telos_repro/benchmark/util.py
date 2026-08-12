"""
Small helpers for benchmark orchestration: strict path coercion and CSV-safe float rounding.
"""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any


def as_path(v: Any, name: str) -> Path:
    """
    Convert a benchmark YAML field to :class:`~pathlib.Path`.

    Raises:
        ValueError: If ``v`` is ``None`` or blank after ``str()`` (field ``name`` is cited).
    """
    if v is None or str(v).strip() == "":
        raise ValueError(f"Missing required benchmark field: {name}")
    return Path(str(v))


def round_float_metrics_in_row(row: dict[str, Any], *, ndigits: int = 2) -> None:
    """
    In-place round all real-valued metrics in a summary row for stable CSV output.

    Skips booleans and integers. Mutates ``row`` values that are :class:`numbers.Real` (including
    numpy scalars that register as Real).
    """
    for k, v in list(row.items()):
        if isinstance(v, bool):
            continue
        if isinstance(v, numbers.Integral):
            continue
        if isinstance(v, numbers.Real):
            row[k] = round(float(v), ndigits)
