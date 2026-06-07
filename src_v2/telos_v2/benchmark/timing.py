"""Timestamped logging and CSV timing records for benchmark runs."""

from __future__ import annotations

import csv
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_ts(message: str, *, file: Any = None) -> None:
    """Print a line prefixed with an ISO UTC timestamp."""
    out = file if file is not None else sys.stdout
    print(f"[{utc_now_iso()}] {message}", file=out, flush=True)


def elapsed_s(start: float) -> float:
    return round(time.perf_counter() - start, 3)


@contextmanager
def timed_phase(
    *,
    phase: str,
    timing_csv: Path | None,
    extra: dict[str, str] | None = None,
) -> Iterator[None]:
    """Log start/end of a phase and optionally append one row to ``timing_csv``."""
    fields = {"phase": phase, **(extra or {})}
    log_ts(f"{phase} start" + (f" ({fields})" if extra else ""))
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dur = elapsed_s(t0)
        log_ts(f"{phase} end elapsed_s={dur}")
        if timing_csv is not None:
            append_timing_row(
                timing_csv,
                {
                    **fields,
                    "started_at": datetime.fromtimestamp(t0, tz=timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "ended_at": utc_now_iso(),
                    "elapsed_s": str(dur),
                },
            )


def append_timing_row(timing_csv: Path, row: dict[str, str]) -> None:
    timing_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not timing_csv.is_file() or timing_csv.stat().st_size == 0
    fieldnames = list(row.keys())
    with timing_csv.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
