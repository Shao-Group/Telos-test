#!/usr/bin/env python3
"""Compare local benchmark_summary.csv cells to ../Telos-test/runs (not goldens/)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from telos_repro.parity import compare_summaries


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-root", type=Path, required=True)
    p.add_argument("--ref-root", type=Path, required=True, help="Telos-test/runs/cross_annotation_repro")
    p.add_argument("--pair-glob", default="*__train_gencode__test_gencode")
    p.add_argument("--abs-tol", type=float, default=1e-6)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    local_root = args.local_root.resolve()
    ref_root = args.ref_root.resolve()
    cells = sorted(local_root.glob(args.pair_glob))
    results = []
    all_ok = True
    for cell in cells:
        if not cell.is_dir():
            continue
        local_csv = cell / "reports" / "benchmark_summary.csv"
        ref_csv = ref_root / cell.name / "reports" / "benchmark_summary.csv"
        row = {
            "cell": cell.name,
            "local": str(local_csv),
            "ref": str(ref_csv),
        }
        if not local_csv.is_file():
            row["status"] = "missing_local"
            all_ok = False
            results.append(row)
            continue
        if not ref_csv.is_file():
            row["status"] = "missing_ref"
            all_ok = False
            results.append(row)
            continue
        cmp = compare_summaries(local_csv, ref_csv, abs_tol=args.abs_tol)
        row["compare"] = cmp
        row["status"] = "ok" if cmp.get("ok") else "drift"
        if not cmp.get("ok"):
            all_ok = False
        results.append(row)

    out = {
        "local_root": str(local_root),
        "ref_root": str(ref_root),
        "pair_glob": args.pair_glob,
        "n_cells": len(results),
        "all_ok": all_ok,
        "cells": results,
        "note": (
            "Reference is Telos-test/runs (paper benchmark outputs), not goldens/. "
            "Retrain with fresh features may drift vs frozen shared-train metrics."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {args.report}", file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
