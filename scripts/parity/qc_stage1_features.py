#!/usr/bin/env python3
"""Fresh Stage I feature extract + anomaly report (no disk cache)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _qc_frame(df: pd.DataFrame, *, label: str) -> dict:
    num = df.select_dtypes(include=[np.number])
    report: dict = {
        "label": label,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_numeric_cols": int(num.shape[1]),
    }
    if num.empty:
        report["status"] = "fail_no_numeric"
        return report

    nan_frac = num.isna().mean()
    inf_counts = {c: int(np.isinf(num[c].to_numpy(dtype=float, copy=False)).sum()) for c in num.columns}
    zero_frac = (num == 0).mean()
    const_cols = [c for c in num.columns if num[c].nunique(dropna=False) <= 1]
    all_nan = [c for c, f in nan_frac.items() if f >= 1.0 - 1e-15]
    high_nan = [c for c, f in nan_frac.items() if 0.5 < f < 1.0]
    any_inf = [c for c, n in inf_counts.items() if n > 0]
    all_zero = [c for c, f in zero_frac.items() if f >= 1.0 - 1e-15]

    # Dual-name Stage I quirk expected by frozen models / paper path.
    has_up_down = "up_down_stream_ratio" in df.columns
    has_upstream = "upstream_downstream_ratio" in df.columns
    both_present_same_row = False
    if has_up_down and has_upstream:
        both_present_same_row = bool(
            (df["up_down_stream_ratio"].notna() & df["upstream_downstream_ratio"].notna()).any()
        )

    report.update(
        {
            "has_up_down_stream_ratio": has_up_down,
            "has_upstream_downstream_ratio": has_upstream,
            "both_ratio_cols_same_row": both_present_same_row,
            "n_all_nan_cols": len(all_nan),
            "all_nan_cols": all_nan[:40],
            "n_high_nan_cols": len(high_nan),
            "high_nan_cols": high_nan[:40],
            "n_inf_cols": len(any_inf),
            "inf_cols": {c: inf_counts[c] for c in any_inf[:40]},
            "n_constant_cols": len(const_cols),
            "constant_cols_sample": const_cols[:40],
            "n_all_zero_cols": len(all_zero),
            "all_zero_cols_sample": all_zero[:40],
            "numeric_min": {c: float(num[c].min()) for c in list(num.columns)[:15]},
            "numeric_max": {c: float(num[c].max()) for c in list(num.columns)[:15]},
        }
    )

    # Soft fail rules: hard anomalies that block a clean paper re-run.
    # Stream-ratio quirk applies only to site feature tables (df_all), not df_cov.
    hard = []
    if report["n_rows"] < 1:
        hard.append("empty_feature_table")
    if all_nan:
        hard.append("all_nan_columns")
    if any_inf:
        hard.append("infinite_values")
    if label == "df_all":
        if not (has_up_down or has_upstream):
            hard.append("missing_stream_ratio_feature")
        # Expected quirk: populated branch uses up_down_stream_ratio; empty uses
        # upstream_downstream_ratio. Both non-null on one row is unexpected.
        if both_present_same_row:
            hard.append("both_ratio_names_on_same_row_unexpected")

    report["hard_anomalies"] = hard
    report["status"] = "ok" if not hard else "anomaly"
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bam", type=Path, required=True)
    p.add_argument("--gtf", type=Path, required=True)
    p.add_argument("--stage1-config", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--dump-parquet", type=Path, default=None, help="Optional df_all dump")
    args = p.parse_args()

    # Force no cache regardless of YAML / ambient env.
    os.environ.pop("TELOS_STAGE1_CACHE_DIR", None)

    from telos.config_loader import load_mapping_config
    from telos.pipeline_core import build_stage1_inputs, build_stage1_runtime_config

    cfg = load_mapping_config(args.stage1_config.resolve())
    # Belt-and-suspenders: null cache in resolved runtime.
    fe = cfg.setdefault("stage1", {}).setdefault("feature_extraction", {})
    fe["cache_dir"] = None
    runtime = build_stage1_runtime_config(cfg, cli_no_parallel=False, cli_n_workers=None)
    if runtime.cache_dir is not None:
        print(f"ERROR: cache still enabled: {runtime.cache_dir}", file=sys.stderr)
        return 2

    print(f"[qc] fresh extract bam={args.bam}", flush=True)
    print(f"[qc] gtf={args.gtf}", flush=True)
    print(f"[qc] cache_dir={runtime.cache_dir!r}", flush=True)
    df_cov, df_all = build_stage1_inputs(bam=args.bam, gtf=args.gtf, runtime_cfg=runtime)
    print(f"[qc] df_cov={df_cov.shape} df_all={df_all.shape}", flush=True)

    cov_rep = _qc_frame(df_cov, label="df_cov")
    all_rep = _qc_frame(df_all, label="df_all")
    out = {
        "bam": str(args.bam.resolve()),
        "gtf": str(args.gtf.resolve()),
        "stage1_config": str(args.stage1_config.resolve()),
        "cache_dir": None,
        "df_cov": cov_rep,
        "df_all": all_rep,
        "status": "ok"
        if cov_rep["status"] == "ok" and all_rep["status"] == "ok"
        else "anomaly",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print(f"[qc] wrote {args.report}", flush=True)

    if args.dump_parquet is not None:
        args.dump_parquet.parent.mkdir(parents=True, exist_ok=True)
        # Prefer parquet; fall back to pickle (no pyarrow required).
        try:
            df_all.to_parquet(args.dump_parquet)
            print(f"[qc] wrote {args.dump_parquet}", flush=True)
        except ImportError:
            pkl = args.dump_parquet.with_suffix(".pkl")
            df_all.to_pickle(pkl)
            print(f"[qc] parquet unavailable; wrote {pkl}", flush=True)

    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
