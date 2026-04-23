"""
Benchmark summary CSV: stable column ordering and placeholder rows for invalid test configs.

The orchestrator appends one dict per test; this module normalizes keys for the final ``benchmark_summary.csv`` artifact.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from telos_v2.benchmark.util import round_float_metrics_in_row

SUMMARY_CSV_ORDERED_KEYS = [
    "test_id",
    "assembler_id",
    "status",
    "predict_code",
    "stage1_test_aupr_tss_rf",
    "stage1_test_aupr_tss_xgb",
    "stage1_test_aupr_tss_baseline",
    "stage1_test_aupr_tss_novel_rf",
    "stage1_test_aupr_tss_novel_xgb",
    "stage1_test_aupr_tss_novel_baseline",
    "stage1_test_n_novel_tss",
    "stage1_test_n_eval_tss",
    "stage1_test_aupr_tes_rf",
    "stage1_test_aupr_tes_xgb",
    "stage1_test_aupr_tes_baseline",
    "stage1_test_aupr_tes_novel_rf",
    "stage1_test_aupr_tes_novel_xgb",
    "stage1_test_aupr_tes_novel_baseline",
    "stage1_test_n_novel_tes",
    "stage1_test_n_eval_tes",
    "stage2_test_aupr_rf",
    "stage2_test_aupr_xgb",
    "stage2_test_aupr_baseline",
    "stage2_test_aupr_novel_rf",
    "stage2_test_aupr_novel_xgb",
    "stage2_test_aupr_novel_baseline",
    "stage2_test_n_novel_pos_tx",
    "stage2_test_n_eval_tx",
    "transcript_pr_auc_model_rf",
    "transcript_pr_auc_model_xgb",
    "transcript_pr_auc_baseline",
    "error",
    "transcript_pr_error",
]


def stub_test_row(
    test_id: str,
    *,
    status: str,
    error: str,
) -> dict[str, Any]:
    """
    Build a minimal result row when a test entry cannot be turned into a :class:`~telos_v2.config_models.PredictIO`.

    Fills empty paths and ``predict_code=2``; applies :func:`~telos_v2.benchmark.util.round_float_metrics_in_row`.
    """
    row: dict[str, Any] = {
        "test_id": test_id,
        "assembler_id": "",
        "predict_code": "2",
        "status": status,
        "error": error,
        "predict_outdir": "",
        "ranked_rf_tsv": "",
        "ranked_xgb_tsv": "",
    }
    round_float_metrics_in_row(row)
    return row


def write_benchmark_summary_csv(rows: list[dict[str, Any]], summary_csv: Path) -> None:
    """
    Write ``benchmark_summary.csv`` with columns ordered by :data:`SUMMARY_CSV_ORDERED_KEYS` first.

    Any additional keys present in any row are **dropped** here (only ordered keys that appear in the
    union of row keys are written). Missing values become empty strings. Creates parent directories.
    """
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    fields = [k for k in SUMMARY_CSV_ORDERED_KEYS if k in all_keys]
    compact_rows: list[dict[str, Any]] = []
    for r in rows:
        compact_rows.append({k: r.get(k, "") for k in fields})
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(compact_rows)
