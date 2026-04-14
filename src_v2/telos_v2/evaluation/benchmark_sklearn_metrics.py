"""Optional sklearn metrics: ranked transcripts × bundle tmap (not legacy gtfcuff PR)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from telos_v2.labels.transcript_labels import load_tmap_labels


def sklearn_metrics_ranked_vs_bundle_tmap(ranked_tsv: Path, tmap_path: Path) -> dict[str, Any]:
    """
    Classification metrics from merging ``transcripts.ranked.*.tsv`` with **bundle** tmap labels.

    This is a fast diagnostic only; legacy transcript AUPR uses :mod:`telos_v2.evaluation.transcript_pr_pipeline`
    (gffcompare + gtfcuff). Do not compare these numbers to gtfcuff AUC directly.
    """
    ranked = pd.read_csv(ranked_tsv, sep="\t")
    if "transcript_id" not in ranked.columns or "pred_prob" not in ranked.columns:
        raise ValueError(f"ranked TSV missing required columns: {ranked_tsv}")
    if "pred_label" not in ranked.columns:
        ranked["pred_label"] = (ranked["pred_prob"].astype(float) >= 0.5).astype(int)

    labels = load_tmap_labels(tmap_path)
    merged = ranked.merge(labels, on="transcript_id", how="inner")
    if merged.empty:
        return {
            "n_eval": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "aupr": None,
        }

    y_true = merged["label"].astype(int)
    y_pred = merged["pred_label"].astype(int)
    y_prob = merged["pred_prob"].astype(float)
    out: dict[str, Any] = {
        "n_eval": int(len(merged)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_true.nunique() >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["aupr"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
        out["aupr"] = None
    return out


def suffix_keys(metrics: dict[str, Any], suf: str) -> dict[str, Any]:
    return {f"{k}_{suf}": v for k, v in metrics.items()}
