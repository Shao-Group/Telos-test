"""
Stage I **test** metrics after benchmark predict: AUPR on held-out chromosomes / sites.

Joins ``sites.scored.tsv`` to assembly coverage coordinates and reference GTF sites so metrics
reflect the same candidates the model scored, not the training split alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score

from telos_v2.backends.gtfformat import build_cov_dataframe
from telos_v2.labels.site_labels import (
    label_sites_by_proximity,
    novel_reference_sites_from_gtf,
    normalize_chrom_name,
    reference_sites_from_gtf,
)


def safe_aupr(y_true: pd.Series, scores: pd.Series) -> float | None:
    """
    Average precision with guards for missing labels or degenerate single-class subsets.

    Aligns ``y_true`` and ``scores`` on non-NaN positions, requires at least two distinct labels in
    ``y_true`` after filtering, then calls :func:`sklearn.metrics.average_precision_score`.
    """
    yt = pd.to_numeric(y_true, errors="coerce")
    sc = pd.to_numeric(scores, errors="coerce")
    mask = yt.notna() & sc.notna()
    if not bool(mask.any()):
        return None
    y = yt[mask].astype(int)
    s = sc[mask].astype(float)
    if y.nunique() < 2:
        return None
    return float(average_precision_score(y, s))


def evaluate_stage1_test_aupr(
    sites_scored_tsv: Path,
    assembly_gtf: Path,
    ref_gtf: Path,
    *,
    tolerance_bp: int,
) -> dict[str, Any]:
    """
    Compute Stage I test AUPR for RF, XGB, and coverage baseline per site type.

    Reads ``sites_scored_tsv`` (must include ``site_type``, ``chrom``, ``position``, ``strand``,
    ``p_site_rf``, ``p_site_xgb``). Builds reference site table from ``ref_gtf`` and coverage from
    ``assembly_gtf``. For TSS and TES separately: inner-join scored rows to coverage on matching
    coordinates (after chromosome normalization), labels via :func:`~telos_v2.labels.site_labels.label_sites_by_proximity`,
    then three AUPR values per type.

    Returns:
        Flat dict with keys like ``stage1_test_aupr_tss_rf``; empty dict if inputs are unusable.
    """
    out: dict[str, Any] = {}
    df = pd.read_csv(
        sites_scored_tsv,
        sep="\t",
        low_memory=False,
        dtype={
            "transcript_id": str,
            "site_type": str,
            "chrom": str,
            "position": "Int64",
            "strand": str,
            "p_site_rf": float,
            "p_site_xgb": float,
        },
    )
    if df.empty:
        return out
    needed = {"site_type", "chrom", "position", "strand", "p_site_rf", "p_site_xgb"}
    if not needed.issubset(df.columns):
        return out
    ref_df = reference_sites_from_gtf(ref_gtf)
    if ref_df.empty:
        return out
    novel_ref_df = novel_reference_sites_from_gtf(ref_gtf)
    cov = build_cov_dataframe(assembly_gtf)

    for st in ("TSS", "TES"):
        sub = df[df["site_type"].astype(str).str.upper() == st].copy()
        if sub.empty:
            continue
        if st == "TSS":
            csub = cov[["tss_chrom", "tss_pos", "strand", "coverage"]].copy()
            csub["tss_chrom"] = csub["tss_chrom"].map(normalize_chrom_name)
            if len(csub) > csub[["tss_chrom", "tss_pos", "strand"]].drop_duplicates().shape[0]:
                csub = csub.groupby(["tss_chrom", "tss_pos", "strand"], as_index=False)["coverage"].mean()
            sub = sub.copy()
            sub["chrom"] = sub["chrom"].map(normalize_chrom_name)
            sub = sub.merge(
                csub,
                left_on=["chrom", "position", "strand"],
                right_on=["tss_chrom", "tss_pos", "strand"],
                how="inner",
            )
        else:
            csub = cov[["tes_chrom", "tes_pos", "strand", "coverage"]].copy()
            csub["tes_chrom"] = csub["tes_chrom"].map(normalize_chrom_name)
            if len(csub) > csub[["tes_chrom", "tes_pos", "strand"]].drop_duplicates().shape[0]:
                csub = csub.groupby(["tes_chrom", "tes_pos", "strand"], as_index=False)["coverage"].mean()
            sub = sub.copy()
            sub["chrom"] = sub["chrom"].map(normalize_chrom_name)
            sub = sub.merge(
                csub,
                left_on=["chrom", "position", "strand"],
                right_on=["tes_chrom", "tes_pos", "strand"],
                how="inner",
            )
        if sub.empty:
            n_before = len(df[df["site_type"].astype(str).str.upper() == st])
            if n_before > 0:
                print(
                    f"[telos_v2] benchmark Stage I test AUPR: no rows for {st} after cov coordinate join "
                    f"({n_before} scored sites; check chrom names vs assembly GTF / normalize_chrom_name)."
                )
            continue
        labels = label_sites_by_proximity(sub, ref_df, st, tolerance_bp)
        out[f"stage1_test_aupr_{st.lower()}_rf"] = safe_aupr(labels, sub["p_site_rf"])
        out[f"stage1_test_aupr_{st.lower()}_xgb"] = safe_aupr(labels, sub["p_site_xgb"])
        out[f"stage1_test_aupr_{st.lower()}_baseline"] = safe_aupr(labels, sub["coverage"])
        if not novel_ref_df.empty:
            novel_labels = label_sites_by_proximity(sub, novel_ref_df, st, tolerance_bp)
            out[f"stage1_test_aupr_{st.lower()}_novel_rf"] = safe_aupr(novel_labels, sub["p_site_rf"])
            out[f"stage1_test_aupr_{st.lower()}_novel_xgb"] = safe_aupr(novel_labels, sub["p_site_xgb"])
            out[f"stage1_test_aupr_{st.lower()}_novel_baseline"] = safe_aupr(
                novel_labels, sub["coverage"]
            )
            out[f"stage1_test_n_novel_{st.lower()}"] = int(pd.Series(novel_labels).astype(int).sum())
            out[f"stage1_test_n_eval_{st.lower()}"] = int(len(sub))
    return out
