"""
Transcript-level precision–recall / AUC by re-annotating the assembly with model scores.

Pipeline steps:

1. ``gtfformat update-transcript-cov`` (Python backend) — copy model scores into each transcript’s
   ``coverage`` field in a derived assembly GTF.
2. ``gffcompare -r <ref.gtf>`` — produce a fresh ``.tmap`` for that derived GTF (distinct from any
   bundle-supplied training tmap).
3. Python ``gtfcuff``-style ROC/AUC — sweep ranked transcripts against the new tmap; ``ref-size`` is
   the count of **multi-exon** reference transcripts.

**Baseline:** run gffcompare on the **original** assembly GTF (assembler coverage only), then the
same ROC/AUC step.

**Chromosomes:** when ``filter_validation_chroms`` is true (default), both assembly and reference
GTFs pass through ``gtfformat filter-chrom`` so evaluation uses the same validation chromosomes as
Stage I/II training (derived from ``split_policy``). ``chromosomes_file`` optionally supplies an
explicit shared chromosome list for both inputs.

This path is **not** the same as joining ``transcripts.ranked.tsv`` to a fixed tmap inside pandas:
that skips re-annotation and can mis-align labels with how scores were produced.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from telos_v2.backends.gtfcuff import run_gtfcuff_auc, run_gtfcuff_roc
from telos_v2.backends.gtfformat import run_filter_chrom, run_update_transcript_cov
from telos_v2.gtf_attributes import parse_transcript_id
from telos_v2.models.chrom_split import seqnames_on_validation_split_from_gtf

_ROC_LINE = re.compile(
    r"sensitivity\s*=\s*([\d.]+)\s+precision\s*=\s*([\d.]+)",
)

def resolve_gffcompare_executable(gffcompare_bin: str | None) -> str:
    """
    Resolve ``gffcompare`` executable string for ``subprocess``: non-empty YAML override wins, then env
    ``GFFCOMPARE``, then :func:`shutil.which`. Raises :class:`FileNotFoundError` if none resolve.
    """
    if gffcompare_bin and str(gffcompare_bin).strip():
        return str(gffcompare_bin).strip()
    env = os.environ.get("GFFCOMPARE", "").strip()
    if env:
        return env
    found = shutil.which("gffcompare")
    if found:
        return found
    raise FileNotFoundError(
        "gffcompare not found: install it, put it on PATH, set GFFCOMPARE, or set "
        "analysis.pr_vs_baseline.gffcompare_bin in the benchmark YAML."
    )


def count_multi_exon_reference_transcripts(ref_gtf: Path) -> int:
    """
    Count reference transcripts that have more than one exon row in ``ref_gtf``.

    Scans exon features, buckets by ``transcript_id`` from column 9, counts exons per id, then
    returns how many ids have count > 1. Single-exon transcripts are excluded because the ROC
    definition used here mirrors the multi-exon reference denominator used in transcript PR.
    """
    exon_counts: dict[str, int] = {}
    with ref_gtf.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "exon":
                continue
            tid = parse_transcript_id(cols[8])
            if tid:
                exon_counts[tid] = exon_counts.get(tid, 0) + 1
    return sum(1 for c in exon_counts.values() if c > 1)


def write_predictions_for_update_cov(ranked_tsv: Path, out_tsv: Path, *, score_col: str = "pred_prob") -> None:
    """
    Write TSV consumed by :func:`telos_v2.backends.gtfformat.run_update_transcript_cov`: header + rows.

    Sorts by ``score_col`` descending and keeps first row per ``transcript_id`` so one score maps to
    each transcript when duplicates exist in the ranked table.
    """
    df = pd.read_csv(ranked_tsv, sep="\t", dtype={"transcript_id": str})
    if "transcript_id" not in df.columns or score_col not in df.columns:
        raise ValueError(f"ranked TSV needs transcript_id and {score_col}: {ranked_tsv}")
    sub = df[["transcript_id", score_col]].copy()
    sub = sub.sort_values(score_col, ascending=False).drop_duplicates("transcript_id", keep="first")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_tsv, sep="\t", index=False, header=True)


def _find_gffcompare_tmap(work_dir: Path, prefix: str, query_gtf: Path) -> Path:
    """
    Locate gffcompare’s ``.tmap`` under ``work_dir`` after ``-o prefix``.

    Prefers ``{prefix}.{query_gtf.name}.tmap``; if missing, accepts exactly one ``{prefix}*.tmap``.
    """
    expected = work_dir / f"{prefix}.{query_gtf.name}.tmap"
    if expected.is_file():
        return expected
    matches = sorted(work_dir.glob(f"{prefix}*.tmap"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"gffcompare tmap not found (expected {expected.name}); files in {work_dir}: "
            f"{[p.name for p in work_dir.iterdir()]}"
        )
    raise FileNotFoundError(f"Ambiguous gffcompare tmap matches for prefix {prefix!r}: {matches}")


def run_gffcompare(
    query_gtf: Path,
    ref_gtf: Path,
    output_prefix: str,
    work_dir: Path,
    *,
    gffcompare_bin: str | None = None,
) -> Path:
    """
    Run ``gffcompare -r ref_gtf -o output_prefix query_gtf`` with cwd ``work_dir``.

    Returns path to the resulting ``.tmap`` via :func:`_find_gffcompare_tmap`. Raises :class:`RuntimeError`
    on non-zero exit.
    """
    exe = resolve_gffcompare_executable(gffcompare_bin)
    work_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            exe,
            "-r",
            str(ref_gtf.resolve()),
            "-o",
            output_prefix,
            str(query_gtf.resolve()),
        ],
        cwd=str(work_dir.resolve()),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gffcompare failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return _find_gffcompare_tmap(work_dir, output_prefix, query_gtf)


def parse_gtfcuff_roc_stdout(text: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse recall/precision pairs from :func:`telos_v2.backends.gtfcuff.run_gtfcuff_roc` text output.

    Accepts ``recall\\tprecision`` data lines (skips header), or lines containing ``ROC:`` with
    ``sensitivity = … precision = …`` percentages scaled to [0,1]. Raises if no points parsed.
    """
    recalls: list[float] = []
    precs: list[float] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.lower().startswith("recall\t"):
            continue
        if "\t" in raw:
            parts = raw.split("\t")
            if len(parts) >= 2:
                try:
                    recalls.append(float(parts[0]))
                    precs.append(float(parts[1]))
                    continue
                except ValueError:
                    pass
        if "ROC:" in raw:
            m = _ROC_LINE.search(raw)
            if not m:
                continue
            recalls.append(float(m.group(1)) / 100.0)
            precs.append(float(m.group(2)) / 100.0)
    if not recalls:
        raise ValueError("No ROC lines parsed from gtfcuff roc output")
    return np.asarray(recalls, dtype=float), np.asarray(precs, dtype=float)


def _tmap_positive_negative(tmap_path: Path) -> tuple[int, int]:
    """
    Scan ``tmap_path`` data rows: count total rows with single-char class code and how many are ``=``.

    Returns ``(n_total, n_positive)`` for logging / summary sidecars.
    """
    n_pos = n_tot = 0
    with tmap_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split("\t")
            if len(parts) < 3:
                parts = raw.split()
            if len(parts) < 3:
                continue
            if parts[0] == "ref_gene_id":
                continue
            code = parts[2]
            if len(code) != 1:
                continue
            n_tot += 1
            if code == "=":
                n_pos += 1
    return n_tot, n_pos


def _write_temp_chrom_list(names: list[str]) -> Path:
    """
    Create a temp file with one chromosome name per line for :func:`run_filter_chrom`.

    Caller must ``unlink`` the path when finished (see :func:`_apply_pr_chrom_filter` ``finally``).
    """
    fd, raw = tempfile.mkstemp(prefix="telos_valchrom_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + ("\n" if names else ""))
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(raw)
        except OSError:
            pass
        raise
    return Path(raw)


def _apply_pr_chrom_filter(
    *,
    assembly_gtf: Path,
    ref_gtf: Path,
    work_dir: Path,
    chromosomes_file: Path | None,
    filter_validation_chroms: bool,
    autosome_train_range: tuple[int, int] | None,
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Optionally filter assembly + reference to the same chromosome set used for validation-split PR.

    Returns:
        ``(assembly_out, ref_out, meta)``. If no filtering applies, paths are originals and ``meta``
        marks ``pr_chrom_filter`` as ``none``. Explicit file mode runs filter-chrom with that list;
        auto mode derives lists from :func:`~telos_v2.models.chrom_split.seqnames_on_validation_split_from_gtf`
        per GTF. Temp list files are deleted in a ``finally`` block when used.
    """
    meta: dict[str, Any] = {
        "pr_chrom_filter": "none",
        "pr_chromosomes_file": "",
        "pr_filtered_assembly_gtf": "",
        "pr_filtered_ref_gtf": "",
    }
    chrom_file = chromosomes_file
    use_explicit = chrom_file is not None and chrom_file.is_file()
    use_auto = (
        not use_explicit
        and filter_validation_chroms
        and autosome_train_range is not None
    )
    if not use_explicit and not use_auto:
        return assembly_gtf.resolve(), ref_gtf.resolve(), meta

    if use_explicit:
        meta["pr_chrom_filter"] = "explicit_file"
        meta["pr_chromosomes_file"] = str(chrom_file.resolve())
        asm_out = work_dir / "query_validation_chrom.gtf"
        ref_out = work_dir / "ref_validation_chrom.gtf"
        run_filter_chrom(None, assembly_gtf, chrom_file, asm_out)
        run_filter_chrom(None, ref_gtf, chrom_file, ref_out)
        meta["pr_filtered_assembly_gtf"] = str(asm_out.resolve())
        meta["pr_filtered_ref_gtf"] = str(ref_out.resolve())
        return asm_out, ref_out, meta

    assert autosome_train_range is not None
    asm_names = seqnames_on_validation_split_from_gtf(assembly_gtf, autosome_train_range)
    ref_names = seqnames_on_validation_split_from_gtf(ref_gtf, autosome_train_range)
    if not asm_names and not ref_names:
        return assembly_gtf.resolve(), ref_gtf.resolve(), {
            **meta,
            "pr_chrom_filter": "skipped_empty_validation_seqnames",
        }

    meta["pr_chrom_filter"] = "split_policy"
    asm_out: Path
    ref_out: Path
    tmp_chrom_files: list[Path] = []
    try:
        if asm_names:
            list_asm = _write_temp_chrom_list(asm_names)
            tmp_chrom_files.append(list_asm)
            asm_out = work_dir / "query_validation_chrom.gtf"
            run_filter_chrom(None, assembly_gtf, list_asm, asm_out)
            meta["pr_filtered_assembly_gtf"] = str(asm_out.resolve())
        else:
            asm_out = assembly_gtf.resolve()
        if ref_names:
            list_ref = _write_temp_chrom_list(ref_names)
            tmp_chrom_files.append(list_ref)
            ref_out = work_dir / "ref_validation_chrom.gtf"
            run_filter_chrom(None, ref_gtf, list_ref, ref_out)
            meta["pr_filtered_ref_gtf"] = str(ref_out.resolve())
        else:
            ref_out = ref_gtf.resolve()
        return asm_out, ref_out, meta
    finally:
        for p in tmp_chrom_files:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def _chrom_meta_as_transcript_pr(chrom_meta: dict[str, Any]) -> dict[str, Any]:
    """
    Prefix chrom-filter metadata for benchmark CSV: keys ``pr_foo`` become ``transcript_pr_foo``;
    other keys get ``transcript_pr_`` prepended.
    """
    out: dict[str, Any] = {}
    for k, v in chrom_meta.items():
        if k.startswith("pr_"):
            out["transcript_pr_" + k[3:]] = v
        else:
            out[f"transcript_pr_{k}"] = v
    return out


def run_transcript_pr_benchmark(
    *,
    assembly_gtf: Path,
    ref_gtf: Path,
    ranked_tsv: Path,
    reports_pr_dir: Path,
    work_rel: str,
    prefix: str,
    gtfcuff_bin: Path | None = None,
    gffcompare_bin: str | None = None,
    measure: str = "cov",
    score_col: str = "pred_prob",
    plot: bool = True,
    plot_filename: str = "transcript_pr.png",
    chromosomes_file: Path | None = None,
    filter_validation_chroms: bool = True,
    autosome_train_range: tuple[int, int] | None = None,
    save_pr_tables: bool = False,
    ephemeral_workdir: bool = False,
) -> dict[str, Any]:
    """
    Run the full transcript-level PR benchmark for one model output (one ranked TSV).

    **Ordered steps**

    1. Create ``work_dir`` — either ``reports_pr_dir / work_rel`` or a temp directory when
       ``ephemeral_workdir`` is true.
    2. Optionally restrict chromosomes via :func:`_apply_pr_chrom_filter` (explicit file and/or
       validation split from ``autosome_train_range``).
    3. Count multi-exon reference transcripts on the (possibly filtered) reference GTF.
    4. Materialize ``pred_tsv`` from ``ranked_tsv`` (dedupe transcripts, keep highest score).
    5. Run ``update-transcript-cov`` to bake scores into a derived assembly GTF.
    6. Run gffcompare for **model** path (updated GTF vs ref) and **baseline** path (raw assembly vs ref).
    7. Run Python gtfcuff ROC + AUC for both tmaps with the same ``measure`` and ``ref_multi`` denominator.
    8. Optionally write PR point TSVs and a small summary CSV under ``reports_pr_dir``.
    9. Optionally render an ROC-style PR plot with matplotlib.
    10. Delete temp ``work_dir`` when ephemeral.

    Args:
        assembly_gtf: Unfiltered assembly GTF path (bundle).
        ref_gtf: Reference annotation GTF.
        ranked_tsv: Model rankings with ``transcript_id`` and ``score_col``.
        reports_pr_dir: Where plots/tables land; also parent of persistent ``work_rel`` when not ephemeral.
        work_rel: Subdirectory name under ``reports_pr_dir`` for non-ephemeral intermediates.
        prefix: Prefix for gffcompare outputs and intermediate filenames.
        gtfcuff_bin: Unused placeholder for historical CLI symmetry; Python backend ignores it.
        gffcompare_bin: Optional explicit gffcompare executable; else ``GFFCOMPARE`` or ``PATH``.
        measure: Abundance column name passed through to ROC (typically ``cov``).
        score_col: Column in ``ranked_tsv`` used as the transcript score.
        plot: Whether to attempt matplotlib PNG output.
        plot_filename: PNG basename under ``reports_pr_dir``.
        chromosomes_file: Optional explicit chromosome list file for filter-chrom on both inputs.
        filter_validation_chroms: If true and ``autosome_train_range`` is set, filter to validation chroms.
        autosome_train_range: Inclusive numeric autosome range from Stage I split policy parsing.
        save_pr_tables: If true, write model/baseline PR TSVs and AUPR summary CSV.
        ephemeral_workdir: If true, use ``tempfile.mkdtemp`` and delete it after extracting metrics.

    Returns:
        Dict with string keys including ``transcript_pr_auc_model``, ``transcript_pr_auc_baseline``,
        tmap row counts, optional paths to saved tables/plot, optional ``transcript_pr_work_dir``,
        and chrom-filter metadata prefixed with ``transcript_pr_``.
    """
    cleanup_dir: Path | None = None
    if ephemeral_workdir:
        cleanup_dir = Path(tempfile.mkdtemp(prefix="telos_transcript_pr_"))
        work_dir = cleanup_dir
    else:
        work_dir = (reports_pr_dir / work_rel).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    assembly_eff, ref_eff, chrom_meta = _apply_pr_chrom_filter(
        assembly_gtf=assembly_gtf,
        ref_gtf=ref_gtf,
        work_dir=work_dir,
        chromosomes_file=chromosomes_file,
        filter_validation_chroms=filter_validation_chroms,
        autosome_train_range=autosome_train_range,
    )

    ref_multi = count_multi_exon_reference_transcripts(ref_eff)

    pred_tsv = work_dir / f"{prefix}_predictions.tsv"
    write_predictions_for_update_cov(ranked_tsv, pred_tsv, score_col=score_col)

    updated_gtf = work_dir / f"{prefix}_updated_cov.gtf"
    run_update_transcript_cov(None, assembly_eff, pred_tsv, updated_gtf)

    prefix_model = f"{prefix}_model"
    tmap_model = run_gffcompare(
        updated_gtf, ref_eff, prefix_model, work_dir, gffcompare_bin=gffcompare_bin
    )

    roc_model_text = run_gtfcuff_roc(gtfcuff_bin, tmap_model, ref_multi, measure=measure)
    auc_model = run_gtfcuff_auc(gtfcuff_bin, tmap_model, ref_multi)

    prefix_base = f"{prefix}_baseline"
    tmap_base = run_gffcompare(
        assembly_eff, ref_eff, prefix_base, work_dir, gffcompare_bin=gffcompare_bin
    )
    roc_base_text = run_gtfcuff_roc(gtfcuff_bin, tmap_base, ref_multi, measure=measure)
    auc_baseline = run_gtfcuff_auc(gtfcuff_bin, tmap_base, ref_multi)

    rec_m, pre_m = parse_gtfcuff_roc_stdout(roc_model_text)
    rec_b, pre_b = parse_gtfcuff_roc_stdout(roc_base_text)

    n_m, pos_m = _tmap_positive_negative(tmap_model)
    n_b, pos_b = _tmap_positive_negative(tmap_base)

    reports_pr_dir.mkdir(parents=True, exist_ok=True)
    model_curve = reports_pr_dir / f"{prefix}_model_pr.tsv"
    baseline_curve = reports_pr_dir / f"{prefix}_baseline_pr.tsv"
    summary_csv = reports_pr_dir / f"{prefix}_aupr_summary.csv"
    curve_model_path = ""
    curve_base_path = ""
    summary_path = ""
    if save_pr_tables:
        pd.DataFrame({"recall": rec_m, "precision": pre_m}).to_csv(model_curve, sep="\t", index=False)
        pd.DataFrame({"recall": rec_b, "precision": pre_b}).to_csv(baseline_curve, sep="\t", index=False)
        pd.DataFrame(
            [
                {
                    "curve": "model",
                    "transcript_pr_auc": auc_model,
                    "n_tmap_rows": n_m,
                    "n_class_eq": pos_m,
                    "ref_multi_exon": ref_multi,
                    "measure": measure,
                },
                {
                    "curve": "baseline",
                    "transcript_pr_auc": auc_baseline,
                    "n_tmap_rows": n_b,
                    "n_class_eq": pos_b,
                    "ref_multi_exon": ref_multi,
                    "measure": measure,
                },
            ]
        ).to_csv(summary_csv, index=False)
        curve_model_path = str(model_curve.resolve())
        curve_base_path = str(baseline_curve.resolve())
        summary_path = str(summary_csv.resolve())

    out: dict[str, Any] = {
        "transcript_pr_auc_model": float(auc_model),
        "transcript_pr_auc_baseline": float(auc_baseline),
        "transcript_pr_n_tmap_rows": n_m,
        "transcript_pr_n_class_eq": pos_m,
        "transcript_pr_abundance_measure": measure,
        "transcript_pr_model_points_tsv": curve_model_path,
        "transcript_pr_baseline_points_tsv": curve_base_path,
        "transcript_pr_metrics_summary_csv": summary_path,
        "transcript_pr_plot_png": "",
        "transcript_pr_work_dir": "" if cleanup_dir is not None else str(work_dir.resolve()),
        **_chrom_meta_as_transcript_pr(chrom_meta),
    }

    if plot:
        try:
            import matplotlib.pyplot as plt

            plot_path = reports_pr_dir / plot_filename
            fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
            ax.plot(rec_b, pre_b, label=f"Baseline (AUC={auc_baseline:.4f})", alpha=0.85)
            ax.plot(rec_m, pre_m, label=f"Model (AUC={auc_model:.4f})", alpha=0.85)
            ax.set_xlabel("Recall (sensitivity)")
            ax.set_ylabel("Precision")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower left")
            ax.set_title("Transcript PR (gffcompare + ranking sweep)")
            fig.tight_layout()
            fig.savefig(plot_path)
            plt.close(fig)
            out["transcript_pr_plot_png"] = str(plot_path)
        except ImportError:
            pass

    if cleanup_dir is not None:
        shutil.rmtree(cleanup_dir, ignore_errors=True)

    return out
