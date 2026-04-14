"""
Transcript-level PR / AUC the same way as legacy ``src/generate_roc_data.py``:

1. ``gtfformat update-transcript-cov`` — write model scores into transcript ``coverage`` in the assembly GTF.
2. ``gffcompare -r <ref.gtf>`` on that GTF — new ``.tmap`` (not the pre-baked bundle tmap).
3. ``gtfcuff roc`` / ``gtfcuff auc`` on that tmap with ``ref-size`` = multi-exon reference transcript count.

Baseline: gffcompare on the **original** assembly GTF (coverage from the assembler), then gtfcuff.

When ``filter_validation_chroms`` is true (default), assembly and reference are passed through
``gtfformat filter-chrom`` to **validation chromosomes** only — same split as Stage I/II training
(``split_policy``), matching legacy ``generate_baseline`` / ``generate_roc_data``. Optional
``chromosomes_file`` forces the same explicit list for both GTFs (legacy text file).

This differs from merging ``transcripts.ranked.tsv`` with a static tmap + sklearn PR, which skips
re-annotation and mis-aligns labels with the ranking procedure.
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
from telos_v2.models.chrom_split import seqnames_on_validation_split_from_gtf

_ROC_LINE = re.compile(
    r"sensitivity\s*=\s*([\d.]+)\s+precision\s*=\s*([\d.]+)",
)

# Default Linux build used in this workspace when ``gffcompare`` is not on PATH and YAML omits
# ``analysis.pr_vs_baseline.gffcompare_bin``. Override with env ``GFFCOMPARE`` or YAML.
_DEFAULT_GFFCOMPARE = Path(
    "/datadisk1/ixk5174/tools/gffcompare-0.12.10.Linux_x86_64/gffcompare"
)


def resolve_gffcompare_executable(gffcompare_bin: str | None) -> str:
    """
    Configured path, then ``GFFCOMPARE`` env, then ``PATH``, then workspace default if present.
    """
    if gffcompare_bin and str(gffcompare_bin).strip():
        return str(gffcompare_bin).strip()
    env = os.environ.get("GFFCOMPARE", "").strip()
    if env:
        return env
    found = shutil.which("gffcompare")
    if found:
        return found
    if _DEFAULT_GFFCOMPARE.is_file():
        return str(_DEFAULT_GFFCOMPARE)
    raise FileNotFoundError(
        "gffcompare not found: not on PATH, GFFCOMPARE unset, and default missing at "
        f"{_DEFAULT_GFFCOMPARE}. Install gffcompare or set analysis.pr_vs_baseline.gffcompare_bin "
        "or GFFCOMPARE."
    )


def count_multi_exon_reference_transcripts(ref_gtf: Path) -> int:
    """Match legacy ``ROCPipeline._count_multi_exon_transcripts`` (exon rows per transcript_id)."""
    exon_counts: dict[str, int] = {}
    with ref_gtf.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "exon":
                continue
            m = re.search(r'transcript_id\s+"([^"]+)"', cols[8])
            if m:
                tid = m.group(1)
                exon_counts[tid] = exon_counts.get(tid, 0) + 1
    return sum(1 for c in exon_counts.values() if c > 1)


def write_predictions_for_update_cov(ranked_tsv: Path, out_tsv: Path, *, score_col: str = "pred_prob") -> None:
    """
    Two-column tab file for ``gtfformat update-transcript-cov`` (see ``genome1::update_transcript_coverage``):
    header row, then ``transcript_id`` + score (column 0 and 1).
    """
    df = pd.read_csv(ranked_tsv, sep="\t", dtype={"transcript_id": str})
    if "transcript_id" not in df.columns or score_col not in df.columns:
        raise ValueError(f"ranked TSV needs transcript_id and {score_col}: {ranked_tsv}")
    sub = df[["transcript_id", score_col]].copy()
    sub = sub.sort_values(score_col, ascending=False).drop_duplicates("transcript_id", keep="first")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_tsv, sep="\t", index=False, header=True)


def _find_gffcompare_tmap(work_dir: Path, prefix: str, query_gtf: Path) -> Path:
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
    """Parse ``ROC: ... sensitivity = X precision = Y`` lines; return recall, precision in [0,1]."""
    recalls: list[float] = []
    precs: list[float] = []
    for line in text.splitlines():
        if "ROC:" not in line:
            continue
        m = _ROC_LINE.search(line)
        if not m:
            continue
        recalls.append(float(m.group(1)) / 100.0)
        precs.append(float(m.group(2)) / 100.0)
    if not recalls:
        raise ValueError("No ROC lines parsed from gtfcuff roc output")
    return np.asarray(recalls, dtype=float), np.asarray(precs, dtype=float)


def _tmap_positive_negative(tmap_path: Path) -> tuple[int, int]:
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
    """Write one seqname per line; caller should unlink when done."""
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
    gtfformat_bin: Path,
    assembly_gtf: Path,
    ref_gtf: Path,
    work_dir: Path,
    chromosomes_file: Path | None,
    filter_validation_chroms: bool,
    autosome_train_range: tuple[int, int] | None,
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Optionally filter assembly + reference to the same chromosome set as legacy ROC prep.

    Returns ``(assembly_out, ref_out, meta)`` where ``meta`` describes what was done.
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
        run_filter_chrom(gtfformat_bin, assembly_gtf, chrom_file, asm_out)
        run_filter_chrom(gtfformat_bin, ref_gtf, chrom_file, ref_out)
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
            run_filter_chrom(gtfformat_bin, assembly_gtf, list_asm, asm_out)
            meta["pr_filtered_assembly_gtf"] = str(asm_out.resolve())
        else:
            asm_out = assembly_gtf.resolve()
        if ref_names:
            list_ref = _write_temp_chrom_list(ref_names)
            tmp_chrom_files.append(list_ref)
            ref_out = work_dir / "ref_validation_chrom.gtf"
            run_filter_chrom(gtfformat_bin, ref_gtf, list_ref, ref_out)
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


def run_transcript_pr_benchmark_gtfcuff(
    *,
    assembly_gtf: Path,
    ref_gtf: Path,
    ranked_tsv: Path,
    reports_pr_dir: Path,
    work_rel: str,
    prefix: str,
    gtfformat_bin: Path,
    gtfcuff_bin: Path,
    gffcompare_bin: str | None = None,
    measure: str = "cov",
    score_col: str = "pred_prob",
    plot: bool = True,
    plot_filename: str = "transcript_pr.png",
    chromosomes_file: Path | None = None,
    filter_validation_chroms: bool = True,
    autosome_train_range: tuple[int, int] | None = None,
    save_pr_tables: bool = False,
) -> dict[str, Any]:
    """
    Legacy-style transcript PR: filter-chrom (optional) → update-transcript-cov → gffcompare → gtfcuff.

    Required work files under ``reports_pr_dir / work_rel``: predictions TSV, cov-updated GTF, gffcompare
    outputs. Optional **curve/summary TSV** files are written only when ``save_pr_tables`` is true;
    the benchmark summary row always gets AUC from stdout parsing either way.
    """
    work_dir = (reports_pr_dir / work_rel).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    assembly_eff, ref_eff, chrom_meta = _apply_pr_chrom_filter(
        gtfformat_bin=gtfformat_bin,
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
    run_update_transcript_cov(gtfformat_bin, assembly_eff, pred_tsv, updated_gtf)

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
                    "gtfcuff_auc": auc_model,
                    "n_tmap_rows": n_m,
                    "n_class_eq": pos_m,
                    "ref_multi_exon": ref_multi,
                    "measure": measure,
                },
                {
                    "curve": "baseline",
                    "gtfcuff_auc": auc_baseline,
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
        "pr_aupr_model": float(auc_model),
        "pr_aupr_baseline": float(auc_baseline),
        "pr_n_items": n_m,
        "pr_n_positive": pos_m,
        "pr_abundance_column": measure,
        "pr_model_curve_tsv": curve_model_path,
        "pr_baseline_curve_tsv": curve_base_path,
        "pr_aupr_summary_csv": summary_path,
        "pr_plot_png": "",
        "pr_gtfcuff_work_dir": str(work_dir),
        **chrom_meta,
    }

    if plot:
        try:
            import matplotlib.pyplot as plt

            plot_path = reports_pr_dir / plot_filename
            fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
            ax.plot(rec_b, pre_b, label=f"Baseline (gtfcuff AUC={auc_baseline:.4f})", alpha=0.85)
            ax.plot(rec_m, pre_m, label=f"Model cov (gtfcuff AUC={auc_model:.4f})", alpha=0.85)
            ax.set_xlabel("Recall (sensitivity)")
            ax.set_ylabel("Precision")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower left")
            ax.set_title("Transcript PR (gffcompare + gtfcuff roc)")
            fig.tight_layout()
            fig.savefig(plot_path)
            plt.close(fig)
            out["pr_plot_png"] = str(plot_path)
        except ImportError:
            pass

    return out
