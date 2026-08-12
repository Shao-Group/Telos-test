"""
RefSeq-novel evaluation on existing cross_annotation_repro runs (analysis-only).

Train on RefSeq; tests are gencode/ensembl cells from ``runs/cross_annotation_repro``.
For each test:
  - filter assembly GTF to multi-exon transcripts not identical to RefSeq (Python genome1-style),
  - re-run transcript PR (gtfcuff AUC) on filtered assembly vs full test annotation,
  - compute TSS/TES AUPR on all endpoint-novel sites vs RefSeq.

Usage:
  PYTHONPATH=src python src/experiments/evaluate_refseq_novel_cross_annotation.py
  PYTHONPATH=src python src/experiments/evaluate_refseq_novel_cross_annotation.py \\
    --root runs/cross_annotation_repro --only-cell sr__train_refseq__test_gencode
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from telos_repro.config_loader import default_stage1_config_path, load_mapping_config
from telos_repro.evaluation.transcript_pr_pipeline import (
    count_multi_exon_reference_transcripts,
    run_transcript_pr_benchmark,
)
from telos_repro.labels.novel_vs_refseq import (
    classify_transcripts_in_gtf,
    evaluate_novel_sites_aupr,
    filter_gtf_by_transcript_ids,
    filter_ranked_tsv,
    get_or_build_refseq_endpoint_sites,
    get_or_build_refseq_index,
)
from telos.models import TRANSCRIPTS_RANKED_RF_TSV, TRANSCRIPTS_RANKED_XGB_TSV
from telos.models.chrom_split import parse_split_policy

CELL_RE = re.compile(
    r"^(?P<data_type>.+?)__train_(?P<train>refseq)__test_(?P<test>gencode|ensembl)$"
)
DEFAULT_GFFCOMPARE_BIN = (
    "/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare"
)


def _resolve_gffcompare_bin(explicit: str | None, mapping: dict[str, Any]) -> str | None:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    analysis = mapping.get("analysis") or {}
    if isinstance(analysis, dict):
        pr = analysis.get("pr_vs_baseline") or {}
        if isinstance(pr, dict):
            raw = pr.get("gffcompare_bin")
            if raw and str(raw).strip():
                p = Path(str(raw))
                if p.is_file():
                    return str(p)
    default = Path(DEFAULT_GFFCOMPARE_BIN)
    if default.is_file():
        return str(default)
    return None


def _load_benchmark_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid benchmark YAML: {path}")
    return data


def _parse_cell_id(run_id: str) -> tuple[str, str] | None:
    m = CELL_RE.match(run_id.strip())
    if not m:
        return None
    return m.group("data_type"), m.group("test")


def _iter_target_cells(root: Path, only_cell: str | None) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for cell_dir in sorted(root.iterdir()):
        if not cell_dir.is_dir():
            continue
        parsed = _parse_cell_id(cell_dir.name)
        if parsed is None:
            continue
        if only_cell and cell_dir.name != only_cell:
            continue
        yaml_path = cell_dir / "reports" / "generated_benchmark.yaml"
        if not yaml_path.is_file():
            continue
        out.append((cell_dir.name, cell_dir))
    return out


def _stage1_tolerance_and_split(config_path: Path) -> tuple[int, tuple[int, int] | None]:
    cfg = load_mapping_config(config_path)
    tol = int(cfg.get("stage1", {}).get("training", {}).get("site_label_tolerance_bp", 50))
    split_pol = str(cfg.get("stage1", {}).get("training", {}).get("split_policy", "chr1-10"))
    try:
        autosome = parse_split_policy(split_pol)
    except ValueError:
        autosome = None
    return tol, autosome


def _build_filtered_assembly(
    *,
    assembly_gtf: Path,
    index,
    filtered_dir: Path,
    test_id: str,
) -> tuple[Path, pd.DataFrame, set[str]]:
    classified = classify_transcripts_in_gtf(assembly_gtf, index)
    classified_path = filtered_dir / f"{test_id}.classification.tsv"
    classified_path.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(classified_path, sep="\t", index=False)
    keep = set(
        classified.loc[classified["is_novel_vs_refseq"] == 1, "transcript_id"].astype(str).tolist()
    )
    out_gtf = filtered_dir / f"{test_id}.novel_vs_refseq.gtf"
    filter_gtf_by_transcript_ids(assembly_gtf, keep, out_gtf)
    return out_gtf, classified, keep


def _run_transcript_pr_row(
    *,
    test_id: str,
    model: str,
    filtered_asm: Path,
    ref_gtf: Path,
    ranked_tsv: Path,
    pr_dir: Path,
    autosome_range: tuple[int, int] | None,
    gffcompare_bin: str | None,
    ref_multi_cache: dict[str, int],
) -> dict[str, Any]:
    if not ranked_tsv.is_file() or not filtered_asm.is_file():
        return {"status": "missing_inputs"}
    ref_key = str(ref_gtf.resolve())
    if ref_key not in ref_multi_cache:
        ref_multi_cache[ref_key] = count_multi_exon_reference_transcripts(ref_gtf)
    print(
        f"[telos_repro] refseq-novel-eval PR start: {test_id} {model} "
        f"(ranked={ranked_tsv.name}, ref_multi={ref_multi_cache[ref_key]})",
        flush=True,
    )
    pr = run_transcript_pr_benchmark(
        assembly_gtf=filtered_asm,
        ref_gtf=ref_gtf,
        ranked_tsv=ranked_tsv,
        reports_pr_dir=pr_dir,
        work_rel=f"work_{test_id}_{model}",
        prefix=f"novel_tx_pr_{model}",
        gffcompare_bin=gffcompare_bin,
        measure="cov",
        score_col="pred_prob",
        plot=False,
        save_pr_tables=False,
        ephemeral_workdir=True,
        filter_validation_chroms=True,
        autosome_train_range=autosome_range,
    )
    auc_m = float(pr.get("transcript_pr_auc_model", 0.0))
    auc_b = float(pr.get("transcript_pr_auc_baseline", 0.0))
    print(
        f"[telos_repro] refseq-novel-eval PR done: {test_id} {model} "
        f"auc_model={auc_m:.2f} auc_baseline={auc_b:.2f}",
        flush=True,
    )
    return {
        "status": "ok",
        "auc_model": auc_m,
        "auc_baseline": auc_b,
        "auc_lift": auc_m - auc_b,
        "n_tmap_class_eq": int(pr.get("transcript_pr_n_class_eq", 0)),
        "ref_multi_exon": ref_multi_cache[ref_key],
    }


def run_evaluation(
    *,
    root: Path,
    refseq_gtf: Path,
    cache_dir: Path,
    outdir: Path,
    site_tolerance_bp: int | None,
    only_cell: str | None,
    gffcompare_bin: str | None,
    score_threshold: float,
    max_tests: int | None,
    skip_transcript_pr: bool,
) -> int:
    root = root.resolve()
    cache_dir = cache_dir.resolve()
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    filtered_root = outdir.parent / "filtered_assembly"
    filtered_root.mkdir(parents=True, exist_ok=True)

    index_path = cache_dir / "refseq_transcript_index.pkl"
    sites_cache = cache_dir / "refseq_endpoint_sites.tsv"
    index = get_or_build_refseq_index(refseq_gtf.resolve(), index_path)
    refseq_sites = get_or_build_refseq_endpoint_sites(refseq_gtf.resolve(), sites_cache)

    tx_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    ref_multi_cache: dict[str, int] = {}
    cells = _iter_target_cells(root, only_cell)
    if not cells:
        print(f"[telos_repro] refseq-novel-eval: no refseq→gencode/ensembl cells under {root}")
        return 2

    stage1_default = default_stage1_config_path()
    tol_default = 50
    if stage1_default.is_file():
        tol_default, _ = _stage1_tolerance_and_split(stage1_default)
    site_tol = int(site_tolerance_bp) if site_tolerance_bp is not None else tol_default

    for cell_id, cell_dir in cells:
        parsed = _parse_cell_id(cell_id)
        if parsed is None:
            continue
        data_type, test_annotation = parsed
        cfg_path = cell_dir / "reports" / "generated_benchmark.yaml"
        mapping = _load_benchmark_yaml(cfg_path)
        gffcompare = _resolve_gffcompare_bin(gffcompare_bin, mapping)
        if not gffcompare:
            print(f"[telos_repro] skip cell {cell_id}: gffcompare not found")
            continue
        tests = mapping.get("tests") or []
        if not isinstance(tests, list):
            continue
        print(f"[telos_repro] refseq-novel-eval cell: {cell_id} ({len(tests)} tests)", flush=True)
        n_done = 0

        for t in tests:
            if max_tests is not None and n_done >= max_tests:
                break
            if not isinstance(t, dict):
                continue
            test_id = str(t.get("id", "")).strip()
            if not test_id:
                continue
            asm_path = Path(str(t.get("gtf", ""))).resolve()
            ref_gtf = Path(str(t.get("ref_gtf", ""))).resolve()
            cfg = Path(str(t.get("config", stage1_default))).resolve()
            assembler_id = str(t.get("assembler_id", "")).strip().lower()
            predict_dir = cell_dir / "tests" / test_id
            tol, autosome = _stage1_tolerance_and_split(cfg if cfg.is_file() else stage1_default)

            if not asm_path.is_file() or not ref_gtf.is_file():
                print(f"[telos_repro] skip {test_id}: missing assembly or ref GTF", flush=True)
                continue

            print(
                f"[telos_repro] refseq-novel-eval test {n_done + 1}: {test_id} "
                f"(classify + filter assembly)",
                flush=True,
            )
            filt_dir = filtered_root / cell_id
            filtered_asm, classified, keep_ids = _build_filtered_assembly(
                assembly_gtf=asm_path,
                index=index,
                filtered_dir=filt_dir,
                test_id=test_id,
            )
            n_multi = int(len(classified))
            n_novel_asm = int(len(keep_ids))
            print(
                f"[telos_repro] refseq-novel-eval {test_id}: "
                f"multi_exon={n_multi} novel_vs_refseq={n_novel_asm}",
                flush=True,
            )

            pr_dir = outdir / "novel_transcript_pr" / cell_id / test_id
            if not skip_transcript_pr:
                for model, ranked_name in (
                    ("rf", TRANSCRIPTS_RANKED_RF_TSV),
                    ("xgb", TRANSCRIPTS_RANKED_XGB_TSV),
                ):
                    ranked_src = predict_dir / "predictions" / ranked_name
                    ranked_filt = filt_dir / f"{test_id}.{model}.novel_vs_refseq.ranked.tsv"
                    n_ranked = 0
                    pr_result: dict[str, Any] = {"status": "skipped_empty_novel_assembly"}
                    if not keep_ids:
                        pass
                    elif ranked_src.is_file():
                        n_ranked = filter_ranked_tsv(ranked_src, keep_ids, ranked_filt)
                        try:
                            pr_result = _run_transcript_pr_row(
                                test_id=test_id,
                                model=model,
                                filtered_asm=filtered_asm,
                                ref_gtf=ref_gtf,
                                ranked_tsv=ranked_filt,
                                pr_dir=pr_dir,
                                autosome_range=autosome,
                                gffcompare_bin=gffcompare,
                                ref_multi_cache=ref_multi_cache,
                            )
                        except (OSError, RuntimeError, FileNotFoundError, ValueError) as exc:
                            pr_result = {"status": f"failed:{exc}"}
                            print(
                                f"[telos_repro] refseq-novel-eval PR failed: {test_id} {model}: {exc}",
                                flush=True,
                            )
                    else:
                        pr_result = {"status": "missing_ranked_tsv"}
                    tx_rows.append(
                        {
                            "analysis_scope": "transcript_filtered_assembly",
                            "entity": "transcript",
                            "cell_id": cell_id,
                            "data_type": data_type,
                            "test_annotation": test_annotation,
                            "test_id": test_id,
                            "assembler_id": assembler_id,
                            "model": model,
                            "n_assembly_multi_exon": n_multi,
                            "n_assembly_novel_vs_refseq": n_novel_asm,
                            "n_ranked_novel": n_ranked,
                            "query_size": n_novel_asm,
                            "ref_size": pr_result.get("ref_multi_exon"),
                            "n_tmap_class_eq": pr_result.get("n_tmap_class_eq"),
                            "auc_model": pr_result.get("auc_model"),
                            "auc_baseline": pr_result.get("auc_baseline"),
                            "auc_lift": pr_result.get("auc_lift"),
                            "pr_status": pr_result.get("status", "ok"),
                        }
                    )

            print(f"[telos_repro] refseq-novel-eval {test_id}: site AUPR ...", flush=True)
            sites_scored = predict_dir / "predictions" / "sites.scored.tsv"
            if sites_scored.is_file():
                site_metrics = evaluate_novel_sites_aupr(
                    sites_scored_tsv=sites_scored,
                    assembly_gtf=asm_path,
                    test_ref_gtf=ref_gtf,
                    refseq_sites=refseq_sites,
                    tolerance_bp=site_tol,
                )
                for entity, metrics in site_metrics.items():
                    for model, score_col in (
                        ("rf", "aupr_novel_rf"),
                        ("xgb", "aupr_novel_xgb"),
                    ):
                        auc_m = metrics.get(score_col)
                        auc_b = metrics.get("aupr_novel_baseline")
                        lift = (auc_m - auc_b) if auc_m is not None and auc_b is not None else None
                        site_rows.append(
                            {
                                "analysis_scope": "sites_all_novel",
                                "entity": entity,
                                "cell_id": cell_id,
                                "data_type": data_type,
                                "test_annotation": test_annotation,
                                "test_id": test_id,
                                "assembler_id": assembler_id,
                                "model": model,
                                "query_size": metrics.get("n_novel_sites"),
                                "n_novel_pos": metrics.get("n_novel_pos"),
                                "ref_size": "",
                                "auc_model": auc_m,
                                "auc_baseline": auc_b,
                                "auc_lift": lift,
                                "site_tolerance_bp": site_tol,
                                "score_threshold": score_threshold,
                            }
                        )

            n_done += 1

    reports = outdir
    reports.mkdir(parents=True, exist_ok=True)
    tx_df = pd.DataFrame(tx_rows)
    site_df = pd.DataFrame(site_rows)
    tx_df.to_csv(reports / "novel_transcript_pr_by_test.tsv", sep="\t", index=False)
    site_df.to_csv(reports / "novel_stage1_by_test.tsv", sep="\t", index=False)

    summary_parts: list[pd.DataFrame] = []
    if not tx_df.empty:
        summary_parts.append(tx_df)
    if not site_df.empty:
        summary_parts.append(site_df)
    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True, sort=False)
        summary.to_csv(reports / "novel_eval_summary.tsv", sep="\t", index=False)

    readme = reports / "novel_eval_README.txt"
    readme.write_text(
        "\n".join(
            [
                "RefSeq-novel evaluation (analysis-only on cross_annotation_repro).",
                "",
                "Transcript (analysis_scope=transcript_filtered_assembly):",
                "  Assembly GTF filtered to multi-exon transcripts not identical to RefSeq.",
                "  gtfcuff AUC from run_transcript_pr_benchmark vs full test annotation ref.",
                "  Compare auc_model vs auc_baseline (coverage); report auc_lift.",
                "",
                "TSS/TES (analysis_scope=sites_all_novel):",
                "  Eval universe = all sites endpoint-novel vs RefSeq (±site_tolerance_bp).",
                "  label = match to full test annotation; AUPR uses sklearn AP on that subset.",
                "",
                f"site_tolerance_bp default: {site_tol}",
                f"refseq index cache: {index_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[telos_repro] refseq-novel-eval wrote: {reports / 'novel_eval_summary.tsv'}")
    return 0 if (tx_rows or site_rows) else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate RefSeq-novel metrics on cross_annotation_repro.")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("runs/cross_annotation_repro"),
        help="cross_annotation_repro root",
    )
    p.add_argument(
        "--refseq-gtf",
        type=Path,
        default=Path(
            "genome/refseq/GCF_000001405.40_GRCh38.p14_genomic.gffcmp.gtf"
        ),
        help="RefSeq reference GTF for novelty indexing",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("runs/refseq_novel_eval/cache"),
        help="Cache for RefSeq transcript index",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("runs/refseq_novel_eval/reports"),
        help="Report output directory",
    )
    p.add_argument(
        "--site-tolerance-bp",
        type=int,
        default=None,
        help="Endpoint novelty tolerance vs RefSeq (default: stage1 YAML)",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Reserved for optional threshold recall on novel positives",
    )
    p.add_argument(
        "--only-cell",
        type=str,
        default=None,
        help="Run one cell only, e.g. sr__train_refseq__test_gencode",
    )
    p.add_argument(
        "--gffcompare-bin",
        type=str,
        default=None,
        help="Optional gffcompare executable",
    )
    p.add_argument(
        "--max-tests",
        type=int,
        default=None,
        help="Process at most N tests per cell (debug/smoke)",
    )
    p.add_argument(
        "--skip-transcript-pr",
        action="store_true",
        help="Skip gtfcuff transcript PR (site metrics only; faster)",
    )
    args = p.parse_args()
    return run_evaluation(
        root=args.root,
        refseq_gtf=args.refseq_gtf,
        cache_dir=args.cache_dir,
        outdir=args.outdir,
        site_tolerance_bp=args.site_tolerance_bp,
        only_cell=args.only_cell,
        gffcompare_bin=args.gffcompare_bin,
        score_threshold=float(args.score_threshold),
        max_tests=args.max_tests,
        skip_transcript_pr=bool(args.skip_transcript_pr),
    )


if __name__ == "__main__":
    raise SystemExit(main())
