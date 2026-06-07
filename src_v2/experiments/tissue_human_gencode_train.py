"""
Benchmark tissue bundles with **per–data-type** human GENCODE models (train skipped).

For each modality (sr, cdna, drna, pacbio), load models trained on the canonical human
GENCODE bundle for that data type, then predict on the matching tissue bundle under
``data/tissue/<data_type>/``. Same genome and annotation as training (GRCh38 / GENCODE v49).

Models are expected at::

    <human_root>/<data_type>__train_gencode/models/

Typically ``human_root`` is the Phase A shared-train directory, e.g.::

    runs/novel_phase_a_cross_annotation/_phase_a_shared_train

Tissue bundle layout (from ``scripts/pipeline-tissue.sh`` / ``rnaseq_pipeline``)::

    data/tissue/<data_type>/
      align/aln.sorted.bam
      stringtie.gtf, stringtie.stringtie.gtf.tmap
      isoquant.gtf, isoquant.isoquant.gtf.tmap   (long-read types)
      scallop2.gtf, scallop2.scallop2.gtf.tmap   (sr)

Usage::

    PYTHONPATH=src_v2 python src_v2/experiments/tissue_human_gencode_train.py \\
      --human-root runs/novel_phase_a_cross_annotation/_phase_a_shared_train

Optional::

    PYTHONPATH=src_v2 python src_v2/experiments/tissue_human_gencode_train.py \\
      --human-root runs/novel_phase_a_cross_annotation/_phase_a_shared_train \\
      --outdir runs/tissue_human_gencode \\
      --tissue-bundle-root data/tissue \\
      --ref-gtf genome/gencode/gencode.v49.primary_assembly.basic.annotation.gtf
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from telos_v2.benchmark.matrix import benchmark_mapping_to_yaml_text
from telos_v2.benchmark.orchestrator import run_benchmark
from telos_v2.config_loader import default_stage1_config_path
from telos_v2.config_models import BenchmarkIO
from telos_v2.config_validation import validate_benchmark_config
from telos_v2.models import (
    STAGE1_BACKENDS,
    stage1_bundle_path,
    stage2_feature_names_json_for_backend,
    stage2_model_joblib_for_backend,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def shared_models_ready(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    for site_type in ("TSS", "TES"):
        for backend in STAGE1_BACKENDS:
            if not (model_dir / stage1_bundle_path(site_type, backend)).is_file():
                return False
    for backend in STAGE1_BACKENDS:
        if not (model_dir / stage2_model_joblib_for_backend(backend)).is_file():
            return False
        if not (model_dir / stage2_feature_names_json_for_backend(backend)).is_file():
            return False
    return True


def _default_analysis_block() -> dict[str, Any]:
    return {
        "enabled": True,
        "benchmark_mode": "minimal",
        "debug": {"keep_pr_work": False},
        "pr_vs_baseline": {
            "enabled": True,
            "measure": "cov",
            "plot": True,
            "filter_validation_chroms": True,
            "chromosomes_file": None,
            "save_pr_tables": True,
            "gffcompare_bin": "/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare",
        },
    }


def _assemblers_for_data_type(data_type: str) -> tuple[str, ...]:
    dt = data_type.strip().lower()
    if dt == "sr":
        return ("stringtie", "scallop2")
    if dt in ("cdna", "drna", "pacbio"):
        return ("stringtie", "isoquant")
    raise ValueError(f"Unknown data_type {data_type!r}; expected sr|cdna|drna|pacbio")


def build_tissue_tests(
    *,
    tissue_dt_dir: Path,
    data_type: str,
    ref_gtf: Path,
    stage1_config: Path,
) -> list[dict[str, Any]]:
    tissue_dt_dir = tissue_dt_dir.resolve()
    bam = tissue_dt_dir / "align" / "aln.sorted.bam"
    if not bam.is_file():
        raise FileNotFoundError(f"Missing tissue BAM (run rnaseq_pipeline first): {bam}")

    tests: list[dict[str, Any]] = []
    for aid in _assemblers_for_data_type(data_type):
        gtf = tissue_dt_dir / f"{aid}.gtf"
        tmap = tissue_dt_dir / f"{aid}.{aid}.gtf.tmap"
        if not gtf.is_file():
            continue
        if not tmap.is_file():
            raise FileNotFoundError(f"Missing gffcompare tmap for {aid}: {tmap}")
        tid = f"tissue_{data_type}__{aid}"
        tests.append(
            {
                "id": tid,
                "assembler_id": aid,
                "bam": str(bam),
                "gtf": str(gtf),
                "ref_gtf": str(ref_gtf.resolve()),
                "tmap": str(tmap),
                "config": str(stage1_config.resolve()),
            }
        )
    if not tests:
        raise ValueError(
            f"No assembler GTF+tmap pairs found under {tissue_dt_dir} "
            f"(expected at least stringtie for {data_type})."
        )
    return tests


def build_benchmark_mapping(
    *,
    data_type: str,
    human_root: Path,
    tissue_dt_dir: Path,
    ref_gtf: Path,
    stage1_config: Path,
) -> dict[str, Any]:
    shared_parent = human_root / f"{data_type}__train_gencode"
    model_dir = shared_parent / "models"
    if not shared_models_ready(model_dir):
        raise FileNotFoundError(
            f"Human GENCODE models not ready under {model_dir}. "
            "Train or copy Phase A shared models for this data_type first."
        )

    tests = build_tissue_tests(
        tissue_dt_dir=tissue_dt_dir,
        data_type=data_type,
        ref_gtf=ref_gtf,
        stage1_config=stage1_config,
    )

    train_block: dict[str, Any] = {
        "mode": "skip",
        "model_dir": str(model_dir.resolve()),
        "outdir": str(shared_parent.resolve()),
        "config": str(stage1_config.resolve()),
    }

    return {
        "generated_by": "experiments.tissue_human_gencode_train",
        "schema_version": 3,
        "train": train_block,
        "tests": tests,
        "execution": {"stop_on_error": False},
        "analysis": _default_analysis_block(),
    }


def run_all(
    *,
    human_root: Path,
    outdir: Path,
    tissue_bundle_root: Path,
    ref_gtf: Path,
    stage1_config: Path,
    data_types: tuple[str, ...],
) -> int:
    human_root = human_root.resolve()
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    reports = outdir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    all_ok = True

    for dt in data_types:
        run_id = f"{dt}__train_gencode__test_tissue"
        combo_outdir = outdir / run_id
        tissue_dt = tissue_bundle_root / dt
        print(f"[telos_v2] tissue benchmark: {run_id}")
        try:
            mapping = build_benchmark_mapping(
                data_type=dt,
                human_root=human_root,
                tissue_dt_dir=tissue_dt,
                ref_gtf=ref_gtf,
                stage1_config=stage1_config,
            )
            validate_benchmark_config(mapping)
        except (FileNotFoundError, ValueError, OSError) as exc:
            all_ok = False
            rows.append(
                {
                    "run_id": run_id,
                    "data_type": dt,
                    "exit_code": "2",
                    "status": "setup_failed",
                    "error": str(exc),
                    "outdir": str(combo_outdir),
                    "summary_csv": str(combo_outdir / "reports" / "benchmark_summary.csv"),
                    "benchmark_yaml": "",
                }
            )
            print(f"[telos_v2] setup failed: {exc}")
            continue

        combo_reports = combo_outdir / "reports"
        combo_reports.mkdir(parents=True, exist_ok=True)
        cfg_path = combo_reports / "generated_benchmark_tissue_gencode.yaml"
        header = (
            "# Tissue evaluation: train.mode=skip; per-datatype GENCODE models from human_root.\n"
            f"# human_root={human_root}\n"
            f"# tissue_bundle_root={tissue_bundle_root.resolve()}\n"
            f"# ref_gtf={ref_gtf.resolve()}\n\n"
        )
        cfg_path.write_text(header + benchmark_mapping_to_yaml_text(mapping), encoding="utf-8")
        code = run_benchmark(BenchmarkIO(config=cfg_path, outdir=combo_outdir))
        ok = code == 0
        all_ok = all_ok and ok
        rows.append(
            {
                "run_id": run_id,
                "data_type": dt,
                "exit_code": str(code),
                "status": "ok" if ok else "failed",
                "error": "",
                "outdir": str(combo_outdir),
                "summary_csv": str(combo_outdir / "reports" / "benchmark_summary.csv"),
                "benchmark_yaml": str(cfg_path),
            }
        )

    index_csv = reports / "tissue_human_gencode_runs.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "run_id",
                "data_type",
                "exit_code",
                "status",
                "error",
                "outdir",
                "summary_csv",
                "benchmark_yaml",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[telos_v2] run index: {index_csv}")
    return 0 if all_ok else 1


def main() -> int:
    repo = _repo_root()
    p = argparse.ArgumentParser(
        description="Benchmark: per-datatype human GENCODE models (skip train) → tissue data/tissue bundles."
    )
    p.add_argument(
        "--human-root",
        type=Path,
        required=True,
        help="Directory containing <data_type>__train_gencode/ (e.g. .../_phase_a_shared_train).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("runs/tissue_human_gencode"),
        help="Parent directory for per–data_type benchmark outputs.",
    )
    p.add_argument(
        "--tissue-bundle-root",
        type=Path,
        default=repo / "data" / "tissue",
        help="Root with subdirs sr, cdna, drna, pacbio (default: <repo>/data/tissue).",
    )
    p.add_argument(
        "--ref-gtf",
        type=Path,
        default=repo / "genome" / "gencode" / "gencode.v49.primary_assembly.basic.annotation.gtf",
        help="Reference GTF for gffcompare tmaps (must match pipeline-tissue.sh).",
    )
    p.add_argument(
        "--stage1-config",
        type=Path,
        default=None,
        help="Stage I YAML (default: telos_v2 default).",
    )
    p.add_argument(
        "--data-types",
        nargs="+",
        default=["sr", "cdna", "drna", "pacbio"],
        help="Subset of data types to run (default: all four).",
    )
    args = p.parse_args()

    stage1 = args.stage1_config if args.stage1_config is not None else default_stage1_config_path()
    if not stage1.is_file():
        print(f"ERROR: stage1 config not found: {stage1}", file=sys.stderr)
        return 2
    if not args.ref_gtf.is_file():
        print(f"ERROR: ref GTF not found: {args.ref_gtf}", file=sys.stderr)
        return 2
    if not args.tissue_bundle_root.is_dir():
        print(f"ERROR: tissue bundle root not found: {args.tissue_bundle_root}", file=sys.stderr)
        return 2

    dtypes = tuple(str(x).strip().lower() for x in args.data_types)
    return run_all(
        human_root=args.human_root,
        outdir=args.outdir,
        tissue_bundle_root=args.tissue_bundle_root.resolve(),
        ref_gtf=args.ref_gtf.resolve(),
        stage1_config=stage1,
        data_types=dtypes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
