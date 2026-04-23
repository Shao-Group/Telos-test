"""
Phase A cross-annotation benchmarks: train on annotation only; evaluate tests against augmented ref.

Training uses the standard matrix (original ``train.ref_gtf`` and train ``tmap``). For each test
row, ``ref_gtf`` is replaced with the augmented reference for ``(test_ref_id, modality, sample)``
(from ``augmented_refs_index.csv``), and ``tmap`` is replaced with a gffcompare ``.tmap`` produced
by comparing the assembly GTF to that same augmented reference (from
``augmented_tmaps_index.csv``, see ``generate_augmented_tmaps_all_bundles.py``). Stage I test AUPR,
Stage II AUPR-on-tmap, and transcript PR then align with novel-inclusive reference + labels.

Usage:
  PYTHONPATH=src_v2 python src_v2/experiments/novel_phase_a_cross_annotation.py

Prerequisite:
  PYTHONPATH=src_v2 python src_v2/experiments/generate_augmented_tmaps_all_bundles.py

Optional CLI overrides:
  PYTHONPATH=src_v2 python src_v2/experiments/novel_phase_a_cross_annotation.py \\
    --outdir runs/novel_phase_a_cross_annotation \\
    --augmented-index runs/novel_ref_all/reports/augmented_refs_index.csv \\
    --augmented-tmap-index runs/novel_ref_all/reports/augmented_tmaps_index.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from telos_v2.benchmark.matrix import (
    ANNOTATION_TO_REF_ID,
    DATA_TYPE_TO_MODALITY,
    benchmark_mapping_to_yaml_text,
    build_benchmark_yaml_mapping,
    resolve_bundles_root,
)
from telos_v2.benchmark.orchestrator import run_benchmark
from telos_v2.config_loader import default_stage1_config_path
from telos_v2.config_models import BenchmarkIO
from telos_v2.config_validation import validate_benchmark_config


def load_augmented_ref_lookup(index_csv: Path) -> dict[tuple[str, str, str], str]:
    """
    Map (ref_id, modality, sample) -> absolute path to augmented_ref.gtf.

    Expects rows from ``augment_annotation_with_novel_all_bundles.py`` index CSV with columns
    ``ref_id``, ``modality``, ``sample``, ``status``, ``out_gtf``.
    """
    if not index_csv.is_file():
        raise FileNotFoundError(f"Augmented index not found: {index_csv}")
    out: dict[tuple[str, str, str], str] = {}
    with index_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=",")
        for row in reader:
            if str(row.get("status", "")).strip().lower() != "ok":
                continue
            ref_id = str(row.get("ref_id", "")).strip()
            modality = str(row.get("modality", "")).strip()
            sample = str(row.get("sample", "")).strip()
            gtf = str(row.get("out_gtf", "")).strip()
            if not (ref_id and modality and sample and gtf):
                continue
            p = Path(gtf).resolve()
            if not p.is_file():
                continue
            out[(ref_id, modality, sample)] = str(p)
    if not out:
        raise ValueError(f"No ok rows with existing out_gtf in {index_csv}")
    return out


def load_augmented_tmap_lookup(index_csv: Path) -> dict[tuple[str, str, str, str], str]:
    """
    Map (ref_id, modality, sample, assembler_id) -> absolute path to ``.tmap`` vs augmented ref.

    Expects ``augmented_tmaps_index.csv`` from ``generate_augmented_tmaps_all_bundles.py``.
    """
    if not index_csv.is_file():
        raise FileNotFoundError(
            f"Augmented tmap index not found: {index_csv}\n"
            "Generate it with:\n"
            "  PYTHONPATH=src_v2 python src_v2/experiments/generate_augmented_tmaps_all_bundles.py"
        )
    out: dict[tuple[str, str, str, str], str] = {}
    with index_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("status", "")).strip().lower() != "ok":
                continue
            ref_id = str(row.get("ref_id", "")).strip()
            modality = str(row.get("modality", "")).strip()
            sample = str(row.get("sample", "")).strip()
            aid = str(row.get("assembler_id", "")).strip().lower()
            tp = str(row.get("tmap_path", "")).strip()
            if not (ref_id and modality and sample and aid and tp):
                continue
            p = Path(tp).resolve()
            if not p.is_file():
                continue
            out[(ref_id, modality, sample, aid)] = str(p)
    if not out:
        raise ValueError(f"No ok rows with existing tmap_path in {index_csv}")
    return out


def apply_phase_a_test_eval_paths(
    mapping: dict[str, Any],
    *,
    test_ref_id: str,
    modality: str,
    ref_lookup: dict[tuple[str, str, str], str],
    tmap_lookup: dict[tuple[str, str, str, str], str],
) -> None:
    """Replace each test's ``ref_gtf`` and ``tmap`` with augmented-ref evaluation artifacts."""
    tests = mapping.get("tests")
    if not isinstance(tests, list):
        raise ValueError("mapping missing tests list")
    missing_ref: list[str] = []
    missing_tmap: list[str] = []
    for t in tests:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id", "")).strip()
        if "__" in tid:
            sample, _ = tid.split("__", 1)
        else:
            sample = tid
        ref_key = (test_ref_id, modality, sample)
        aug_ref = ref_lookup.get(ref_key)
        if not aug_ref:
            missing_ref.append(f"id={tid!r} key={ref_key!r}")
            continue
        t["ref_gtf"] = aug_ref

        aid = str(t.get("assembler_id", "")).strip().lower()
        if not aid:
            missing_tmap.append(f"id={tid!r} missing assembler_id")
            continue
        tmap_key = (test_ref_id, modality, sample, aid)
        aug_tmap = tmap_lookup.get(tmap_key)
        if not aug_tmap:
            missing_tmap.append(f"id={tid!r} key={tmap_key!r}")
            continue
        t["tmap"] = aug_tmap

    if missing_ref:
        raise ValueError(
            "Missing augmented ref_gtf for some test rows (build augment index first):\n"
            + "\n".join(missing_ref[:20])
            + (f"\n... and {len(missing_ref) - 20} more" if len(missing_ref) > 20 else "")
        )
    if missing_tmap:
        raise ValueError(
            "Missing augmented tmap for some test rows (run generate_augmented_tmaps_all_bundles.py):\n"
            + "\n".join(missing_tmap[:20])
            + (f"\n... and {len(missing_tmap) - 20} more" if len(missing_tmap) > 20 else "")
        )


def run_phase_a_cross_annotation_benchmarks(
    *,
    outdir: Path,
    augmented_index: Path,
    augmented_tmap_index: Path,
    bundles_root: Path | None,
    stage1_config: Path | None,
    data_types: tuple[str, ...],
    annotations: tuple[str, ...],
    include_same_annotation: bool,
) -> int:
    root = resolve_bundles_root(bundles_root)
    stage1 = stage1_config if stage1_config is not None else default_stage1_config_path()
    if not stage1.is_file():
        print(f"[telos_v2] novel phase A: stage1 config not found: {stage1}")
        return 2

    ref_lookup = load_augmented_ref_lookup(augmented_index.resolve())
    tmap_lookup = load_augmented_tmap_lookup(augmented_tmap_index.resolve())
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    reports_dir = outdir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    all_ok = True
    total = 0
    for dt in data_types:
        modality = DATA_TYPE_TO_MODALITY[dt]
        for tr in annotations:
            for te in annotations:
                if not include_same_annotation and tr == te:
                    continue
                total += 1
                test_ref_id = ANNOTATION_TO_REF_ID[te]
                combo_id = f"{dt}__train_{tr}__test_{te}__phase_a_novel_ref"
                combo_outdir = outdir / combo_id
                print(f"[telos_v2] novel phase A {total}: {combo_id}")
                try:
                    mapping = build_benchmark_yaml_mapping(
                        data_type=dt,
                        train_annotation=tr,
                        test_annotation=te,
                        bundles_root=root,
                        stage1_config=stage1,
                        train_outdir=combo_outdir / "train",
                    )
                    apply_phase_a_test_eval_paths(
                        mapping,
                        test_ref_id=test_ref_id,
                        modality=modality,
                        ref_lookup=ref_lookup,
                        tmap_lookup=tmap_lookup,
                    )
                    validate_benchmark_config(mapping)
                except (FileNotFoundError, ValueError, OSError) as exc:
                    all_ok = False
                    rows.append(
                        {
                            "run_id": combo_id,
                            "data_type": dt,
                            "train_annotation": tr,
                            "test_annotation": te,
                            "exit_code": "2",
                            "status": "failed",
                            "error": str(exc),
                            "outdir": str(combo_outdir),
                            "summary_csv": str(combo_outdir / "reports" / "benchmark_summary.csv"),
                            "benchmark_yaml": "",
                        }
                    )
                    print(f"[telos_v2] novel phase A setup failed: {exc}")
                    continue

                combo_reports = combo_outdir / "reports"
                combo_reports.mkdir(parents=True, exist_ok=True)
                cfg_path = combo_reports / "generated_benchmark_phase_a.yaml"
                header = (
                    "# Phase A (novel eval): train uses original annotation; each test ref_gtf and tmap "
                    "use augmented reference + gffcompare tmap for that (ref_id, modality, sample, assembler).\n"
                    f"# augmented_index={augmented_index.resolve()}\n"
                    f"# augmented_tmap_index={augmented_tmap_index.resolve()}\n\n"
                )
                cfg_path.write_text(
                    header + benchmark_mapping_to_yaml_text(mapping),
                    encoding="utf-8",
                )
                code = run_benchmark(BenchmarkIO(config=cfg_path, outdir=combo_outdir))
                ok = code == 0
                all_ok = all_ok and ok
                rows.append(
                    {
                        "run_id": combo_id,
                        "data_type": dt,
                        "train_annotation": tr,
                        "test_annotation": te,
                        "exit_code": str(code),
                        "status": "ok" if ok else "failed",
                        "error": "",
                        "outdir": str(combo_outdir),
                        "summary_csv": str(combo_outdir / "reports" / "benchmark_summary.csv"),
                        "benchmark_yaml": str(cfg_path),
                    }
                )

    run_index = reports_dir / "novel_phase_a_runs.csv"
    with run_index.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "run_id",
                "data_type",
                "train_annotation",
                "test_annotation",
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
    print(f"[telos_v2] novel phase A index: {run_index}")
    return 0 if all_ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Phase A cross-annotation benchmarks with augmented test ref_gtf.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("runs/novel_phase_a_cross_annotation"),
        help="Root output directory for all matrix cells",
    )
    p.add_argument(
        "--augmented-index",
        type=Path,
        default=Path("runs/novel_ref_all/reports/augmented_refs_index.csv"),
        help="CSV from augment_annotation_with_novel_all_bundles.py",
    )
    p.add_argument(
        "--augmented-tmap-index",
        type=Path,
        default=Path("runs/novel_ref_all/reports/augmented_tmaps_index.csv"),
        help="CSV from generate_augmented_tmaps_all_bundles.py",
    )
    p.add_argument(
        "--bundles-root",
        type=Path,
        default=None,
        help="Override TELOS_BUNDLES_ROOT / auto-resolve",
    )
    p.add_argument(
        "--stage1-config",
        type=Path,
        default=None,
        help="Stage I YAML (default: telos_v2 default)",
    )
    p.add_argument(
        "--include-same-annotation",
        action="store_true",
        help="Include train_annotation == test_annotation cells",
    )
    args = p.parse_args()
    return run_phase_a_cross_annotation_benchmarks(
        outdir=args.outdir,
        augmented_index=args.augmented_index,
        augmented_tmap_index=args.augmented_tmap_index,
        bundles_root=args.bundles_root,
        stage1_config=args.stage1_config,
        data_types=("sr", "cdna", "drna", "pacbio"),
        annotations=("refseq", "gencode", "ensembl"),
        include_same_annotation=args.include_same_annotation,
    )


if __name__ == "__main__":
    raise SystemExit(main())
