"""
Driver: build strict novel-augmented reference GTFs for all bundle manifests.

Walks bundles root:
  <bundles_root>/<ref_id>/<modality>/<sample>/bundle_manifest.yaml

For each manifest, uses manifest["ref_annotation_gtf"] as base annotation, runs strict
augmentation, and writes outputs under:
  <out-root>/<ref_id>/<modality>/<sample>/
    augmented_ref.gtf
    reports/novel_kept.tsv
    reports/novel_dropped.tsv

Also writes an index CSV:
  <out-root>/reports/augmented_refs_index.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from telos_v2.benchmark.matrix import resolve_bundles_root
from telos_v2.config_loader import default_stage1_config_path

from augment_annotation_with_novel import build_augmented_reference


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read bundle manifests.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest (not a mapping): {path}")
    return data


def _iter_bundle_manifests(bundles_root: Path) -> list[tuple[str, str, str, Path]]:
    rows: list[tuple[str, str, str, Path]] = []
    if not bundles_root.is_dir():
        raise FileNotFoundError(f"Bundles root not found: {bundles_root}")
    for ref_dir in sorted(p for p in bundles_root.iterdir() if p.is_dir()):
        for modality_dir in sorted(p for p in ref_dir.iterdir() if p.is_dir()):
            for sample_dir in sorted(p for p in modality_dir.iterdir() if p.is_dir()):
                mf = sample_dir / "bundle_manifest.yaml"
                if mf.is_file():
                    rows.append((ref_dir.name, modality_dir.name, sample_dir.name, mf))
    return rows


def run(args: argparse.Namespace) -> int:
    bundles_root = resolve_bundles_root(args.bundles_root).resolve()
    stage1_config = (
        args.stage1_config.resolve() if args.stage1_config is not None else default_stage1_config_path().resolve()
    )
    if not stage1_config.is_file():
        raise FileNotFoundError(f"Stage1 config not found: {stage1_config}")

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifests = _iter_bundle_manifests(bundles_root)
    if not manifests:
        raise ValueError(f"No bundle_manifest.yaml files found under {bundles_root}")

    index_rows: list[dict[str, str]] = []
    for i, (ref_id, modality, sample, mf) in enumerate(manifests, start=1):
        rel = f"{ref_id}/{modality}/{sample}"
        out_dir = out_root / ref_id / modality / sample
        out_gtf = out_dir / "augmented_ref.gtf"
        reports_dir = out_dir / "reports"
        print(f"[telos_v2] augment-all {i}/{len(manifests)}: {rel}")
        try:
            manifest = _load_manifest(mf)
            ann_raw = manifest.get("ref_annotation_gtf")
            if not ann_raw:
                raise ValueError("manifest missing ref_annotation_gtf")
            ann_gtf = Path(str(ann_raw)).resolve()
            if not ann_gtf.is_file():
                raise FileNotFoundError(f"ref_annotation_gtf not found: {ann_gtf}")
            summary = build_augmented_reference(
                bundle_manifest=mf.resolve(),
                annotation_gtf=ann_gtf,
                stage1_config=stage1_config,
                out_gtf=out_gtf,
                reports_dir=reports_dir,
                min_support=args.min_support,
                assemblers_csv=args.assemblers,
            )
            index_rows.append(
                {
                    "ref_id": ref_id,
                    "modality": modality,
                    "sample": sample,
                    "bundle_manifest": str(mf.resolve()),
                    "annotation_gtf": str(ann_gtf),
                    "status": "ok",
                    "error": "",
                    "novel_kept": str(summary["novel_kept"]),
                    "novel_dropped": str(summary["novel_dropped"]),
                    "out_gtf": str(out_gtf.resolve()),
                    "novel_kept_tsv": summary["novel_kept_tsv"],
                    "novel_dropped_tsv": summary["novel_dropped_tsv"],
                }
            )
        except (OSError, ValueError, RuntimeError) as exc:
            index_rows.append(
                {
                    "ref_id": ref_id,
                    "modality": modality,
                    "sample": sample,
                    "bundle_manifest": str(mf.resolve()),
                    "annotation_gtf": "",
                    "status": "failed",
                    "error": str(exc),
                    "novel_kept": "",
                    "novel_dropped": "",
                    "out_gtf": str(out_gtf.resolve()),
                    "novel_kept_tsv": str((reports_dir / "novel_kept.tsv").resolve()),
                    "novel_dropped_tsv": str((reports_dir / "novel_dropped.tsv").resolve()),
                }
            )

    reports_root = out_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    index_csv = reports_root / "augmented_refs_index.csv"
    with index_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ref_id",
                "modality",
                "sample",
                "bundle_manifest",
                "annotation_gtf",
                "status",
                "error",
                "novel_kept",
                "novel_dropped",
                "out_gtf",
                "novel_kept_tsv",
                "novel_dropped_tsv",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    ok = sum(1 for r in index_rows if r["status"] == "ok")
    print("[telos_v2] augment-all complete")
    print(f"  bundles_root={bundles_root}")
    print(f"  stage1_config={stage1_config}")
    print(f"  min_support={args.min_support}")
    print(f"  processed={len(index_rows)}")
    print(f"  ok={ok}")
    print(f"  failed={len(index_rows) - ok}")
    print(f"  index_csv={index_csv.resolve()}")
    return 0 if ok == len(index_rows) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build augmented reference GTFs for all bundle manifests.")
    p.add_argument(
        "--bundles-root",
        type=Path,
        default=None,
        help="Root containing per-reference bundle trees (defaults to TELOS_BUNDLES_ROOT or auto-resolve)",
    )
    p.add_argument(
        "--stage1-config",
        type=Path,
        default=None,
        help="Stage1 config path; defaults to telos_v2 default stage1 config",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/novel_ref_all"),
        help="Root directory for per-bundle augmented refs and reports",
    )
    p.add_argument(
        "--min-support",
        type=int,
        default=2,
        help="Minimum distinct assembler support (strict mode; default: 2)",
    )
    p.add_argument(
        "--assemblers",
        type=str,
        default="",
        help="Optional comma-separated assembler_id allowlist",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
