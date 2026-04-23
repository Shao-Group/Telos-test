"""
Run gffcompare for each bundle assembly GTF against that bundle's augmented reference GTF.

For every row in ``augmented_refs_index.csv`` with ``status=ok``, loads ``bundle_manifest.yaml``,
iterates ``assemblies[]`` entries that have a ``gtf`` path, runs::

  gffcompare -r <augmented_ref.gtf> -o <prefix> <assembly.gtf>

and copies the resulting ``.tmap`` to a stable path::

  <bundle_dir>/reports/gffcompare_augmented_ref/<assembler_id>.tmap

Writes ``augmented_tmaps_index.csv`` (default: next to the refs index) for use by
``novel_phase_a_cross_annotation.py`` to set each benchmark test's ``tmap``.

Usage:
  PYTHONPATH=src_v2 python src_v2/experiments/generate_augmented_tmaps_all_bundles.py

  # optional
  PYTHONPATH=src_v2 python src_v2/experiments/generate_augmented_tmaps_all_bundles.py \\
    --refs-index runs/novel_ref_all/reports/augmented_refs_index.csv \\
    --gffcompare-bin /path/to/gffcompare
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from telos_v2.evaluation.transcript_pr_pipeline import run_gffcompare


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read bundle manifests.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest (not a mapping): {path}")
    return data


def _assemblies_with_gtf(manifest: dict[str, Any], bundle_dir: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for ass in manifest.get("assemblies") or []:
        if not isinstance(ass, dict):
            continue
        aid = str(ass.get("assembler_id", "")).strip().lower()
        if not aid:
            continue
        raw = ass.get("gtf")
        if raw is None:
            continue
        gtf = Path(str(raw))
        if not gtf.is_absolute():
            gtf = (bundle_dir / gtf).resolve()
        out.append((aid, gtf))
    return out


def _read_refs_index(index_csv: Path) -> list[dict[str, str]]:
    if not index_csv.is_file():
        raise FileNotFoundError(f"Augmented refs index not found: {index_csv}")
    rows: list[dict[str, str]] = []
    with index_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: (v or "").strip() if v is not None else "" for k, v in row.items()})
    return rows


def run(
    *,
    refs_index: Path,
    tmap_index_out: Path,
    gffcompare_bin: str | None,
) -> int:
    ref_rows = _read_refs_index(refs_index.resolve())
    index_rows: list[dict[str, str]] = []
    for i, row in enumerate(ref_rows, start=1):
        if row.get("status", "").lower() != "ok":
            continue
        ref_id = row.get("ref_id", "")
        modality = row.get("modality", "")
        sample = row.get("sample", "")
        mf_s = row.get("bundle_manifest", "")
        aug_s = row.get("out_gtf", "")
        if not (ref_id and modality and sample and mf_s and aug_s):
            continue
        manifest_path = Path(mf_s).resolve()
        augmented_ref = Path(aug_s).resolve()
        bundle_dir = manifest_path.parent
        rel = f"{ref_id}/{modality}/{sample}"
        print(f"[telos_v2] augmented tmaps {i}: {rel}")
        if not manifest_path.is_file():
            index_rows.append(
                {
                    "ref_id": ref_id,
                    "modality": modality,
                    "sample": sample,
                    "assembler_id": "",
                    "assembly_gtf": "",
                    "augmented_ref_gtf": str(augmented_ref),
                    "tmap_path": "",
                    "status": "failed",
                    "error": f"manifest not found: {manifest_path}",
                }
            )
            continue
        if not augmented_ref.is_file():
            index_rows.append(
                {
                    "ref_id": ref_id,
                    "modality": modality,
                    "sample": sample,
                    "assembler_id": "",
                    "assembly_gtf": "",
                    "augmented_ref_gtf": str(augmented_ref),
                    "tmap_path": "",
                    "status": "failed",
                    "error": f"augmented ref not found: {augmented_ref}",
                }
            )
            continue
        try:
            manifest = _load_manifest(manifest_path)
            assemblies = _assemblies_with_gtf(manifest, bundle_dir)
        except (OSError, ValueError, RuntimeError) as exc:
            index_rows.append(
                {
                    "ref_id": ref_id,
                    "modality": modality,
                    "sample": sample,
                    "assembler_id": "",
                    "assembly_gtf": "",
                    "augmented_ref_gtf": str(augmented_ref),
                    "tmap_path": "",
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue
        if not assemblies:
            index_rows.append(
                {
                    "ref_id": ref_id,
                    "modality": modality,
                    "sample": sample,
                    "assembler_id": "",
                    "assembly_gtf": "",
                    "augmented_ref_gtf": str(augmented_ref),
                    "tmap_path": "",
                    "status": "failed",
                    "error": "no assemblies with gtf in manifest",
                }
            )
            continue

        out_dir = bundle_dir / "reports" / "gffcompare_augmented_ref"
        out_dir.mkdir(parents=True, exist_ok=True)
        for aid, asm_gtf in assemblies:
            if not asm_gtf.is_file():
                index_rows.append(
                    {
                        "ref_id": ref_id,
                        "modality": modality,
                        "sample": sample,
                        "assembler_id": aid,
                        "assembly_gtf": str(asm_gtf),
                        "augmented_ref_gtf": str(augmented_ref),
                        "tmap_path": "",
                        "status": "failed",
                        "error": "assembly gtf not found",
                    }
                )
                continue
            dest = out_dir / f"{aid}.tmap"
            prefix = f"aug_{aid}"
            try:
                # Some gffcompare builds emit prefix.* outputs next to query GTF rather than cwd.
                # Run in the assembly directory, then copy the resolved .tmap to a stable reports path.
                produced = run_gffcompare(
                    asm_gtf,
                    augmented_ref,
                    prefix,
                    asm_gtf.parent,
                    gffcompare_bin=gffcompare_bin,
                )
                dest.write_bytes(produced.read_bytes())
            except (OSError, RuntimeError, FileNotFoundError) as exc:
                index_rows.append(
                    {
                        "ref_id": ref_id,
                        "modality": modality,
                        "sample": sample,
                        "assembler_id": aid,
                        "assembly_gtf": str(asm_gtf),
                        "augmented_ref_gtf": str(augmented_ref),
                        "tmap_path": str(dest.resolve()),
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                continue
            index_rows.append(
                {
                    "ref_id": ref_id,
                    "modality": modality,
                    "sample": sample,
                    "assembler_id": aid,
                    "assembly_gtf": str(asm_gtf),
                    "augmented_ref_gtf": str(augmented_ref),
                    "tmap_path": str(dest.resolve()),
                    "status": "ok",
                    "error": "",
                }
            )

    tmap_index_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ref_id",
        "modality",
        "sample",
        "assembler_id",
        "assembly_gtf",
        "augmented_ref_gtf",
        "tmap_path",
        "status",
        "error",
    ]
    with tmap_index_out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(index_rows)

    n_ok = sum(1 for r in index_rows if r["status"] == "ok")
    n_fail = sum(1 for r in index_rows if r["status"] == "failed")
    print("[telos_v2] augmented tmaps index complete")
    print(f"  refs_index={refs_index.resolve()}")
    print(f"  tmap_index={tmap_index_out.resolve()}")
    print(f"  per-assembler rows: ok={n_ok} failed={n_fail}")
    return 0 if n_fail == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Generate gffcompare .tmap vs augmented ref for all bundles.")
    p.add_argument(
        "--refs-index",
        type=Path,
        default=Path("runs/novel_ref_all/reports/augmented_refs_index.csv"),
        help="Output from augment_annotation_with_novel_all_bundles.py",
    )
    p.add_argument(
        "--tmap-index-out",
        type=Path,
        default=None,
        help="Where to write augmented_tmaps_index.csv (default: same dir as refs index)",
    )
    p.add_argument(
        "--gffcompare-bin",
        type=str,
        default=None,
        help="Optional gffcompare executable (else GFFCOMPARE env or PATH)",
    )
    args = p.parse_args()
    out_path = (
        args.tmap_index_out
        if args.tmap_index_out is not None
        else (args.refs_index.resolve().parent / "augmented_tmaps_index.csv")
    )
    return run(
        refs_index=args.refs_index,
        tmap_index_out=out_path,
        gffcompare_bin=args.gffcompare_bin,
    )


if __name__ == "__main__":
    raise SystemExit(main())
