#!/usr/bin/env python3
"""Reassemble IsoQuant outputs from existing bundle BAMs via rnaseq_pipeline subprocesses."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _ref_fasta_for_bundle(bundle_name: str, repo_root: Path) -> Path:
    key = bundle_name.lower()
    if "ensembl" in key:
        return repo_root / "genome" / "ensembl" / "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    if "gencode" in key:
        return repo_root / "genome" / "gencode" / "GRCh38.primary_assembly.genome.fa"
    if "refseq" in key:
        return repo_root / "genome" / "refseq" / "GCF_000001405.40_GRCh38.p14_genomic.fna"
    raise ValueError(f"Cannot infer reference FASTA from bundle name: {bundle_name}")


def _preset_for_modality(modality: str) -> str:
    mod = modality.lower()
    if mod == "pacbio":
        return "pacbio"
    if mod == "ont_drna":
        return "nanopore"
    if mod == "ont_cdna":
        return "nanopore"
    raise ValueError(f"Unsupported modality for IsoQuant reassembly: {modality}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Scan data/bundles and rerun only IsoQuant where isoquant.gtf is missing/empty "
            "using existing align/aln.sorted.bam."
        )
    )
    p.add_argument("--root", type=Path, default=Path("data/bundles"))
    p.add_argument("--python", default=sys.executable, help="Python executable to run subprocesses.")
    p.add_argument("--conda-env", default="irtesam-berth")
    p.add_argument("--isoquant-conda-env", default=None)
    p.add_argument("--isoquant-script", default=None)
    p.add_argument("--threads-isoquant", type=int, default=20)
    p.add_argument(
        "--only-empty-or-missing",
        action="store_true",
        help="Process only samples with missing/empty isoquant.gtf (default true behavior).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Process all discovered long-read samples even if isoquant.gtf exists and is non-empty.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Bundles root not found: {root}")

    candidates: list[tuple[Path, Path, str, Path]] = []
    for bam in sorted(root.glob("**/align/aln.sorted.bam")):
        sample_dir = bam.parent.parent
        if sample_dir.parent == root:
            # Root may already be modality-level; cannot infer bundle reliably.
            continue
        modality = sample_dir.parent.name
        bundle = sample_dir.parent.parent.name
        try:
            preset = _preset_for_modality(modality)
            ref_fasta = _ref_fasta_for_bundle(bundle, repo_root)
        except ValueError:
            continue
        iso_gtf = sample_dir / "isoquant.gtf"
        needs = (not iso_gtf.is_file()) or (iso_gtf.stat().st_size == 0)
        if not args.force and not needs:
            continue
        candidates.append((sample_dir, bam, preset, ref_fasta))

    if args.limit > 0:
        candidates = candidates[: args.limit]
    if not candidates:
        print("[reassemble-isoquant] no matching samples found")
        return 1

    print(f"[reassemble-isoquant] targets={len(candidates)}")
    ok = 0
    for i, (sample_dir, bam, preset, ref_fasta) in enumerate(candidates, start=1):
        cmd = [
            args.python,
            "-m",
            "rnaseq_pipeline",
            "run-isoquant-only",
            "--preset",
            preset,
            "--work-dir",
            str(sample_dir),
            "--bam",
            str(bam),
            "--ref-fasta",
            str(ref_fasta),
            "--conda-env",
            args.conda_env,
            "--threads-isoquant",
            str(args.threads_isoquant),
        ]
        if args.isoquant_conda_env:
            cmd.extend(["--isoquant-conda-env", args.isoquant_conda_env])
        if args.isoquant_script:
            cmd.extend(["--isoquant-script", args.isoquant_script])
        print(f"[{i}/{len(candidates)}] {' '.join(cmd)}")
        if args.dry_run:
            continue
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "src")
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
        if proc.returncode == 0:
            ok += 1
        else:
            print(f"[reassemble-isoquant] failed for {sample_dir} (exit={proc.returncode})")

    print(f"[reassemble-isoquant] complete ok={ok}/{len(candidates)}")
    return 0 if ok == len(candidates) or args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
