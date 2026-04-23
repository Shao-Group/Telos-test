"""
Reproducibility runner for cross-annotation benchmark experiments.

Usage:
  PYTHONPATH=src_v2 python src_v2/experiments/cross_annotation_repro.py
"""

from __future__ import annotations

from pathlib import Path

from telos_v2.benchmark.cross_annotation import run_cross_annotation_benchmarks

# ---- Experiment knobs (edit here, keep CLI surface minimal) ----
OUTDIR = Path("runs/cross_annotation_repro")
BUNDLES_ROOT: Path | None = None  # e.g. Path("/abs/path/to/data/bundles")
STAGE1_CONFIG: Path | None = None  # e.g. Path("src_v2/configs/stage1.defaults.yaml")

DATA_TYPES = ("sr", "cdna", "drna", "pacbio")
ANNOTATIONS = ("refseq", "gencode", "ensembl")
INCLUDE_SAME_ANNOTATION = False

# Cache is read from TELOS_STAGE1_CACHE_DIR (or stage1.feature_extraction.cache_dir in YAML).
# Example:
#   export TELOS_STAGE1_CACHE_DIR=/datadisk1/ixk5174/telos_stage1_cache


def main() -> int:
    return run_cross_annotation_benchmarks(
        outdir=OUTDIR,
        bundles_root=BUNDLES_ROOT,
        stage1_config=STAGE1_CONFIG,
        data_types=DATA_TYPES,
        annotations=ANNOTATIONS,
        include_same_annotation=INCLUDE_SAME_ANNOTATION,
    )


if __name__ == "__main__":
    raise SystemExit(main())

