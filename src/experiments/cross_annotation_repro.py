"""
Reproducibility runner for cross-annotation benchmark experiments.

Mirrors ``novel_phase_a_cross_annotation.py`` structure: shared train per
``(data_type, train_annotation)``, optional parallel tests, and a run index CSV.
Uses standard bundle ``ref_gtf`` / ``tmap`` (no augmented novel references).

Usage:
  PYTHONPATH=src python src/experiments/cross_annotation_repro.py

  PYTHONPATH=src python src/experiments/cross_annotation_repro.py \\
    --outdir runs/cross_annotation_repro \\
    --max-parallel-cells 18 --max-parallel-trains 12 --max-parallel-tests 4 \\
    --total-cpus 80 --no-pr-tables

Speed tips:
  - Set ``TELOS_STAGE1_CACHE_DIR`` so Stage I features are reused across predict runs.
  - ``--max-parallel-cells``: run many train/test combos at once (e.g. gencode→refseq ∥ ensembl→gencode).
  - ``--max-parallel-trains``: train unique (data_type, train_annotation) axes in parallel first.
  - ``--total-cpus``: caps ``cells × tests`` so you do not oversubscribe (e.g. 80).
  - Use ``--no-pr-tables`` if you only need summary metrics.

Rerun only same-annotation cells (after a cross-only grid), reusing shared train + cache:

  PYTHONPATH=src python src/experiments/cross_annotation_repro.py \\
    --outdir runs/cross_annotation_repro \\
    --only-same-annotation \\
    --max-parallel-cells 12 --max-parallel-trains 12 --max-parallel-tests 4 --total-cpus 80

Run only specific train/test annotation pairs (all data types):

  PYTHONPATH=src python src/experiments/cross_annotation_repro.py \\
    --outdir runs/cross_annotation_repro \\
    --annotation-pairs gencode-refseq gencode-gencode
"""

from __future__ import annotations

import argparse
from pathlib import Path

from telos_repro.benchmark.cross_annotation import run_cross_annotation_benchmarks


def _parse_annotation_pairs(raw: list[str] | None) -> tuple[tuple[str, str], ...] | None:
    if not raw:
        return None
    pairs: list[tuple[str, str]] = []
    for item in raw:
        parts = item.split("-", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise SystemExit(
                f"invalid --annotation-pairs entry {item!r} (expected train-test, e.g. gencode-refseq)"
            )
        pairs.append((parts[0], parts[1]))
    return tuple(pairs)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Cross-annotation benchmark grid with shared training directories."
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("runs/cross_annotation_repro"),
        help="Root output directory for all matrix cells",
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
        help="Stage I YAML (default: telos_repro default)",
    )
    p.add_argument(
        "--annotation-pairs",
        nargs="+",
        metavar="TRAIN-TEST",
        help="Run only these train/test pairs across all data types (e.g. gencode-refseq gencode-gencode)",
    )
    p.add_argument(
        "--include-same-annotation",
        action="store_true",
        help="Include train_annotation == test_annotation cells (full grid + diagonal)",
    )
    p.add_argument(
        "--only-same-annotation",
        action="store_true",
        help="Run only train_annotation == test_annotation cells (12 combos); merges run index",
    )
    p.add_argument(
        "--max-parallel-cells",
        type=int,
        default=1,
        help="Run up to N matrix cells in parallel (different train/test annotation combos)",
    )
    p.add_argument(
        "--max-parallel-trains",
        type=int,
        default=1,
        help="Train up to N unique (data_type, train_annotation) axes in parallel before cells",
    )
    p.add_argument(
        "--max-parallel-tests",
        type=int,
        default=1,
        help="Run up to N test rows in parallel within each benchmark cell",
    )
    p.add_argument(
        "--total-cpus",
        type=int,
        default=None,
        help="Cap cells×tests parallelism (default: os.cpu_count()); e.g. 80 on your machine",
    )
    p.add_argument(
        "--no-pr-tables",
        action="store_true",
        help="Disable writing PR curve TSV tables (faster; AUPR still in benchmark_summary.csv)",
    )
    args = p.parse_args()
    if args.annotation_pairs and args.only_same_annotation:
        raise SystemExit("--annotation-pairs and --only-same-annotation are mutually exclusive")
    return run_cross_annotation_benchmarks(
        outdir=args.outdir,
        bundles_root=args.bundles_root,
        stage1_config=args.stage1_config,
        data_types=("sr", "cdna", "drna", "pacbio"),
        annotations=("refseq", "gencode", "ensembl"),
        annotation_pairs=_parse_annotation_pairs(args.annotation_pairs),
        include_same_annotation=args.include_same_annotation,
        only_same_annotation=args.only_same_annotation,
        max_parallel_cells=int(args.max_parallel_cells),
        max_parallel_trains=int(args.max_parallel_trains),
        max_parallel_tests=int(args.max_parallel_tests),
        total_cpus=args.total_cpus,
        save_pr_tables=not args.no_pr_tables,
    )


if __name__ == "__main__":
    raise SystemExit(main())

