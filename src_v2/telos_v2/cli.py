from __future__ import annotations

import argparse
from pathlib import Path

from telos_v2.commands.benchmark import run_benchmark
from telos_v2.commands.benchmark_matrix import run_benchmark_matrix
from telos_v2.commands.filter_gtf import run_filter_gtf
from telos_v2.commands.predict import run_predict
from telos_v2.commands.train import run_train
from telos_v2.config_models import BenchmarkIO, FilterGtfIO, PredictIO, TrainIO


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="telos_v2",
        description="Telos v2 scaffold CLI (isolated from legacy src).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train Stage I/II models")
    train.add_argument("--bam", type=Path, required=True)
    train.add_argument("--gtf", type=Path, required=True)
    train.add_argument("--ref-gtf", type=Path, required=True)
    train.add_argument(
        "--tmap",
        type=Path,
        required=True,
        help="gffcompare .tmap (qry_id / class_code) for Stage II transcript labels.",
    )
    train.add_argument("--outdir", type=Path, required=True)
    train.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON config for feature/model parameters.",
    )
    train.add_argument(
        "--stage1-no-parallel",
        action="store_true",
        help="Disable multiprocessing for Stage I BAM feature extraction.",
    )
    train.add_argument(
        "--stage1-workers",
        type=int,
        default=None,
        metavar="N",
        help="Stage I feature pool size (default: config or min(CPU, 8)). Implies parallel when >1.",
    )

    predict = sub.add_parser("predict", help="Run inference with pretrained models")
    predict.add_argument("--bam", type=Path, required=True)
    predict.add_argument("--gtf", type=Path, required=True)
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--outdir", type=Path, required=True)
    predict.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON config for feature/model parameters.",
    )
    predict.add_argument(
        "--stage1-no-parallel",
        action="store_true",
        help="Disable multiprocessing for Stage I BAM feature extraction.",
    )
    predict.add_argument(
        "--stage1-workers",
        type=int,
        default=None,
        metavar="N",
        help="Stage I feature pool size (default: config or min(CPU, 8)). Implies parallel when >1.",
    )

    bench = sub.add_parser("benchmark", help="Run benchmark sweeps")
    bench.add_argument("--config", type=Path, required=True)
    bench.add_argument("--outdir", type=Path, required=True)

    bmat = sub.add_parser(
        "benchmark-matrix",
        help=(
            "Generate benchmark YAML from data_type + train/test annotation, then run benchmark "
            "(see docs/benchmark-matrix-convention.md)."
        ),
    )
    bmat.add_argument(
        "--data-type",
        choices=("sr", "cdna", "drna", "pacbio"),
        required=True,
        help="Sequencing modality (fixed train sample per type).",
    )
    bmat.add_argument(
        "--train-annotation",
        choices=("refseq", "gencode", "ensembl"),
        required=True,
        dest="train_annotation",
        help="Reference bundle for training (BAM/GTF/tmap/ref GTF).",
    )
    bmat.add_argument(
        "--test-annotation",
        choices=("refseq", "gencode", "ensembl"),
        required=True,
        dest="test_annotation",
        help="Reference bundle for test samples (same modality, other samples).",
    )
    bmat.add_argument("--outdir", type=Path, required=True, help="Benchmark root (train/, tests/, reports/).")
    bmat.add_argument(
        "--bundles-root",
        type=Path,
        default=None,
        help="Directory containing <ref_id>/<modality>/<sample>/ (default: TELOS_BUNDLES_ROOT or ./data/bundles).",
    )
    bmat.add_argument(
        "--stage1-config",
        type=Path,
        default=None,
        help="Telos stage1 YAML (default: src_v2/configs/stage1.defaults.yaml next to package).",
    )
   
    fg = sub.add_parser(
        "filter-gtf",
        help="Filter GTF with gtfformat remove-fp / remove-fp-threshold (site predictions TSV).",
    )
    fg.add_argument("--gtf", type=Path, required=True)
    fg.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="TSV: site_type, chrom, pos, ... pred in column index 4 (gtfformat convention).",
    )
    fg.add_argument("--out-gtf", type=Path, required=True)
    fg.add_argument(
        "--mode",
        choices=("exact", "threshold"),
        default="threshold",
        help="exact=remove-fp; threshold=remove-fp-threshold.",
    )
    fg.add_argument("--hard-mode", type=int, default=0, help="Passed to gtfformat (0 or 1).")
    fg.add_argument(
        "--bp-threshold",
        type=int,
        default=0,
        help="For --mode threshold: match TSS/TES within this many bp.",
    )
    fg.add_argument(
        "--gtfformat-bin",
        type=Path,
        default=None,
        help="Path to gtfformat binary (overrides config and env).",
    )
    fg.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON for rnaseqtools.gtfformat_bin.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return run_train(
            TrainIO(
                bam=args.bam,
                gtf=args.gtf,
                ref_gtf=args.ref_gtf,
                tmap=args.tmap,
                outdir=args.outdir,
                config_file=args.config,
                stage1_no_parallel=args.stage1_no_parallel,
                stage1_n_workers=args.stage1_workers,
            )
        )
    if args.command == "predict":
        return run_predict(
            PredictIO(
                bam=args.bam,
                gtf=args.gtf,
                model_dir=args.model_dir,
                outdir=args.outdir,
                config_file=args.config,
                stage1_no_parallel=args.stage1_no_parallel,
                stage1_n_workers=args.stage1_workers,
            )
        )
    if args.command == "benchmark":
        return run_benchmark(BenchmarkIO(config=args.config, outdir=args.outdir))
    if args.command == "benchmark-matrix":
        return run_benchmark_matrix(
            data_type=args.data_type,
            train_annotation=args.train_annotation,
            test_annotation=args.test_annotation,
            outdir=args.outdir,
            bundles_root=args.bundles_root,
            stage1_config=args.stage1_config
        )
    if args.command == "filter-gtf":
        return run_filter_gtf(
            FilterGtfIO(
                gtf=args.gtf,
                predictions=args.predictions,
                out_gtf=args.out_gtf,
                hard_mode=args.hard_mode,
                mode=args.mode,
                bp_threshold=args.bp_threshold,
                gtfformat_bin=args.gtfformat_bin,
                config_file=args.config,
            )
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
