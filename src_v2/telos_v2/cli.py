"""
Command-line entry for Telos v2.

Defines ``argparse`` subcommands and maps parsed arguments to frozen IO dataclasses
(:mod:`telos_v2.config_models`) and the thin handlers in :mod:`telos_v2.commands`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from telos_v2.commands.benchmark import run_benchmark
from telos_v2.commands.benchmark_matrix import run_benchmark_matrix
from telos_v2.commands.predict import run_predict
from telos_v2.commands.train import run_train
from telos_v2.config_loader import default_stage1_config_path
from telos_v2.config_models import BenchmarkIO, PredictIO, TrainIO


def _default_outdir(command: str) -> Path:
    """
    Build a timestamped output directory under the current working directory.

    Pattern: ``./telos_v2_<command>_<YYYYMMDD_HHMMSS>/`` (resolved absolute). Used when ``train``,
    ``predict``, or ``benchmark`` omits ``--outdir``.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path.cwd() / f"telos_v2_{command}_{stamp}").resolve()


def build_parser() -> argparse.ArgumentParser:
    """
    Construct the root parser with four required subcommands.

    Subcommands:

    - ``train`` — paths for BAM, assembly GTF, reference GTF, gffcompare tmap; optional config and Stage I parallelism flags.
    - ``predict`` — BAM, GTF, ``--model-dir``; optional config and Stage I parallelism flags.
    - ``benchmark`` — benchmark YAML path; optional outdir.
    - ``benchmark-matrix`` — data type and train/test annotation choices; required benchmark ``--outdir``; optional bundle root and stage1 config.

    Returns:
        Configured :class:`argparse.ArgumentParser` (caller must invoke ``parse_args``).
    """
    p = argparse.ArgumentParser(
        prog="telos_v2",
        description="Telos v2: Stage I/II train, predict, and benchmark.",
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
    train.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: ./telos_v2_train_<timestamp>).",
    )
    train.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML/JSON config (default: bundled configs/stage1.defaults.yaml).",
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
    predict.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: ./telos_v2_predict_<timestamp>).",
    )
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
    bench.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: ./telos_v2_benchmark_<timestamp>).",
    )

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

    return p


def main(argv: list[str] | None = None) -> int:
    """
    Parse CLI arguments and dispatch to the appropriate command handler.

    Exit codes are delegated to ``run_train``, ``run_predict``, ``run_benchmark``, or
    ``run_benchmark_matrix`` (typically ``0`` success, ``1`` partial benchmark failure, ``2`` config/preflight).

    Args:
        argv: Argument vector; ``None`` means use ``sys.argv[1:]`` inside :func:`argparse.ArgumentParser.parse_args`.

    Returns:
        Integer exit code suitable for :func:`sys.exit`.
    """
    args = build_parser().parse_args(argv)
    if args.command == "train":
        outdir = args.outdir if args.outdir is not None else _default_outdir("train")
        train_cfg = args.config if args.config is not None else default_stage1_config_path()
        return run_train(
            TrainIO(
                bam=args.bam,
                gtf=args.gtf,
                ref_gtf=args.ref_gtf,
                tmap=args.tmap,
                outdir=outdir,
                config_file=train_cfg,
                stage1_no_parallel=args.stage1_no_parallel,
                stage1_n_workers=args.stage1_workers,
            )
        )
    if args.command == "predict":
        outdir = args.outdir if args.outdir is not None else _default_outdir("predict")
        return run_predict(
            PredictIO(
                bam=args.bam,
                gtf=args.gtf,
                model_dir=args.model_dir,
                outdir=outdir,
                config_file=args.config,
                stage1_no_parallel=args.stage1_no_parallel,
                stage1_n_workers=args.stage1_workers,
            )
        )
    if args.command == "benchmark":
        outdir = args.outdir if args.outdir is not None else _default_outdir("benchmark")
        return run_benchmark(BenchmarkIO(config=args.config, outdir=outdir))
    if args.command == "benchmark-matrix":
        return run_benchmark_matrix(
            data_type=args.data_type,
            train_annotation=args.train_annotation,
            test_annotation=args.test_annotation,
            outdir=args.outdir,
            bundles_root=args.bundles_root,
            stage1_config=args.stage1_config,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
