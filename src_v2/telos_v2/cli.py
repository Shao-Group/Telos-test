from __future__ import annotations

import argparse
from pathlib import Path

from telos_v2.commands.benchmark import run_benchmark
from telos_v2.commands.predict import run_predict
from telos_v2.commands.train import run_train
from telos_v2.config_models import BenchmarkIO, PredictIO, TrainIO


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
    train.add_argument("--tmap", type=Path, default=None)
    train.add_argument("--outdir", type=Path, required=True)

    predict = sub.add_parser("predict", help="Run inference with pretrained models")
    predict.add_argument("--bam", type=Path, required=True)
    predict.add_argument("--gtf", type=Path, required=True)
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--outdir", type=Path, required=True)

    bench = sub.add_parser("benchmark", help="Run benchmark sweeps")
    bench.add_argument("--config", type=Path, required=True)
    bench.add_argument("--outdir", type=Path, required=True)

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
            )
        )
    if args.command == "predict":
        return run_predict(
            PredictIO(
                bam=args.bam,
                gtf=args.gtf,
                model_dir=args.model_dir,
                outdir=args.outdir,
            )
        )
    if args.command == "benchmark":
        return run_benchmark(BenchmarkIO(config=args.config, outdir=args.outdir))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
