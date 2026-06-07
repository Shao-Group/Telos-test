"""
CLI for v2 experiment plots. Benchmark bar figures (PDF) go under ``plot_v2/<run_name>/``.

Usage::

  PYTHONPATH=src_v2 python -m telos_v2.plotting.plot_experiments mouse
  PYTHONPATH=src_v2 python -m telos_v2.plotting.plot_experiments tissue
  PYTHONPATH=src_v2 python -m telos_v2.plotting.plot_experiments cross-annotation
  PYTHONPATH=src_v2 python -m telos_v2.plotting.plot_experiments novel-phase-a-cross
  PYTHONPATH=src_v2 python -m telos_v2.plotting.plot_experiments refseq-novel
  PYTHONPATH=src_v2 python -m telos_v2.plotting.plot_experiments feature-importance
  PYTHONPATH=src_v2 python -m telos_v2.plotting.plot_experiments window
  PYTHONPATH=src_v2 python -m telos_v2.plotting.plot_experiments all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from telos_v2.plotting.grouped_aupr_bars import (
    DEFAULT_PLOT_V2_ROOT,
    apply_plot_style,
    load_benchmark_summaries,
    load_novel_summary,
    plot_benchmark_aupr_bars,
    plot_cross_annotation_per_dataset_bars,
    plot_novel_per_dataset_bars,
)

DEFAULT_CROSS_PAIRINGS: tuple[str, ...] = (
    "gencode->gencode",
    "refseq->ensembl",
    "gencode->ensembl",
    "refseq->gencode",
    "gencode->refseq",
    "ensembl->gencode",
)

# Phase-A novel-ref cross-annotation: only gencode→gencode (compare later vs cross_annotation_repro).
NOVEL_PHASE_A_CROSS_PAIRINGS: tuple[str, ...] = ("gencode->gencode",)


def _plot_v2_outdir(root: Path, outdir: Path | None) -> Path:
    if outdir is not None:
        return outdir.resolve()
    return (DEFAULT_PLOT_V2_ROOT / root.name).resolve()


def _parse_pairing(s: str) -> tuple[str, str]:
    if "->" not in s:
        raise ValueError(f"Invalid pairing {s!r}; expected train->test (e.g. gencode->gencode)")
    train, test = s.split("->", 1)
    return train.strip(), test.strip()


def _pairing_dir_name(train: str, test: str) -> str:
    return f"{train}__to__{test}"


def _filter_train_test(df: pd.DataFrame, train: str, test: str) -> pd.DataFrame:
    return df[
        (df["train_annotation"].astype(str) == train)
        & (df["test_annotation"].astype(str) == test)
    ].copy()


def run_mouse(root: Path, outdir: Path) -> None:
    df = load_benchmark_summaries(
        root,
        row_filter=lambda d: _filter_train_test(d, "gencode", "mouse"),
    )
    plot_benchmark_aupr_bars(df, outdir, long_csv_name="mouse_aupr_long.csv")


def run_tissue(root: Path, outdir: Path) -> None:
    df = load_benchmark_summaries(
        root,
        row_filter=lambda d: _filter_train_test(d, "gencode", "tissue"),
    )
    plot_benchmark_aupr_bars(df, outdir, long_csv_name="tissue_aupr_long.csv")


def run_cross_annotation(root: Path, outdir: Path, pairings: tuple[str, ...]) -> None:
    df = load_benchmark_summaries(root)
    for pairing in pairings:
        train, test = _parse_pairing(pairing)
        sub = _filter_train_test(df, train, test)
        if sub.empty:
            print(f"[telos_v2] cross-annotation: skip {pairing} (no rows)")
            continue
        pair_out = outdir / _pairing_dir_name(train, test)
        plot_cross_annotation_per_dataset_bars(
            sub,
            pair_out,
            long_csv_name=f"aupr_per_dataset_long_{train}__to__{test}.csv",
        )
        print(f"[telos_v2] cross-annotation {pairing} -> {pair_out}")


def run_novel_phase_a_cross_annotation(root: Path, outdir: Path) -> None:
    """
    Phase-A cross-annotation with augmented test reference: stacked AUPR bars for gencode→gencode only.

    Same per-dataset layout as ``cross-annotation`` (``aupr_bars_stringtie.pdf`` +
    ``aupr_bars_other.pdf`` under ``gencode__to__gencode/``), for side-by-side comparison with
    ``runs/cross_annotation_repro`` without the augmented test ref.
    """
    run_cross_annotation(root, outdir, NOVEL_PHASE_A_CROSS_PAIRINGS)


def run_refseq_novel(reports: Path, outdir: Path, *, facet: bool) -> None:
    summary = load_novel_summary(reports)
    facet_col = "test_annotation" if facet else None
    plot_novel_per_dataset_bars(summary, outdir, facet_col=facet_col)


def run_feature_importance(run_root: Path, outdir: Path, *, top_n: int = 10) -> None:
    from telos_v2.plotting.feature_window_plots import run_feature_importance_plots

    fi_dir = run_root / "reports" / "feature_importance"
    run_feature_importance_plots(fi_dir, outdir, top_n=top_n)


def run_window_sweep(root: Path, outdir: Path) -> None:
    from telos_v2.plotting.feature_window_plots import run_window_plots

    run_window_plots(root, outdir)


def main(argv: list[str] | None = None) -> int:
    apply_plot_style()
    p = argparse.ArgumentParser(description="Grouped AUPR bar plots → plot_v2/")
    sub = p.add_subparsers(dest="mode", required=True)

    pm = sub.add_parser("mouse", help="Human GENCODE → mouse benchmarks")
    pm.add_argument("--root", type=Path, default=Path("runs/mouse_cross_species_gencode"))

    pt = sub.add_parser("tissue", help="Human GENCODE → tissue benchmarks")
    pt.add_argument("--root", type=Path, default=Path("runs/tissue_human_gencode"))

    pc = sub.add_parser("cross-annotation", help="Cross-annotation pairing bar plots")
    pc.add_argument("--root", type=Path, default=Path("runs/cross_annotation_repro"))
    pc.add_argument(
        "--pairings",
        nargs="+",
        default=list(DEFAULT_CROSS_PAIRINGS),
        help="Train->test pairs (includes gencode->gencode by default)",
    )

    ppa = sub.add_parser(
        "novel-phase-a-cross",
        help="Phase-A cross-annotation: gencode->gencode AUPR bars only (augmented test ref)",
    )
    ppa.add_argument(
        "--root",
        type=Path,
        default=Path("runs/novel_phase_a_cross_annotation"),
    )

    pn = sub.add_parser("refseq-novel", help="RefSeq-novel eval bar plots")
    pn.add_argument("--reports", type=Path, default=Path("runs/refseq_novel_eval/reports"))
    pn.add_argument(
        "--no-facet",
        action="store_true",
        help="Do not split by test_annotation (default: facet gencode / ensembl)",
    )

    pf = sub.add_parser("feature-importance", help="Stacked top-N feature-importance PDFs (RF, XGB)")
    pf.add_argument("--root", type=Path, default=Path("runs/stage1_feature_importance_gencode"))
    pf.add_argument("--top-n", type=int, default=10)

    pw = sub.add_parser("window", help="Stacked feature-window sweep PDF")
    pw.add_argument("--root", type=Path, default=Path("runs/human_gencode_feature_window"))

    pa = sub.add_parser("all", help="All benchmark bars + refseq-novel + FI + window → plot_v2/")
    pa.add_argument("--cross-root", type=Path, default=Path("runs/cross_annotation_repro"))
    pa.add_argument("--mouse-root", type=Path, default=Path("runs/mouse_cross_species_gencode"))
    pa.add_argument("--tissue-root", type=Path, default=Path("runs/tissue_human_gencode"))
    pa.add_argument("--novel-reports", type=Path, default=Path("runs/refseq_novel_eval/reports"))
    pa.add_argument(
        "--fi-root", type=Path, default=Path("runs/stage1_feature_importance_gencode")
    )
    pa.add_argument("--window-root", type=Path, default=Path("runs/human_gencode_feature_window"))
    pa.add_argument("--top-n", type=int, default=10)

    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Override output directory (default: plot_v2/<run_folder_name>/)",
    )

    args = p.parse_args(argv)
    out_override = getattr(args, "outdir", None)

    if args.mode == "mouse":
        root = args.root.resolve()
        out = _plot_v2_outdir(root, out_override)
        run_mouse(root, out)
        print(f"[telos_v2] mouse -> {out}")
    elif args.mode == "tissue":
        root = args.root.resolve()
        out = _plot_v2_outdir(root, out_override)
        run_tissue(root, out)
        print(f"[telos_v2] tissue -> {out}")
    elif args.mode == "cross-annotation":
        root = args.root.resolve()
        out = _plot_v2_outdir(root, out_override)
        run_cross_annotation(root, out, tuple(args.pairings))
        print(f"[telos_v2] cross-annotation -> {out}")
    elif args.mode == "novel-phase-a-cross":
        root = args.root.resolve()
        out = _plot_v2_outdir(root, out_override)
        run_novel_phase_a_cross_annotation(root, out)
        print(f"[telos_v2] novel-phase-a-cross -> {out}")
    elif args.mode == "refseq-novel":
        reports = args.reports.resolve()
        out = out_override or (DEFAULT_PLOT_V2_ROOT / "refseq_novel_eval").resolve()
        run_refseq_novel(reports, out, facet=not args.no_facet)
        print(f"[telos_v2] refseq-novel -> {out}")
    elif args.mode == "feature-importance":
        root = args.root.resolve()
        out = out_override or (DEFAULT_PLOT_V2_ROOT / root.name).resolve()
        run_feature_importance(root, out, top_n=args.top_n)
        print(f"[telos_v2] feature-importance -> {out}")
    elif args.mode == "window":
        root = args.root.resolve()
        out = out_override or (DEFAULT_PLOT_V2_ROOT / root.name).resolve()
        run_window_sweep(root, out)
        print(f"[telos_v2] window sweep -> {out}")
    elif args.mode == "all":
        mroot = args.mouse_root.resolve()
        troot = args.tissue_root.resolve()
        croot = args.cross_root.resolve()
        firoot = args.fi_root.resolve()
        wroot = args.window_root.resolve()
        run_mouse(mroot, _plot_v2_outdir(mroot, out_override))
        run_tissue(troot, _plot_v2_outdir(troot, out_override))
        run_cross_annotation(
            croot,
            _plot_v2_outdir(croot, out_override),
            tuple(DEFAULT_CROSS_PAIRINGS),
        )
        novel_out = out_override or (DEFAULT_PLOT_V2_ROOT / "refseq_novel_eval").resolve()
        run_refseq_novel(args.novel_reports.resolve(), novel_out, facet=True)
        run_feature_importance(
            firoot,
            out_override or (DEFAULT_PLOT_V2_ROOT / firoot.name).resolve(),
            top_n=args.top_n,
        )
        run_window_sweep(
            wroot,
            out_override or (DEFAULT_PLOT_V2_ROOT / wroot.name).resolve(),
        )
        print(f"[telos_v2] all figures under {DEFAULT_PLOT_V2_ROOT.resolve()}")
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
