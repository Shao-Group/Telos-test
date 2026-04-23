"""
Plot novel Phase A cross-annotation benchmark results without overwriting base plots.

Usage:
  conda run -n irtesam-berth python src_v2/experiments/plot_novel_phase_a_results.py
"""

from __future__ import annotations

from pathlib import Path

from plot_cross_annotation_results import run_plotting

ROOT = Path("runs/novel_phase_a_cross_annotation")
OUTDIR = ROOT / "reports" / "plots_phase_a_novel"


def main() -> int:
    return run_plotting(ROOT.resolve(), OUTDIR.resolve())


if __name__ == "__main__":
    raise SystemExit(main())

