"""Load ``configs/experiments.yaml`` experiment + plot registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from telos_repro.paths import find_repo_root


@dataclass(frozen=True)
class Experiment:
    id: str
    description: str
    kind: str  # experiment | pipeline_step
    module: str | None = None
    script: str | None = None
    default_args: list[str] = field(default_factory=list)
    outdir_arg: str | None = "--outdir"
    default_outdir: str | None = None
    summary_glob: str | None = "**/reports/benchmark_summary.csv"
    golden_run_subdir: str | None = None
    plot_id: str | None = None
    notes: str = ""

    def command_target(self, repo_root: Path) -> list[str]:
        """Return argv prefix: either ``-m package.module`` or a script path."""
        if self.module:
            return ["-m", self.module]
        if self.script:
            return [str(repo_root / self.script)]
        raise ValueError(f"experiment {self.id!r} has neither module nor script")


@dataclass(frozen=True)
class PlotSpec:
    id: str
    description: str
    plot_mode: str
    default_args: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    golden_figure_subdir: str | None = None
    notes: str = ""


def _require_mapping(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    return raw


def load_registry(repo_root: Path | None = None) -> tuple[dict[str, Experiment], dict[str, PlotSpec]]:
    root = repo_root or find_repo_root()
    path = root / "configs" / "experiments.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment registry: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    data = _require_mapping(raw, "experiments.yaml")

    experiments: dict[str, Experiment] = {}
    for eid, body in _require_mapping(data.get("experiments") or {}, "experiments").items():
        b = _require_mapping(body, f"experiments.{eid}")
        experiments[eid] = Experiment(
            id=eid,
            description=str(b.get("description") or ""),
            kind=str(b.get("kind") or "experiment"),
            module=b.get("module"),
            script=b.get("script"),
            default_args=[str(x) for x in (b.get("default_args") or [])],
            outdir_arg=b.get("outdir_arg", "--outdir"),
            default_outdir=b.get("default_outdir"),
            summary_glob=b.get("summary_glob", "**/reports/benchmark_summary.csv"),
            golden_run_subdir=b.get("golden_run_subdir"),
            plot_id=b.get("plot"),
            notes=str(b.get("notes") or ""),
        )

    plots: dict[str, PlotSpec] = {}
    for pid, body in _require_mapping(data.get("plots") or {}, "plots").items():
        b = _require_mapping(body, f"plots.{pid}")
        plots[pid] = PlotSpec(
            id=pid,
            description=str(b.get("description") or ""),
            plot_mode=str(b["plot_mode"]),
            default_args=[str(x) for x in (b.get("default_args") or [])],
            expected_outputs=[str(x) for x in (b.get("expected_outputs") or [])],
            golden_figure_subdir=b.get("golden_figure_subdir"),
            notes=str(b.get("notes") or ""),
        )

    return experiments, plots
