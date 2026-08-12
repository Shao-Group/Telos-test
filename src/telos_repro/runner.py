"""Subprocess wrappers around experiment / plot entrypoints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from telos_repro.paths import load_paths, path_value
from telos_repro.registry import Experiment, PlotSpec, load_registry


def _env_for_run(repo_root: Path, paths: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    # Single src/ tree: telos_repro (paper stack) + experiments + experiments; then product Telos.
    path_parts: list[str] = [str(repo_root / "src")]
    telos_checkout = paths.get("telos_checkout")
    if telos_checkout:
        telos_src = str(Path(telos_checkout) / "src")
        if telos_src not in path_parts:
            path_parts.append(telos_src)
    existing = env.get("PYTHONPATH", "")
    prefix = os.pathsep.join(path_parts)
    env["PYTHONPATH"] = prefix if not existing else f"{prefix}{os.pathsep}{existing}"

    if "bundles_root" in paths:
        env["TELOS_BUNDLES_ROOT"] = str(path_value(paths, "bundles_root"))
    if "telos_stage1_cache_dir" in paths:
        env["TELOS_STAGE1_CACHE_DIR"] = str(path_value(paths, "telos_stage1_cache_dir"))
    if "gffcompare_bin" in paths:
        env["GFFCOMPARE"] = str(path_value(paths, "gffcompare_bin"))
    if "backend" in paths and paths["backend"]:
        env.setdefault("TELOS_REPRO_BACKEND", str(paths["backend"]))
    return env


def _resolve_outdir(repo_root: Path, paths: dict[str, Any], rel: str | None) -> Path | None:
    if not rel:
        return None
    p = Path(rel)
    if p.is_absolute():
        return p
    # Prefer writable runs_root from paths.yaml when relative path starts with runs/
    if rel.startswith("runs/") or rel == "runs":
        runs_root = path_value(paths, "runs_root")
        suffix = rel[len("runs/") :] if rel.startswith("runs/") else ""
        return (runs_root / suffix) if suffix else runs_root
    return (repo_root / p).resolve()


def build_experiment_argv(
    exp: Experiment,
    repo_root: Path,
    paths: dict[str, Any],
    extra_args: list[str] | None = None,
) -> list[str]:
    argv = [sys.executable, *exp.command_target(repo_root), *exp.default_args]
    # Inject --outdir / equivalent unless caller already overrides it.
    extra = list(extra_args or [])
    outdir_flag = exp.outdir_arg
    if outdir_flag and exp.default_outdir:
        already = any(a == outdir_flag or a.startswith(f"{outdir_flag}=") for a in extra)
        if not already:
            out = _resolve_outdir(repo_root, paths, exp.default_outdir)
            if out is not None:
                argv.extend([outdir_flag, str(out)])
    argv.extend(extra)
    return argv


def build_plot_argv(
    plot: PlotSpec,
    repo_root: Path,
    paths: dict[str, Any],
    extra_args: list[str] | None = None,
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "telos_repro.plotting.plot_experiments",
        plot.plot_mode,
        *plot.default_args,
    ]
    extra = list(extra_args or [])
    if "--outdir" not in extra and not any(a.startswith("--outdir=") for a in extra):
        figures_root = path_value(paths, "figures_root")
        # plot_experiments defaults to plot_v2/<run_name>; pass parent override only when useful.
        # Keep default relative behavior unless figures_root differs from repo plot_v2.
        default_plot = (repo_root / "plot_v2").resolve()
        if figures_root.resolve() != default_plot:
            # Subcommand-specific roots still write under figures_root/<name> via CWD;
            # ensure CWD is repo_root and figures_root exists.
            figures_root.mkdir(parents=True, exist_ok=True)
    argv.extend(extra)
    return argv


def run_argv(
    argv: list[str],
    *,
    repo_root: Path,
    paths: dict[str, Any],
    dry_run: bool = False,
) -> int:
    env = _env_for_run(repo_root, paths)
    print("+", " ".join(argv), flush=True)
    if dry_run:
        return 0
    # Ensure writable output roots exist.
    for key in ("runs_root", "figures_root", "telos_stage1_cache_dir"):
        if key in paths:
            path_value(paths, key).mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(argv, cwd=str(repo_root), env=env)
    return int(proc.returncode)


def status_experiment(
    exp: Experiment,
    repo_root: Path,
    paths: dict[str, Any],
) -> dict[str, Any]:
    local_outdir = _resolve_outdir(repo_root, paths, exp.default_outdir)
    local_summaries: list[str] = []
    if local_outdir and local_outdir.is_dir() and exp.summary_glob:
        local_summaries = sorted(str(p) for p in local_outdir.glob(exp.summary_glob))

    golden_summaries: list[str] = []
    golden_root = None
    if exp.golden_run_subdir and "goldens_runs_root" in paths:
        golden_root = path_value(paths, "goldens_runs_root") / exp.golden_run_subdir
        if golden_root.is_dir() and exp.summary_glob:
            golden_summaries = sorted(str(p) for p in golden_root.glob(exp.summary_glob))

    return {
        "id": exp.id,
        "outdir": str(local_outdir) if local_outdir else None,
        "outdir_exists": bool(local_outdir and local_outdir.is_dir()),
        "local_summary_count": len(local_summaries),
        "golden_root": str(golden_root) if golden_root else None,
        "golden_exists": bool(golden_root and golden_root.is_dir()),
        "golden_summary_count": len(golden_summaries),
    }


def get_context(repo_root: Path | None = None) -> tuple[Path, dict[str, Any], dict[str, Experiment], dict[str, PlotSpec]]:
    from telos_repro.paths import find_repo_root

    root = repo_root or find_repo_root()
    paths = load_paths(root)
    experiments, plots = load_registry(root)
    return root, paths, experiments, plots
