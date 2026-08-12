"""``telos-repro`` CLI: list | status | run | plot | parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from telos_repro import __version__
from telos_repro.runner import (
    build_experiment_argv,
    build_plot_argv,
    get_context,
    run_argv,
    status_experiment,
)


def _cmd_list(args: argparse.Namespace) -> int:
    root, _paths, experiments, plots = get_context()
    print(f"repo: {root}")
    print(f"paths: {_paths.get('_config_path')}")
    print(f"backend: {_paths.get('backend', 'telos')}")
    print()
    print("Experiments:")
    for eid, exp in experiments.items():
        kind = exp.kind
        plot = f"  plot={exp.plot_id}" if exp.plot_id else ""
        print(f"  {eid:40s} [{kind}]{plot}")
        if args.verbose and exp.description:
            print(f"      {exp.description}")
    print()
    print("Plots:")
    for pid, plot in plots.items():
        print(f"  {pid:40s} mode={plot.plot_mode}")
        if args.verbose and plot.description:
            print(f"      {plot.description}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    root, paths, experiments, _plots = get_context()
    ids = [args.experiment] if args.experiment else list(experiments)
    rows = []
    for eid in ids:
        if eid not in experiments:
            print(f"unknown experiment: {eid}", file=sys.stderr)
            return 2
        rows.append(status_experiment(experiments[eid], root, paths))

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    for row in rows:
        local = "yes" if row["outdir_exists"] else "no"
        golden = "yes" if row["golden_exists"] else "no"
        print(
            f"{row['id']:40s}  local={local} summaries={row['local_summary_count']}  "
            f"golden={golden} summaries={row['golden_summary_count']}"
        )
        if args.verbose:
            print(f"  outdir:  {row['outdir']}")
            print(f"  golden:  {row['golden_root']}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    root, paths, experiments, _plots = get_context()
    if args.experiment not in experiments:
        print(f"unknown experiment: {args.experiment}", file=sys.stderr)
        print("Use: telos-repro list", file=sys.stderr)
        return 2
    exp = experiments[args.experiment]
    argv = build_experiment_argv(exp, root, paths, extra_args=list(args.extra or []))
    return run_argv(argv, repo_root=root, paths=paths, dry_run=args.dry_run)


def _cmd_plot(args: argparse.Namespace) -> int:
    root, paths, experiments, plots = get_context()
    key = args.target
    # Allow experiment id → linked plot, or direct plot id.
    if key in experiments and experiments[key].plot_id:
        key = experiments[key].plot_id  # type: ignore[assignment]
    if key not in plots:
        print(f"unknown plot (or experiment without plot=): {args.target}", file=sys.stderr)
        print("Use: telos-repro list", file=sys.stderr)
        return 2
    plot = plots[key]
    argv = build_plot_argv(plot, root, paths, extra_args=list(args.extra or []))
    return run_argv(argv, repo_root=root, paths=paths, dry_run=args.dry_run)


def _cmd_parity(args: argparse.Namespace) -> int:
    from telos_repro.parity import run_tier0, run_tier1_compare, write_report
    from telos_repro.paths import find_repo_root, load_paths, path_value

    root = find_repo_root()
    paths = load_paths(root)
    reports_dir = path_value(paths, "runs_root") / "parity_reports"

    if args.tier == "0":
        report = run_tier0(
            backend=args.backend,
            skip_train=args.skip_train,
            skip_predict=args.skip_predict,
            dry_run=args.dry_run,
        )
    elif args.tier == "1":
        local = (
            Path(args.local_summary)
            if args.local_summary
            else path_value(paths, "runs_root")
            / "parity_tier1"
            / "sr__train_gencode__test_gencode"
            / "reports"
            / "benchmark_summary.csv"
        )
        golden = Path(args.golden_summary) if args.golden_summary else None
        report = run_tier1_compare(local_summary=local, golden_summary=golden, abs_tol=args.abs_tol)
    else:
        print(f"unknown tier: {args.tier}", file=sys.stderr)
        return 2

    out = Path(args.report) if args.report else reports_dir / f"tier{args.tier}.json"
    write_report(report, out)
    print(json.dumps(report, indent=2))
    print(f"wrote {out}", file=sys.stderr)
    status = report.get("status")
    if status in ("ok", "dry_run"):
        return 0
    if status and str(status).startswith("blocked"):
        return 3
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="telos-repro",
        description=(
            "Paper reproduction add-on for product Telos: list/status/run/plot/parity. "
            "Train/predict use product telos via telos_repro.backend."
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("list", help="List registered experiments and plots")
    pl.add_argument("-v", "--verbose", action="store_true")
    pl.set_defaults(func=_cmd_list)

    ps = sub.add_parser("status", help="Show local vs golden summary presence")
    ps.add_argument("experiment", nargs="?", help="Experiment id (default: all)")
    ps.add_argument("-v", "--verbose", action="store_true")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=_cmd_status)

    pr = sub.add_parser(
        "run",
        help="Run a registered experiment (forwards extra args after --)",
    )
    pr.add_argument("-n", "--dry-run", action="store_true", help="Print command only")
    pr.add_argument("experiment", help="Experiment id from the registry")
    pr.set_defaults(func=_cmd_run, extra=[])

    pp = sub.add_parser(
        "plot",
        help="Run a registered plot (or experiment id that has plot=)",
    )
    pp.add_argument("-n", "--dry-run", action="store_true")
    pp.add_argument("target", help="Plot id or experiment id with a linked plot")
    pp.set_defaults(func=_cmd_plot, extra=[])

    pa = sub.add_parser("parity", help="Track B Tier-0/1 parity vs frozen goldens")
    pa.add_argument("tier", choices=("0", "1"), help="0=import/smoke train+predict; 1=CSV compare")
    pa.add_argument("--backend", choices=("telos",), default=None)
    pa.add_argument(
        "--skip-train",
        action="store_true",
        help="Tier-0: reuse frozen shared-train models (predict-only smoke)",
    )
    pa.add_argument("--skip-predict", action="store_true", help="Tier-0: stop after train/import")
    pa.add_argument("-n", "--dry-run", action="store_true", help="Tier-0: resolve paths only")
    pa.add_argument("--local-summary", help="Tier-1: local benchmark_summary.csv")
    pa.add_argument("--golden-summary", help="Tier-1: frozen golden CSV (default: sr gencode→gencode)")
    pa.add_argument("--abs-tol", type=float, default=1e-6, help="Tier-1 absolute metric tolerance")
    pa.add_argument("--report", help="Write JSON report path")
    pa.set_defaults(func=_cmd_parity)

    return p


def _split_forwarded_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``cmd ... -- forwarded`` so ``-n`` is not swallowed by REMAINDER."""
    if "--" not in argv:
        return argv, []
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1 :]


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    own, forwarded = _split_forwarded_argv(raw)
    parser = build_parser()
    args = parser.parse_args(own)
    args.extra = forwarded
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
