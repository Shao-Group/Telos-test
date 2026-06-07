"""
Self-contained Stage I feature-importance study: GENCODE vs GENCODE, window = 100 bp.

For each data type (sr, cdna, drna, pacbio), runs a GENCODE train/test benchmark, then reads
models and cache. The four modalities run in parallel by default.

Layout under ``runs/stage1_feature_importance_gencode`` (default)::

    sr/train/models/              Stage I bundles
    sr/reports/generated_benchmark.yaml
    stage1_cache/                 shared feature cache
    reports/stage1.window100.yaml
    reports/feature_importance/   output TSVs

Usage::

    # Full pipeline: 4 parallel GENCODE benchmarks + built-in + permutation importance
    PYTHONPATH=src_v2 python src_v2/experiments/stage1_feature_importance_gencode.py

    # Re-summarize only (benchmarks and cache must already exist)
    PYTHONPATH=src_v2 python src_v2/experiments/stage1_feature_importance_gencode.py --importance-only

    # Skip permutation (faster; built-in importances only)
    PYTHONPATH=src_v2 python src_v2/experiments/stage1_feature_importance_gencode.py --no-permutation
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml

from telos_v2.analysis.stage1_importance import (
    builtin_importances_from_bundle,
    group_importance_summary,
    iter_stage1_model_paths,
    permutation_importance_stage1,
    stage1_feature_group,
)
from telos_v2.benchmark.matrix import resolve_bundles_root, run_benchmark_matrix
from telos_v2.config_loader import default_stage1_config_path, load_mapping_config
from telos_v2.labels.site_labels import label_sites_by_proximity, reference_sites_from_gtf
from telos_v2.models import STAGE1_BACKENDS, stage1_bundle_path
from telos_v2.models.chrom_split import parse_split_policy, split_train_val_masks
from telos_v2.models.stage1_train import load_stage1_bundle
from telos_v2.pipeline_core import build_stage1_runtime_config, stage1_cache_dir_for

WINDOW_SIZE = 100
ANNOTATION = "gencode"
DATA_TYPES = ("sr", "cdna", "drna", "pacbio")
DEFAULT_RUN_ROOT_NAME = "stage1_feature_importance_gencode"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_run_root() -> Path:
    return _repo_root() / "runs" / DEFAULT_RUN_ROOT_NAME


def default_cache_root(run_root: Path) -> Path:
    return run_root / "stage1_cache"


def data_type_dir(run_root: Path, data_type: str) -> Path:
    return run_root / data_type


def models_dir(run_root: Path, data_type: str) -> Path:
    return data_type_dir(run_root, data_type) / "train" / "models"


def materialize_stage1_config(*, base_config: Path, run_root: Path) -> Path:
    cfg = copy.deepcopy(load_mapping_config(base_config.resolve()))
    stage1 = cfg.setdefault("stage1", {})
    if not isinstance(stage1, dict):
        raise ValueError("stage1 block must be a mapping")
    fe = stage1.setdefault("feature_extraction", {})
    if not isinstance(fe, dict):
        raise ValueError("stage1.feature_extraction must be a mapping")
    fe["window_size"] = WINDOW_SIZE
    fe["density_window"] = WINDOW_SIZE
    fe["coverage_window"] = WINDOW_SIZE
    fe["gradient_analysis_range"] = WINDOW_SIZE
    fe["cache_dir"] = str(default_cache_root(run_root).resolve())

    out_path = run_root / "reports" / "stage1.window100.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    return out_path.resolve()


@contextmanager
def _stage1_cache_env(cache_root: Path) -> Iterator[None]:
    key = "TELOS_STAGE1_CACHE_DIR"
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    saved = os.environ.get(key)
    os.environ[key] = str(cache_root)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved


def stage1_models_ready(run_root: Path, data_types: tuple[str, ...] = DATA_TYPES) -> bool:
    for dt in data_types:
        md = models_dir(run_root, dt)
        if not md.is_dir():
            return False
        for site in ("TSS", "TES"):
            for backend in STAGE1_BACKENDS:
                if not (md / stage1_bundle_path(site, backend)).is_file():
                    return False
    return True


def _worker_benchmark_matrix(job: dict[str, str]) -> tuple[str, int]:
    """Run one modality (process-pool worker)."""
    dt = job["data_type"]
    run_root = Path(job["run_root"])
    if stage1_models_ready(run_root, (dt,)):
        return dt, 0
    with _stage1_cache_env(Path(job["cache_root"])):
        code = run_benchmark_matrix(
            data_type=dt,
            train_annotation=ANNOTATION,
            test_annotation=ANNOTATION,
            outdir=run_root / dt,
            bundles_root=Path(job["bundles_root"]) if job.get("bundles_root") else None,
            stage1_config=Path(job["stage1_path"]),
        )
    return dt, code


def run_gencode_experiments(
    run_root: Path,
    *,
    bundles_root: Path | None,
    base_stage1_config: Path | None,
    sequential: bool = False,
) -> int:
    """Run GENCODE train/test benchmarks; sr/cdna/drna/pacbio in parallel unless ``sequential``."""
    run_root.mkdir(parents=True, exist_ok=True)
    root = resolve_bundles_root(bundles_root)
    base = base_stage1_config or default_stage1_config_path()
    if not base.is_file():
        print(f"[stage1_fi] stage1 base config not found: {base}")
        return 2

    stage1_path = materialize_stage1_config(base_config=base, run_root=run_root)
    cache_root = default_cache_root(run_root)
    pending = [dt for dt in DATA_TYPES if not stage1_models_ready(run_root, (dt,))]

    print(
        f"[stage1_fi] run_root={run_root} window={WINDOW_SIZE} cache={cache_root}",
        flush=True,
    )
    if not pending:
        print("[stage1_fi] all data types already trained", flush=True)
        return 0

    job_base = {
        "run_root": str(run_root),
        "bundles_root": str(root),
        "stage1_path": str(stage1_path),
        "cache_root": str(cache_root),
    }
    worst = 0

    if sequential:
        with _stage1_cache_env(cache_root):
            for dt in pending:
                print(f"[stage1_fi] benchmark {dt}", flush=True)
                _, code = _worker_benchmark_matrix({**job_base, "data_type": dt})
                if code != 0:
                    worst = code
        return worst

    with ProcessPoolExecutor(max_workers=len(pending)) as pool:
        futures = {
            pool.submit(_worker_benchmark_matrix, {**job_base, "data_type": dt}): dt
            for dt in pending
        }
        for fut in as_completed(futures):
            dt = futures[fut]
            try:
                _, code = fut.result()
            except Exception as exc:
                print(f"[stage1_fi] {dt} failed: {exc}", flush=True)
                worst = 1
                continue
            if code != 0:
                worst = code
                print(f"[stage1_fi] {dt} exited {code}", flush=True)
    return worst


def _condition_meta(data_type: str) -> dict[str, str | int]:
    return {
        "data_type": data_type,
        "window_size": str(WINDOW_SIZE),
        "annotation": ANNOTATION,
    }


def collect_builtin_importance(run_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for dt in DATA_TYPES:
        md = models_dir(run_root, dt)
        if not md.is_dir():
            continue
        meta = _condition_meta(dt)
        for site_type, backend, bundle_path in iter_stage1_model_paths(md):
            fi = builtin_importances_from_bundle(bundle_path)
            fi["site_type"] = site_type
            fi["backend"] = backend
            for k, v in meta.items():
                fi[k] = v
            fi["model_path"] = str(bundle_path)
            rows.append(fi)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _summary_group_keys(long_df: pd.DataFrame) -> list[str]:
    keys = ["data_type", "site_type", "backend"]
    if "window_size" in long_df.columns and long_df["window_size"].nunique(dropna=True) > 1:
        keys.insert(1, "window_size")
    return keys


def _write_group_summary(long_df: pd.DataFrame, out_dir: Path, tag: str) -> None:
    if long_df.empty:
        return
    parts: list[pd.DataFrame] = []
    present = _summary_group_keys(long_df)
    for group_keys, sub in long_df.groupby(present, dropna=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        g = group_importance_summary(sub)
        for k, v in zip(present, group_keys):
            g[k] = v
        parts.append(g)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(
            out_dir / f"stage1_feature_importance_{tag}_by_group.tsv",
            sep="\t",
            index=False,
        )


def _rank_stability(long_df: pd.DataFrame, out_path: Path) -> None:
    if long_df.empty or "data_type" not in long_df.columns:
        return
    present = _summary_group_keys(long_df)
    ranked = long_df.copy()
    ranked["rank"] = ranked.groupby(present)["importance_norm"].rank(ascending=False, method="average")
    stab = (
        ranked.groupby("feature", as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            mean_importance=("importance_norm", "mean"),
            n_conditions=("rank", "count"),
        )
        .sort_values("mean_rank")
    )
    stab["group"] = stab["feature"].map(stage1_feature_group)
    stab.to_csv(out_path, sep="\t", index=False)


def _load_train_block(run_root: Path, data_type: str) -> dict[str, Any] | None:
    cfg_path = data_type_dir(run_root, data_type) / "reports" / "generated_benchmark.yaml"
    if not cfg_path.is_file():
        return None
    bench = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    train = bench.get("train")
    return train if isinstance(train, dict) else None


def _try_load_cached_df_all(*, train: dict[str, Any], cache_root: Path) -> pd.DataFrame | None:
    bam = Path(str(train.get("bam", "")))
    gtf = Path(str(train.get("gtf", "")))
    cfg_path = Path(str(train.get("config", "")))
    if not bam.is_file() or not gtf.is_file() or not cfg_path.is_file():
        return None
    cfg_map = load_mapping_config(cfg_path)
    runtime = build_stage1_runtime_config(cfg_map, cli_no_parallel=False, cli_n_workers=None)
    cache_dir = stage1_cache_dir_for(bam=bam, gtf=gtf, runtime_cfg=runtime, cache_root=cache_root)
    pkl = cache_dir / "df_all.pkl"
    if pkl.is_file():
        return pd.read_pickle(pkl)
    return None


def _run_permutation_for_data_type(
    run_root: Path,
    data_type: str,
    cache_root: Path,
    *,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    meta = _condition_meta(data_type)
    train = _load_train_block(run_root, data_type)
    if train is None:
        return pd.DataFrame()
    df_all = _try_load_cached_df_all(train=train, cache_root=cache_root)
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    ref_gtf = Path(str(train.get("ref_gtf", "")))
    if not ref_gtf.is_file():
        return pd.DataFrame()
    cfg_path = Path(str(train.get("config", "")))
    cfg_map = load_mapping_config(cfg_path)
    tol = int(cfg_map.get("stage1", {}).get("training", {}).get("site_label_tolerance_bp", 50))
    policy = str(cfg_map.get("stage1", {}).get("training", {}).get("split_policy", "chr1-10"))
    train_range = parse_split_policy(policy)
    ref_df = reference_sites_from_gtf(ref_gtf)
    md = models_dir(run_root, data_type)
    if not md.is_dir():
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for site_type in ("TSS", "TES"):
        labeled = df_all[df_all["site_type"].str.upper() == site_type].copy()
        labeled["label"] = label_sites_by_proximity(labeled, ref_df, site_type, tol)
        _, val_mask = split_train_val_masks(labeled, train_range)
        sub = labeled.loc[val_mask]
        if sub.empty or sub["label"].nunique() < 2:
            continue
        for backend in ("rf", "xgb"):
            bundle_path = md / f"stage1_{site_type.lower()}_{backend}_model.joblib"
            if not bundle_path.is_file():
                continue
            bundle = load_stage1_bundle(bundle_path)
            feats = list(bundle["feature_names"])
            try:
                fi = permutation_importance_stage1(
                    bundle["model"],
                    feats,
                    sub,
                    sub["label"],
                    n_repeats=n_repeats,
                    random_state=random_state,
                )
            except Exception:
                continue
            fi["site_type"] = site_type
            fi["backend"] = backend
            for k, v in meta.items():
                fi[k] = v
            rows.append(fi)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_importance(
    run_root: Path,
    *,
    out_dir: Path | None = None,
    do_permutation: bool = False,
    n_repeats: int = 3,
    random_state: int = 42,
) -> int:
    run_root = run_root.resolve()
    out_dir = (out_dir or (run_root / "reports" / "feature_importance")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = default_cache_root(run_root)

    long_builtin = collect_builtin_importance(run_root)
    if long_builtin.empty:
        print(f"[stage1_fi] no stage1 models under {run_root}")
        return 1

    long_builtin.to_csv(out_dir / "stage1_builtin_importance_long.tsv", sep="\t", index=False)
    _write_group_summary(long_builtin, out_dir, "builtin")
    _rank_stability(long_builtin, out_dir / "stage1_builtin_importance_stability.tsv")

    top_rows: list[dict] = []
    for (dt, st, be), sub in long_builtin.groupby(["data_type", "site_type", "backend"], dropna=False):
        agg = sub.groupby("feature", as_index=False)["importance_norm"].mean()
        top = agg.nlargest(15, "importance_norm")
        top_rows.append(
            {
                "data_type": dt,
                "site_type": st,
                "backend": be,
                "window_size": WINDOW_SIZE,
                "top_features": top.to_dict(orient="records"),
            }
        )
    (out_dir / "stage1_builtin_top15_by_condition.json").write_text(
        json.dumps(top_rows, indent=2),
        encoding="utf-8",
    )
    print(
        f"[stage1_fi] wrote {out_dir / 'stage1_builtin_importance_long.tsv'} "
        f"({len(long_builtin)} rows, window={WINDOW_SIZE})"
    )

    if do_permutation:
        perm_parts: list[pd.DataFrame] = []
        for dt in DATA_TYPES:
            part = _run_permutation_for_data_type(
                run_root,
                dt,
                cache_root,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            if not part.empty:
                perm_parts.append(part)
                print(f"[stage1_fi] permutation ok: {dt} ({len(part)} rows)")
            else:
                print(f"[stage1_fi] permutation skipped: {dt}")
        if perm_parts:
            perm_long = pd.concat(perm_parts, ignore_index=True)
            perm_long.to_csv(out_dir / "stage1_permutation_importance_long.tsv", sep="\t", index=False)
            _write_group_summary(
                perm_long.rename(columns={"importance_mean": "importance_norm"}),
                out_dir,
                "permutation",
            )

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage I feature importance from GENCODE vs GENCODE benchmarks (window=100 bp)",
    )
    ap.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help=f"Experiment root (default: runs/{DEFAULT_RUN_ROOT_NAME})",
    )
    ap.add_argument("--outdir", type=Path, default=None, help="Importance TSV directory")
    ap.add_argument(
        "--importance-only",
        action="store_true",
        help="Skip benchmarks; only write importance TSVs from existing runs",
    )
    ap.add_argument(
        "--no-permutation",
        action="store_true",
        help="Skip validation permutation importance (default: run it)",
    )
    ap.add_argument("--bundles-root", type=Path, default=None)
    ap.add_argument("--stage1-config", type=Path, default=None)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--sequential", action="store_true", default=False)
    args = ap.parse_args()

    run_root = (args.run_root or default_run_root()).resolve()

    if not args.importance_only:
        code = run_gencode_experiments(
            run_root,
            bundles_root=args.bundles_root,
            base_stage1_config=args.stage1_config,
            sequential=args.sequential,
        )
        if code != 0:
            return code

    return summarize_importance(
        run_root,
        out_dir=args.outdir,
        do_permutation=not args.no_permutation,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    raise SystemExit(main())
