"""Tier-0 / Tier-1 parity helpers vs frozen goldens."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from telos_repro.backend import get_backend_name, predict, resolve_backend, train
from telos_repro.backend.telos_backend import ensure_telos_importable
from telos_repro.backend.types import PredictRequest, TrainRequest
from telos_repro.paths import find_repo_root, load_paths, path_value

# Figure-driving columns to compare for Tier-1 (fail closed on drift).
DEFAULT_METRIC_COLUMNS: tuple[str, ...] = (
    "stage2_test_aupr_rf",
    "stage2_test_aupr_xgb",
    "stage2_test_aupr_baseline",
    "transcript_pr_auc_model_rf",
    "transcript_pr_auc_model_xgb",
    "transcript_pr_auc_baseline",
)


def _bundle_paths(
    bundles_root: Path,
    *,
    genome_root: Path,
    sample: str = "SRR307903",
) -> dict[str, Path]:
    base = bundles_root / "GRCh38_gencode49" / "sr" / sample
    ref_gtf = genome_root / "gencode" / "gencode.v49.primary_assembly.basic.annotation.gtf"
    # Prefer manifest ref if present.
    manifest = base / "bundle_manifest.yaml"
    if manifest.is_file():
        import yaml

        raw = yaml.safe_load(manifest.read_text()) or {}
        ref = raw.get("ref_annotation_gtf")
        if ref:
            ref_gtf = Path(str(ref))
        aln = (raw.get("alignments") or {}).get("sorted_bam")
        bam = Path(str(aln)) if aln else base / "align" / "aln.sorted.bam"
    else:
        bam = base / "align" / "aln.sorted.bam"
    return {
        "bam": bam,
        "gtf": base / "stringtie.gtf",
        "tmap": base / "stringtie.stringtie.gtf.tmap",
        "ref_gtf": ref_gtf,
    }


def run_tier0(
    *,
    backend: str | None = None,
    skip_train: bool = False,
    skip_predict: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Tier-0 smoke: resolve backend, optionally train once + predict once on a single bundle.

    Full SR BAM train is expensive; use ``skip_train`` with frozen shared models for a
    faster predict-only smoke.
    """
    root = find_repo_root()
    paths = load_paths(root)
    name = get_backend_name(explicit=backend)
    os.environ["TELOS_REPRO_BACKEND"] = name
    backend_obj = resolve_backend(explicit=name)

    report: dict[str, Any] = {
        "tier": 0,
        "backend": name,
        "backend_class": type(backend_obj).__name__,
        "import_ok": True,
    }

    if name == "telos":
        telos_src = ensure_telos_importable()
        import telos

        report["telos_version"] = getattr(telos, "__version__", None)
        report["telos_src"] = str(telos_src)
        pin = paths.get("telos_pin_sha")
        report["telos_pin_sha"] = pin

    bundles = path_value(paths, "bundles_root")
    bp = _bundle_paths(bundles, genome_root=path_value(paths, "genome_root"))
    runs = path_value(paths, "runs_root")
    out_root = runs / "parity_tier0" / name
    train_out = out_root / "train"
    pred_out = out_root / "predict"
    stage1 = root / "configs" / "stage1.defaults.yaml"

    report["paths"] = {k: str(v) for k, v in bp.items()}
    report["outdir"] = str(out_root)

    missing = [k for k, v in bp.items() if not v.exists()]
    if missing:
        report["import_ok"] = True
        report["data_ok"] = False
        report["missing"] = missing
        report["status"] = "blocked_missing_inputs"
        return report
    report["data_ok"] = True

    if dry_run:
        report["status"] = "dry_run"
        report["would_train"] = not skip_train
        report["would_predict"] = not skip_predict
        return report

    # Prefer frozen shared models when skipping train.
    golden_models = (
        path_value(paths, "goldens_runs_root")
        / "cross_annotation_repro"
        / "_cross_annotation_shared_train"
        / "sr__train_gencode"
        / "models"
    )
    model_dir = train_out / "models"
    train_code = None
    if skip_train:
        if not golden_models.is_dir():
            report["status"] = "blocked_no_frozen_models"
            report["golden_models"] = str(golden_models)
            return report
        model_dir = golden_models
        report["train"] = {"skipped": True, "model_dir": str(model_dir)}
    else:
        train_code = train(
            TrainRequest(
                bam=bp["bam"],
                gtf=bp["gtf"],
                tmap=bp["tmap"],
                ref_gtf=bp["ref_gtf"],
                outdir=train_out,
                config_file=stage1 if stage1.is_file() else None,
            )
        )
        report["train"] = {"code": train_code, "outdir": str(train_out)}
        if train_code != 0:
            report["status"] = "train_failed"
            return report

    pred_code = None
    if not skip_predict:
        pred_code = predict(
            PredictRequest(
                bam=bp["bam"],
                gtf=bp["gtf"],
                model_dir=model_dir,
                outdir=pred_out,
                config_file=stage1 if stage1.is_file() else None,
            )
        )
        report["predict"] = {"code": pred_code, "outdir": str(pred_out)}
        if pred_code != 0:
            report["status"] = "predict_failed"
            return report
    else:
        report["predict"] = {"skipped": True}

    report["status"] = "ok"
    return report


def _load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _normalize_assembler_id(row: dict[str, str]) -> str:
    aid = str(row.get("assembler_id") or "").strip()
    if aid:
        return aid
    # Manual YAML cells often omit assembler_id; recover from <sample>__<assembler>.
    tid = str(row.get("test_id") or "")
    if "__" in tid:
        return tid.rsplit("__", 1)[-1].strip()
    return ""


def _row_key(row: dict[str, str], key_fields: tuple[str, ...]) -> tuple[str, ...]:
    parts: list[str] = []
    for field in key_fields:
        if field == "assembler_id":
            parts.append(_normalize_assembler_id(row))
        else:
            parts.append(str(row.get(field) or "").strip())
    return tuple(parts)


def compare_summaries(
    local_csv: Path,
    golden_csv: Path,
    *,
    columns: tuple[str, ...] = DEFAULT_METRIC_COLUMNS,
    abs_tol: float = 1e-6,
    key_fields: tuple[str, ...] = ("test_id", "assembler_id"),
    require_all_golden_keys: bool = False,
) -> dict[str, Any]:
    """
    Fail-closed compare of figure-driving metric columns.

    By default only rows present in ``local_csv`` are required (small Tier-1 slice).
    Set ``require_all_golden_keys=True`` to also fail if any golden row is missing locally.
    """
    local_rows = {_row_key(r, key_fields): r for r in _load_summary_rows(local_csv)}
    golden_rows = {_row_key(r, key_fields): r for r in _load_summary_rows(golden_csv)}
    diffs: list[dict[str, Any]] = []
    missing_in_golden = sorted(set(local_rows) - set(golden_rows))
    missing_local = sorted(set(golden_rows) - set(local_rows)) if require_all_golden_keys else []
    for key, lrow in local_rows.items():
        if key not in golden_rows:
            continue
        grow = golden_rows[key]
        for col in columns:
            gv = grow.get(col, "")
            lv = lrow.get(col, "")
            if gv in ("", None) and lv in ("", None):
                continue
            try:
                gnum = float(gv)
                lnum = float(lv)
            except (TypeError, ValueError):
                if str(gv) != str(lv):
                    diffs.append({"key": list(key), "column": col, "golden": gv, "local": lv})
                continue
            if abs(gnum - lnum) > abs_tol:
                diffs.append(
                    {
                        "key": list(key),
                        "column": col,
                        "golden": gnum,
                        "local": lnum,
                        "abs_diff": abs(gnum - lnum),
                    }
                )
    ok = not diffs and not missing_in_golden and not missing_local
    return {
        "ok": ok,
        "n_local": len(local_rows),
        "n_golden": len(golden_rows),
        "missing_in_golden_keys": [list(k) for k in missing_in_golden],
        "missing_local_keys": [list(k) for k in missing_local],
        "diffs": diffs,
        "columns": list(columns),
        "abs_tol": abs_tol,
        "require_all_golden_keys": require_all_golden_keys,
    }


def run_tier1_compare(
    *,
    local_summary: Path,
    golden_summary: Path | None = None,
    abs_tol: float = 1e-6,
) -> dict[str, Any]:
    """Compare a local ``benchmark_summary.csv`` to the frozen golden (default: sr gencode→gencode)."""
    root = find_repo_root()
    paths = load_paths(root)
    if golden_summary is None:
        golden_summary = (
            path_value(paths, "goldens_runs_root")
            / "cross_annotation_repro"
            / "sr__train_gencode__test_gencode"
            / "reports"
            / "benchmark_summary.csv"
        )
    report: dict[str, Any] = {
        "tier": 1,
        "local": str(local_summary),
        "golden": str(golden_summary),
        "backend": get_backend_name(),
    }
    if not local_summary.is_file():
        report["status"] = "missing_local"
        return report
    if not golden_summary.is_file():
        report["status"] = "missing_golden"
        return report
    cmp_ = compare_summaries(local_summary, golden_summary, abs_tol=abs_tol)
    report["compare"] = cmp_
    report["status"] = "ok" if cmp_["ok"] else "drift"
    return report


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
