"""Markdown run summaries (``reports/summary.md``) for train, predict, and benchmark."""

from __future__ import annotations

import json
import numbers
from pathlib import Path
from typing import Any


def _esc(s: str) -> str:
    return s.replace("|", "\\|")


def _section(title: str, body_lines: list[str]) -> list[str]:
    out = [f"## {title}", ""]
    out.extend(body_lines)
    out.append("")
    return out


def _kv_lines(items: list[tuple[str, str]]) -> list[str]:
    return [f"- **{_esc(k)}**: {_esc(v)}" for k, v in items]


def write_text_summary(reports_dir: Path, filename: str, content: str) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def build_train_summary_md(
    *,
    manifest_path: Path,
    metrics_payload: dict[str, Any],
    paths: dict[str, Path | str],
) -> str:
    lines: list[str] = [
        "# Telos v2 — train run summary",
        "",
        "Two-stage training completed. Primary artifacts are under `models/` and `predictions/`.",
        "",
    ]
    lines.extend(
        _section(
            "Inputs and layout",
            _kv_lines(
                [
                    ("Run manifest", str(paths.get("manifest", manifest_path))),
                    ("BAM", str(paths.get("bam", ""))),
                    ("Transcript GTF", str(paths.get("gtf", ""))),
                    ("Reference GTF", str(paths.get("ref_gtf", ""))),
                    ("tmap (Stage II labels)", str(paths.get("tmap", ""))),
                    ("Output directory", str(paths.get("outdir", ""))),
                ]
            ),
        )
    )

    def _stage1_block(key: str, label: str) -> list[str]:
        m = metrics_payload.get(key)
        if not isinstance(m, dict):
            return [f"_No metrics for {label}._"]
        parts: list[str] = []
        for name in ("n_train", "n_val", "accuracy", "precision", "recall", "f1", "roc_auc", "aupr"):
            if name in m and m[name] is not None:
                parts.append(f"{name}={m[name]}")
        if parts:
            return ["`" + ", ".join(parts) + "`"]
        return [json.dumps(m, indent=2)[:4000]]

    for st in ("TSS", "TES"):
        for b in ("rf", "xgb"):
            key = f"{st.lower()}_{b}"
            lines.extend(
                _section(f"Stage I — {st} ({b.upper()})", _stage1_block(key, f"{st} {b}"))
            )

    for b in ("rf", "xgb"):
        s2 = metrics_payload.get(f"stage2_{b}")
        if isinstance(s2, dict):
            s2_lines = []
            for name in ("n_features", "n_train", "n_val", "accuracy", "roc_auc", "aupr"):
                if name in s2 and s2[name] is not None:
                    s2_lines.append(f"{name}={s2[name]}")
            if s2_lines:
                lines.extend(
                    _section(f"Stage II — validation ({b.upper()})", ["`" + ", ".join(s2_lines) + "`"])
                )
            else:
                lines.extend(
                    _section(f"Stage II — validation ({b.upper()})", [json.dumps(s2, indent=2)[:4000]])
                )
        else:
            lines.extend(_section(f"Stage II — validation ({b.upper()})", ["_No Stage II metrics._"]))

    lines.extend(
        _section(
            "Primary outputs",
            _kv_lines(
                [
                    ("Stage I models", str(paths.get("stage1_models", ""))),
                    ("Stage II models", str(paths.get("stage2_models", paths.get("stage2_model", "")))),
                    ("Sites scored", str(paths.get("sites_scored", ""))),
                    ("Transcripts ranked (RF)", str(paths.get("transcripts_ranked_rf", ""))),
                    ("Transcripts ranked (XGB)", str(paths.get("transcripts_ranked_xgb", ""))),
                    ("Train metrics JSON", str(paths.get("train_metrics_json", ""))),
                ]
            ),
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def build_predict_summary_md(
    *,
    manifest_path: Path,
    paths: dict[str, Path | str],
    kept_tx: int | None = None,
    total_tx: int | None = None,
    filter_threshold: float | None = None,
    n_ranked_transcripts: int | str | None = None,
    kept_tx_xgb: int | None = None,
    total_tx_xgb: int | None = None,
) -> str:
    filt_note_rf = ""
    if kept_tx is not None and total_tx is not None:
        filt_note_rf = f" ({kept_tx} kept of {total_tx} transcripts)"
    filt_note_xgb = ""
    if kept_tx_xgb is not None and total_tx_xgb is not None:
        filt_note_xgb = f" ({kept_tx_xgb} kept of {total_tx_xgb} transcripts)"
    thr = ""
    if filter_threshold is not None:
        thr = f" (threshold={filter_threshold})"

    lines: list[str] = [
        "# Telos v2 — predict run summary",
        "",
        "Annotation-free inference: Stage I site scores and Stage II transcript ranking.",
        "",
    ]
    lines.extend(
        _section(
            "Inputs",
            _kv_lines(
                [
                    ("Run manifest", str(paths.get("manifest", manifest_path))),
                    ("BAM", str(paths.get("bam", ""))),
                    ("Transcript GTF", str(paths.get("gtf", ""))),
                    ("Model directory", str(paths.get("model_dir", ""))),
                    ("Output directory", str(paths.get("outdir", ""))),
                ]
            ),
        )
    )
    prim: list[tuple[str, str]] = [
        ("Sites scored", str(paths.get("sites_scored", ""))),
        ("Transcripts ranked (RF)", str(paths.get("transcripts_ranked_rf", ""))),
        ("Transcripts ranked (XGB)", str(paths.get("transcripts_ranked_xgb", ""))),
        ("Filtered GTF (RF)", str(paths.get("filtered_gtf_rf", paths.get("filtered_gtf", ""))) + thr + filt_note_rf),
        ("Filtered GTF (XGB)", str(paths.get("filtered_gtf_xgb", "")) + thr + filt_note_xgb),
    ]
    if n_ranked_transcripts is not None:
        prim.append(("Ranked rows", str(n_ranked_transcripts)))
    lines.extend(_section("Primary outputs", _kv_lines(prim)))
    return "\n".join(lines).rstrip() + "\n"


def build_benchmark_summary_md(
    *,
    config_path: Path,
    outdir: Path,
    manifest_path: Path,
    summary_tsv: Path,
    summary_json: Path,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# Telos v2 — benchmark summary",
        "",
        "Matrix benchmark: train once (or skip), predict on each test entry, optional tmap metrics.",
        "",
    ]
    lines.extend(
        _section(
            "Run",
            _kv_lines(
                [
                    ("Config", str(config_path)),
                    ("Output directory", str(outdir)),
                    ("Manifest", str(manifest_path)),
                    ("Summary CSV", str(summary_tsv)),
                    ("Summary JSON", str(summary_json)),
                ]
            ),
        )
    )
    train_aupr_lines: list[str] = []
    if train_rows:
        tr0 = train_rows[0]
        if isinstance(tr0, dict):
            for key in (
                "train_val_aupr_stage1_tss_rf",
                "train_val_aupr_stage1_tss_xgb",
                "train_val_aupr_stage1_tes_rf",
                "train_val_aupr_stage1_tes_xgb",
                "train_val_aupr_stage2_rf",
                "train_val_aupr_stage2_xgb",
            ):
                if key in tr0 and tr0[key] is not None:
                    v = tr0[key]
                    if isinstance(v, numbers.Real) and not isinstance(v, bool):
                        disp = f"{float(v):.2f}"
                    else:
                        disp = str(v)
                    train_aupr_lines.append(f"- `{key}`: {disp}")
    if train_aupr_lines:
        lines.extend(
            _section(
                "Training validation AUPR (from train/reports/train_metrics.json)",
                train_aupr_lines,
            )
        )
    lines.extend(
        _section(
            "Training",
            ["```json", json.dumps(train_rows, indent=2)[:12000], "```"],
        )
    )
    lines.extend(
        _section(
            "Tests",
            ["```json", json.dumps(test_rows, indent=2)[:24000], "```"],
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def write_train_summary(
    reports_dir: Path,
    *,
    manifest_path: Path,
    metrics_payload: dict[str, Any],
    paths: dict[str, Path | str],
) -> Path:
    md = build_train_summary_md(manifest_path=manifest_path, metrics_payload=metrics_payload, paths=paths)
    return write_text_summary(reports_dir, "summary.md", md)


def write_predict_summaries(
    reports_dir: Path,
    *,
    manifest_path: Path,
    paths: dict[str, Path | str],
    kept_tx: int | None = None,
    total_tx: int | None = None,
    filter_threshold: float | None = None,
    n_ranked_transcripts: int | str | None = None,
    kept_tx_xgb: int | None = None,
    total_tx_xgb: int | None = None,
) -> tuple[Path, Path]:
    """Writes ``summary.md`` (required) and ``predict_summary.md`` (command-contract alias)."""
    md = build_predict_summary_md(
        manifest_path=manifest_path,
        paths=paths,
        kept_tx=kept_tx,
        total_tx=total_tx,
        filter_threshold=filter_threshold,
        n_ranked_transcripts=n_ranked_transcripts,
        kept_tx_xgb=kept_tx_xgb,
        total_tx_xgb=total_tx_xgb,
    )
    p1 = write_text_summary(reports_dir, "summary.md", md)
    p2 = write_text_summary(reports_dir, "predict_summary.md", md)
    return p1, p2


def write_benchmark_summary(
    reports_dir: Path,
    *,
    config_path: Path,
    outdir: Path,
    manifest_path: Path,
    summary_tsv: Path,
    summary_json: Path,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> Path:
    md = build_benchmark_summary_md(
        config_path=config_path,
        outdir=outdir,
        manifest_path=manifest_path,
        summary_tsv=summary_tsv,
        summary_json=summary_json,
        train_rows=train_rows,
        test_rows=test_rows,
    )
    return write_text_summary(reports_dir, "benchmark_summary.md", md)
