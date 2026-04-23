"""
Run a full cross-annotation benchmark grid and write a run-index CSV.

Grid axes:
- data_type: sr, cdna, drna, pacbio
- train_annotation: refseq, gencode, ensembl
- test_annotation: refseq, gencode, ensembl (excluding same-as-train by default)
"""

from __future__ import annotations

import csv
from pathlib import Path

from telos_v2.benchmark.matrix import (
    ANNOTATION_TO_REF_ID,
    DATA_TYPE_TO_MODALITY,
    resolve_bundles_root,
    run_benchmark_matrix,
)


def run_cross_annotation_benchmarks(
    *,
    outdir: Path,
    bundles_root: Path | None = None,
    stage1_config: Path | None = None,
    data_types: tuple[str, ...] | None = None,
    annotations: tuple[str, ...] | None = None,
    include_same_annotation: bool = False,
) -> int:
    """
    Execute all matrix benchmarks for the selected cross-annotation grid.

    Writes a compact run index at ``<outdir>/reports/cross_annotation_runs.csv``.
    Returns ``0`` only if every combination exits successfully.
    """
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    reports_dir = outdir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    root = resolve_bundles_root(bundles_root)
    dtypes = data_types if data_types is not None else tuple(DATA_TYPE_TO_MODALITY.keys())
    anns = annotations if annotations is not None else tuple(ANNOTATION_TO_REF_ID.keys())

    rows: list[dict[str, str]] = []
    all_ok = True
    total = 0
    for dt in dtypes:
        for tr in anns:
            for te in anns:
                if not include_same_annotation and tr == te:
                    continue
                total += 1
                combo_id = f"{dt}__train_{tr}__test_{te}"
                combo_outdir = outdir / combo_id
                print(f"[telos_v2] cross-annotation {total}: {combo_id}")
                code = run_benchmark_matrix(
                    data_type=dt,
                    train_annotation=tr,
                    test_annotation=te,
                    outdir=combo_outdir,
                    bundles_root=root,
                    stage1_config=stage1_config,
                )
                ok = code == 0
                all_ok = all_ok and ok
                rows.append(
                    {
                        "run_id": combo_id,
                        "data_type": dt,
                        "train_annotation": tr,
                        "test_annotation": te,
                        "exit_code": str(code),
                        "status": "ok" if ok else "failed",
                        "outdir": str(combo_outdir),
                        "summary_csv": str(combo_outdir / "reports" / "benchmark_summary.csv"),
                    }
                )

    run_index = reports_dir / "cross_annotation_runs.csv"
    with run_index.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "run_id",
                "data_type",
                "train_annotation",
                "test_annotation",
                "exit_code",
                "status",
                "outdir",
                "summary_csv",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[telos_v2] cross-annotation index: {run_index}")
    return 0 if all_ok else 1

