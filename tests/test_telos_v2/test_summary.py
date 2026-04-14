"""Tests for Markdown run summaries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from telos_v2.reporting.summary import (
    build_predict_summary_md,
    build_train_summary_md,
    write_benchmark_summary,
    write_predict_summaries,
    write_train_summary,
)


class TestSummaryMarkdown(unittest.TestCase):
    def test_train_summary_contains_sections(self) -> None:
        md = build_train_summary_md(
            manifest_path=Path("/tmp/run_manifest.json"),
            metrics_payload={
                "tss_rf": {"n_train": 1, "n_val": 2, "accuracy": 0.9},
                "tss_xgb": {"n_train": 1, "n_val": 2, "accuracy": 0.88},
                "tes_rf": {"n_train": 3, "n_val": 4, "accuracy": 0.8},
                "tes_xgb": {"n_train": 3, "n_val": 4, "accuracy": 0.79},
                "stage2_rf": {"n_features": 10, "accuracy": 0.85, "roc_auc": 0.7},
                "stage2_xgb": {"n_features": 10, "accuracy": 0.84, "roc_auc": 0.69},
            },
            paths={
                "manifest": Path("/out/reports/run_manifest.json"),
                "bam": Path("/data/a.bam"),
                "gtf": Path("/data/t.gtf"),
                "ref_gtf": Path("/ref/r.gtf"),
                "tmap": Path("/ref/m.tmap"),
                "outdir": Path("/out"),
                "stage1_models": "a.joblib, b.joblib",
                "stage2_models": "/out/models/s2_rf.joblib, /out/models/s2_xgb.joblib",
                "sites_scored": Path("/out/predictions/sites.scored.tsv"),
                "transcripts_ranked_rf": Path("/out/predictions/transcripts.ranked.rf.tsv"),
                "transcripts_ranked_xgb": Path("/out/predictions/transcripts.ranked.xgb.tsv"),
                "train_metrics_json": Path("/out/reports/train_metrics.json"),
            },
        )
        self.assertIn("# Telos v2 — train run summary", md)
        self.assertIn("Stage I — TSS (RF)", md)
        self.assertIn("Stage II — validation (RF)", md)

    def test_predict_summary_and_dual_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td) / "reports"
            p1, p2 = write_predict_summaries(
                rd,
                manifest_path=Path(td) / "run_manifest.json",
                paths={
                    "manifest": rd / "run_manifest.json",
                    "bam": Path("/x.bam"),
                    "gtf": Path("/y.gtf"),
                    "model_dir": Path("/m"),
                    "outdir": Path(td),
                    "sites_scored": Path("/s.tsv"),
                    "transcripts_ranked_rf": Path("/r_rf.tsv"),
                    "transcripts_ranked_xgb": Path("/r_xgb.tsv"),
                    "filtered_gtf_rf": Path("/f_rf.gtf"),
                    "filtered_gtf_xgb": Path("/f_xgb.gtf"),
                },
                kept_tx=3,
                total_tx=10,
                filter_threshold=0.5,
                n_ranked_transcripts="rf=9, xgb=9",
                kept_tx_xgb=4,
                total_tx_xgb=10,
            )
            self.assertTrue(p1.exists())
            self.assertTrue(p2.exists())
            self.assertEqual(p1.read_text(encoding="utf-8"), p2.read_text(encoding="utf-8"))
            self.assertEqual(p1.name, "summary.md")
            self.assertEqual(p2.name, "predict_summary.md")

    def test_predict_md_escape_pipe(self) -> None:
        md = build_predict_summary_md(
            manifest_path=Path("/m.json"),
            paths={"gtf": Path("/a|b.gtf"), "bam": Path("/x.bam")},
        )
        self.assertIn("\\|", md)

    def test_write_train_summary_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td) / "reports"
            out = write_train_summary(
                rd,
                manifest_path=Path(td) / "run_manifest.json",
                metrics_payload={
                    "tss_rf": {"accuracy": 1.0},
                    "tss_xgb": {"accuracy": 1.0},
                    "tes_rf": {"accuracy": 1.0},
                    "tes_xgb": {"accuracy": 1.0},
                    "stage2_rf": {},
                    "stage2_xgb": {},
                },
                paths={"outdir": Path(td), "bam": Path("/b.bam")},
            )
            self.assertTrue(out.exists())
            self.assertEqual(out.name, "summary.md")

    def test_benchmark_summary_writes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td) / "reports"
            p = write_benchmark_summary(
                rd,
                config_path=Path("/cfg.yaml"),
                outdir=Path(td),
                manifest_path=rd / "run_manifest.json",
                summary_tsv=rd / "benchmark_summary.csv",
                summary_json=rd / "benchmark_summary.json",
                train_rows=[{"status": "ok"}],
                test_rows=[{"test_id": "t1", "status": "ok"}],
            )
            self.assertTrue(p.exists())
            self.assertIn("benchmark", p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
