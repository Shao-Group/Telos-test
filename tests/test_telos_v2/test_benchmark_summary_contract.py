"""Benchmark summary CSV stable columns (phase 6/7)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from telos_v2.benchmark.report import SUMMARY_CSV_ORDERED_KEYS, write_benchmark_summary_csv


class TestBenchmarkSummaryContract(unittest.TestCase):
    def test_writer_emits_ordered_core_columns(self) -> None:
        rows = [
            {
                "test_id": "x",
                "status": "ok",
                "predict_code": "0",
                "stage2_test_aupr_rf": 0.5,
                "transcript_pr_auc_model_rf": 0.5,
                "extra_noise": "drop_me",
            }
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "benchmark_summary.csv"
            write_benchmark_summary_csv(rows, p)
            with p.open(encoding="utf-8", newline="") as fh:
                r = csv.DictReader(fh)
                fieldnames = list(r.fieldnames or [])
                self.assertTrue(set(fieldnames) <= set(SUMMARY_CSV_ORDERED_KEYS))
                first = next(r)
                self.assertEqual(first["test_id"], "x")
                self.assertEqual(fieldnames[:4], ["test_id", "status", "predict_code", "stage2_test_aupr_rf"])
                self.assertNotIn("extra_noise", first)
                # Transcript PR columns are part of the public summary schema when enabled.


if __name__ == "__main__":
    unittest.main()
