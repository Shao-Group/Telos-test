"""Unit tests for Track B backend facade (no BAM training)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from telos_repro.backend.select import get_backend_name, resolve_backend
from telos_repro.backend.types import PredictRequest, TrainRequest, as_predict_request, as_train_request
from telos_repro.parity import compare_summaries


class TestBackendSelect(unittest.TestCase):
    def test_env_defaults_telos(self) -> None:
        with mock.patch.dict(os.environ, {"TELOS_REPRO_BACKEND": "telos"}):
            self.assertEqual(get_backend_name(), "telos")

    def test_explicit_wins(self) -> None:
        with mock.patch.dict(os.environ, {"TELOS_REPRO_BACKEND": "telos"}):
            self.assertEqual(get_backend_name(explicit="telos"), "telos")

    def test_telos_v2_removed(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            get_backend_name(explicit="telos_v2")
        self.assertIn("removed", str(ctx.exception).lower())

    def test_resolve_classes(self) -> None:
        self.assertEqual(resolve_backend(explicit="telos").name, "telos")


class TestRequestConversion(unittest.TestCase):
    def test_train_from_duck(self) -> None:
        class IO:
            bam = Path("/b.bam")
            gtf = Path("/a.gtf")
            outdir = Path("/out")
            ref_gtf = Path("/ref.gtf")
            tmap = Path("/t.tmap")
            config_file = None
            gtf_pool = None
            tmap_pool = None
            stage1_no_parallel = False
            stage1_n_workers = None

        req = as_train_request(IO())
        self.assertIsInstance(req, TrainRequest)
        self.assertEqual(req.bam, Path("/b.bam"))
        self.assertEqual(req.ref_gtf, Path("/ref.gtf"))

    def test_predict_from_duck(self) -> None:
        class IO:
            bam = Path("/b.bam")
            gtf = Path("/a.gtf")
            outdir = Path("/out")
            model_dir = Path("/models")
            config_file = None
            stage1_no_parallel = True
            stage1_n_workers = 2

        req = as_predict_request(IO())
        self.assertIsInstance(req, PredictRequest)
        self.assertTrue(req.stage1_no_parallel)
        self.assertEqual(req.backend, "xgb")


class TestCompareSummaries(unittest.TestCase):
    def test_exact_match(self) -> None:
        header = (
            "test_id,assembler_id,stage2_test_aupr_rf,stage2_test_aupr_xgb,"
            "stage2_test_aupr_baseline,transcript_pr_auc_model_rf,"
            "transcript_pr_auc_model_xgb,transcript_pr_auc_baseline\n"
        )
        row = "t1,stringtie,0.7,0.71,0.6,100.0,101.0,90.0\n"
        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / "local.csv"
            golden = Path(td) / "golden.csv"
            local.write_text(header + row)
            golden.write_text(header + row)
            out = compare_summaries(local, golden)
            self.assertTrue(out["ok"])
            self.assertEqual(out["diffs"], [])

    def test_drift_fails(self) -> None:
        header = (
            "test_id,assembler_id,stage2_test_aupr_rf,stage2_test_aupr_xgb,"
            "stage2_test_aupr_baseline,transcript_pr_auc_model_rf,"
            "transcript_pr_auc_model_xgb,transcript_pr_auc_baseline\n"
        )
        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / "local.csv"
            golden = Path(td) / "golden.csv"
            local.write_text(header + "t1,stringtie,0.9,0.71,0.6,100.0,101.0,90.0\n")
            golden.write_text(header + "t1,stringtie,0.7,0.71,0.6,100.0,101.0,90.0\n")
            out = compare_summaries(local, golden)
            self.assertFalse(out["ok"])
            self.assertTrue(any(d["column"] == "stage2_test_aupr_rf" for d in out["diffs"]))

    def test_empty_assembler_id_normalized(self) -> None:
        header = (
            "test_id,assembler_id,stage2_test_aupr_rf,stage2_test_aupr_xgb,"
            "stage2_test_aupr_baseline,transcript_pr_auc_model_rf,"
            "transcript_pr_auc_model_xgb,transcript_pr_auc_baseline\n"
        )
        with tempfile.TemporaryDirectory() as td:
            local = Path(td) / "local.csv"
            golden = Path(td) / "golden.csv"
            local.write_text(header + "SRR1__stringtie,,0.7,0.71,0.6,100.0,101.0,90.0\n")
            golden.write_text(header + "SRR1__stringtie,stringtie,0.7,0.71,0.6,100.0,101.0,90.0\n")
            out = compare_summaries(local, golden)
            self.assertTrue(out["ok"], out)


if __name__ == "__main__":
    unittest.main()
