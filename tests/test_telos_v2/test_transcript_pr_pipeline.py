"""Legacy-style transcript PR pipeline (parse helpers)."""

from __future__ import annotations

import shutil
import unittest

import numpy as np

from telos_v2.evaluation.transcript_pr_pipeline import (
    parse_gtfcuff_roc_stdout,
    resolve_gffcompare_executable,
)


class TestTranscriptPRPipeline(unittest.TestCase):
    def test_resolve_gffcompare_explicit(self) -> None:
        self.assertEqual(resolve_gffcompare_executable("/bin/true"), "/bin/true")

    def test_resolve_gffcompare_uses_path_when_on_path(self) -> None:
        which = shutil.which("gffcompare")
        if which:
            self.assertEqual(resolve_gffcompare_executable(None), which)

    def test_parse_gtfcuff_roc_stdout(self) -> None:
        text = """
ROC: reference = 100 prediction = 50 correct = 10 sensitivity = 5.00 precision = 20.00 | coverage = 1.000
ROC: reference = 100 prediction = 40 correct = 8 sensitivity = 8.00 precision = 20.00 | coverage = 0.500
""".strip()
        rec, pre = parse_gtfcuff_roc_stdout(text)
        np.testing.assert_array_almost_equal(rec, [0.05, 0.08])
        np.testing.assert_array_almost_equal(pre, [0.2, 0.2])


if __name__ == "__main__":
    unittest.main()
