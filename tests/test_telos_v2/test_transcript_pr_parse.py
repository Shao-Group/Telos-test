"""Transcript PR point parsing (TSV + legacy log lines)."""

from __future__ import annotations

import unittest

import numpy as np

from telos_v2.evaluation.transcript_pr_pipeline import parse_gtfcuff_roc_stdout


class TestParseGtfcuffRocStdout(unittest.TestCase):
    def test_tsv_rows(self) -> None:
        text = "recall\tprecision\n0.1\t0.2\n0.3\t0.4\n"
        r, p = parse_gtfcuff_roc_stdout(text)
        np.testing.assert_array_almost_equal(r, [0.1, 0.3])
        np.testing.assert_array_almost_equal(p, [0.2, 0.4])

    def test_legacy_roc_line(self) -> None:
        text = 'ROC: sensitivity = 50.000000 precision = 75.500000\n'
        r, p = parse_gtfcuff_roc_stdout(text)
        np.testing.assert_array_almost_equal(r, [0.5])
        np.testing.assert_array_almost_equal(p, [0.755])


if __name__ == "__main__":
    unittest.main()
