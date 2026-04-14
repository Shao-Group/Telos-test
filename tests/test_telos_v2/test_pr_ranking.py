"""Tests for PR ranking vs abundance baseline."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from telos_v2.evaluation.pr_ranking import (
    compute_pr_curves_vs_baseline,
    transcript_pr_vs_baseline_from_files,
)


class TestPRRanking(unittest.TestCase):
    def test_model_beats_random_baseline_synthetic(self) -> None:
        rng = np.random.default_rng(0)
        n = 500
        # Scores correlated with label → high AUPR
        y = rng.integers(0, 2, size=n)
        logits = rng.normal(size=n) + y.astype(float) * 2.0
        model = 1 / (1 + np.exp(-logits))
        baseline = rng.uniform(0, 1, size=n)
        r = compute_pr_curves_vs_baseline(y, model, baseline, abundance_column="coverage")
        self.assertGreater(r.model.aupr, r.baseline.aupr)
        self.assertEqual(r.n_items, n)
        self.assertEqual(r.n_positive, int(y.sum()))

    def test_transcript_from_files(self) -> None:
        ranked = pd.DataFrame(
            {
                "transcript_id": ["a", "b", "c", "d"],
                "pred_prob": [0.9, 0.7, 0.4, 0.1],
            }
        )
        cov = pd.DataFrame(
            {
                "transcript_id": ["a", "b", "c", "d"],
                "coverage": [100.0, 50.0, 10.0, 1.0],
            }
        )
        tmap = io.StringIO(
            "qry_id\tref_gene_id\tref_id\tclass_code\n"
            "a\tg1\tr1\t=\n"
            "b\tg2\tr2\t=\n"
            "c\tg3\tr3\tx\n"
            "d\tg4\tr4\tx\n"
        )
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            ranked.to_csv(rd / "ranked.tsv", sep="\t", index=False)
            cov.to_csv(rd / "cov.tsv", sep="\t", index=False)
            (rd / "m.tmap").write_text(tmap.getvalue(), encoding="utf-8")
            r = transcript_pr_vs_baseline_from_files(
                rd / "ranked.tsv",
                rd / "cov.tsv",
                rd / "m.tmap",
            )
            self.assertEqual(r.n_positive, 2)
            self.assertGreaterEqual(r.model.aupr, 0.0)
            self.assertGreaterEqual(r.baseline.aupr, 0.0)


if __name__ == "__main__":
    unittest.main()
