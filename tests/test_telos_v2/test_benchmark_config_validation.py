"""Strict benchmark YAML validation (phase 2/6 contract)."""

from __future__ import annotations

import unittest

from telos_v2.config_validation import validate_benchmark_config


class TestBenchmarkConfigValidation(unittest.TestCase):
    def test_minimal_valid_config(self) -> None:
        validate_benchmark_config(
            {
                "train": {
                    "mode": "skip",
                    "model_dir": "/m",
                },
                "tests": [
                    {
                        "id": "t1",
                        "bam": "/a.bam",
                        "gtf": "/a.gtf",
                        "ref_gtf": "/r.gtf",
                        "tmap": "/a.tmap",
                    }
                ],
                "analysis": {
                    "enabled": True,
                    "benchmark_mode": "minimal",
                    "debug": {"keep_pr_work": False},
                    "pr_vs_baseline": {"enabled": False},
                },
            }
        )

    def test_accepts_matrix_metadata_keys(self) -> None:
        validate_benchmark_config(
            {
                "generated_by": "telos_v2.benchmark.matrix",
                "schema_version": 3,
                "train": {"mode": "skip", "model_dir": "/m"},
                "tests": [
                    {"id": "t1", "bam": "/b", "gtf": "/g", "ref_gtf": "/r", "tmap": "/t"},
                ],
                "analysis": {
                    "enabled": True,
                    "benchmark_mode": "minimal",
                    "debug": {"keep_pr_work": False},
                    "pr_vs_baseline": {"enabled": False},
                },
            }
        )

    def test_rejects_bad_benchmark_mode(self) -> None:
        with self.assertRaises(ValueError):
            validate_benchmark_config(
                {
                    "train": {"mode": "skip", "model_dir": "/m"},
                    "tests": [{"id": "t1", "bam": "/b", "gtf": "/g", "ref_gtf": "/r", "tmap": "/t"}],
                    "analysis": {"benchmark_mode": "lite"},
                }
            )


if __name__ == "__main__":
    unittest.main()
