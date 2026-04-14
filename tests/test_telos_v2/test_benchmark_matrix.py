"""benchmark_matrix path resolution and YAML mapping."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from telos_v2.benchmark_matrix import build_benchmark_yaml_mapping
from telos_v2.reporting.benchmark_columns import train_validation_aupr_columns


def _write_min_manifest(path: Path, *, bam: str, gtf: str, tmap: str, ref_gtf: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "alignments": {"sorted_bam": bam},
        "ref_annotation_gtf": ref_gtf,
        "assemblies": [
            {
                "assembler_id": "stringtie",
                "gtf": gtf,
                "gffcompare": {"tmap": tmap},
            }
        ],
    }
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")


class TestBenchmarkMatrix(unittest.TestCase):
    def test_cross_annotation_train_test(self) -> None:
        td = tempfile.mkdtemp(prefix="telos_mtx_")
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        root = Path(td)
        mod = "sr"
        # Train: refseq layout, SRR307903
        tr = root / "GRCh38_refseq_p14" / mod / "SRR307903"
        _write_min_manifest(
            tr / "bundle_manifest.yaml",
            bam="/t.bam",
            gtf="/t.gtf",
            tmap="/t.tmap",
            ref_gtf="/ref_r.gtf",
        )
        # Test: gencode layout, exclude SRR307903, one test sample
        for sid in ("SRR307903", "SRR999"):
            d = root / "GRCh38_gencode49" / mod / sid
            _write_min_manifest(
                d / "bundle_manifest.yaml",
                bam=f"/{sid}.bam",
                gtf=f"/{sid}.gtf",
                tmap=f"/{sid}.tmap",
                ref_gtf="/ref_g.gtf",
            )

        stage1 = Path(td) / "stage1.yaml"
        stage1.write_text("x: 1\n", encoding="utf-8")
        m = build_benchmark_yaml_mapping(
            data_type="sr",
            train_annotation="refseq",
            test_annotation="gencode",
            bundles_root=root,
            stage1_config=stage1,
        )

        self.assertEqual(m["train"]["bam"], "/t.bam")
        self.assertEqual(m["train"]["ref_gtf"], "/ref_r.gtf")
        self.assertEqual(len(m["tests"]), 1)
        self.assertEqual(m["tests"][0]["id"], "SRR999__stringtie")
        self.assertEqual(m["tests"][0]["assembler_id"], "stringtie")
        self.assertEqual(m["tests"][0]["ref_gtf"], "/ref_g.gtf")
        self.assertEqual(m["tests"][0]["config"], str(stage1.resolve()))
        self.assertEqual(m.get("schema_version"), 2)
        pr = m["analysis"]["pr_vs_baseline"]
        self.assertTrue(pr.get("filter_validation_chroms"))
        self.assertIn("chromosomes_file", pr)
        self.assertTrue(pr.get("save_pr_tables"))
        self.assertEqual(
            pr.get("gffcompare_bin"),
            "/datadisk1/ixk5174/tools/gffcompare-0.12.10.Linux_x86_64/gffcompare",
        )

    def test_multi_assembler_expands_test_rows(self) -> None:
        td = tempfile.mkdtemp(prefix="telos_mtx2_")
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        root = Path(td)
        mod = "pacbio"
        tr = root / "GRCh38_refseq_p14" / mod / "ENCFF450VAU"
        _write_min_manifest(
            tr / "bundle_manifest.yaml",
            bam="/t.bam",
            gtf="/t_st.gtf",
            tmap="/t_st.tmap",
            ref_gtf="/ref_r.gtf",
        )
        d = root / "GRCh38_gencode49" / mod / "ENCFF370NFS"
        d.mkdir(parents=True, exist_ok=True)
        multi = {
            "alignments": {"sorted_bam": "/s.bam"},
            "ref_annotation_gtf": "/ref_g.gtf",
            "assemblies": [
                {
                    "assembler_id": "isoquant",
                    "gtf": "/s_iq.gtf",
                    "gffcompare": {"tmap": "/s_iq.tmap"},
                },
                {
                    "assembler_id": "stringtie",
                    "gtf": "/s_st.gtf",
                    "gffcompare": {"tmap": "/s_st.tmap"},
                },
            ],
        }
        (d / "bundle_manifest.yaml").write_text(yaml.safe_dump(multi), encoding="utf-8")

        stage1 = Path(td) / "stage1.yaml"
        stage1.write_text("x: 1\n", encoding="utf-8")
        m = build_benchmark_yaml_mapping(
            data_type="pacbio",
            train_annotation="refseq",
            test_annotation="gencode",
            bundles_root=root,
            stage1_config=stage1,
        )
        self.assertEqual(len(m["tests"]), 2)
        self.assertEqual(m["tests"][0]["id"], "ENCFF370NFS__stringtie")
        self.assertEqual(m["tests"][0]["assembler_id"], "stringtie")
        self.assertEqual(m["tests"][0]["gtf"], "/s_st.gtf")
        self.assertEqual(m["tests"][1]["id"], "ENCFF370NFS__isoquant")
        self.assertEqual(m["tests"][1]["assembler_id"], "isoquant")
        self.assertEqual(m["tests"][1]["gtf"], "/s_iq.gtf")

    def test_train_validation_aupr_columns_from_json(self) -> None:
        td = tempfile.mkdtemp(prefix="telos_tm_")
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        rep = Path(td) / "reports"
        rep.mkdir(parents=True)
        (rep / "train_metrics.json").write_text(
            '{"tss_rf": {"aupr": 0.1}, "tes_xgb": {"aupr": 0.2}, "stage2_rf": {"aupr": 0.3}}\n',
            encoding="utf-8",
        )
        cols = train_validation_aupr_columns(rep)
        self.assertAlmostEqual(cols["train_val_aupr_stage1_tss_rf"], 0.1)
        self.assertAlmostEqual(cols["train_val_aupr_stage1_tes_xgb"], 0.2)
        self.assertAlmostEqual(cols["train_val_aupr_stage2_rf"], 0.3)


if __name__ == "__main__":
    unittest.main()
