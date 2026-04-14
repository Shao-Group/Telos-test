"""Validation-seqname extraction for PR chromosome filtering."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from telos_v2.models.chrom_split import seqnames_on_validation_split_from_gtf


def _mini_gtf(lines: list[str]) -> Path:
    td = tempfile.mkdtemp(prefix="telos_chrom_")
    p = Path(td) / "t.gtf"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


class TestSeqnamesValidationSplit(unittest.TestCase):
    def test_chr1_10_train_policy_marks_chr11_as_validation(self) -> None:
        gtf = _mini_gtf(
            [
                'chr1\t.\tgene\t1\t100\t.\t+\t.\tgene_id "a";',
                'chr11\t.\tgene\t1\t100\t.\t+\t.\tgene_id "b";',
            ]
        )
        self.addCleanup(lambda: shutil.rmtree(gtf.parent, ignore_errors=True))
        names = seqnames_on_validation_split_from_gtf(gtf, (1, 10))
        self.assertEqual(names, ["chr11"])

    def test_nc_and_chr_equivalence_for_train_range(self) -> None:
        gtf = _mini_gtf(
            [
                'NC_000001.11\t.\tgene\t1\t100\t.\t+\t.\tgene_id "a";',
                'NC_000011.10\t.\tgene\t1\t100\t.\t+\t.\tgene_id "b";',
            ]
        )
        self.addCleanup(lambda: shutil.rmtree(gtf.parent, ignore_errors=True))
        names = seqnames_on_validation_split_from_gtf(gtf, (1, 10))
        self.assertEqual(names, ["NC_000011.10"])


if __name__ == "__main__":
    unittest.main()
