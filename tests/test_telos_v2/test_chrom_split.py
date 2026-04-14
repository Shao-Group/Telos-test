"""Chromosome split policy: adaptive autosome indexing across naming schemes."""

from __future__ import annotations

import unittest

import pandas as pd

from telos_v2.models.chrom_split import (
    parse_split_policy,
    primary_autosome_number,
    split_train_val_masks,
)


class TestPrimaryAutosomeNumber(unittest.TestCase):
    def test_gencode_ucsc(self) -> None:
        self.assertEqual(primary_autosome_number("chr1"), 1)
        self.assertEqual(primary_autosome_number("CHR22"), 22)
        self.assertIsNone(primary_autosome_number("chrX"))
        self.assertIsNone(primary_autosome_number("chrM"))

    def test_ensembl_bare(self) -> None:
        self.assertEqual(primary_autosome_number("1"), 1)
        self.assertEqual(primary_autosome_number("22"), 22)
        self.assertIsNone(primary_autosome_number("X"))

    def test_refseq_nc(self) -> None:
        self.assertEqual(primary_autosome_number("NC_000001.11"), 1)
        self.assertEqual(primary_autosome_number("NC_000010.11"), 10)
        self.assertEqual(primary_autosome_number("NC_000022.11"), 22)
        self.assertIsNone(primary_autosome_number("NC_000023.11"))  # X
        self.assertIsNone(primary_autosome_number("NC_012920.1"))  # MT

    def test_chr_prefixed_refseq_from_normalize(self) -> None:
        self.assertEqual(primary_autosome_number("chrNC_000001.11"), 1)

    def test_ncbi_contig_prefixes_not_autosomes(self) -> None:
        self.assertIsNone(primary_autosome_number("NT_187361.1"))
        self.assertIsNone(primary_autosome_number("chrNT_187361.1"))
        self.assertIsNone(primary_autosome_number("NW_003315933.1"))
        self.assertIsNone(primary_autosome_number("NS_12345.1"))

    def test_ucsc_patch_and_un_random(self) -> None:
        self.assertIsNone(primary_autosome_number("GL000191.1"))
        self.assertIsNone(primary_autosome_number("KI270733.1"))
        self.assertIsNone(primary_autosome_number("chrUn_KI270302v1"))


class TestSplitTrainValMasks(unittest.TestCase):
    def test_refseq_names_policy_chr1_10(self) -> None:
        df = pd.DataFrame(
            {
                "chrom": [
                    "NC_000001.11",
                    "NC_000010.11",
                    "NC_000011.11",
                    "NC_000023.11",
                ],
                "x": [1, 2, 3, 4],
            }
        )
        lo, hi = parse_split_policy("chr1-10")
        tr, va = split_train_val_masks(df, (lo, hi))
        self.assertListEqual(tr.tolist(), [True, True, False, False])
        self.assertListEqual(va.tolist(), [False, False, True, True])

    def test_mixed_ensembl_and_chr(self) -> None:
        df = pd.DataFrame({"chrom": ["1", "chr5", "GL000191.1", "15"]})
        tr, va = split_train_val_masks(df, parse_split_policy("1-10"))
        self.assertTrue(tr[0] and tr[1] and not tr[2] and not tr[3])
        self.assertTrue(va[2] and va[3])


if __name__ == "__main__":
    unittest.main()
