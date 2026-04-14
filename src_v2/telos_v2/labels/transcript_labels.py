from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_tmap_labels(tmap_path: Path) -> pd.DataFrame:
    """
    Legacy parity: gffcompare .tmap — class code '=' is positive.
    Same as src/ml_utils.load_tmap_labels.
    """
    df = pd.read_csv(tmap_path, sep="\t", comment="#", header=0)
    df = df[["qry_id", "class_code"]].rename(columns={"qry_id": "transcript_id"})
    df["label"] = (df["class_code"] == "=").astype(int)
    return df[["transcript_id", "label"]]
