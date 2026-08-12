#!/usr/bin/env python
import pandas as pd
from pathlib import Path

cache = Path("runs/telos_stage1_feature_cache/b01475da9d5d4eadfcc8")
print("files", sorted(p.name for p in cache.iterdir()))
df = pd.read_pickle(cache / "df_all.pkl")
print("shape", df.shape)
print("has up_down_stream_ratio", "up_down_stream_ratio" in df.columns)
print("has upstream_downstream_ratio", "upstream_downstream_ratio" in df.columns)
for c in ("up_down_stream_ratio", "upstream_downstream_ratio"):
    if c in df.columns:
        s = df[c]
        print(c, "null", float(s.isna().mean()), "describe", s.describe().to_dict())
