#!/usr/bin/env python
import json
import re
from pathlib import Path

import joblib

p = Path(
    "/datadisk1/ixk5174/project_repo/Telos-test/runs/cross_annotation_repro/"
    "_cross_annotation_shared_train/sr__train_gencode/models/stage1_tss_rf_model.joblib"
)
b = joblib.load(p)
print("bundle type", type(b))
if isinstance(b, dict):
    print("keys", sorted(b.keys()))
    fn = b.get("feature_names")
    print("n_features", len(fn) if fn else None)
    for name in fn or []:
        if "stream" in name.lower() or "entropy" in name.lower():
            print("S1_FEATURE", name)

v2 = Path("src/telos_v2/features/stage1.py").read_text()
te = Path("/datadisk1/ixk5174/project_repo/Telos/src/telos/features/stage1.py").read_text()
for label, t in [("v2", v2), ("telos", te)]:
    for m in re.findall(r'features\["([^"]*stream[^"]*)"\]', t):
        print(label, "assigns", m)

p2 = Path(
    "/datadisk1/ixk5174/project_repo/Telos-test/runs/cross_annotation_repro/"
    "_cross_annotation_shared_train/sr__train_gencode/models/stage2_feature_names_rf.json"
)
fn2 = json.loads(p2.read_text())
print("stage2 n", len(fn2))
for n in fn2:
    if "stream" in n.lower():
        print("S2", n)

# stage2_train build_stage2_inference_frame differences: which site columns flow in
print("--- stage2.py function defs ---")
for label, path in [
    ("v2", "src/telos_v2/features/stage2.py"),
    ("telos", "/datadisk1/ixk5174/project_repo/Telos/src/telos/features/stage2.py"),
]:
    text = Path(path).read_text()
    for fn in re.findall(r"^def (\w+)", text, flags=re.M):
        print(label, fn)
