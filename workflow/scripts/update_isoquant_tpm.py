#!/usr/bin/env python3
"""Write IsoQuant TPM values into transcript ``cov`` attributes in a GTF."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

_TX_ID = re.compile(r'transcript_id "([^"]+)"')
_COV = re.compile(r'\bcov(?:erage)? "[^"]*"')


def load_tpm(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            out[parts[0].strip()] = float(parts[1])
        except ValueError:
            continue
    return out


def set_cov(attrs: str, value: float) -> str:
    lit = f'cov "{value:.6f}"'
    if _COV.search(attrs):
        return _COV.sub(lit, attrs, count=1)
    attrs = attrs.rstrip()
    if attrs and not attrs.endswith(";"):
        attrs += ";"
    return f"{attrs} {lit};" if attrs else f"{lit};"


def update_gtf(gtf_path: Path, tpm_path: Path, out_gtf: Path) -> None:
    tpm = load_tpm(tpm_path)
    out_gtf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(out_gtf.parent), delete=False, suffix=".gtf"
    ) as tmp:
        tmp_path = Path(tmp.name)
        with gtf_path.open(encoding="utf-8", errors="replace") as src:
            for raw in src:
                cols = raw.rstrip("\n").split("\t")
                if len(cols) >= 9 and cols[2] == "transcript":
                    m = _TX_ID.search(cols[8])
                    if m and m.group(1) in tpm:
                        cols[8] = set_cov(cols[8], tpm[m.group(1)])
                        tmp.write("\t".join(cols) + "\n")
                        continue
                tmp.write(raw)
    tmp_path.replace(out_gtf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtf", type=Path, required=True)
    ap.add_argument("--tpm", type=Path, required=True)
    ap.add_argument("--out-gtf", type=Path, required=True)
    args = ap.parse_args()
    update_gtf(args.gtf, args.tpm, args.out_gtf)


if __name__ == "__main__":
    main()
