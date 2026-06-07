"""
RefSeq-novel evaluation entrypoint (deprecated runner).

This experiment is **analysis-only** on existing cross-annotation benchmark outputs.
Use:

  PYTHONPATH=src_v2 python src_v2/experiments/evaluate_refseq_novel_cross_annotation.py \\
    --root runs/cross_annotation_repro

See ``runs/refseq_novel_eval/reports/novel_eval_README.txt`` after a run.
"""

from __future__ import annotations

import sys

from evaluate_refseq_novel_cross_annotation import main as eval_main


def main() -> int:
    print(
        "[telos_v2] refseq_train_gencode_ensembl_novel.py delegates to "
        "evaluate_refseq_novel_cross_annotation.py",
        flush=True,
    )
    return eval_main()


if __name__ == "__main__":
    raise SystemExit(main())
