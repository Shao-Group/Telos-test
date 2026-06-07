"""
V1 plotter conventions (``src/plotters/plot_stage1_aupr_barplot.py``).

- Two tool panels: ``stringtie`` (all StringTie) and ``other`` (Scallop2 on SR, IsoQuant on long-read).
- 18 datasets per panel: 9 short-read + 3 cDNA + 3 dRNA + 3 PacBio.
- Pretty x-axis labels from ``dataset_to_run_accession``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# v1 group order (plot_stage1_aupr_barplot.classify_group / group_order)
V1_GROUP_ORDER: tuple[str, ...] = ("pacbio", "cdna", "drna", "srr")

V1_GROUP_TITLE: dict[str, str] = {
    "pacbio": "PacBio",
    "cdna": "cDNA",
    "drna": "dRNA",
    "srr": "Short Reads",
}

# Canonical dataset keys per group (matches v1 barplot ordering).
V1_CANONICAL_DATASET_ORDER: tuple[str, ...] = (
    "pacbio_ENCFF694DIE",
    "pacbio_ENCFF563QZR",
    "pacbio_ENCFF370NFS",
    "cDNA-K562",
    "cDNA-ENCFF263YFG",
    "cDNA-NA12878",
    "dRNA-Hek293T",
    "dRNA-ENCFF771DIX",
    "dRNA-ENCFF155CFF",
    "SRR307911",
    "SRR545695",
    "SRR315334",
    "SRR534307",
    "SRR545723",
    "SRR315323",
    "SRR534319",
    "SRR534291",
    "SRR387661",
)

_V1_SORT_INDEX: dict[str, int] = {k: i for i, k in enumerate(V1_CANONICAL_DATASET_ORDER)}

# (data_type, test_id base without __assembler) -> v1 dataset key
_V2_BASE_TO_V1_KEY: dict[tuple[str, str], str] = {
    ("cdna", "ENCFF263YFG"): "cDNA-ENCFF263YFG",
    ("cdna", "NA12878-cDNA_All_Guppy_4.2.2"): "cDNA-NA12878",
    ("cdna", "SGNex_K562_cDNA_replicate1_run3"): "cDNA-K562",
    ("drna", "ENCFF155CFF"): "dRNA-ENCFF155CFF",
    ("drna", "ENCFF771DIX"): "dRNA-ENCFF771DIX",
    ("drna", "SGNex_Hek293T_directRNA_replicate1_run1"): "dRNA-Hek293T",
    ("pacbio", "ENCFF370NFS"): "pacbio_ENCFF370NFS",
    ("pacbio", "ENCFF563QZR"): "pacbio_ENCFF563QZR",
    ("pacbio", "ENCFF694DIE"): "pacbio_ENCFF694DIE",
}

_FALLBACK_DATASET_TO_PRETTY: dict[str, str] = {
    "dRNA-ENCFF155CFF": "ENCFF155CFF",
    "dRNA-ENCFF771DIX": "ENCFF771DIX",
    "dRNA-NA12878": "dRNA NA12878",
    "dRNA-Hek293T": "ERR6053059",
    "cDNA-K562": "ERR6053079",
    "cDNA-NA12878": "cDNA NA12878",
    "cDNA-ENCFF263YFG": "ENCFF263YFG",
    "cDNA-ENCFF023EXJ": "ENCFF023EXJ",
    "pacbio_ENCFF450VAU": "ENCFF450VAU",
    "pacbio_ENCFF694DIE": "ENCFF694DIE",
    "pacbio_ENCFF563QZR": "ENCFF563QZR",
    "pacbio_ENCFF370NFS": "ENCFF370NFS",
    "SRR307903": "SRR307903",
    "SRR307911": "SRR307911",
    "SRR545695": "SRR545695",
    "SRR315334": "SRR315334",
    "SRR534307": "SRR534307",
    "SRR545723": "SRR545723",
    "SRR315323": "SRR315323",
    "SRR534319": "SRR534319",
    "SRR534291": "SRR534291",
    "SRR387661": "SRR387661",
}


def _load_dataset_to_run_accession() -> dict[str, str]:
    try:
        repo = Path(__file__).resolve().parents[4]
        plotters = repo / "src" / "plotters"
        if str(plotters) not in sys.path:
            sys.path.insert(0, str(plotters))
        from calculate_true_false_stats import dataset_to_run_accession  # type: ignore

        return dict(dataset_to_run_accession)
    except Exception:
        return dict(_FALLBACK_DATASET_TO_PRETTY)


DATASET_TO_RUN_ACCESSION: dict[str, str] = _load_dataset_to_run_accession()

# Cross-annotation bar panels (v1 tool_group names).
CROSS_ANNOTATION_TOOL_PANELS: tuple[str, ...] = ("stringtie", "other")

PANEL_DISPLAY: dict[str, str] = {
    "stringtie": "StringTie",
    "other": "IsoQuant\nScallop2",
}


def test_id_base(test_id: str) -> str:
    tid = str(test_id).strip()
    if "__" in tid:
        return tid.rsplit("__", 1)[0]
    return tid


def v1_dataset_key(data_type: str, test_id: str) -> str:
    """Map v2 (data_type, test_id) to v1 ``dataset`` key used in plotters CSVs."""
    dt = str(data_type).strip().lower()
    base = test_id_base(test_id)
    if dt == "sr":
        return base
    return _V2_BASE_TO_V1_KEY.get((dt, base), base)


def v1_group_for_datatype(data_type: str) -> str:
    dt = str(data_type).strip().lower()
    if dt == "sr":
        return "srr"
    if dt in V1_GROUP_ORDER:
        return dt
    return "other"


def pretty_dataset_label(v1_key: str) -> str:
    return DATASET_TO_RUN_ACCESSION.get(v1_key, v1_key)


def row_matches_tool_panel(row, panel: str) -> bool:
    """
    v1 ``tool_group`` filtering (plot_stage1_aupr_barplot / gather_auc_results).

    - stringtie: assembler stringtie for every data type.
    - other: scallop2 on short-read; isoquant on cDNA / dRNA / PacBio.
    """
    asm = str(row.get("assembler_id", row.get("assembler", ""))).strip().lower()
    dt = str(row.get("data_type", "")).strip().lower()
    if panel == "stringtie":
        return asm == "stringtie"
    if panel == "other":
        if dt == "sr":
            return asm == "scallop2"
        return asm == "isoquant"
    return False


def sort_index_v1(row) -> int:
    key = row.get("v1_dataset_key") or v1_dataset_key(
        str(row.get("data_type", "")), str(row.get("test_id", ""))
    )
    return _V1_SORT_INDEX.get(key, 999)


def annotate_v1_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add v1_dataset_key, v1_group, dataset_label columns."""
    out = df.copy()
    out["v1_dataset_key"] = [
        v1_dataset_key(str(r.get("data_type", "")), str(r.get("test_id", "")))
        for _, r in out.iterrows()
    ]
    out["v1_group"] = [v1_group_for_datatype(str(r.get("data_type", ""))) for _, r in out.iterrows()]
    out["dataset_label"] = [pretty_dataset_label(k) for k in out["v1_dataset_key"]]
    out["_v1_ord"] = out.apply(sort_index_v1, axis=1)
    return out.sort_values(["_v1_ord", "dataset_label"], kind="stable").drop(columns="_v1_ord")


def expected_panel_size() -> int:
    return len(V1_CANONICAL_DATASET_ORDER)
