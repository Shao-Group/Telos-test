"""
Fixed benchmark matrix: modality (data type), train/test annotation ref bundles, train sample per modality.

See docs/benchmark-matrix-convention.md for the human-readable contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# ref_id folder names under data/bundles/<ref_id>/<modality>/<sample>/
ANNOTATION_TO_REF_ID: dict[str, str] = {
    "refseq": "GRCh38_refseq_p14",
    "gencode": "GRCh38_gencode49",
    "ensembl": "GRCh38_ensembl115",
}

# Bundle modality folder (rnaseq_pipeline batch layout)
DATA_TYPE_TO_MODALITY: dict[str, str] = {
    "sr": "sr",
    "cdna": "ont_cdna",
    "drna": "ont_drna",
    "pacbio": "pacbio",
}

# Canonical train sample (directory name under the modality folder)
TRAIN_SAMPLE_BY_DATA_TYPE: dict[str, str] = {
    "sr": "SRR307903",
    "cdna": "ENCFF023EXJ",
    "drna": "NA12878-DirectRNA_All_Guppy_4.2.2",
    "pacbio": "ENCFF450VAU",
}

# Test rows: one benchmark row per (sample, assembler), in this order when all exist.
_ASSEMBLER_BENCHMARK_ORDER = ("stringtie", "isoquant", "scallop2")


def default_stage1_config_path() -> Path:
    """``src_v2/configs/stage1.defaults.yaml`` relative to this package."""
    telos_v2_dir = Path(__file__).resolve().parent
    return (telos_v2_dir.parent / "configs" / "stage1.defaults.yaml").resolve()


def resolve_bundles_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    env = os.environ.get("TELOS_BUNDLES_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for base in (Path.cwd(), *here.parents):
        cand = (base / "data" / "bundles").resolve()
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Could not locate data/bundles (walked up from cwd and package). "
        "Set TELOS_BUNDLES_ROOT or pass --bundles-root."
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read bundle_manifest.yaml") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest (not a mapping): {path}")
    return data


def _manifest_bam_ref(manifest: dict[str, Any]) -> tuple[Path, Path]:
    try:
        bam = Path(manifest["alignments"]["sorted_bam"])
        ref_gtf = Path(manifest["ref_annotation_gtf"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Manifest missing alignments/ref_annotation_gtf: {exc}") from exc
    return bam, ref_gtf


def _assembler_sort_key(assembler_id: str) -> tuple[int, str]:
    aid = assembler_id.strip().lower()
    try:
        return (_ASSEMBLER_BENCHMARK_ORDER.index(aid), aid)
    except ValueError:
        return (len(_ASSEMBLER_BENCHMARK_ORDER), aid)


def _assemblies_with_gffcompare_tmap(manifest: dict[str, Any]) -> list[tuple[str, Path, Path]]:
    """
    Each entry: (assembler_id, gtf, tmap). Sorted stringtie → isoquant → scallop2 → others;
    duplicate assembler_id in the manifest keeps the first in that order.
    """
    raw: list[tuple[str, Path, Path]] = []
    for ass in manifest.get("assemblies") or []:
        if not isinstance(ass, dict):
            continue
        aid = str(ass.get("assembler_id", "")).strip().lower()
        if not aid:
            continue
        gc = ass.get("gffcompare")
        if not isinstance(gc, dict):
            continue
        try:
            gtf = Path(ass["gtf"])
            tmap = Path(gc["tmap"])
        except (KeyError, TypeError):
            continue
        raw.append((aid, gtf, tmap))
    raw.sort(key=lambda t: _assembler_sort_key(t[0]))
    seen: set[str] = set()
    out: list[tuple[str, Path, Path]] = []
    for aid, gtf, tmap in raw:
        if aid in seen:
            continue
        seen.add(aid)
        out.append((aid, gtf, tmap))
    return out


def _stringtie_paths(manifest: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    """Training: bam, stringtie gtf, ref gtf, stringtie tmap (unchanged convention)."""
    bam, ref_gtf = _manifest_bam_ref(manifest)
    for ass in manifest.get("assemblies") or []:
        if not isinstance(ass, dict):
            continue
        if str(ass.get("assembler_id", "")).strip().lower() != "stringtie":
            continue
        try:
            gtf = Path(ass["gtf"])
            tmap = Path(ass["gffcompare"]["tmap"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Stringtie assembly incomplete in manifest: {exc}") from exc
        return bam, gtf, ref_gtf, tmap
    raise ValueError("No stringtie assembly in manifest assemblies[]")


def _manifest_path(bundles_root: Path, ref_id: str, modality: str, sample_id: str) -> Path:
    p = bundles_root / ref_id / modality / sample_id / "bundle_manifest.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"Missing bundle manifest: {p}")
    return p


def list_test_sample_ids(
    bundles_root: Path,
    ref_id: str,
    modality: str,
    train_sample_id: str,
) -> list[str]:
    """All sample directories with a manifest under ref/modality, except the fixed train sample."""
    base = bundles_root / ref_id / modality
    if not base.is_dir():
        raise FileNotFoundError(f"Bundle modality directory not found: {base}")
    out: list[str] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or entry.name == train_sample_id:
            continue
        if (entry / "bundle_manifest.yaml").is_file():
            out.append(entry.name)
    return out


def build_benchmark_yaml_mapping(
    *,
    data_type: str,
    train_annotation: str,
    test_annotation: str,
    bundles_root: Path,
    stage1_config: Path,
    train_outdir: Path | None = None,
) -> dict[str, Any]:
    """
    Build the mapping equivalent to a hand-written benchmark YAML (train + tests + execution + analysis).

    Train bundle uses (train_annotation, data_type, fixed train sample).
    Each test row uses (test_annotation, data_type, other samples).
    """
    dt = data_type.strip().lower()
    ta = train_annotation.strip().lower()
    test_a = test_annotation.strip().lower()

    if dt not in DATA_TYPE_TO_MODALITY:
        raise ValueError(f"Unknown data_type {data_type!r}; expected one of {sorted(DATA_TYPE_TO_MODALITY)}")
    if ta not in ANNOTATION_TO_REF_ID:
        raise ValueError(f"Unknown train annotation {train_annotation!r}; expected one of {sorted(ANNOTATION_TO_REF_ID)}")
    if test_a not in ANNOTATION_TO_REF_ID:
        raise ValueError(f"Unknown test annotation {test_annotation!r}; expected one of {sorted(ANNOTATION_TO_REF_ID)}")

    modality = DATA_TYPE_TO_MODALITY[dt]
    train_ref = ANNOTATION_TO_REF_ID[ta]
    test_ref = ANNOTATION_TO_REF_ID[test_a]
    train_sample = TRAIN_SAMPLE_BY_DATA_TYPE[dt]

    train_mf = _load_manifest(_manifest_path(bundles_root, train_ref, modality, train_sample))
    tbam, tgtf, tref, ttmap = _stringtie_paths(train_mf)

    test_ids = list_test_sample_ids(bundles_root, test_ref, modality, train_sample)
    if not test_ids:
        raise ValueError(
            f"No test bundles found under {bundles_root / test_ref / modality} "
            f"(excluding train sample {train_sample!r})."
        )

    stage1_path = str(stage1_config.resolve())

    tests: list[dict[str, Any]] = []
    for sid in test_ids:
        mf = _load_manifest(_manifest_path(bundles_root, test_ref, modality, sid))
        bam, ref_gtf = _manifest_bam_ref(mf)
        per_asm = _assemblies_with_gffcompare_tmap(mf)
        if not per_asm:
            raise ValueError(
                f"No assemblies with gffcompare.tmap in manifest for test sample {sid!r} "
                f"({bundles_root / test_ref / modality / sid})"
            )
        for aid, gtf, tmap in per_asm:
            tests.append(
                {
                    "id": f"{sid}__{aid}",
                    "assembler_id": aid,
                    "bam": str(bam),
                    "gtf": str(gtf),
                    "ref_gtf": str(ref_gtf),
                    "tmap": str(tmap),
                    "config": stage1_path,
                }
            )

    train_block: dict[str, Any] = {
        "mode": "run",
        "bam": str(tbam),
        "gtf": str(tgtf),
        "ref_gtf": str(tref),
        "tmap": str(ttmap),
        "config": stage1_path,
    }
    if train_outdir is not None:
        train_block["outdir"] = str(train_outdir.resolve())

    return {
        "generated_by": "telos_v2.benchmark_matrix",
        "schema_version": 2,
        "train": train_block,
        "tests": tests,
        "execution": {"stop_on_error": False},
        "analysis": {
            "enabled": True,
            "pr_vs_baseline": {
                "enabled": True,
                # gtfcuff subcommand: roc <tmap> <ref_multi_exon_count> <cov|TPM|FPKM>
                "measure": "cov",
                "plot": True,
                # Like legacy generate_roc_data: filter assembly + ref to validation chromosomes
                # (stage1.training.split_policy in the stage1 config). Set false to use full GTFs.
                "filter_validation_chroms": True,
                # Optional: one gtfformat chrom list for both GTFs; null = derive per-file lists.
                "chromosomes_file": None,
                # When true, write per-backend PR curve TSV + small summary CSV under reports/pr/.
                "save_pr_tables": True,
                "gffcompare_bin": "/datadisk1/ixk5174/tools/gffcompare-0.12.10.Linux_x86_64/gffcompare",
                "gtfcuff_bin": None,
            },
        },
    }


def benchmark_mapping_to_yaml_text(mapping: dict[str, Any]) -> str:
    import yaml  # type: ignore

    return yaml.safe_dump(
        mapping,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
