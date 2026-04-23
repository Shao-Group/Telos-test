"""
Generate benchmark YAML from a **fixed experimental matrix** and optionally run the benchmark.

The matrix is parameterized by short-read vs long-read **data type**, which **reference annotation**
bundle to use for training vs testing, and a on-disk **bundle tree** (see repo docs for layout).
Training always uses one canonical sample per data type; tests enumerate all other samples under the
test annotation’s bundle path, crossed with every assembler in each sample’s ``bundle_manifest.yaml``
that has gffcompare outputs.

The CLI subcommand ``benchmark-matrix`` dispatches to :func:`run_benchmark_matrix`; the implementation
also lives here so tests can call :func:`build_benchmark_yaml_mapping` without subprocess overhead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from telos_v2.config_loader import default_stage1_config_path
from telos_v2.config_models import BenchmarkIO

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


def resolve_bundles_root(explicit: Path | None) -> Path:
    """
    Locate the root directory that contains per-reference, per-modality bundle folders.

    Resolution order:

    1. If ``explicit`` is not ``None``, return its resolved path (caller override).
    2. Else if environment variable ``TELOS_BUNDLES_ROOT`` is non-empty, use that path.
    3. Else walk ``Path.cwd()`` then each ancestor of this file’s directory; return the first
       ``<base>/data/bundles`` that exists **and** is a directory.

    Raises:
        FileNotFoundError: No directory matched step 3 and no explicit path was given.

    Returns:
        Absolute :class:`~pathlib.Path` to the bundles root (e.g. ``.../data/bundles``).
    """
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
    """
    Parse ``bundle_manifest.yaml`` into a plain ``dict``.

    Uses PyYAML ``safe_load``. Empty or all-comment files become ``{}``. The top-level value must
    be a mapping or :class:`ValueError` is raised.

    Raises:
        RuntimeError: PyYAML is not installed.
        ValueError: File is not a YAML mapping.
    """
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to read bundle_manifest.yaml") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest (not a mapping): {path}")
    return data


def _manifest_bam_ref(manifest: dict[str, Any]) -> tuple[Path, Path]:
    """
    Extract aligned-read BAM and reference GTF paths from a loaded manifest dict.

    Expects keys ``alignments.sorted_bam`` and ``ref_annotation_gtf`` with string values convertible
    to paths. Used for both train and test rows so BAM and ref annotation stay consistent with the
    bundle that produced the assembly GTF.

    Raises:
        ValueError: Missing keys or wrong structure.
    """
    try:
        bam = Path(manifest["alignments"]["sorted_bam"])
        ref_gtf = Path(manifest["ref_annotation_gtf"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Manifest missing alignments/ref_annotation_gtf: {exc}") from exc
    return bam, ref_gtf


def _assembler_sort_key(assembler_id: str) -> tuple[int, str]:
    """
    Sort key for assembler ids: stringtie, isoquant, scallop2 first (in that order), then others alphabetically.

    Unknown assemblers sort after the three named ones. The integer primary key is the index in
    ``_ASSEMBLER_BENCHMARK_ORDER`` or ``len(order)`` if not listed.
    """
    aid = assembler_id.strip().lower()
    try:
        return (_ASSEMBLER_BENCHMARK_ORDER.index(aid), aid)
    except ValueError:
        return (len(_ASSEMBLER_BENCHMARK_ORDER), aid)


def _assemblies_with_gffcompare_tmap(manifest: dict[str, Any]) -> list[tuple[str, Path, Path]]:
    """
    List assemblies that have both a ``gtf`` path and ``gffcompare.tmap`` in the manifest.

    Iterates ``manifest["assemblies"]``, skips non-dicts or entries without ``assembler_id`` or
    ``gffcompare`` block. Sorts by :func:`_assembler_sort_key`, then **deduplicates** by
    ``assembler_id`` keeping the first occurrence after sort.

    Returns:
        ``(assembler_id, assembly_gtf, tmap_path)`` tuples for benchmark test rows.
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
    """
    Return (bam, stringtie_gtf, ref_gtf, stringtie_tmap) for the **training** bundle.

    Searches ``assemblies`` for ``assembler_id == "stringtie"`` (case-insensitive). Requires
    ``gtf`` and ``gffcompare.tmap`` under that entry, and uses :func:`_manifest_bam_ref` for BAM/ref.

    Raises:
        ValueError: No stringtie assembly or incomplete paths.
    """
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


def _train_pool_paths(manifest: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    """
    Additional train assemblies (excluding stringtie) with paired ``gtf`` and ``gffcompare.tmap``.
    """
    gtfs: list[Path] = []
    tmaps: list[Path] = []
    for aid, gtf, tmap in _assemblies_with_gffcompare_tmap(manifest):
        if aid == "stringtie":
            continue
        gtfs.append(gtf)
        tmaps.append(tmap)
    return gtfs, tmaps


def _manifest_path(bundles_root: Path, ref_id: str, modality: str, sample_id: str) -> Path:
    """
    Absolute path to ``bundle_manifest.yaml`` for one sample under the bundle tree.

    Raises:
        FileNotFoundError: The file does not exist (caller gets a clear path in the message).
    """
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
    """
    Enumerate sample directory names under ``bundles_root/ref_id/modality`` that have a manifest.

    Skips non-directories and skips ``train_sample_id`` (the canonical training sample for that
    modality, which must not appear twice as train and test). Only includes entries where
    ``bundle_manifest.yaml`` exists. Sorted by directory name for stable benchmark ordering.

    Raises:
        FileNotFoundError: The modality directory does not exist.
    """
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
    Build the in-memory dict that serializes to a full benchmark YAML (train, tests, execution, analysis).

    Steps:

    1. Normalize ``data_type``, ``train_annotation``, ``test_annotation`` and map to ``modality`` and
       ``ref_id`` folder names via :data:`DATA_TYPE_TO_MODALITY` and :data:`ANNOTATION_TO_REF_ID`.
    2. Load the **train** manifest at the fixed train sample for that data type under the **train**
       ref tree; extract stringtie BAM/GTF/ref/tmap via :func:`_stringtie_paths`.
    3. List **test** sample ids under the **test** ref tree (excluding the train sample id).
    4. For each test sample, load its manifest, get BAM/ref from :func:`_manifest_bam_ref`, and for
       each assembly with gffcompare tmap append one test dict with ``id`` ``{sample}__{assembler}``.
    5. Attach the same resolved ``stage1_config`` path string to train and every test (unless ``train_outdir`` adds only to the train block).

    Raises:
        ValueError: Unknown annotation/data type, no test bundles, or manifest missing gffcompare data.

    Returns:
        Mapping suitable for :func:`benchmark_mapping_to_yaml_text` and then :func:`run_benchmark`.
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
    pool_gtfs, pool_tmaps = _train_pool_paths(train_mf)

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
        "gtf_pool": [str(p) for p in pool_gtfs],
        "tmap_pool": [str(p) for p in pool_tmaps],
        "config": stage1_path,
    }
    if train_outdir is not None:
        train_block["outdir"] = str(train_outdir.resolve())

    return {
        "generated_by": "telos_v2.benchmark.matrix",
        "schema_version": 3,
        "train": train_block,
        "tests": tests,
        "execution": {"stop_on_error": False},
        "analysis": {
            "enabled": True,
            "benchmark_mode": "minimal",
            "debug": {"keep_pr_work": False},
            "pr_vs_baseline": {
                "enabled": True,
                "measure": "cov",
                "plot": True,
                "filter_validation_chroms": True,
                "chromosomes_file": None,
                "save_pr_tables": True,
                "gffcompare_bin": "/datadisk1/shared/tools/gffcompare/gffcompare-0.11.2.Linux_x86_64/gffcompare",
            },
        },
    }


def benchmark_mapping_to_yaml_text(mapping: dict[str, Any]) -> str:
    """
    Serialize a benchmark mapping to multi-line YAML text (no flow style, stable key order).

    ``sort_keys=False`` preserves insertion order so ``train`` stays before ``tests`` as built.
    """
    import yaml  # type: ignore

    return yaml.safe_dump(
        mapping,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def run_benchmark_matrix(
    *,
    data_type: str,
    train_annotation: str,
    test_annotation: str,
    outdir: Path,
    bundles_root: Path | None = None,
    stage1_config: Path | None = None,
) -> int:
    """
    CLI/backend entry: materialize ``reports/generated_benchmark.yaml`` and invoke the benchmark orchestrator.

    1. Resolves bundle root via :func:`resolve_bundles_root`.
    2. Picks Stage I config from ``stage1_config`` or
       :func:`~telos_v2.config_loader.default_stage1_config_path`; returns exit code ``2`` if missing.
    3. Builds the mapping with :func:`build_benchmark_yaml_mapping`, with train output under
       ``outdir/train``.
    4. Writes YAML to ``outdir/reports/generated_benchmark.yaml`` with a short header comment block.
    5. Calls :func:`~telos_v2.benchmark.orchestrator.run_benchmark` with ``BenchmarkIO`` pointing at
       that file and ``outdir``.

    Returns:
        Process exit code from :func:`~telos_v2.benchmark.orchestrator.run_benchmark`, or ``2`` on
        setup failures (missing config, bad bundles, etc.).
    """
    root = resolve_bundles_root(bundles_root)
    stage1 = stage1_config if stage1_config is not None else default_stage1_config_path()
    if not stage1.is_file():
        print(f"[telos_v2] benchmark-matrix: stage1 config not found: {stage1}")
        return 2

    outdir = outdir.resolve()
    train_out = outdir / "train"
    try:
        mapping = build_benchmark_yaml_mapping(
            data_type=data_type,
            train_annotation=train_annotation,
            test_annotation=test_annotation,
            bundles_root=root,
            stage1_config=stage1,
            train_outdir=train_out,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[telos_v2] benchmark-matrix: {exc}")
        return 2

    text = benchmark_mapping_to_yaml_text(mapping)
    reports = outdir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    cfg_path = reports / "generated_benchmark.yaml"
    header = (
        "# Auto-generated by: python -m telos_v2.cli benchmark-matrix (...)\n"
        "# Tests: one row per (sample, assembler) with id <sample>__<assembler_id> "
        "(stringtie, isoquant long-read, scallop2 short-read when present in bundle_manifest).\n"
        "# Transcript PR (when analysis.pr_vs_baseline.enabled): "
        "gtfformat filter-chrom on BOTH assembly GTF and ref_gtf to validation chromosomes only "
        "(same split as stage1.training.split_policy unless chromosomes_file is set) → "
        "update-transcript-cov → gffcompare -r filtered ref → gtfcuff roc/auc.\n"
        "# Optional chromosomes_file overrides auto filter (one shared list for both GTFs).\n"
        "# save_pr_tables: false avoids extra PR curve tables; AUC still in benchmark_summary.csv.\n"
        "# Each test must include ref_gtf (this generator fills it from bundle_manifest.yaml).\n"
        "# Set gffcompare_bin below if gffcompare is not on PATH.\n\n"
    )
    cfg_path.write_text(header + text, encoding="utf-8")
    print(f"[telos_v2] benchmark-matrix: wrote {cfg_path}")

    from telos_v2.benchmark.orchestrator import run_benchmark

    return run_benchmark(BenchmarkIO(config=cfg_path, outdir=outdir))


__all__ = [
    "ANNOTATION_TO_REF_ID",
    "DATA_TYPE_TO_MODALITY",
    "TRAIN_SAMPLE_BY_DATA_TYPE",
    "benchmark_mapping_to_yaml_text",
    "build_benchmark_yaml_mapping",
    "list_test_sample_ids",
    "resolve_bundles_root",
    "run_benchmark_matrix",
]
