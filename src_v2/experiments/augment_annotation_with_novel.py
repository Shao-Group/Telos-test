"""
Build an augmented reference GTF by appending strict multi-assembler novel transcripts.

Strict consensus rule:
- Support must come from >= N distinct assemblers.
- Transcript endpoints (TSS and TES) must both match within tolerance.
- Tolerance is taken directly from stage1.training.site_label_tolerance_bp in config.

Novel exclusion rule:
- A consensus transcript is dropped only if an annotation transcript exists on the same
  chrom+strand with BOTH endpoints within tolerance.

Usage example:
  PYTHONPATH=src_v2 python src_v2/experiments/augment_annotation_with_novel.py \
    --bundle-manifest /abs/path/bundle_manifest.yaml \
    --annotation-gtf /abs/path/ref.gtf \
    --stage1-config src_v2/configs/stage1.defaults.yaml \
    --out-gtf runs/novel_ref/augmented_ref.gtf \
    --min-support 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from telos_v2.config_loader import get_nested, load_mapping_config
from telos_v2.gtf_attributes import parse_transcript_id

_GENE_ID_RE = re.compile(r'gene_id\s+"([^"]+)"')
@dataclass(frozen=True)
class TranscriptRecord:
    assembler_id: str
    source_gtf: Path
    transcript_id: str
    chrom: str
    strand: str
    tss: int
    tes: int
    lines: list[str]


@dataclass
class ConsensusGroup:
    chrom: str
    strand: str
    tss_center: float
    tes_center: float
    members: list[TranscriptRecord]
    assemblers: set[str]


def _replace_or_add_attr(attrs: str, key: str, value: str) -> str:
    patt = re.compile(rf'{re.escape(key)}\s+"[^"]*"')
    literal = f'{key} "{value}"'
    if patt.search(attrs):
        return patt.sub(literal, attrs, count=1)
    raw = attrs.strip()
    if raw and not raw.endswith(";"):
        raw = f"{raw};"
    return f"{raw} {literal};".strip()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read bundle manifests.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest (not a mapping): {path}")
    return data


def _select_assemblies(manifest: dict[str, Any], allowed: set[str] | None) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for ass in manifest.get("assemblies") or []:
        if not isinstance(ass, dict):
            continue
        aid = str(ass.get("assembler_id", "")).strip().lower()
        if not aid:
            continue
        if allowed is not None and aid not in allowed:
            continue
        gtf_raw = ass.get("gtf")
        if gtf_raw is None:
            continue
        out.append((aid, Path(str(gtf_raw))))
    if not out:
        raise ValueError("No assemblies found in manifest after filtering.")
    return out


def _read_transcripts_from_gtf(gtf: Path, assembler_id: str) -> list[TranscriptRecord]:
    rows_by_tx: dict[str, list[str]] = {}
    tx_meta: dict[str, tuple[str, str, int, int]] = {}
    with gtf.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 9:
                continue
            chrom, feature, start_s, end_s, strand, attrs = cols[0], cols[2], cols[3], cols[4], cols[6], cols[8]
            txid = parse_transcript_id(attrs)
            if not txid:
                continue
            rows_by_tx.setdefault(txid, []).append(line)
            if feature != "transcript":
                continue
            if strand not in {"+", "-"}:
                continue
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            tss = start + 1 if strand == "+" else end
            tes = end if strand == "+" else start + 1
            tx_meta[txid] = (chrom, strand, tss, tes)
    out: list[TranscriptRecord] = []
    for txid, lines in rows_by_tx.items():
        meta = tx_meta.get(txid)
        if meta is None:
            continue
        chrom, strand, tss, tes = meta
        out.append(
            TranscriptRecord(
                assembler_id=assembler_id,
                source_gtf=gtf,
                transcript_id=txid,
                chrom=chrom,
                strand=strand,
                tss=tss,
                tes=tes,
                lines=lines,
            )
        )
    return out


def _read_annotation_endpoints(annotation_gtf: Path) -> dict[tuple[str, str], list[tuple[int, int]]]:
    by_key: dict[tuple[str, str], list[tuple[int, int]]] = {}
    with annotation_gtf.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 9 or cols[2] != "transcript":
                continue
            chrom, start_s, end_s, strand = cols[0], cols[3], cols[4], cols[6]
            if strand not in {"+", "-"}:
                continue
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            tss = start + 1 if strand == "+" else end
            tes = end if strand == "+" else start + 1
            by_key.setdefault((chrom, strand), []).append((tss, tes))
    return by_key


def _fits_group(tx: TranscriptRecord, group: ConsensusGroup, tol: int) -> bool:
    return abs(tx.tss - group.tss_center) <= tol and abs(tx.tes - group.tes_center) <= tol


def _build_consensus_groups(transcripts: list[TranscriptRecord], tolerance_bp: int) -> list[ConsensusGroup]:
    groups: list[ConsensusGroup] = []
    for tx in transcripts:
        matched = False
        for g in groups:
            if tx.chrom != g.chrom or tx.strand != g.strand:
                continue
            if not _fits_group(tx, g, tolerance_bp):
                continue
            g.members.append(tx)
            g.assemblers.add(tx.assembler_id)
            n = len(g.members)
            g.tss_center = ((g.tss_center * (n - 1)) + tx.tss) / n
            g.tes_center = ((g.tes_center * (n - 1)) + tx.tes) / n
            matched = True
            break
        if matched:
            continue
        groups.append(
            ConsensusGroup(
                chrom=tx.chrom,
                strand=tx.strand,
                tss_center=float(tx.tss),
                tes_center=float(tx.tes),
                members=[tx],
                assemblers={tx.assembler_id},
            )
        )
    return groups


def _is_annotation_match(
    chrom: str,
    strand: str,
    tss: int,
    tes: int,
    ann_by_key: dict[tuple[str, str], list[tuple[int, int]]],
    tolerance_bp: int,
) -> bool:
    for atss, ates in ann_by_key.get((chrom, strand), []):
        if abs(tss - atss) <= tolerance_bp and abs(tes - ates) <= tolerance_bp:
            return True
    return False


def _choose_representative(members: list[TranscriptRecord]) -> TranscriptRecord:
    return sorted(members, key=lambda m: (m.assembler_id, m.transcript_id))[0]


def _rewrite_transcript_lines(lines: list[str], new_gene_id: str, new_transcript_id: str) -> list[str]:
    out: list[str] = []
    for line in lines:
        cols = line.split("\t")
        if len(cols) < 9:
            continue
        attrs = cols[8]
        attrs = _replace_or_add_attr(attrs, "gene_id", new_gene_id)
        attrs = _replace_or_add_attr(attrs, "transcript_id", new_transcript_id)
        cols[8] = attrs
        out.append("\t".join(cols))
    return out


def _write_tsv(path: Path, rows: list[dict[str, Any]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _canonical_novel_key(chrom: str, strand: str, tss: int, tes: int) -> str:
    return f"{chrom}|{strand}|{tss}|{tes}"


def _novel_ids_from_key(canonical_key: str) -> tuple[str, str]:
    digest = hashlib.sha1(canonical_key.encode("utf-8")).hexdigest()[:16].upper()
    return f"NOVEL_GENE_{digest}", f"NOVEL_TX_{digest}"


def build_augmented_reference(
    *,
    bundle_manifest: Path,
    annotation_gtf: Path,
    stage1_config: Path,
    out_gtf: Path,
    reports_dir: Path,
    min_support: int = 2,
    assemblers_csv: str = "",
) -> dict[str, Any]:
    manifest = _load_manifest(bundle_manifest.resolve())
    allow = {x.strip().lower() for x in assemblers_csv.split(",")} if assemblers_csv else None
    assemblies = _select_assemblies(manifest, allow)

    cfg_map = load_mapping_config(stage1_config.resolve())
    tol = int(get_nested(cfg_map, ["stage1", "training", "site_label_tolerance_bp"], 50))
    min_support = int(min_support)
    if min_support < 2:
        raise ValueError("min_support must be >= 2 for strict multi-assembler consensus.")

    all_tx: list[TranscriptRecord] = []
    for aid, gtf in assemblies:
        all_tx.extend(_read_transcripts_from_gtf(gtf.resolve(), aid))
    if not all_tx:
        raise ValueError("No transcript records parsed from selected assembly GTFs.")

    groups = _build_consensus_groups(all_tx, tol)
    ann_endpoints = _read_annotation_endpoints(annotation_gtf.resolve())

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    appended_lines: list[str] = []
    seen_novel_keys: set[str] = set()

    for g in groups:
        support = len(g.assemblers)
        rep = _choose_representative(g.members)
        row = {
            "chrom": g.chrom,
            "strand": g.strand,
            "tss_center": int(round(g.tss_center)),
            "tes_center": int(round(g.tes_center)),
            "support_assemblers": support,
            "assemblers": ",".join(sorted(g.assemblers)),
            "representative_assembler": rep.assembler_id,
            "representative_transcript_id": rep.transcript_id,
            "representative_gtf": str(rep.source_gtf),
        }
        canonical_key = _canonical_novel_key(
            row["chrom"], row["strand"], int(row["tss_center"]), int(row["tes_center"])
        )
        row["canonical_key"] = canonical_key
        if support < min_support:
            dropped.append({"reason": "support_below_min", **row})
            continue
        if _is_annotation_match(
            g.chrom, g.strand, int(round(g.tss_center)), int(round(g.tes_center)), ann_endpoints, tol
        ):
            dropped.append({"reason": "annotation_match", **row})
            continue
        if canonical_key in seen_novel_keys:
            dropped.append({"reason": "duplicate_novel_key", **row})
            continue
        new_gid, new_tid = _novel_ids_from_key(canonical_key)
        rewritten = _rewrite_transcript_lines(rep.lines, new_gid, new_tid)
        if not rewritten:
            dropped.append({"reason": "representative_has_no_lines", **row})
            continue
        seen_novel_keys.add(canonical_key)
        appended_lines.extend(rewritten)
        kept.append({"new_gene_id": new_gid, "new_transcript_id": new_tid, **row})

    out_gtf = out_gtf.resolve()
    out_gtf.parent.mkdir(parents=True, exist_ok=True)
    last_line_had_newline = True
    with annotation_gtf.resolve().open("r", encoding="utf-8", errors="replace") as src, out_gtf.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw in src:
            dst.write(raw)
            last_line_had_newline = raw.endswith("\n")
        if appended_lines and not last_line_had_newline:
            dst.write("\n")
        for line in appended_lines:
            dst.write(f"{line}\n")

    reports_dir = reports_dir.resolve()
    _write_tsv(
        reports_dir / "novel_kept.tsv",
        kept,
        [
            "new_gene_id",
            "new_transcript_id",
            "chrom",
            "strand",
            "tss_center",
            "tes_center",
            "support_assemblers",
            "assemblers",
            "representative_assembler",
            "representative_transcript_id",
            "representative_gtf",
            "canonical_key",
        ],
    )
    _write_tsv(
        reports_dir / "novel_dropped.tsv",
        dropped,
        [
            "reason",
            "chrom",
            "strand",
            "tss_center",
            "tes_center",
            "support_assemblers",
            "assemblers",
            "representative_assembler",
            "representative_transcript_id",
            "representative_gtf",
            "canonical_key",
        ],
    )

    return {
        "tolerance_bp": tol,
        "min_support": min_support,
        "selected_assemblers": ",".join(sorted(a for a, _ in assemblies)),
        "transcripts_total": len(all_tx),
        "consensus_groups": len(groups),
        "novel_kept": len(kept),
        "novel_dropped": len(dropped),
        "out_gtf": str(out_gtf),
        "novel_kept_tsv": str((reports_dir / "novel_kept.tsv").resolve()),
        "novel_dropped_tsv": str((reports_dir / "novel_dropped.tsv").resolve()),
    }


def run(args: argparse.Namespace) -> int:
    summary = build_augmented_reference(
        bundle_manifest=args.bundle_manifest,
        annotation_gtf=args.annotation_gtf,
        stage1_config=args.stage1_config,
        out_gtf=args.out_gtf,
        reports_dir=args.reports_dir,
        min_support=args.min_support,
        assemblers_csv=args.assemblers,
    )
    print("[telos_v2] augmented reference build complete")
    print(f"  tolerance_bp(from stage1 config)={summary['tolerance_bp']}")
    print(f"  min_support={summary['min_support']}")
    print(f"  selected_assemblers={summary['selected_assemblers']}")
    print(f"  transcripts_total={summary['transcripts_total']}")
    print(f"  consensus_groups={summary['consensus_groups']}")
    print(f"  novel_kept={summary['novel_kept']}")
    print(f"  novel_dropped={summary['novel_dropped']}")
    print(f"  out_gtf={summary['out_gtf']}")
    print(f"  novel_kept_tsv={summary['novel_kept_tsv']}")
    print(f"  novel_dropped_tsv={summary['novel_dropped_tsv']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Augment annotation GTF with strict multi-assembler novel transcripts.")
    p.add_argument("--bundle-manifest", type=Path, required=True, help="Path to bundle_manifest.yaml")
    p.add_argument("--annotation-gtf", type=Path, required=True, help="Base annotation GTF to augment")
    p.add_argument(
        "--stage1-config",
        type=Path,
        required=True,
        help="Stage1 YAML/JSON config; site_label_tolerance_bp is used for consensus merge tolerance",
    )
    p.add_argument("--out-gtf", type=Path, required=True, help="Output augmented reference GTF path")
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("runs/novel_ref/reports"),
        help="Directory for novel_kept.tsv and novel_dropped.tsv",
    )
    p.add_argument(
        "--min-support",
        type=int,
        default=2,
        help="Minimum distinct assembler support (strict mode; default: 2)",
    )
    p.add_argument(
        "--assemblers",
        type=str,
        default="",
        help="Optional comma-separated assembler_id allowlist (e.g. stringtie,isoquant,scallop2)",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
