"""Unit tests for RefSeq-relative novelty labeling."""

from __future__ import annotations

from pathlib import Path

from telos_v2.labels.novel_vs_refseq import (
    RefSeqTranscriptIndex,
    TranscriptModel,
    build_refseq_transcript_index,
    classify_transcripts_in_gtf,
    filter_gtf_by_transcript_ids,
)


def _write_gtf(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_identical_and_novel_chain(tmp_path: Path) -> None:
    ref = tmp_path / "ref.gtf"
    qry = tmp_path / "qry.gtf"
    # Two-exon transcript on chr1 +
    base = [
        'chr1\tRef\texon\t100\t200\t.\t+\t.\tgene_id "G1"; transcript_id "TX1";',
        'chr1\tRef\texon\t300\t400\t.\t+\t.\tgene_id "G1"; transcript_id "TX1";',
    ]
    _write_gtf(ref, base)
    _write_gtf(
        qry,
        base
        + [
            'chr1\tRef\texon\t500\t600\t.\t+\t.\tgene_id "G2"; transcript_id "TX2";',
            'chr1\tRef\texon\t700\t800\t.\t+\t.\tgene_id "G2"; transcript_id "TX2";',
        ],
    )
    idx = build_refseq_transcript_index(ref)
    df = classify_transcripts_in_gtf(qry, idx)
    by_id = df.set_index("transcript_id")
    assert int(by_id.loc["TX1", "is_novel_vs_refseq"]) == 0
    assert by_id.loc["TX1", "match_type"] == "identical"
    assert int(by_id.loc["TX2", "is_novel_vs_refseq"]) == 1


def test_strand_mismatch_is_novel(tmp_path: Path) -> None:
    ref = tmp_path / "ref.gtf"
    qry = tmp_path / "qry.gtf"
    _write_gtf(
        ref,
        [
            'chr1\tRef\texon\t100\t200\t.\t+\t.\tgene_id "G1"; transcript_id "TX1";',
            'chr1\tRef\texon\t300\t400\t.\t+\t.\tgene_id "G1"; transcript_id "TX1";',
        ],
    )
    _write_gtf(
        qry,
        [
            'chr1\tRef\texon\t100\t200\t.\t-\t.\tgene_id "G1"; transcript_id "TX1m";',
            'chr1\tRef\texon\t300\t400\t.\t-\t.\tgene_id "G1"; transcript_id "TX1m";',
        ],
    )
    idx = build_refseq_transcript_index(ref)
    df = classify_transcripts_in_gtf(qry, idx)
    assert df.iloc[0]["match_type"] == "strand_mismatch"
    assert int(df.iloc[0]["is_novel_vs_refseq"]) == 1


def test_filter_gtf_keeps_novel_ids(tmp_path: Path) -> None:
    src = tmp_path / "asm.gtf"
    _write_gtf(
        src,
        [
            'chr1\tRef\texon\t100\t200\t.\t+\t.\tgene_id "G1"; transcript_id "KEEP";',
            'chr1\tRef\texon\t300\t400\t.\t+\t.\tgene_id "G1"; transcript_id "KEEP";',
            'chr1\tRef\texon\t500\t600\t.\t+\t.\tgene_id "G2"; transcript_id "DROP";',
        ],
    )
    out = tmp_path / "out.gtf"
    n = filter_gtf_by_transcript_ids(src, {"KEEP"}, out)
    assert n == 2
    text = out.read_text(encoding="utf-8")
    assert "KEEP" in text
    assert "DROP" not in text


def test_index_classify_direct() -> None:
    idx = RefSeqTranscriptIndex()
    tx = TranscriptModel("T", "chr1", "+", [(100, 200), (300, 400)])
    idx.add(tx)
    match, nov = idx.classify(tx)
    assert match == "identical"
    assert nov == 0
