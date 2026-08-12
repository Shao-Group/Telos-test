# Telos RNA-seq data-generation (Snakemake)

Self-contained workflow: FASTQ + genome/annotation → Telos bundles
(`align/aln.sorted.bam`, `stringtie.gtf`, `isoquant.gtf` or `scallop2.gtf`,
gffcompare `.tmap`, `bundle_manifest.yaml`). Intermediates (IsoQuant work dir,
TPM TSV, gffcompare `.stats`/`.loci`/`.refmap`/etc.) are deleted after each step.

**Accessions, download links, and public-data policy:** see [`docs/DATA.md`](../docs/DATA.md).

## Requirements

- Snakemake ≥ 7 with conda support (`--use-conda`)
- Channels: `conda-forge`, `bioconda`

Pinned tool envs (under `envs/`):

| Env | Pins |
|-----|------|
| hisat2 | HISAT2 2.2.0, samtools |
| minimap2 | minimap2 2.20, samtools |
| stringtie | StringTie 3.0.3 |
| isoquant | IsoQuant 3.6.3 |
| scallop2 | Scallop2 1.1.2 |
| gffcompare | gffcompare 0.12.10 |

Optional absolute binary overrides go in `config.tools` (non-null values).

## Quick start

From the **repository root**:

```bash
snakemake -s workflow/Snakefile \
  --configfile workflow/config/config.example.yaml \
  --use-conda --cores 16
```

Dry-run (DAG only):

```bash
snakemake -s workflow/Snakefile \
  --configfile workflow/config/config.example.yaml \
  --use-conda -n
```

## Config

- [`config/config.example.yaml`](config/config.example.yaml) — `bundles_root`, `references`, threads, tool overrides
- [`config/samples.example.yaml`](config/samples.example.yaml) — sample list

Copy and edit for your runs:

```bash
cp workflow/config/config.example.yaml workflow/config/config.yaml
cp workflow/config/samples.example.yaml workflow/config/samples.yaml
# edit samples_file path inside config.yaml, then:
snakemake -s workflow/Snakefile --configfile workflow/config/config.yaml --use-conda --cores 16
```

### Sample fields

| Field | Description |
|-------|-------------|
| `sample_id` | Output directory name |
| `ref_id` | Key under `config.references` |
| `modality` | `sr` \| `cdna` \| `drna` \| `pacbio` |
| `fastq` | One path (long-read) or `[R1, R2]` (short-read) |
| `hisat2_strandness` | Optional per-sample override (`RF` / `FR` / …) |

Output layout:

```
{bundles_root}/{ref_id}/{sr|ont_cdna|ont_drna|pacbio}/{sample_id}/
  align/aln.sorted.bam(+ .bai)
  stringtie.gtf
  isoquant.gtf | scallop2.gtf
  stringtie.stringtie.gtf.tmap
  isoquant.isoquant.gtf.tmap | scallop2.scallop2.gtf.tmap
  bundle_manifest.yaml
```

## Modality → tools

- **cdna / drna / pacbio**: minimap2 → StringTie `-L` + IsoQuant → gffcompare
- **sr**: HISAT2 index (built here) → HISAT2 → StringTie + Scallop2 → gffcompare

## Annotation filter for gffcompare

`config.references.*.ref_gtf` may contain `gene` / CDS / UTR rows. Before gffcompare,
the workflow writes:

`genome/derived/<ref_id>/annotation.gffcompare.gtf`

keeping only `transcript` and `exon` lines that have both `gene_id` and `transcript_id`
(same logic as `scripts/run_all_fastq_bundles.sh` for RefSeq `*.gffcmp.gtf`). That file
is what gffcompare and `bundle_manifest.yaml` use.

Note: older `genome/derived/*/annotation.filtered.gtf` from `genome_prepare.sh` is a
**different** artifact (`gffread` for HISAT2/RSeQC), not this gffcompare filter.
