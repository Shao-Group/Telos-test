# Public data path — accessions + Snakemake bundle rebuild

**Policy:** do **not** publish BAM/GTF bundle tarballs. Public reproduction downloads
raw reads by accession, prepares references, then builds Telos bundles with Snakemake.

Internal development may still point `TELOS_BUNDLES_ROOT` at the frozen Telos-test
tree (read-only). That is a convenience path, not the public distribution path.

## End-to-end flow

```text
1. Download FASTQ by accession  →  fastq/<modality>/…
2. Obtain genome + annotation   →  genome/…
3. Snakemake workflow           →  data/bundles/<ref>/<modality>/<sample>/…
4. telos-repro / benchmark      →  runs/…  (compare to goldens/)
```

## Short-read (sr) — NCBI SRA

Paper cross-annotation SR grid (train = `SRR307903`; others = held-out tests):

| Accession | Role |
|-----------|------|
| SRR307903 | Shared train sample |
| SRR307911 | Test |
| SRR315323 | Test |
| SRR315334 | Test |
| SRR387661 | Test |
| SRR534291 | Test |
| SRR534307 | Test |
| SRR534319 | Test |
| SRR545695 | Test |
| SRR545723 | Test |

Portal: https://www.ncbi.nlm.nih.gov/sra (search by `SRR…`).

Helper (sra-toolkit `prefetch` + `fasterq-dump`):

```bash
# Discovers SRR* under ENCODE10_DIR and writes paired FASTQ under fastq/sr/
ENCODE10_DIR=/path/with/SRR*dirs \
  OUT_DIR=fastq/sr \
  PREFETCH_MAX_SIZE=100G \
  ./scripts/data/download_sra_encode10_fastq.sh
```

Expected local layout after dump (paired):

```text
fastq/sr/SRR307903/SRR307903.sra_1.fastq
fastq/sr/SRR307903/SRR307903.sra_2.fastq
```

## Long-read / ENCODE / other public files

Human long-read bundles used in paper grids (sample_id = local folder / file stem):

| Modality | Sample IDs |
|----------|------------|
| ont_cdna | ENCFF023EXJ, ENCFF263YFG, NA12878-cDNA_All_Guppy_4.2.2, SGNex_K562_cDNA_replicate1_run3 |
| ont_drna | ENCFF155CFF, ENCFF771DIX, NA12878-DirectRNA_All_Guppy_4.2.2, SGNex_Hek293T_directRNA_replicate1_run1 |
| pacbio | ENCFF370NFS, ENCFF450VAU, ENCFF563QZR, ENCFF694DIE |

- **ENCFF\*** — ENCODE portal: https://www.encodeproject.org/ (search file accession; download FASTQ).
- **NA12878-*** / **SGNex_*** — public long-read community datasets; place as
  `fastq/cdna/…` or `fastq/drna/…` matching the sample_id filename used in
  `workflow/config/samples.yaml`.

### Mouse / tissue extras (optional paper experiments)

| Set | Accessions / files (under `fastq/`) |
|-----|--------------------------------------|
| mouse sr | ENCFF006WNS, ENCFF521IDK |
| mouse cdna | ENCFF683TBO |
| mouse drna | ENCFF765AEC |
| mouse pacbio | ENCFF874VSI |
| tissue cdna | SRR31255649 |
| tissue drna | SRR36400176 |
| tissue pacbio | ENCFF185VYD |
| tissue sr | ENCFF431RAQ (R1), ENCFF008OVI (R2) — paired |

## Reference genomes

Configure under `workflow/config/config.yaml` → `references`:

| `ref_id` | Typical use |
|----------|-------------|
| `GRCh38_gencode49` | Primary human paper grids |
| `GRCh38_ensembl115` | Cross-annotation |
| `GRCh38_refseq_p14` | Cross-annotation / novel eval |

Obtain primary assembly FASTA + matching annotation GTF from GENCODE / Ensembl / NCBI
RefSeq as appropriate; point `genome_fasta` and `ref_gtf` at local paths. Do not
commit large FASTA/GTF blobs to git.

## Snakemake: FASTQ → bundles

From the **repository root** of this clone:

```bash
cp workflow/config/config.example.yaml workflow/config/config.yaml
cp workflow/config/samples.example.yaml workflow/config/samples.yaml
# edit samples.yaml (accessions → local fastq paths) and config.yaml references

snakemake -s workflow/Snakefile \
  --configfile workflow/config/config.yaml \
  --use-conda --cores 16
```

Dry-run:

```bash
snakemake -s workflow/Snakefile --configfile workflow/config/config.yaml --use-conda -n
```

Details: [`workflow/README.md`](../workflow/README.md).

Outputs:

```text
{bundles_root}/{ref_id}/{sr|ont_cdna|ont_drna|pacbio}/{sample_id}/
  align/aln.sorted.bam(+.bai)
  stringtie.gtf
  isoquant.gtf | scallop2.gtf
  *.tmap
  bundle_manifest.yaml
```

Then set:

```bash
export TELOS_BUNDLES_ROOT=$PWD/data/bundles
export TELOS_REPRO_BACKEND=telos
```

and run experiments (`telos-repro run …`).

## Notes on numeric match

Rebuilding bundles from FASTQ can introduce aligner/assembler version drift versus
any prior reference runs. Treat rebuilt bundles as the public reproduction path;
pin tool versions where you need tight numeric agreement.
