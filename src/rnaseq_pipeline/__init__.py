"""
RNA-seq preprocessing for Telos: FASTQ → aligned BAM → assemblies → GFFCompare.

This package mirrors the workflows in ``generate-assemblies.sh`` (nanopore),
``generate-assemblies-pacbio.sh`` (PacBio), and ``generate_sr_assemblies.sh``
(short reads + Scallop2).

**Layout**

- ``config.py`` — paths, conda env, thread counts, tool defaults (edit here first).
- ``exec_env.py`` — run commands inside ``conda run -n <env>``.
- ``fastq_input.py`` — discover FASTQ pairs / single-end reads.
- ``align.py`` — minimap2 (long) or HISAT2 (short) → sorted, indexed BAM.
- ``assemble.py`` — StringTie3, IsoQuant, Scallop2, optional gtfformat.
- ``gffcompare_run.py`` — reference comparison and ``gffcmp/`` outputs.
- ``pipeline.py`` — high-level ``RnaseqAssemblyPipeline``.
- ``cli.py`` — command-line entry.
- ``hisat2_index.py`` — ``hisat2-build`` (+ optional GTF splice extraction).

**Build HISAT2 index (once per genome)**

.. code-block:: bash

   PYTHONPATH=src python -m rnaseq_pipeline build-hisat2-index \\
     --ref-fasta genome/GRCh38.primary_assembly.genome.fa \\
     --index-prefix /path/to/index/GRCh38_primary \\
     --ref-gtf genome/gencode.annotation.gtf

Then pass ``--hisat2-index /path/to/index/GRCh38_primary`` to ``run`` with ``--preset short_paired``.

**Typical use (full run)**

.. code-block:: bash

   cd /path/to/Telos-test
   PYTHONPATH=src python -m rnaseq_pipeline run \\
     --preset nanopore --nanopore-library cdna \\
     --fastq fastq/sample.fastq \\
     --work-dir data/my_sample \\
     --ref-fasta genome/GRCh38.primary_assembly.genome.fa \\
     --ref-gtf genome/gencode.basic.annotation.gtf

Outputs include ``stringtie.gtf``, ``isoquant.gtf`` or ``scallop2.gtf``, and ``gffcmp/``.

**Batch Telos data bundles** (all samples under ``fastq/``, multiple reference annotations): repo script
``scripts/run_all_fastq_bundles.sh`` (after ``scripts/prepare_all_genome_annotations.sh``). Writes
``bundle_manifest.yaml`` per sample via ``scripts/write_bundle_manifest.py``.

Presets: ``nanopore`` (use ``--nanopore-library drna`` for direct RNA), ``pacbio``,
``short_paired`` (StringTie + Scallop2; pass ``--hisat2-index``).
"""

from rnaseq_pipeline.config import LibraryPreset, NanoporeLibraryKind, RnaseqToolConfig
from rnaseq_pipeline.hisat2_index import build_hisat2_index
from rnaseq_pipeline.pipeline import RnaseqAssemblyPipeline, PipelineResult

__all__ = [
    "LibraryPreset",
    "NanoporeLibraryKind",
    "RnaseqToolConfig",
    "RnaseqAssemblyPipeline",
    "PipelineResult",
    "build_hisat2_index",
]
