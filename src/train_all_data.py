import pandas as pd
from install import install
import subprocess
import os

RNASEQ_DIR = "../../tools/rnaseqtools"
OUTPUT_DIR = "train_output"
GENCODE_REF = "data/GRCh38_gencode.gtf"
ENSEMBLE_REF = "data/GRCh38_ensembl.gtf"
PROJECT_CONFIG_DIR = "project_config"
MODEL_CONFIG_DIR = "project_config"
GFFCOMPARE_ENV = "gffcompare"
LOG_DIR = "logs"

def train_data(prefix, rnaseq_dir, output_dir, bam_file, gtf_file, ref_anno_gtf, tmap_file):
    output_dir = os.path.join(output_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{prefix}.log")
    with open(log_file, "w") as log:
        log.write(f"====== Processing {prefix}... ========\n")
        print(f"Installing {prefix}...")
        config_file_path = install(prefix, rnaseq_dir, output_dir, bam_file, gtf_file, ref_anno_gtf, tmap_file)
        print(f"Installation of {prefix} completed.")

        print(f"Running extract_features.py for {prefix}...")
        p = subprocess.run(
            ["python", "src/extract_features.py", "--config", config_file_path],
             capture_output=True, text=True, check=True
        )
        log.write(p.stdout)
        log.write(p.stderr)
        log.flush()
        if p.returncode != 0:
            print(f"❌ Error in feature extraction for {prefix}: {p.stderr}")
            exit(1)
        else:
            print(f"✅ Feature extraction for {prefix} completed.")

        print(f"Labeling candidates for {prefix}...")
        p = subprocess.run(
            ["python", "src/label_candidates.py", "--config", config_file_path],
            capture_output=True, text=True
        )
        log.write(p.stdout)
        log.write(p.stderr)
        log.flush()
        if p.returncode != 0:
            print(f"❌ Error in candidate labeling for {prefix}: {p.stderr}")
            exit(1)
        else:
            print(f"✅ Candidate labeling for {prefix} completed.")

        print(f"Training model for {prefix}...")
        p = subprocess.run(
            ["python", "src/train_model.py", "--project-config", config_file_path, "--model-config-folder", MODEL_CONFIG_DIR],
            capture_output=True, text=True
        )
        log.write(p.stdout)
        log.write(p.stderr)
        log.flush()
        if p.returncode != 0:
            print(f"❌ Error in model training for {prefix}: {p.stderr}")
            exit(1)
        else:
            print(f"✅ Model training for {prefix} completed.")

        print(f"Generating ROC data for {prefix}...")
        p = subprocess.run(
            ["python", "src/generate_roc_data.py", "--project-config", config_file_path, "--gffcompare-env", GFFCOMPARE_ENV],
            capture_output=True, text=True
        )
        log.write(p.stdout)
        log.write(p.stderr)
        log.flush()
        if p.returncode != 0:
            print(f"❌ Error in ROC data generation for {prefix}: {p.stderr}")
            exit(1)
        else:
            print(f"✅ ROC data generation for {prefix} completed.")


def init():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_configs = {
        "prefix": ["cDNA-NA1278","dNA-NA1278", "pacbio_ENCFF450VAU", "SRR307903"],
        "bam_file": ["data/nanopore_cDNA_NA12878/NA12878-cDNA.sorted.bam",
                     "data/nanopore_dRNA_NA12878/NA12878-DirectRNA.sorted.bam",
                     "data/pacbio_ENCFF450VAU/ENCFF450VAU.sorted.bam",
                     "data/SRR307903_hisat/hisat.sorted.bam"],
        "gtf_file1": ["data/nanopore_cDNA_NA12878/stringtie.gtf", 
                      "data/nanopore_dRNA_NA12878/stringtie.gtf",
                      "data/pacbio_ENCFF450VAU/stringtie.gtf",
                      "data/SRR307903_hisat/stringtie.gtf"],
        "gtf_file2": ["data/nanopore_cDNA_NA12878/isoquant.gtf", 
                      "data/nanopore_dRNA_NA12878/isoquant.gtf",
                      "data/pacbio_ENCFF450VAU/isoquant.gtf",
                      "data/SRR307903_hisat/scallop2.gtf"],
        "tmap_file1":  ["data/nanopore_cDNA_NA12878/stringtie.stringtie.gtf.tmap",
                        "data/nanopore_dRNA_NA12878/stringtie.stringtie.gtf.tmap",
                       "data/pacbio_ENCFF450VAU/stringtie.stringtie.gtf.tmap",
                       "data/SRR307903_hisat/stringtie.stringtie.gtf.tmap"],
        "tmap_file2":  ["data/nanopore_cDNA_NA12878/isoquant.isoquant.gtf.tmap",
                        "data/nanopore_dRNA_NA12878/isoquant.isoquant.gtf.tmap",
                       "data/pacbio_ENCFF450VAU/isoquant.isoquant.gtf.tmap",
                       "data/SRR307903_hisat/scallop2.scallop2.gtf.tmap"],
        "ref_anno_gtf": [GENCODE_REF, GENCODE_REF, GENCODE_REF, ENSEMBLE_REF]

    }
    train_configs_df = pd.DataFrame(train_configs)
    for index, row in train_configs_df.iterrows():
        prefix = row["prefix"]
        bam_file = row["bam_file"]
        gtf_file1 = row["gtf_file1"]
        gtf_file2 = row["gtf_file2"]
        tmap_file1 = row["tmap_file1"]
        tmap_file2 = row["tmap_file2"]
        ref = row["ref_anno_gtf"]

        print(f"Processing {prefix}...")
        prefix1 = (prefix + "_stringtie") 
        prefix2 = (prefix + "_isoquant") if not prefix.startswith("SRR") else (prefix + "_scallop2")
        # def install(prefix, rnaseq_dir, output_dir, bam_file, gtf_file, ref_anno_gtf, tmap_file):
        train_data(prefix1, RNASEQ_DIR, OUTPUT_DIR, bam_file, gtf_file1, ref, tmap_file1)
        train_data(prefix2, RNASEQ_DIR, OUTPUT_DIR, bam_file, gtf_file2, ref, tmap_file2)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    init()
    

if __name__ == "__main__":
    main()