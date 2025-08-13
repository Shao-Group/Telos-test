import pandas as pd
from install import install
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import sys
# sys.path.append('src')
from config import load_config, save_config
import os

# ----------- Directory paths -------------------
RNASEQ_DIR = "../../tools/rnaseqtools"
OUTPUT_DIR = "test_output"
GENCODE_REF = "data/GRCh38_gencode.gtf"
ENSEMBLE_REF = "data/GRCh38_ensembl.gtf"
PROJECT_CONFIG_DIR = "project_config"
MODEL_CONFIG_DIR = "project_config"
GFFCOMPARE_ENV = "gffcompare"
LOG_DIR = "logs"
STAGE1_MODELS = ["xgboost", "randomforest"]
# -----------------------------------------------

def generate_model_paths(model_folder):
    """
    Generate paths for pretrained models.
    """
    model_paths = []
    
    m1 = {
        "tss": os.path.join(model_folder, f"tss_randomforest_model.joblib"),
        "tes": os.path.join(model_folder, f"tes_randomforest_model.joblib"),
        "stage2": os.path.join(model_folder, f"xgboost_stage2_model.json"),
        "model_type": "randomforest"
    }
    model_paths.append(m1)
    m2 = {
        "tss": os.path.join(model_folder, f"tss_xgboost_model.json"),
        "tes": os.path.join(model_folder, f"tes_xgboost_model.json"),
        "stage2": os.path.join(model_folder, f"xgboost_stage2_model.json"),
        "model_type": "xgboost"
    }
    model_paths.append(m2)

    return model_paths

def test_with_pretrained(prefix, rnaseq_dir, output_dir, bam_file, gtf_file, ref_anno_gtf, tmap_file, pretrained_model_folder):
    output_dir = os.path.join(output_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{prefix}.log")
    with open(log_file, "w") as log:
        log.write(f"====== Processing {prefix}... ========\n")
        print(f"Installing {prefix}...")
        config_file_path = install(prefix, rnaseq_dir, output_dir, bam_file, gtf_file, ref_anno_gtf, tmap_file)
        print(f"Installation of {prefix} completed.")

        print(f"Extracting soft-clipped sequences for soft-clipped sequences for {prefix}...")
        p = subprocess.run(
            ["python", "src/extract_soft_clip_sequences.py", "--config", config_file_path],
            capture_output=True, text=True
        )
        log.write(p.stdout)
        log.write(p.stderr)
        log.flush()
        if p.returncode != 0:
            print(f"❌ Error in soft-clip sequence extraction for {prefix}: {p.stderr}")
            exit(1)
        else:
            print(f"✅ Soft-clip sequence extraction for {prefix} completed.")

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

        print(f"Setting up CNN model paths for {prefix}...")
        # Set up CNN model paths from the pretrained model folder
        

        # config_path = config_file_path
        # pretrained_folder = pretrained_model_folder

        cfg = load_config(config_file_path)

        # Set site CNN model paths from pretrained folder
        start_model_path = os.path.join(pretrained_model_folder, 'site_cnn_start_clips_model.pth')
        end_model_path = os.path.join(pretrained_model_folder, 'site_cnn_end_clips_model.pth')

        if os.path.exists(start_model_path):
            cfg.set_cnn_start_model_path(start_model_path)
            print(f'Set start clip model path: {start_model_path}')
        else:
            print(f'Warning: Start clip model not found at {start_model_path}')

        if os.path.exists(end_model_path):
            cfg.set_cnn_end_model_path(end_model_path)
            print(f'Set end clip model path: {end_model_path}')
        else:
            print(f'Warning: End clip model not found at {end_model_path}')

        save_config(config_file_path)
        print('Site CNN model paths configured successfully')


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
            

        print(f"Testing model for {prefix}...")
        # Generate paths for pretrained models
        model_paths = generate_model_paths(pretrained_model_folder)
        for model in model_paths:
            model_type = model["model_type"]
            tss_model = model["tss"]
            tes_model = model["tes"]
            stage2_model = model["stage2"]
        
            p = subprocess.run(
                ["python", "src/validate_with_pretrained.py", "--project-config", config_file_path,
                 "--model-config-folder", MODEL_CONFIG_DIR, "--pretrained_tss_model", tss_model,
                 "--pretrained_tes_model", tes_model, "--pretrained_stage2_model", stage2_model,
                 "--model_type", model_type ] ,
                capture_output=True, text=True
            )
            log.write(p.stdout)
            log.write(p.stderr)
            log.flush()
            if p.returncode != 0:
                print(f"❌ Error in model testing for {prefix}: {p.stderr}")
                exit(1)
            else:
                print(f"✅ Model testing for {prefix} with {model_type} completed.")

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

def process_dataset_tools_test(row_data, rnaseq_dir, output_dir):
    """Process both tools for a single dataset sequentially"""
    prefix, bam_file, gtf_file1, gtf_file2, tmap_file1, tmap_file2, ref, pretrained_folder1, pretrained_folder2 = row_data
    
    print(f"Processing {prefix}...")
    prefix1 = prefix + "_stringtie"
    prefix2 = (prefix + "_isoquant") if not prefix.startswith("SRR") else (prefix + "_scallop2")
    
    results = []
    
    # Process stringtie first
    try:
        test_with_pretrained(prefix1, rnaseq_dir, output_dir, bam_file, gtf_file1, ref, tmap_file1, pretrained_folder1)
        results.append(f"✅ {prefix1} completed successfully")
    except Exception as exc:
        results.append(f"❌ {prefix1} failed with exception: {exc}")
        return results  # Stop processing if first tool fails
    
    # Process isoquant/scallop2 second
    try:
        test_with_pretrained(prefix2, rnaseq_dir, output_dir, bam_file, gtf_file2, ref, tmap_file2, pretrained_folder2)
        results.append(f"✅ {prefix2} completed successfully")
    except Exception as exc:
        results.append(f"❌ {prefix2} failed with exception: {exc}")
                
    return results

def test_all_parallel():
    """Parallel version of test_all() function"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_configs = {
        "prefix": ["cDNA-K562","dRNA-Hek293T", "pacbio_ENCFF694DIE", "SRR307911"],
        "bam_file": ["data/cv_K562_cDNA/SGNex_K562_cDNA.sorted.bam",
                     "data/cv_Hek293T_dRNA/SGNex_Hek293T_dRNA.sorted.bam",
                     "data/cv_pacbio_ENCFF694DIE/ENCFF694DIE.sorted.bam",
                     "data/cv_SRR307911_hisat/hisat.sorted.bam"],
        "gtf_file1": ["data/cv_K562_cDNA/stringtie.gtf", 
                      "data/cv_Hek293T_dRNA/stringtie.gtf",
                      "data/cv_pacbio_ENCFF694DIE/stringtie.gtf",
                      "data/cv_SRR307911_hisat/stringtie.gtf"],
        "gtf_file2": ["data/cv_K562_cDNA/isoquant.gtf", 
                      "data/cv_Hek293T_dRNA/isoquant.gtf",
                      "data/cv_pacbio_ENCFF694DIE/isoquant.gtf",
                      "data/cv_SRR307911_hisat/scallop2.gtf"],
        "tmap_file1":  ["data/cv_K562_cDNA/stringtie.stringtie.gtf.tmap",
                        "data/cv_Hek293T_dRNA/stringtie.stringtie.gtf.tmap",
                       "data/cv_pacbio_ENCFF694DIE/stringtie.stringtie.gtf.tmap",
                       "data/cv_SRR307911_hisat/stringtie.stringtie.gtf.tmap"],
        "tmap_file2":  ["data/cv_K562_cDNA/isoquant.isoquant.gtf.tmap",
                        "data/cv_Hek293T_dRNA/isoquant.isoquant.gtf.tmap",
                       "data/cv_pacbio_ENCFF694DIE/isoquant.isoquant.gtf.tmap",
                       "data/cv_SRR307911_hisat/scallop2.scallop2.gtf.tmap"],
        "ref_anno_gtf": [GENCODE_REF, GENCODE_REF, GENCODE_REF, ENSEMBLE_REF],
        "pretrained_model_folder1" : ["train_output/cDNA-NA12878_stringtie/models", "train_output/dRNA-NA12878_stringtie/models",
                                      "train_output/pacbio_ENCFF450VAU_stringtie/models", "train_output/SRR307903_stringtie/models"],
        "pretrained_model_folder2" : ["train_output/cDNA-NA12878_isoquant/models", "train_output/dRNA-NA12878_isoquant/models",
                                      "train_output/pacbio_ENCFF450VAU_isoquant/models", "train_output/SRR307903_scallop2/models"]
    }
    
    test_configs_df = pd.DataFrame(test_configs)
    
    # Prepare data for parallel processing
    dataset_tasks = []
    for index, row in test_configs_df.iterrows():
        row_data = (row["prefix"], row["bam_file"], row["gtf_file1"], 
                   row["gtf_file2"], row["tmap_file1"], row["tmap_file2"], 
                   row["ref_anno_gtf"], row["pretrained_model_folder1"], 
                   row["pretrained_model_folder2"])
        dataset_tasks.append(row_data)
    
    # Process datasets in parallel (max 2 concurrent datasets to avoid overwhelming system)
    max_concurrent_datasets = min(4, len(dataset_tasks))
    print(f"🚀 Starting parallel testing with {max_concurrent_datasets} concurrent datasets...")
    
    with ProcessPoolExecutor(max_workers=max_concurrent_datasets) as executor:
        # Submit all dataset processing tasks
        future_to_dataset = {
            executor.submit(process_dataset_tools_test, task, RNASEQ_DIR, OUTPUT_DIR): task[0]
            for task in dataset_tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_dataset):
            dataset_name = future_to_dataset[future]
            try:
                results = future.result()
                print(f"📊 Dataset {dataset_name} testing completed:")
                for result in results:
                    print(f"  {result}")
            except Exception as exc:
                print(f"❌ Dataset {dataset_name} failed with exception: {exc}")

def test_all():
    # data/cv_Hek293T_dRNA  data/cv_K562_cDNA  data/cv_pacbio_ENCFF694DIE  data/cv_SRR307911_hisat
    # data/cv_Hek293T_dRNA/SGNex_Hek293T_dRNA.sorted.bam  data/cv_pacbio_ENCFF694DIE/ENCFF694DIE.sorted.bam
    # data/cv_K562_cDNA/SGNex_K562_cDNA.sorted.bam        data/cv_SRR307911_hisat/hisat.sorted.bam
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_configs = {
        "prefix": ["cDNA-K562","dRNA-Hek293T", "pacbio_ENCFF694DIE", "SRR307911"],
        "bam_file": ["data/cv_K562_cDNA/SGNex_K562_cDNA.sorted.bam",
                     "data/cv_Hek293T_dRNA/SGNex_Hek293T_dRNA.sorted.bam",
                     "data/cv_pacbio_ENCFF694DIE/ENCFF694DIE.sorted.bam",
                     "data/cv_SRR307911_hisat/hisat.sorted.bam"],
        "gtf_file1": ["data/cv_K562_cDNA/stringtie.gtf", 
                      "data/cv_Hek293T_dRNA/stringtie.gtf",
                      "data/cv_pacbio_ENCFF694DIE/stringtie.gtf",
                      "data/cv_SRR307911_hisat/stringtie.gtf"],
        "gtf_file2": ["data/cv_K562_cDNA/isoquant.gtf", 
                      "data/cv_Hek293T_dRNA/isoquant.gtf",
                      "data/cv_pacbio_ENCFF694DIE/isoquant.gtf",
                      "data/cv_SRR307911_hisat/scallop2.gtf"],
        "tmap_file1":  ["data/cv_K562_cDNA/stringtie.stringtie.gtf.tmap",
                        "data/cv_Hek293T_dRNA/stringtie.stringtie.gtf.tmap",
                       "data/cv_pacbio_ENCFF694DIE/stringtie.stringtie.gtf.tmap",
                       "data/cv_SRR307911_hisat/stringtie.stringtie.gtf.tmap"],
        "tmap_file2":  ["data/cv_K562_cDNA/isoquant.isoquant.gtf.tmap",
                        "data/cv_Hek293T_dRNA/isoquant.isoquant.gtf.tmap",
                       "data/cv_pacbio_ENCFF694DIE/isoquant.isoquant.gtf.tmap",
                       "data/cv_SRR307911_hisat/scallop2.scallop2.gtf.tmap"],
        "ref_anno_gtf": [GENCODE_REF, GENCODE_REF, GENCODE_REF, ENSEMBLE_REF],
        "pretrained_model_folder1" : ["train_output/cDNA-NA12878_stringtie/models", "train_output/dRNA-NA12878_stringtie/models",
                                      "train_output/pacbio_ENCFF450VAU_stringtie/models", "train_output/SRR307903_stringtie/models"],
        "pretrained_model_folder2" : ["train_output/cDNA-NA12878_isoquant/models", "train_output/dRNA-NA12878_isoquant/models",
                                      "train_output/pacbio_ENCFF450VAU_isoquant/models", "train_output/SRR307903_scallop2/models"]

    }
    test_configs_df = pd.DataFrame(test_configs)
    for index, row in test_configs_df.iterrows():
        prefix = row["prefix"]
        bam_file = row["bam_file"]
        gtf_file1 = row["gtf_file1"]
        gtf_file2 = row["gtf_file2"]
        tmap_file1 = row["tmap_file1"]
        tmap_file2 = row["tmap_file2"]
        ref = row["ref_anno_gtf"]
        pretrained_model_folder1 = row["pretrained_model_folder1"]
        pretrained_model_folder2 = row["pretrained_model_folder2"]

        print(f"Processing {prefix}...")
        prefix1 = (prefix + "_stringtie") 
        prefix2 = (prefix + "_isoquant") if not prefix.startswith("SRR") else (prefix + "_scallop2")
        # def install(prefix, rnaseq_dir, output_dir, bam_file, gtf_file, ref_anno_gtf, tmap_file):
        test_with_pretrained(prefix1, RNASEQ_DIR, OUTPUT_DIR, bam_file, gtf_file1, ref, tmap_file1, pretrained_model_folder1)
        test_with_pretrained(prefix2, RNASEQ_DIR, OUTPUT_DIR, bam_file, gtf_file2, ref, tmap_file2, pretrained_model_folder2)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Use the parallel version by default
    # Change to test_all() if you want sequential processing
    test_all_parallel()
    print("All tests passed!")

if __name__ == "__main__":
    main()