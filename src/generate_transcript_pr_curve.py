import os
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from config import Config, load_config
from argparse import ArgumentParser

TRANSCRIPT_PLOT_FOLDER = "plots/transcript_pr_curves"

def plot_pr_curves(config:Config):
    """
    Reads all ROC-like files in the specified folder, parses sensitivity (recall)
    and precision values, and plots Precision-Recall curves for each file.
    """
    tool = config.data_name.split('_')[-1]
    plt.figure(figsize=(6, 4))

    # get auc map where key is the tool name and value is the auc in the two column csv
    auc_map = {}
    with open(config.auc_file, 'r') as f:
        for line in f:
            if line.startswith('label'):
                continue
            tool_name, auc = line.strip().split(',')
            auc_map[tool_name] = float(auc)

    # Find all files in the folder
    file_paths = glob.glob(os.path.join(config.transcript_pr_data, '*.roc'))
    file_paths = sorted(file_paths, reverse=True)
    for file_path in file_paths:
        recalls = []
        precisions = []
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('ROC:'):
                    # Extract sensitivity and precision values
                    rec_match = re.search(r'sensitivity\s*=\s*([0-9.]+)', line)
                    prec_match = re.search(r'precision\s*=\s*([0-9.]+)', line)
                    if rec_match and prec_match:
                        recalls.append(float(rec_match.group(1)))
                        precisions.append(float(prec_match.group(1)))
        
        # Plot the curve for this file
        label =  os.path.basename(file_path) 
        label = label + " \n[ AuPR:" + f"{auc_map[label.split('.')[0].split('-updated-cov')[0]]/10000.0:.4f}" + " ]"
        label = tool + " " + label
        label = label.replace('-', ' ').replace('.roc', '').title()
        label = label.replace('Updated Cov', '' ) # '\n(Updated Coverage)')
        label = label.replace('Aupr', 'AuPR')
        
        plt.plot(recalls, precisions, label=label)


    plt.xlabel('Recall (%)')
    plt.ylabel('Precision (%)')
    # plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'{TRANSCRIPT_PLOT_FOLDER}/pr_curves_compare_{config.data_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')



def main():
    # Load the configuration file
    os.makedirs(TRANSCRIPT_PLOT_FOLDER, exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--config_folder', required=True, help='Path to the configuration file')
    args = parser.parse_args()

    train_prefix = ["cDNA-NA1278","dNA-NA1278", "pacbio_ENCFF450VAU", "SRR307903"]
    test_prefix = ["cDNA-K562","dRNA-Hek293T", "pacbio_ENCFF694DIE", "SRR307911"]
    for p in train_prefix + test_prefix:
        prefix1 = (p + "_stringtie")
        prefix2 = (p + "_isoquant") if not p.startswith("SRR") else (p + "_scallop2")
        config1 = load_config(os.path.join(args.config_folder, f"{prefix1}_config.pkl"))
        config2 = load_config(os.path.join(args.config_folder, f"{prefix2}_config.pkl"))

        plot_pr_curves(config1)
        plot_pr_curves(config2)

        print(f"Finished plotting PR curve for {p}")


if __name__ == "__main__":
    main()