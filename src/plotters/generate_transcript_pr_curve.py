import os, sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from config import Config, load_config
from argparse import ArgumentParser

TRANSCRIPT_PLOT_FOLDER = "plots/transcript_pr_curves_compare"

def plot_pr_curves_on_subplot(config: Config, ax, title_prefix="", is_train=False):
    """
    Reads all ROC-like files in the specified folder, parses sensitivity (recall)
    and precision values, and plots Precision-Recall curves on the given subplot axis.
    """
    tool = config.data_name.split('_')[-1]

    # get auc map where key is the tool name and value is the auc in the two column csv
    auc_map = {}
    auc_file = config.auc_file_train if is_train else config.auc_file_val
    with open(auc_file, 'r') as f:
        for line in f:
            if line.startswith('label'):
                continue
            tool_name, auc = line.strip().split(',')
            auc_map[tool_name] = float(auc)

    # Find all files in the folder
    suffix = "train" if is_train else "val"
    file_paths = glob.glob(os.path.join(config.transcript_pr_data, f'*{suffix}.roc'))
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
        label = label.replace('_', '').replace(suffix, '')
        # print(label)
        label = label.replace('-', ' ').replace('.roc', '').title()
        label = label.replace('Updated Cov', '' ) # '\n(Updated Coverage)')
        label = label.replace('Aupr', 'AuPR')
        # label = label.split('_')[0]
        
        ax.plot(recalls, precisions, label=label)

    ax.set_xlabel('Recall (%)', fontsize=12)
    ax.set_ylabel('Precision (%)', fontsize=12)
    ax.set_title(f'{title_prefix} - {config.data_name.split("_")[0]}', fontsize=14)
    ax.legend(loc='lower left', fontsize=11 )
    ax.grid(False)

def main():
    # Load the configuration file
    os.makedirs(TRANSCRIPT_PLOT_FOLDER, exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--config_folder', required=True, help='Path to the configuration file')
    parser.add_argument('--is_train', action='store_true', help='Is training chromosomes')
    args = parser.parse_args()

    train_prefix = ["cDNA-ENCFF023EXJ", "cDNA-NA12878", "dRNA-ENCFF155CFF", "dRNA-ENCFF155CFF", "pacbio_ENCFF450VAU", "SRR307903"]
    test_prefix = ["cDNA-ENCFF263YFG", "cDNA-K562","dRNA-ENCFF771DIX", "dRNA-Hek293T", "pacbio_ENCFF694DIE", "SRR307911"]
    
    # Create pairs of datasets
    for i, (train_p, test_p) in enumerate(zip(train_prefix, test_prefix)):
        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        # fig.suptitle(f'PR Curves Comparison: {train_p} (Train) vs {test_p} (Test)', fontsize=18, fontweight="bold")
        
        # Training datasets - top row
        train_prefix1 = (train_p + "_stringtie")
        train_prefix2 = (train_p + "_isoquant") if not train_p.startswith("SRR") else (train_p + "_scallop2")
        train_config1 = load_config(os.path.join(args.config_folder, f"{train_prefix1}_config.pkl"))
        train_config2 = load_config(os.path.join(args.config_folder, f"{train_prefix2}_config.pkl"))
        
        plot_pr_curves_on_subplot(train_config1, axes[0, 0], "Training Dataset", args.is_train)
        plot_pr_curves_on_subplot(train_config2, axes[0, 1], "Training Dataset", args.is_train)
        
        # Testing datasets - bottom row
        test_prefix1 = (test_p + "_stringtie")
        test_prefix2 = (test_p + "_isoquant") if not test_p.startswith("SRR") else (test_p + "_scallop2")
        test_config1 = load_config(os.path.join(args.config_folder, f"{test_prefix1}_config.pkl"))
        test_config2 = load_config(os.path.join(args.config_folder, f"{test_prefix2}_config.pkl"))
        
        plot_pr_curves_on_subplot(test_config1, axes[1, 0], "Testing Dataset", args.is_train)
        plot_pr_curves_on_subplot(test_config2, axes[1, 1], "Testing Dataset", args.is_train)
        
        plt.tight_layout()
        plt.savefig(f'{TRANSCRIPT_PLOT_FOLDER}/pr_curves_compare_{train_p}_vs_{test_p}{"-train" if args.is_train else "-test"}.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Finished plotting PR curve comparison for {train_p} (train) vs {test_p} (test)")


if __name__ == "__main__":
    main()