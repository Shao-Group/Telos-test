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
    parser = ArgumentParser()
    parser.add_argument('--config_folder', required=True, help='Path to the configuration file')
    parser.add_argument('--is_train', action='store_true', help='Is training chromosomes')
    args = parser.parse_args()

    suffix = "train" if args.is_train else "val"
    TRANSCRIPT_PLOT_FOLDER = f"plots_individual/transcript_pr_curves_extended/{suffix}"
    os.makedirs(TRANSCRIPT_PLOT_FOLDER, exist_ok=True)

    # Discover all config files and plot individually per config
    config_files = sorted([f for f in os.listdir(args.config_folder) if f.endswith('_config.pkl')])
    for cfg_file in config_files:
        cfg_path = os.path.join(args.config_folder, cfg_file)
        print("loading {}".format(cfg_path))
        try:
            config = load_config(cfg_path)
        except Exception as e:
            print(f"Skipping {cfg_file}: {e}")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_pr_curves_on_subplot(config, ax, title_prefix='Transcript PR', is_train=args.is_train)
        plt.tight_layout()
        out_name = os.path.splitext(cfg_file)[0].replace('_config', f'_{suffix}_transcript_pr') + '.pdf'
        out_path = os.path.join(TRANSCRIPT_PLOT_FOLDER, out_name)
        plt.savefig(out_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()