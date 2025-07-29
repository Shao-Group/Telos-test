import os, sys
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from config import load_config, Config


# —— USER SETTINGS ——
data_names = [
    "cDNA-NA12878","dRNA-NA12878", "pacbio_ENCFF450VAU", "SRR307903"
]
name_dict = {
    "cDNA-NA12878" : "NA12878 cDNA",
    "dRNA-NA12878" : "NA12878 dRNA",
    "pacbio_ENCFF450VAU": "ENCFF450VAU",
    "SRR307903" : "SRR307903"
}


STAGE1_PR_PLOT_FOLDER = "plots/"

# ———————

# create a 4×2 grid: 4 datasets × 2 site-types
def plot_pr(config_folder):
    site_types = ['tss', 'tes']
    fig, axes = plt.subplots(
        nrows=len(data_names),
        ncols=2,
        figsize=(12, 3 * len(data_names)),
        sharex=False,
        sharey=False
    )

    for i, name in enumerate(data_names):
        tools =  ["stringtie", "scallop2"] if name.startswith("SRR") else ["stringtie", "isoquant"]

        configs  = {
            tools[0] : load_config(os.path.join(config_folder, f"{name}_{tools[0]}_config.pkl")),
            tools[1] : load_config(os.path.join(config_folder, f"{name}_{tools[1]}_config.pkl"))
        }
        for j, site in enumerate(site_types):
            ax = axes[i, j]
            
            pr_files  = {
                tools[0] : os.listdir(configs[tools[0]].pr_data_dir), 
                tools[1] : os.listdir(configs[tools[1]].pr_data_dir)
            }
            
            
            for tool, files in pr_files.items():
                for fname in files:
                    assert fname.endswith("_pr_data.csv")
                    if site not in fname:
                        continue

                    site_model = fname.replace("_pr_data.csv", "")
                    pr_df = pd.read_csv(os.path.join(configs[tool].pr_data_dir, fname))

                    label = tool + " " +  fname.split("_")[1]
                    # full PR curve
                    ax.plot(
                        pr_df["recall"],
                        pr_df["precision"],
                        label=label.title()
                    )

                # baseline: recall=1, precision = fraction of label==1
                labeled_data_file = configs[tool].tss_labeled_file if site == "tss" else configs[tool].tes_labeled_file
                baseline_df = pd.read_csv(labeled_data_file)
                with open (configs[tool].validation_chromosomes_file, "r") as fp:
                    val_chrom = [line.rstrip("\n") for line in fp]

                baseline_prec = baseline_df.loc[baseline_df["chrom"].isin(val_chrom),"label"].mean()
                ax.scatter(
                    1.0,
                    baseline_prec,
                    marker="o",
                    s=40,
                    edgecolor="k",
                    zorder=10
                )

            # formatting per‐subplot
            ax.set_title(f"{name_dict[name]} - {site.upper()}", fontsize=12)
            if i == len(data_names) - 1:
                ax.set_xlabel("Recall", fontsize=10)
            if j == 0:
                ax.set_ylabel("Precision", fontsize=10)
            # ax.tick_params(labelsize= nine)  # increase tick labels if you wish
            ax.legend(fontsize=12, loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(STAGE1_PR_PLOT_FOLDER,"all_pr_curves_grid.pdf"), dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {STAGE1_PR_PLOT_FOLDER}/all_pr_curves_grid.pdf")


def main():
    os.makedirs(STAGE1_PR_PLOT_FOLDER, exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--config_folder', required=True, help='Path to the configuration file')
    args = parser.parse_args()
    plot_pr(args.config_folder)

if __name__ == "__main__":
    main()