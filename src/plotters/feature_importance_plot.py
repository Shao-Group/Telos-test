import os, sys
from typing import List
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
import numpy as np

def get_tools(data_name):
    low = data_name.lower()
    if data_name.startswith("SRR"):
        return ["stringtie", "scallop2"]
    else:
        return ["stringtie", "isoquant"]

# —— USER SETTINGS ——
data_names = [
    "cDNA-NA1278","dNA-NA1278", "pacbio_ENCFF450VAU", "SRR307903"
]
name_dict = {
    "cDNA-NA1278" : "NA12878 cDNA",
    "dNA-NA1278" : "NA12878 dRNA",
    "pacbio_ENCFF450VAU": "ENCFF450VAU",
    "SRR307903" : "SRR307903"
}

site_types = ["tss", "tes"]
bar_width = 0.35
FEATURE_IMPORTANCE_OUT_DIR = "plots"

# ———————

def plot_feature_importance(config_folder):
    # Prepare figure grid
    n = len(data_names)
    fig, axes = plt.subplots(nrows=n, ncols=2,
                            figsize=(16, 4*n),
                            sharex=False, sharey=False)

    for i, name in enumerate(data_names):
        tools = get_tools(name)
        configs : List[Config] = [load_config(os.path.join(config_folder, f"{name}_{tools[0]}_config.pkl")),
                  load_config(os.path.join(config_folder, f"{name}_{tools[1]}_config.pkl"))]

        # Load & merge feature‐importance for TSS and TES
        data = {}
        for site in site_types:
            dfs = []
            for ti, tool in enumerate(tools):
                path = os.path.join(configs[ti].feature_importance_dir, f"{site}_rf_feature_importance.csv") 
                df = pd.read_csv(path).set_index("feature")
                dfs.append(df.rename(columns={"importance": tool}))
            data[site] = pd.concat(dfs, axis=1).fillna(0)

        # Plot each subplot
        for j, site in enumerate(site_types):
            ax = axes[i, j]
            df = data[site]

            # select top-10 by average importance
            top10 = df.mean(axis=1).nlargest(10).index
            df = df.loc[top10]
            features = df.index.tolist()
            x = np.arange(len(features))

            # grouped bars
            ax.bar(x - bar_width/2, df[tools[0]], bar_width,
                label=tools[0].capitalize())
            ax.bar(x + bar_width/2, df[tools[1]], bar_width,
                label=tools[1].capitalize())

            # formatting
            ax.set_xticks(x)
            ax.set_xticklabels(features, rotation=45, ha="right", fontsize=12)
            ax.set_title(f"{name_dict[name]} — {site.upper()}", fontsize=14)
            if j == 0:
                ax.set_ylabel("Importance", fontsize=12)
            ax.tick_params(axis="y", labelsize=10)

    # single legend on top
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels,
            loc="upper center", ncol=len(tools), fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # save
    out_dir = FEATURE_IMPORTANCE_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "all_datasets_feature_importance.pdf")
    plt.savefig(out_file, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined figure to {out_file}")



def main():
    os.makedirs(FEATURE_IMPORTANCE_OUT_DIR, exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--config_folder', required=True, help='Path to the configuration file')
    args = parser.parse_args()
    plot_feature_importance(args.config_folder)

if __name__ == "__main__":
    main()