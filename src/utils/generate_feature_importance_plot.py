import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_tools(data_name):
    low = data_name.lower()
    if low.startswith(("nanopore", "pacbio")):
        return ["isoquant", "stringtie"]
    elif data_name.startswith("SRR"):
        return ["stringtie", "scallop2"]
    else:
        raise ValueError(f"Unknown prefix for {data_name!r}")

# —— USER SETTINGS ——
data_names = [
    "nanopore_cDNA_NA12878/gencode",
    "nanopore_dRNA_NA12878/gencode",
    "pacbio_ENCFF450VAU/gencode",
    "SRR307903_hisat/ensembl"
]
name_dict = {
    "nanopore_cDNA_NA12878/gencode": "NA12878 cDNA",
    "nanopore_dRNA_NA12878/gencode": "NA12878 dRNA",
    "pacbio_ENCFF450VAU/gencode": "ENCFF450VAU",
    "SRR307903_hisat/ensembl": "SRR307903"
}
site_types = ["tss", "tes"]
bar_width = 0.35
# ———————

# Prepare figure grid
n = len(data_names)
fig, axes = plt.subplots(nrows=n, ncols=2,
                         figsize=(16, 4*n),
                         sharex=False, sharey=False)

for i, name in enumerate(data_names):
    tools = get_tools(name)

    # Load & merge feature‐importance for TSS and TES
    data = {}
    for site in site_types:
        dfs = []
        for tool in tools:
            path = os.path.join("out", name, "reports", site,
                                f"{tool}_rf_feature_importance.csv")
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
out_dir = os.path.join("out", "plots")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "all_datasets_feature_importance.pdf")
plt.savefig(out_file, dpi=300, format="pdf", bbox_inches="tight")
plt.close(fig)

print(f"Saved combined figure to {out_file}")
