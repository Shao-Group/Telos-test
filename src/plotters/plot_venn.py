import os, sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from config import Config, load_config
from argparse import ArgumentParser

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
VENN_OUTPUT_DIR = "plots/"

def load_site_ids(filepath, site_type):
    df = pd.read_csv(filepath)
    return set(site_type + ":" + df['chrom'].astype(str) + ":" 
               + df['position'].astype(str) + ":" 
               + df['strand'].astype(str))

def find_intersection(site1, site2, max_dist=50):
    s1, chrom1, pos1, strand1 = site1.split(":")
    s2, chrom2, pos2, strand2 = site2.split(":")
    return (s1 == s2 and chrom1 == chrom2 
            and strand1 == strand2 
            and abs(int(pos1) - int(pos2)) <= max_dist)

def find_intersection_sets(set1, set2, max_dist=50):
    intersection = set()
    for s1 in set1:
        if any(find_intersection(s1, s2, max_dist) for s2 in set2):
            intersection.add(s1)
    return intersection


# ——————————————————
def plot_venn(config_folder):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, name in zip(axes, data_names):
        tools =  ["stringtie", "scallop2"] if name.startswith("SRR") else ["stringtie", "isoquant"]
        sets = {}
        
        for tool in tools:
            config : Config = load_config(os.path.join(config_folder, f"{name}_{tool}_config.pkl"))
            tss_set : set = load_site_ids(config.tss_labeled_file, "tss")
            tes_set : set = load_site_ids(config.tes_labeled_file, "tes")
            sets[tool] = tss_set | tes_set

        inter = find_intersection_sets(sets[tools[0]], sets[tools[1]])
        only1 = len(sets[tools[0]] - inter)
        only2 = len(sets[tools[1]] - inter)
        both  = len(inter)

        venn = venn2(
            subsets=(only1, only2, both),
            set_labels=(tools[0].capitalize(), tools[1].capitalize()),
            ax=ax
        )
        # bump up the tool‐name labels
        for lbl in venn.set_labels:
            if lbl:
                lbl.set_fontsize(20)

        # bump up the region count labels
        for cnt in venn.subset_labels:
            if cnt:
                cnt.set_fontsize(18)
        ax.set_title(name_dict[name], fontsize=22)
        print(f"{name} done")
        # plt.show()

    plt.tight_layout()
    plt.savefig(os.path.join(VENN_OUTPUT_DIR, "venn_tss_4datasets.pdf"), dpi=300, format="pdf", bbox_inches="tight")
# plt.show()


def main():
    os.makedirs(VENN_OUTPUT_DIR, exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--config_folder', required=True, help='Path to the configuration file')
    args = parser.parse_args()
    plot_venn(args.config_folder)


if __name__ == "__main__":
    main()