import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

def load_site_ids(filepath):
    df = pd.read_csv(filepath)
    return set(df['chrom'].astype(str) + ":" 
               + df['position'].astype(str) + ":" 
               + df['strand'].astype(str))

def find_intersection(site1, site2, max_dist=50):
    chrom1, pos1, strand1 = site1.split(":")
    chrom2, pos2, strand2 = site2.split(":")
    return (chrom1 == chrom2 
            and strand1 == strand2 
            and abs(int(pos1) - int(pos2)) <= max_dist)

def find_intersection_sets(set1, set2, max_dist=50):
    intersection = set()
    for s1 in set1:
        if any(find_intersection(s1, s2, max_dist) for s2 in set2):
            intersection.add(s1)
    return intersection

def get_tools(data_name):
    """Choose tools by data_name prefix."""
    low = data_name.lower()
    if low.startswith(("nanopore", "pacbio")):
        return ["isoquant", "stringtie"]
    elif data_name.startswith("SRR"):
        return ["stringtie", "scallop2"]
    else:
        raise ValueError(f"Unknown prefix for {data_name}")

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

# ——————————————————

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for ax, name in zip(axes, data_names):
    tools = get_tools(name)
    sets = {}
    for tool in tools:
        path = os.path.join("data_train", name, f"{tool}_tss_labeled.csv")
        sets[tool] = load_site_ids(path)

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
    plt.show()

plt.tight_layout()
plt.savefig("out/plots/venn_tss_4datasets.pdf", dpi=300, format="pdf", bbox_inches="tight")
# plt.show()
