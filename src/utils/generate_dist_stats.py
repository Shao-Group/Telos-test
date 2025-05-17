import os
import pandas as pd

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
# ———————

rows = []
for ds in data_names:
    display_ds = ds.split("/")[0]
    tools = get_tools(ds)
    for site in site_types:
        # for each assembler, count and pct
        for asm in tools:
            path = os.path.join("data_train", ds, f"{asm}_{site}_labeled.csv")
            sub = pd.read_csv(path)

            # sub = df[df["assembler"] == asm]
            true_count  = (sub["label"] == 1).sum()
            false_count = (sub["label"] == 0).sum()
            total = true_count + false_count or 1  # avoid div0

            rows.append({
                "Dataset":   name_dict[ds],
                "SiteType":  site.upper(),
                "Assembler": asm.capitalize(),
                "True":  f"{true_count} ({true_count/total*100:.1f}%)",
                "False": f"{false_count} ({false_count/total*100:.1f}%)"
            })

summary = pd.DataFrame(rows)

# pivot so each assembler is a pair of columns
pivot = summary.pivot_table(
    index=["Dataset", "SiteType"],
    columns="Assembler",
    values=["True", "False"],
    aggfunc='first'
)

# flatten the MultiIndex columns
pivot.columns = [f"{stat}_{asm}" for stat, asm in pivot.columns]
pivot = pivot.reset_index()

print(pivot.to_markdown(index=False))
# Save to CSV
output_path = os.path.join("out", "data_summary.csv")
pivot.to_csv(output_path, index=False)
print(f"Saved summary to {output_path}")