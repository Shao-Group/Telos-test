import os, sys
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from argparse import ArgumentParser
from config import load_config, Config


# —— USER SETTINGS ——
data_names = [
    # "cDNA-NA12878","dRNA-ENCFF155CFF", "pacbio_ENCFF450VAU", "SRR307903"
        "cDNA-K562",
        "cDNA-ENCFF263YFG", 
        "cDNA-NA12878", 
        "dRNA-Hek293T", 
        "dRNA-ENCFF771DIX",
        "dRNA-NA12878",
        "pacbio_ENCFF694DIE",
        "pacbio_ENCFF563QZR",
        "pacbio_ENCFF370NFS",
        "SRR307911",
        "SRR545695",
        "SRR315334",
        "SRR534307",
        "SRR545723",
        "SRR307911",
        "SRR315323",
        "SRR534319",
        "SRR534291",
        "SRR387661"
]
name_dict = {
    "cDNA-K562" : "K562 cDNA",
    "cDNA-ENCFF263YFG" : "ENCFF263YFG cDNA",
    "cDNA-NA12878" : "NA12878 cDNA",
    "dRNA-Hek293T" : "Hek293T dRNA",
    "dRNA-ENCFF771DIX" : "ENCFF771DIX dRNA",
    "dRNA-NA12878" : "NA12878 dRNA",
    "pacbio_ENCFF694DIE" : "ENCFF694DIE pacbio",
    "pacbio_ENCFF563QZR" : "ENCFF563QZR pacbio",
    "pacbio_ENCFF370NFS" : "ENCFF370NFS pacbio",
    # "pacbio_ENCFF212HLP" : "ENCFF212HLP pacbio",
    "SRR307911" : "SRR307911",
    "SRR545695" : "SRR545695",
    "SRR315334" : "SRR315334",
    "SRR534307" : "SRR534307",
    "SRR545723" : "SRR545723",
    "SRR307911" : "SRR307911",
    "SRR315323" : "SRR315323",
    "SRR534319" : "SRR534319",
    "SRR534291" : "SRR534291",
    "SRR387661" : "SRR387661"
}

# Define colors for different curve types
colors = {
    # Tool-specific colors
    "assembler1": "#000000",      # black
    "assembler2": "#808080",      # grey
    
    # Model-specific colors (add more as needed)
    "assembler1_xgboost": "#1f77b4", # blue
    "assembler1_randomforest": "#008000", # green
    "assembler2_xgboost": "#ff7f0e", # orange
    "assembler2_randomforest": "#ff0000" # red
}

STAGE1_PR_PLOT_FOLDER = "plots_individual/stage1_pr_curves/"

# ———————

def plot_pr(config_folder, is_train):
    site_types = ['tss', 'tes']
    tss_folder = os.path.join(STAGE1_PR_PLOT_FOLDER, "tss")
    tes_folder = os.path.join(STAGE1_PR_PLOT_FOLDER, "tes")
    os.makedirs(tss_folder, exist_ok=True)
    os.makedirs(tes_folder, exist_ok=True)
    suffix = "train" if is_train else "val"

    for i, name in enumerate(data_names):
        if "ENCFF563QZR" not in name:
            continue
        tools =  ["stringtie", "scallop2"] if name.startswith("SRR") else ["stringtie", "isoquant"]
        tool_map = {
            "stringtie" : "assembler1",
            "scallop2" : "assembler2",
            "isoquant" : "assembler2"
        }

        configs  = {
            tools[0] : load_config(os.path.join(config_folder, f"{name}_{tools[0]}_config.pkl")),
            tools[1] : load_config(os.path.join(config_folder, f"{name}_{tools[1]}_config.pkl"))
        }
        for j, site in enumerate(site_types):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            pr_files  = {
                tools[0] : os.listdir(configs[tools[0]].pr_data_dir), 
                tools[1] : os.listdir(configs[tools[1]].pr_data_dir)
            }
            
            # print(pr_files)
            for tool, files in pr_files.items():
                for fname in files:
                    # if not fname.endswith(f"_{suffix}_pr_data.csv"):
                    #     continue
                    # assert fname.endswith(f"_{suffix}_pr_data.csv")
                    print(tool + " " + fname)
                    if "isoquant" not in tool:
                        continue
                    if site not in fname:
                        continue

                    # site_model = fname.replace(f"_{suffix}_pr_data.csv", "")
                    # if  "randomforest" in site_model :
                    #     continue
                    pr_df = pd.read_csv(os.path.join(configs[tool].pr_data_dir, fname))

                    model_type = fname.split("_")[1]  # e.g., "logistic"
                    label = f"{tool.title()} {model_type.title()}"
                    color_key = f"{tool_map[tool]}_{model_type}"
                    color = colors.get(color_key, colors.get(tool, "#333333"))

                    # print(f"Plotting {label} with color {color}")
                    # print(f"PR data: {pr_df.head(3)}")
                    
                    # full PR curve
                    line = ax.plot(
                        pr_df["recall"],
                        pr_df["precision"],
                        label=label,
                        color=color,
                        linewidth=2
                    )[0]

                # baseline: coverage-based PR curve
                labeled_data_file = configs[tool].tss_labeled_file if site == "tss" else configs[tool].tes_labeled_file
                baseline_df = pd.read_csv(labeled_data_file, dtype={"chrom": str, "position": int})
                # print(f"Number of original points: {len(baseline_df)} for {tool} {site}")

                # print([c for c in baseline_df.columns])
                df_cov = pd.read_csv(configs[tool].cov_file, sep="\t", dtype={"tss_chrom": str, "tes_chrom": str})
                # print(f"Coverage file shape: {df_cov.shape}")

                df_cov = df_cov[[f"{site}_chrom", f"{site}_pos", "coverage"]]
                
                # Check for duplicates in coverage file
                n_unique_positions = df_cov[[f"{site}_chrom", f"{site}_pos"]].drop_duplicates().shape[0]
                print(f"Unique positions in coverage file: {n_unique_positions}, Total rows: {len(df_cov)} for {tool} {site}")
                
                if len(df_cov) > n_unique_positions:
                    # print(f"WARNING: Coverage file has duplicates! Averaging coverage values...")
                    # Average coverage for duplicate positions
                    df_cov = df_cov.groupby([f"{site}_chrom", f"{site}_pos"])['coverage'].mean().reset_index()
                
                baseline_df = baseline_df.merge(df_cov, left_on=["chrom", "position"], right_on=[f"{site}_chrom", f"{site}_pos"], how="inner")
                # print(f"Number of filtered points: {len(baseline_df)} for {tool} {site}")
                # # print sample missing points
                # if baseline_df[baseline_df["coverage"].isna()].shape[0] > 0:
                #     print(baseline_df[baseline_df["coverage"].isna()].sample(min(10, baseline_df[baseline_df["coverage"].isna()].shape[0])))

                with open (configs[tool].validation_chromosomes_file, "r") as fp:
                    val_chrom = [line.rstrip("\n") for line in fp]

                # Filter to validation chromosomes and sort by coverage
                baseline_filtered = baseline_df.loc[baseline_df["chrom"].isin(val_chrom)].copy()
                baseline_filtered = baseline_filtered.sort_values('coverage', ascending=False)
                
                # Generate PR curve by gradually removing low coverage points
                precisions = []
                recalls = []
                n_total = len(baseline_filtered)
                n_positive = baseline_filtered['label'].sum()

                # print(f"Baseline columns: {baseline_filtered.columns}")
                print(baseline_filtered[['chrom', 'position', f'{site}_chrom', f'{site}_pos', 'coverage', 'label']].head(50))
                print(baseline_filtered[['chrom', 'position', f'{site}_chrom', f'{site}_pos', 'coverage', 'label']].tail(50))
                
                for k in range(1, n_total + 1):
                    # Keep top i highest coverage points
                    subset = baseline_filtered.iloc[:k]
                    n_kept = len(subset)
                    n_tp = subset['label'].sum()  # true positives in kept subset
                    n_fp = n_kept - n_tp  # false positives in kept subset
                    
                    # Precision = TP / (TP + FP)
                    precision = n_tp / n_kept 
                    # Recall = TP / (total positive)  
                    recall = n_tp / n_positive 
                    
                    precisions.append(precision)
                    recalls.append(recall)
                
                # Plot coverage-based baseline PR curve
                baseline_label = f"{tool_map[tool].title()} Baseline"
                baseline_color = colors.get(f"{tool_map[tool]}", colors.get(tool, "#333333"))
                
                baseline_line = ax.plot(
                    recalls,
                    precisions,
                    label=baseline_label,
                    # linestyle="dotted",
                    linewidth=2,
                    color=baseline_color,
                    alpha=0.8
                )[0]

                # add a point at the end of the curve
                ax.plot(
                    [recalls[-1]],
                    [precisions[-1]],
                    marker="o",
                    color=baseline_color,
                    markersize=5,
                    alpha=0.5
                )
                
            # formatting per‐subplot
            ax.set_title(f"{name_dict.get(name, name)} - {site.upper()}", fontsize=12)
            ax.set_xlabel("Recall", fontsize=10)
            ax.set_ylabel("Precision", fontsize=10)

            ax.legend(loc='lower left', fontsize=9, frameon=True)

            plt.tight_layout()
            out_name = f"stage1_pr_{name}_{suffix}.pdf"
            out_path = os.path.join(tss_folder if site == "tss" else tes_folder, out_name)
            plt.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")


def main():
    os.makedirs(STAGE1_PR_PLOT_FOLDER, exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--config_folder', required=True, help='Path to the configuration file')
    parser.add_argument('--is_train', action='store_true', help='Is training chromosomes')
    args = parser.parse_args()
    plot_pr(args.config_folder, args.is_train)

if __name__ == "__main__":
    main()