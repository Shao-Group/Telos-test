import os, sys
import re
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
from glob import glob
from collections import defaultdict

# —— USER SETTINGS ——
# data_names = [
#     "cDNA-NA12878","dRNA-ENCFF155CFF", "pacbio_ENCFF450VAU", "SRR307903"
# ]
name_dict = {
    "cDNA-NA12878" : "NA12878 cDNA",
    "dRNA-ENCFF155CFF" : "ENCFF155CFF dRNA",
    "pacbio_ENCFF450VAU": "ENCFF450VAU",
    "SRR307903" : "SRR307903"
}


def load_site_ids(filepath, site_type, validation_chromosomes):
    df = pd.read_csv(filepath, dtype={"chrom": str})
    df = df[df['chrom'].isin(validation_chromosomes)]
    df['strand'] = '+'
    df['chrom'] = df['chrom'].apply(lambda x: f"chr{x}" if re.fullmatch(r'[1-9]|1[0-9]|2[0-2]|X|Y', x) else x)
    return set(site_type + ":" + df['chrom'].astype(str) + ":" 
               + df['position'].astype(str)  
               + ":" 
               + df['strand'].astype(str))

def parse_sites(site_set):
    """Pre-parse all sites into structured data for faster comparisons"""
    parsed_sites = defaultdict(lambda: defaultdict(list))
    site_to_parsed = {}
    
    for site in site_set:
        site_type, chrom, pos, strand = site.split(":")
        pos_int = int(pos)
        key = (site_type, chrom, strand)
        parsed_sites[key][pos_int].append(site)
        site_to_parsed[site] = (site_type, chrom, pos_int, strand)
    
    return parsed_sites, site_to_parsed

def find_intersection_sets_optimized(set1, set2, max_dist=50):
    """Optimized intersection finding using spatial grouping"""
    if not set1 or not set2:
        return set()
    
    # Parse sites into structured format
    parsed1, site_map1 = parse_sites(set1)
    parsed2, site_map2 = parse_sites(set2)
    
    intersection = set()
    
    # Only compare sites with same site_type, chrom, and strand
    for key in parsed1:
        if key not in parsed2:
            continue
            
        positions1 = sorted(parsed1[key].keys())
        positions2 = sorted(parsed2[key].keys())
        
        # Use two-pointer technique for efficient range finding
        j = 0
        for pos1 in positions1:
            # Move j to first position in range
            while j < len(positions2) and positions2[j] < pos1 - max_dist:
                j += 1
            
            # Check all positions in range
            k = j
            while k < len(positions2) and positions2[k] <= pos1 + max_dist:
                # Found intersection - add all sites at pos1
                for site1 in parsed1[key][pos1]:
                    intersection.add(site1)
                break
                k += 1
    
    return intersection

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
def plot_venn(config_folder, is_predictions, model_type=None):
    # Discover datasets from all *_config.pkl files in the folder
    VENN_OUTPUT_DIR = "plots_individual/venn_baseline/"
    if is_predictions:
        assert model_type is not None, "Model type is required for predictions"
        VENN_OUTPUT_DIR = f"plots_individual/venn_predictions/{model_type}"
    else:
        assert model_type is None, "Model type is not required for baseline"
    os.makedirs(VENN_OUTPUT_DIR, exist_ok=True)
    cfg_files = glob(os.path.join(config_folder, "*_config.pkl"))
    datasets_to_tools = {}
    validation_chromosomes = defaultdict(set)
    for path in cfg_files:
        base = os.path.basename(path)
        # expect pattern <name>_<tool>_config.pkl; split from right to preserve underscores in name
        name_tool = base.split("_config.pkl")[0]  # strip _config.pkl
        if "_" not in name_tool:
            continue
        name, tool = name_tool.rsplit("_", 1)
        datasets_to_tools.setdefault(name, set()).add(tool)

        cfg = load_config(path)
        tmp = set()
        with open(cfg.validation_chromosomes_file, 'r') as f:
            tmp.update(f.readlines())
        tmp = {chrom.strip() for chrom in tmp}
        validation_chromosomes[name].update(tmp)

    # Collect summary stats for CSV output
    summary_rows = []

    for name, tools_present in sorted(datasets_to_tools.items()):
        print(f"Processing {name} with tools: {tools_present}")
        # Determine preferred tool pair for this dataset
        preferred_pair = ("stringtie", "scallop2") if name.startswith("SRR") else ("stringtie", "isoquant")
        if all(t in tools_present for t in preferred_pair):
            tools = list(preferred_pair)
        elif len(tools_present) >= 2:
            # fallback to any two tools available
            raise ValueError(f"Skipping {name} - need maximum two tools, found: {tools_present}")
            tools = sorted(list(tools_present))[:2]
        else:
            print(f"Skipping {name} - need at least two tools, found: {tools_present}")
            continue

        sets = {}
        for tool in tools:
            cfg_path = os.path.join(config_folder, f"{name}_{tool}_config.pkl")
            try:
                config : Config = load_config(cfg_path)
            except Exception as e:
                print(f"Skipping {name}/{tool} - cannot load {cfg_path}: {e}")
                sets = {}
                break
            tss_set : set = load_site_ids(config.tss_labeled_file, "tss", validation_chromosomes[name])
            tes_set : set = load_site_ids(config.tes_labeled_file, "tes", validation_chromosomes[name])

            # print(list(tss_set)[:5])
            if is_predictions:
                tss_pred_file = os.path.join(config.predictions_output_dir, f"tss_{model_type}_predictions.csv")
                tes_pred_file = os.path.join(config.predictions_output_dir, f"tes_{model_type}_predictions.csv")
                tss_pred = pd.read_csv(tss_pred_file, dtype={"chrom": str})
                tes_pred = pd.read_csv(tes_pred_file, dtype={"chrom": str})
                
                tss_pred['chrom'] = tss_pred['chrom'].apply(lambda x: f"chr{x}" if re.fullmatch(r'[1-9]|1[0-9]|2[0-2]|X|Y', x) else x)
                tes_pred['chrom'] = tes_pred['chrom'].apply(lambda x: f"chr{x}" if re.fullmatch(r'[1-9]|1[0-9]|2[0-2]|X|Y', x) else x)
                tss_pred['site_type'] = tss_pred['site_type'].apply(lambda x: x.lower())
                tes_pred['site_type'] = tes_pred['site_type'].apply(lambda x: x.lower())
                tss_pred['strand'] = '+'
                tes_pred['strand'] = '+'

                
                # create predictions dictionary
                tss_keys = tss_pred["site_type"] + ":" + tss_pred['chrom'].astype(str) + ":" + tss_pred['position'].astype(str) + ":" + tss_pred['strand'].astype(str)
                tss_pred_dict = dict(zip(tss_keys, tss_pred['probability']))
                # print(tss_keys[:5])
                tes_keys = tes_pred["site_type"] + ":" + tes_pred['chrom'].astype(str) + ":" + tes_pred['position'].astype(str) + ":" + tes_pred['strand'].astype(str)
                tes_pred_dict = dict(zip(tes_keys, tes_pred['probability']))
                # print(tes_keys[:5])
                # filter baseline site ids from predictions based on probability
                tss_set = {site for site in tss_set if site in tss_pred_dict and tss_pred_dict[site] > 0.5}
                tes_set = {site for site in tes_set if site in tes_pred_dict and tes_pred_dict[site] > 0.5}
                
               
            sets[tool] = tss_set | tes_set

        if len(sets) != 2:
            print(f"Skipping {name} - need exactly two tools, found: {tools_present}")
            continue

        inter = find_intersection_sets_optimized(sets[tools[0]], sets[tools[1]])
        only1 = len(sets[tools[0]] - inter)
        only2 = len(sets[tools[1]] - inter)
        both  = len(inter)
        
        # Calculate total and percentages
        total = only1 + only2 + both
        only1_pct = (only1 / total * 100) if total > 0 else 0
        only2_pct = (only2 / total * 100) if total > 0 else 0
        both_pct = (both / total * 100) if total > 0 else 0

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        venn = venn2(
            subsets=(only1, only2, both),
            set_labels=(tools[0].capitalize(), tools[1].capitalize()),
            ax=ax
        )
        for lbl in (venn.set_labels or []):
            if lbl:
                lbl.set_fontsize(20)

        # Update labels to show count and percentage
        if venn.subset_labels[0]:  # only1 region
            venn.subset_labels[0].set_text(f"{only1}\n({only1_pct:.2f}%)")
            venn.subset_labels[0].set_fontsize(16)
        if venn.subset_labels[1]:  # only2 region
            venn.subset_labels[1].set_text(f"{only2}\n({only2_pct:.2f}%)")
            venn.subset_labels[1].set_fontsize(16)
        if venn.subset_labels[2]:  # both region
            venn.subset_labels[2].set_text(f"{both}\n({both_pct:.2f}%)")
            venn.subset_labels[2].set_fontsize(16)
        ax.set_title(name_dict.get(name, name), fontsize=22)

        plt.tight_layout()
        out_path = os.path.join(VENN_OUTPUT_DIR, f"venn_{name}_{tools[0]}_vs_{tools[1]}.pdf")
        plt.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

        # Save stats for CSV output
        summary_rows.append({
            "dataset": name,
            "tool_1": tools[0],
            "tool_2": tools[1],
            "only_tool_1_count": only1,
            "only_tool_2_count": only2,
            "both_count": both,
            "only_tool_1_pct": round(only1_pct, 6),
            "only_tool_2_pct": round(only2_pct, 6),
            "both_pct": round(both_pct, 6),
            "total": total,
            "is_predictions": bool(is_predictions),
            "model_type": model_type if is_predictions else None,
        })

    # Write a single CSV summary per run (predictions vs baseline)
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        csv_name = "venn_summary.csv"
        csv_path = os.path.join(VENN_OUTPUT_DIR, csv_name)
        df_summary.to_csv(csv_path, index=False)
        print(f"Saved summary CSV: {csv_path}")
# plt.show()


def main():
    # os.makedirs(VENN_OUTPUT_DIR, exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--config_folder', required=True, help='Path to the configuration file')
    parser.add_argument('--is_predictions', action='store_true', help='Is predictions')
    parser.add_argument('--model_type', required=False, help='Model type')
    args = parser.parse_args()
    plot_venn(args.config_folder, args.is_predictions, args.model_type)


if __name__ == "__main__":
    main()