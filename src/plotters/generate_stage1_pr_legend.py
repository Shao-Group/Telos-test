import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Reuse styling from the PR curve script to ensure consistency
from generate_stage1_pr_curve import (
    assembler_base_colors,
    model_shade_factors,
    BASELINE_SHADE_FACTOR,
    BASELINE_LINESTYLE,
    shade_hex_color,
    STAGE1_PR_PLOT_FOLDER,
)


def generate_legend_for_tools(tool_names: list[str], output_filename: str) -> str:
    """Create a standalone legend PDF for a given set of tools.

    tool_names should be a list like ["stringtie", "isoquant"] or ["stringtie", "scallop2"].
    Colors are taken from assembler_base_colors using:
      stringtie -> assembler1, isoquant/scallop2 -> assembler2
    """
    os.makedirs(STAGE1_PR_PLOT_FOLDER, exist_ok=True)
    out_path = os.path.join(STAGE1_PR_PLOT_FOLDER, output_filename)

    tool_to_assembler = {
        "stringtie": "assembler1",
        "isoquant": "assembler2",
        "scallop2": "assembler2",
    }
    pretty_name = {
        "stringtie": "StringTie",
        "isoquant": "IsoQuant",
        "scallop2": "Scallop2",
    }

    legend_handles = []
    model_order = ["xgboost", "randomforest"]

    for tool in tool_names:
        assembler_key = tool_to_assembler.get(tool, "assembler2")
        base_color = assembler_base_colors.get(assembler_key, "#333333")

        # Baseline entry
        baseline_color = shade_hex_color(base_color, BASELINE_SHADE_FACTOR)
        legend_handles.append(
            Line2D(
                [0], [0], color=baseline_color, linestyle=BASELINE_LINESTYLE, linewidth=1,
                label=f"{pretty_name.get(tool, tool.title())} Baseline",
            )
        )

        # Model entries
        for model in model_order:
            shade = model_shade_factors.get(model, 1.0)
            model_color = shade_hex_color(base_color, shade)
            legend_handles.append(
                Line2D(
                    [0], [0], color=model_color, linestyle="-", linewidth=2,
                    label=f"{pretty_name.get(tool, tool.title())} {model.title()}",
                )
            )

    # Create figure sized to the legend only
    fig, ax = plt.subplots(figsize=(10, 1.6))
    ax.axis("off")

    ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=3 * len(tool_names) // 1,
        frameon=True,
        fontsize=14,
        columnspacing=1.4,
        handlelength=2.8,
        handletextpad=0.6,
        borderaxespad=0.2,
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    long_reads_path = generate_legend_for_tools(["stringtie", "isoquant"], "stage1_pr_legend_long_reads.pdf")
    print(f"Legend saved: {long_reads_path}")
    short_reads_path = generate_legend_for_tools(["stringtie", "scallop2"], "stage1_pr_legend_short_reads.pdf")
    print(f"Legend saved: {short_reads_path}")


