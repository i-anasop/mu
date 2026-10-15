"""
Regenerates Figure 6 of the paper: retain and forget accuracy on the
document image benchmarks, before and after BEAIS unlearning.

Values are the test-set results reported in Table 6 and produced by
beais/run_tobacco3482.py and beais/run_rvlcdip.py.

Usage:
    python figures/plot_document_results.py
Writes doc_results.png (400 dpi) and doc_results.pdf to the working directory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

# Test-set accuracies, before and after unlearning (see Table 6)
DATA = {
    "Tobacco3482": {"retain": (63.89, 67.13), "forget": (69.57, 0.00)},
    "RVL-CDIP":    {"retain": (58.83, 60.39), "forget": (64.04, 0.00)},
}

C_BEFORE = "#4C72B0"
C_AFTER = "#C44E52"


def main():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)

    for ax, (name, vals) in zip(axes, DATA.items()):
        x = np.arange(2)
        w = 0.34
        before = [vals["retain"][0], vals["forget"][0]]
        after = [vals["retain"][1], vals["forget"][1]]

        bars = [
            ax.bar(x - w / 2, before, w, label="Original",
                   color=C_BEFORE, edgecolor="black", linewidth=0.5),
            ax.bar(x + w / 2, after, w, label="After BEAIS",
                   color=C_AFTER, edgecolor="black", linewidth=0.5),
        ]

        for group in bars:
            for bar in group:
                h = bar.get_height()
                ax.annotate(f"{h:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([r"$\mathcal{D}_r$-Acc.", r"$\mathcal{D}_f$-Acc."])
        ax.set_title(name, fontsize=9.5, pad=6)
        ax.set_ylim(0, 85)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.04))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("doc_results.pdf", bbox_inches="tight")
    fig.savefig("doc_results.png", dpi=400, bbox_inches="tight")
    print("Wrote doc_results.pdf and doc_results.png")


if __name__ == "__main__":
    main()
