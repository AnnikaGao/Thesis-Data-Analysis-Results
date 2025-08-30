import numpy as np
import matplotlib.pyplot as plt
from .theme import PALE_PURPLE, PALE_GREEN, SLIGHTLY_DARK_PURPLE, SLIGHTLY_DARK_GREEN


def create_depth_distribution_histogram(depths_df, ax: plt.Axes) -> None:
    human_depths = depths_df[depths_df["conversation_nature"] == "human"]["depth"]
    ai_depths = depths_df[depths_df["conversation_nature"] == "ai"]["depth"]

    all_depths = depths_df["depth"]
    if len(all_depths) == 0:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", transform=ax.transAxes)
        return

    bin_range = (all_depths.min() - 0.1, all_depths.max() + 0.1)
    bins = np.linspace(bin_range[0], bin_range[1], 16)

    ax.hist(human_depths, bins=bins, alpha=0.75, color=PALE_PURPLE, label=f"Human (n={len(human_depths)})", edgecolor="black", linewidth=0.5)
    ax.hist(ai_depths, bins=bins, alpha=0.75, color=PALE_GREEN, label=f"AI (n={len(ai_depths)})", edgecolor="black", linewidth=0.5)

    if len(human_depths) > 0:
        ax.axvline(human_depths.mean(), color=SLIGHTLY_DARK_PURPLE, linestyle="--", label=f"Human Mean: {human_depths.mean():.2f}", linewidth=2)
    if len(ai_depths) > 0:
        ax.axvline(ai_depths.mean(), color=SLIGHTLY_DARK_GREEN, linestyle="--", label=f"AI Mean: {ai_depths.mean():.2f}", linewidth=2)

    ax.set_xlabel("Response Depth Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Response Depth Distribution: Human vs AI (Interviewee messages)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
