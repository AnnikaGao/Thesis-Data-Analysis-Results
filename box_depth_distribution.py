import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from .theme import PALE_PURPLE, PALE_GREEN, get_depth_level_colors


def create_box_plot_distributions(df, effectiveness, ax: plt.Axes) -> None:
    top_5_types = list(effectiveness.head(5).index)
    plot_df = df[df["probing_type"].isin(top_5_types)].copy()
    plot_df = plot_df.dropna(subset=["response_depth"])
    if len(plot_df) == 0:
        ax.set_visible(False)
        return

    plot_df["response_depth"] = plot_df["response_depth"].round().astype(int)
    depth_levels = sorted(plot_df["response_depth"].unique())

    bar_width = 0.35
    x_positions = np.arange(len(top_5_types))

    # Precompute distinct colors per depth level
    human_level_colors, ai_level_colors = get_depth_level_colors(len(depth_levels))

    for i, pt in enumerate(top_5_types):
        subset = plot_df[plot_df["probing_type"] == pt]
        grouped = subset.groupby(["conversation_nature", "response_depth"]).size().unstack(fill_value=0)

        human_counts = grouped.loc["human"] if "human" in grouped.index else grouped.iloc[0] * 0
        human_counts = human_counts.reindex(depth_levels, fill_value=0)
        human_total = human_counts.sum() if human_counts.sum() > 0 else 1
        human_props = human_counts / human_total
        bottom = 0.0
        for j, level in enumerate(depth_levels):
            height = human_props.loc[level]
            if height > 0:
                ax.bar(i - bar_width / 2, height, bar_width, bottom=bottom, color=human_level_colors[j])
                bottom += height

        ai_counts = grouped.loc["ai"] if "ai" in grouped.index else grouped.iloc[0] * 0
        ai_counts = ai_counts.reindex(depth_levels, fill_value=0)
        ai_total = ai_counts.sum() if ai_counts.sum() > 0 else 1
        ai_props = ai_counts / ai_total
        bottom = 0.0
        for j, level in enumerate(depth_levels):
            height = ai_props.loc[level]
            if height > 0:
                ax.bar(i + bar_width / 2, height, bar_width, bottom=bottom, color=ai_level_colors[j])
                bottom += height

    ax.set_xticks(x_positions)
    ax.set_xticklabels(top_5_types, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion of Responses")
    ax.set_xlabel("Probing Type")
    ax.set_title("Depth Distribution (Proportions): Human vs AI (Top 5)")

    # Custom legends to avoid repeated/confusing entries
    nature_handles = [
        mpatches.Patch(facecolor=PALE_PURPLE, label="Human"),
        mpatches.Patch(facecolor=PALE_GREEN, label="AI"),
    ]
    depth_handles = [mpatches.Patch(facecolor=human_level_colors[j], label=f"Depth {level}") for j, level in enumerate(depth_levels)]

    nature_legend = ax.legend(handles=nature_handles, title="", fontsize=8, loc="upper left")
    ax.add_artist(nature_legend)
    ax.legend(handles=depth_handles, title="Levels", fontsize=8, loc="upper right")


def create_box_plot_distributions_counts(df, effectiveness, ax: plt.Axes) -> None:
    """Same as create_box_plot_distributions but label blocks with counts instead of proportions."""
    top_5_types = list(effectiveness.head(5).index)
    plot_df = df[df["probing_type"].isin(top_5_types)].copy()
    plot_df = plot_df.dropna(subset=["response_depth"])
    if len(plot_df) == 0:
        ax.set_visible(False)
        return

    plot_df["response_depth"] = plot_df["response_depth"].round().astype(int)
    depth_levels = sorted(plot_df["response_depth"].unique())

    bar_width = 0.35
    x_positions = np.arange(len(top_5_types))

    human_level_colors, ai_level_colors = get_depth_level_colors(len(depth_levels))

    for i, pt in enumerate(top_5_types):
        subset = plot_df[plot_df["probing_type"] == pt]
        grouped = subset.groupby(["conversation_nature", "response_depth"]).size().unstack(fill_value=0)

        # Human
        human_counts = grouped.loc["human"] if "human" in grouped.index else grouped.iloc[0] * 0
        human_counts = human_counts.reindex(depth_levels, fill_value=0)
        human_total = human_counts.sum() if human_counts.sum() > 0 else 1
        human_props = human_counts / human_total
        bottom = 0.0
        for j, level in enumerate(depth_levels):
            count = int(human_counts.loc[level])
            height = human_props.loc[level]
            if height > 0:
                rect = ax.bar(i - bar_width / 2, height, bar_width, bottom=bottom, color=human_level_colors[j])
                # Add count label centered in the block
                y_center = bottom + height / 2
                ax.text(i - bar_width / 2, y_center, str(count), ha="center", va="center", fontsize=8, color="#222")
                bottom += height

        # AI
        ai_counts = grouped.loc["ai"] if "ai" in grouped.index else grouped.iloc[0] * 0
        ai_counts = ai_counts.reindex(depth_levels, fill_value=0)
        ai_total = ai_counts.sum() if ai_counts.sum() > 0 else 1
        ai_props = ai_counts / ai_total
        bottom = 0.0
        for j, level in enumerate(depth_levels):
            count = int(ai_counts.loc[level])
            height = ai_props.loc[level]
            if height > 0:
                rect = ax.bar(i + bar_width / 2, height, bar_width, bottom=bottom, color=ai_level_colors[j])
                y_center = bottom + height / 2
                ax.text(i + bar_width / 2, y_center, str(count), ha="center", va="center", fontsize=8, color="#222")
                bottom += height

    ax.set_xticks(x_positions)
    ax.set_xticklabels(top_5_types, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion of Responses (Counts shown)")
    ax.set_xlabel("Probing Type")
    ax.set_title("Depth Distribution (Proportions) with Counts: Human vs AI (Top 5)")

    nature_handles = [
        mpatches.Patch(facecolor=PALE_PURPLE, label="Human"),
        mpatches.Patch(facecolor=PALE_GREEN, label="AI"),
    ]
    depth_handles = [mpatches.Patch(facecolor=human_level_colors[j], label=f"Depth {level}") for j, level in enumerate(depth_levels)]

    nature_legend = ax.legend(handles=nature_handles, title="", fontsize=8, loc="upper left")
    ax.add_artist(nature_legend)
    ax.legend(handles=depth_handles, title="Levels", fontsize=8, loc="upper right")
