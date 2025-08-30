import matplotlib.pyplot as plt
from .theme import PALE_PURPLE, PALE_GREEN


def create_sample_size_breakdown(df, effectiveness, ax: plt.Axes) -> None:
    top_10_types = effectiveness.head(10).index
    sample_breakdown = df[df["probing_type"].isin(top_10_types)].groupby(["probing_type", "conversation_nature"]).size().unstack(fill_value=0)

    y_pos = range(len(top_10_types))
    bar_width = 0.35

    human_counts = [sample_breakdown.loc[pt, "human"] if pt in sample_breakdown.index and "human" in sample_breakdown.columns else 0 for pt in top_10_types]
    ai_counts = [sample_breakdown.loc[pt, "ai"] if pt in sample_breakdown.index and "ai" in sample_breakdown.columns else 0 for pt in top_10_types]

    human_bars_y = [y - bar_width / 2 for y in y_pos]
    ai_bars_y = [y + bar_width / 2 for y in y_pos]

    ax.barh(human_bars_y, human_counts, bar_width, label="Human", color=PALE_PURPLE, alpha=0.9)
    ax.barh(ai_bars_y, ai_counts, bar_width, label="AI", color=PALE_GREEN, alpha=0.9)

    # Annotate counts at the end of each bar
    max_val = max([0] + human_counts + ai_counts)
    pad = max(1, max_val * 0.01)
    for y, c in zip(human_bars_y, human_counts):
        ax.text(c + pad, y, str(int(c)), va="center", ha="left", fontsize=9)
    for y, c in zip(ai_bars_y, ai_counts):
        ax.text(c + pad, y, str(int(c)), va="center", ha="left", fontsize=9)

    # Ensure there is room for the labels on the right
    ax.set_xlim(0, max_val * 1.15 if max_val > 0 else 1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_10_types, fontsize=10)
    ax.set_xlabel("Number of Observations")
    ax.set_title("Sample Size: Human vs AI (Top 10)")
    ax.legend()
