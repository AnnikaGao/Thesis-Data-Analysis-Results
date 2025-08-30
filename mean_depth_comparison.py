import matplotlib.pyplot as plt
from .theme import PALE_PURPLE, PALE_GREEN


def create_mean_depth_comparison(df, effectiveness, ax: plt.Axes) -> None:
    top_types = effectiveness.head(10)
    subset = df[df["probing_type"].isin(top_types.index)]
    human_ai_means = subset.groupby(["probing_type", "conversation_nature"]) ["response_depth"].mean().unstack(fill_value=0)
    human_ai_stds = subset.groupby(["probing_type", "conversation_nature"]) ["response_depth"].std(ddof=1).unstack(fill_value=0).fillna(0)

    y_pos = range(len(top_types))
    bar_width = 0.35

    human_means = [human_ai_means.loc[pt, "human"] if pt in human_ai_means.index and "human" in human_ai_means.columns else 0 for pt in top_types.index]
    ai_means = [human_ai_means.loc[pt, "ai"] if pt in human_ai_means.index and "ai" in human_ai_means.columns else 0 for pt in top_types.index]
    human_stds = [human_ai_stds.loc[pt, "human"] if pt in human_ai_stds.index and "human" in human_ai_stds.columns else 0 for pt in top_types.index]
    ai_stds = [human_ai_stds.loc[pt, "ai"] if pt in human_ai_stds.index and "ai" in human_ai_stds.columns else 0 for pt in top_types.index]

    ax.barh([y - bar_width / 2 for y in y_pos], human_means, bar_width, label="Human", color=PALE_PURPLE, alpha=0.9,
            xerr=human_stds, capsize=3, error_kw={"elinewidth": 1, "alpha": 0.8, "ecolor": "#555"})
    ax.barh([y + bar_width / 2 for y in y_pos], ai_means, bar_width, label="AI", color=PALE_GREEN, alpha=0.9,
            xerr=ai_stds, capsize=3, error_kw={"elinewidth": 1, "alpha": 0.8, "ecolor": "#555"})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_types.index, fontsize=10)
    ax.set_xlabel("Mean Response Depth")
    ax.set_title("Mean Response Depth: Human vs AI (Top 10)")
    ax.legend()

    # Ensure room for error bars and labels
    max_mean = max(human_means + ai_means) if (human_means or ai_means) else 0
    max_std = max(human_stds + ai_stds) if (human_stds or ai_stds) else 0
    ax.set_xlim(0, (max_mean + max_std) * 1.15 if (max_mean + max_std) > 0 else 1)

    for i, (human_val, ai_val) in enumerate(zip(human_means, ai_means)):
        if human_val > 0:
            ax.text(human_val + 0.02, i - bar_width / 2, f"{human_val:.2f}", va="center", fontsize=8, fontweight="bold")
        if ai_val > 0:
            ax.text(ai_val + 0.02, i + bar_width / 2, f"{ai_val:.2f}", va="center", fontsize=8, fontweight="bold")
