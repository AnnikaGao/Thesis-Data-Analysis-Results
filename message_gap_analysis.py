import numpy as np
import matplotlib.pyplot as plt
from .theme import PALE_PURPLE, PALE_GREEN


def create_message_gap_analysis(df, effectiveness, ax: plt.Axes) -> None:
    top_8_types = effectiveness.head(8).index
    gap_breakdown = df[df["probing_type"].isin(top_8_types)].groupby(["probing_type", "conversation_nature"])["message_gap"].mean().unstack(fill_value=0)

    x_pos = range(len(top_8_types))
    bar_width = 0.35

    human_gaps = [gap_breakdown.loc[pt, "human"] if pt in gap_breakdown.index and "human" in gap_breakdown.columns else 0 for pt in top_8_types]
    ai_gaps = [gap_breakdown.loc[pt, "ai"] if pt in gap_breakdown.index and "ai" in gap_breakdown.columns else 0 for pt in top_8_types]

    ax.bar([x - bar_width / 2 for x in x_pos], human_gaps, bar_width, label="Human", color=PALE_PURPLE, alpha=0.9)
    ax.bar([x + bar_width / 2 for x in x_pos], ai_gaps, bar_width, label="AI", color=PALE_GREEN, alpha=0.9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_8_types, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Average Message Gap")
    ax.set_title("Response Timing: Human vs AI")
    ax.legend()
