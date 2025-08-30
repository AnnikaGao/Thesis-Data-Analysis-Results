import numpy as np
import matplotlib.pyplot as plt
from .theme import PALE_PURPLE, PALE_GREEN


def create_goal_relevance_impact(df, effectiveness, ax: plt.Axes) -> None:
    top_8_types = effectiveness.head(8).index
    goal_impact_human = df[(df["probing_type"].isin(top_8_types)) & (df["conversation_nature"] == "human")].groupby(["probing_type", "goal_relevant"])["response_depth"].mean().unstack(fill_value=0)
    goal_impact_ai = df[(df["probing_type"].isin(top_8_types)) & (df["conversation_nature"] == "ai")].groupby(["probing_type", "goal_relevant"])["response_depth"].mean().unstack(fill_value=0)

    x_pos = np.arange(len(top_8_types))
    bar_width = 0.2

    human_goal_rel = [goal_impact_human.loc[pt, True] if pt in goal_impact_human.index and True in goal_impact_human.columns else 0 for pt in top_8_types]
    human_goal_not_rel = [goal_impact_human.loc[pt, False] if pt in goal_impact_human.index and False in goal_impact_human.columns else 0 for pt in top_8_types]
    ai_goal_rel = [goal_impact_ai.loc[pt, True] if pt in goal_impact_ai.index and True in goal_impact_ai.columns else 0 for pt in top_8_types]
    ai_goal_not_rel = [goal_impact_ai.loc[pt, False] if pt in goal_impact_ai.index and False in goal_impact_ai.columns else 0 for pt in top_8_types]

    ax.bar(x_pos - 1.5 * bar_width, human_goal_not_rel, bar_width, label="Human - Not Goal Rel", color=PALE_PURPLE, alpha=0.6)
    ax.bar(x_pos - 0.5 * bar_width, human_goal_rel, bar_width, label="Human - Goal Rel", color=PALE_PURPLE, alpha=0.95)
    ax.bar(x_pos + 0.5 * bar_width, ai_goal_not_rel, bar_width, label="AI - Not Goal Rel", color=PALE_GREEN, alpha=0.6)
    ax.bar(x_pos + 1.5 * bar_width, ai_goal_rel, bar_width, label="AI - Goal Rel", color=PALE_GREEN, alpha=0.95)

    ax.set_xlabel("Probing Type")
    ax.set_ylabel("Mean Response Depth")
    ax.set_title("Goal Relevance Impact: Human vs AI")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_8_types, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
