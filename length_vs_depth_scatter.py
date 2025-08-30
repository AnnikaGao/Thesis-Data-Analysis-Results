import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from .theme import get_gradient_colors


def create_scatter_length_vs_depth(df, effectiveness, ax: plt.Axes) -> None:
    top_5_types = list(effectiveness.head(5).index)
    colors = get_gradient_colors(len(top_5_types))

    # Cap x-axis to reduce extreme right tail compression
    q99 = df[df["probing_type"].isin(top_5_types)]["response_length"].quantile(0.99)
    ax.set_xlim(0, float(q99 * 1.05) if np.isfinite(q99) else None)

    # Plot with slight vertical jitter and clearer styling
    for i, probing_type in enumerate(top_5_types):
        color = colors[i]
        human_mask = (df["probing_type"] == probing_type) & (df["conversation_nature"] == "human")
        ai_mask = (df["probing_type"] == probing_type) & (df["conversation_nature"] == "ai")

        if human_mask.any():
            human_x = df.loc[human_mask, "response_length"]
            human_y = df.loc[human_mask, "response_depth"].astype(float) + np.random.uniform(-0.06, 0.06, size=human_mask.sum())
            ax.scatter(human_x, human_y, alpha=0.55, s=35, color=color, marker="o", edgecolors="white", linewidths=0.4)

        if ai_mask.any():
            ai_x = df.loc[ai_mask, "response_length"]
            ai_y = df.loc[ai_mask, "response_depth"].astype(float) + np.random.uniform(-0.06, 0.06, size=ai_mask.sum())
            ax.scatter(ai_x, ai_y, alpha=0.55, s=35, color=color, marker="^", edgecolors="white", linewidths=0.4)

    ax.set_xlabel("Response Length (characters)")
    ax.set_ylabel("Response Depth")
    ax.set_title("Response Length vs Depth: Human (○) vs AI (△)")
    ax.set_ylim(-0.1, 3.1)
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(True, axis="x", linestyle=":", alpha=0.25)

    # Build clearer legends: colors map probing types; shapes map nature
    color_handles = [Patch(facecolor=colors[i], label=top_5_types[i]) for i in range(len(top_5_types))]
    shape_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="", label="Human", markerfacecolor="gray", markersize=6),
        Line2D([0], [0], marker="^", color="gray", linestyle="", label="AI", markerfacecolor="gray", markersize=6),
    ]

    type_legend = ax.legend(handles=color_handles, title="Probing Type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.add_artist(type_legend)
    ax.legend(handles=shape_handles, title="Nature", bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8)
