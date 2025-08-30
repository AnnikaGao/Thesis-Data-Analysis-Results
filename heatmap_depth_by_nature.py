import seaborn as sns
import matplotlib.pyplot as plt
from .theme import get_purple_green_cmap


def create_heatmap_depth_by_nature(df, effectiveness, ax: plt.Axes) -> None:
    depth_by_nature = df.groupby(["probing_type", "conversation_nature"])["response_depth"].mean().unstack(fill_value=0)
    if len(depth_by_nature) > 10:
        depth_by_nature = depth_by_nature.loc[effectiveness.head(10).index]
    sns.heatmap(depth_by_nature, annot=True, fmt=".2f", cmap=get_purple_green_cmap(include_white_midpoint=True), ax=ax)
    ax.set_title("Mean Depth: Probing Type × Conversation Nature")
    ax.set_xlabel("Conversation Nature")
    ax.set_ylabel("Probing Type")
