import matplotlib.pyplot as plt
from .theme import PALE_PURPLE, PALE_GREEN


def create_human_ai_distribution_pie(df, ax: plt.Axes) -> None:
    nature_counts = df["conversation_nature"].value_counts()
    colors = [PALE_PURPLE, PALE_GREEN]
    labels = [f"{nature.title()} ({count})" for nature, count in nature_counts.items()]
    ax.pie(nature_counts.values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    ax.set_title("Human vs AI Distribution")


