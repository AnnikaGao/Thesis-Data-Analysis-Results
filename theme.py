from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def setup_plot_style():
    """Setup consistent plotting style using the provided theme palette."""
    plt.style.use("default")
    # Two-color theme (from image): pink and teal
    sns.set_palette(["#fcb2af", "#9bdfdf"])  # pink, teal


# Two primary theme colors (RGB from image):
# Pink:  R:252 G:178 B:175 → #fcb2af
# Teal:  R:155 G:223 B:223 → #9bdfdf
THEME_PINK = "#fcb2af"
THEME_TEAL = "#9bdfdf"

# Backwards-compatible aliases used across plots (Human/AI mapping via existing names)
PALE_PURPLE = THEME_PINK
PALE_GREEN = THEME_TEAL

# Slightly darker accents for lines (used for mean lines, etc.)
SLIGHTLY_DARK_PURPLE = "#f28f89"  # darker pink
SLIGHTLY_DARK_GREEN = "#6cbcbc"  # darker teal


def get_purple_green_cmap(include_white_midpoint: bool = False) -> LinearSegmentedColormap:
    """Create a pink→(white)→teal colormap from the theme."""
    if include_white_midpoint:
        return LinearSegmentedColormap.from_list("pink_teal_mid", [PALE_PURPLE, "#ffffff", PALE_GREEN])
    return LinearSegmentedColormap.from_list("pink_teal", [PALE_PURPLE, PALE_GREEN])


def get_gradient_colors(num_colors: int):
    """Return a list of colors blending from lavender to teal (theme)."""
    cmap = get_purple_green_cmap()
    if num_colors <= 1:
        return [PALE_PURPLE]
    return [cmap(i / (num_colors - 1)) for i in range(num_colors)]


def get_depth_level_colors(num_levels: int):
    """Return strong, clearly separated shades for each depth level for human (pink) and AI (teal).

    Uses colormaps white→color and samples towards the saturated end to increase contrast.
    """
    if num_levels <= 0:
        return [], []
    pink_cmap = LinearSegmentedColormap.from_list("pink_levels", ["#ffffff", THEME_PINK])
    teal_cmap = LinearSegmentedColormap.from_list("teal_levels", ["#ffffff", THEME_TEAL])
    # Sample from 0.35 to 0.95 to avoid too-light shades
    positions = np.linspace(0.35, 0.95, num_levels)
    human_colors = [pink_cmap(pos) for pos in positions]
    ai_colors = [teal_cmap(pos) for pos in positions]
    return human_colors, ai_colors
