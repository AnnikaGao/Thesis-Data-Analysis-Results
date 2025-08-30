"""
Goal Mapping Frequency Analysis

Creates bar plots showing the frequency of different goal mappings 
for human vs AI interviews using the core parser.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from src.parsers.core import load_all_scripts
from collections import Counter
from pathlib import Path
import sys

# Add src to path to import our parser
sys.path.append(str(Path(__file__).parent / "src"))


def extract_goal_mappings(scripts):
    """Extract all goal mappings from a list of scripts."""
    all_mappings = []
    for script in scripts:
        for round in script.rounds:
            # goal_mapping returns a list of strings
            mappings = round.goal_mapping
            for mapping in mappings:
                if mapping:  # Only add non-empty mappings
                    all_mappings.append(mapping)
    return all_mappings


def create_goal_mapping_plots():
    """Create bar plots showing goal mapping frequencies for human vs AI."""

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Load all scripts from data directory
    print("Loading scripts...")
    scripts = load_all_scripts(data_root="data", subdirs=["human", "ai"])

    # Separate human and AI scripts
    human_scripts = [s for s in scripts if s.nature == "human"]
    ai_scripts = [s for s in scripts if s.nature == "ai"]

    print(f"Found {len(human_scripts)} human scripts and {len(ai_scripts)} AI scripts")

    # Extract goal mappings
    human_mappings = extract_goal_mappings(human_scripts)
    ai_mappings = extract_goal_mappings(ai_scripts)

    print(f"Extracted {len(human_mappings)} human mappings and {len(ai_mappings)} AI mappings")

    # Count frequencies
    human_counts = Counter(human_mappings)
    ai_counts = Counter(ai_mappings)

    # Get all unique mappings and sort by frequency (combined)
    all_mappings = set(human_mappings + ai_mappings)
    combined_counts = Counter(human_mappings + ai_mappings)
    sorted_mappings = sorted(all_mappings, key=lambda x: combined_counts[x], reverse=True)

    # Prepare data for plotting
    human_freqs = [human_counts.get(mapping, 0) for mapping in sorted_mappings]
    ai_freqs = [ai_counts.get(mapping, 0) for mapping in sorted_mappings]

    # Create the plots with seaborn styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Use seaborn color palette
    colors = sns.color_palette("husl", 2)

    # Human plot with seaborn styling
    bars1 = sns.barplot(x=list(range(len(sorted_mappings))), y=human_freqs, color=colors[0], alpha=0.8, ax=ax1)
    ax1.set_title("Goal Mapping Frequency - Human Interviews", fontsize=16, fontweight="bold", pad=20)
    ax1.set_xlabel("Goal Mapping Categories", fontsize=13, fontweight="medium")
    ax1.set_ylabel("Frequency", fontsize=13, fontweight="medium")
    ax1.set_xticks(range(len(sorted_mappings)))
    ax1.set_xticklabels(sorted_mappings, rotation=45, ha="right", fontsize=11)
    ax1.tick_params(axis="y", labelsize=11)

    # Add value labels on bars
    for i, freq in enumerate(human_freqs):
        if freq > 0:
            ax1.text(i, freq + max(human_freqs) * 0.01, str(freq), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # AI plot with seaborn styling
    bars2 = sns.barplot(x=list(range(len(sorted_mappings))), y=ai_freqs, color=colors[1], alpha=0.8, ax=ax2)
    ax2.set_title("Goal Mapping Frequency - AI Interviews", fontsize=16, fontweight="bold", pad=20)
    ax2.set_xlabel("Goal Mapping Categories", fontsize=13, fontweight="medium")
    ax2.set_ylabel("Frequency", fontsize=13, fontweight="medium")
    ax2.set_xticks(range(len(sorted_mappings)))
    ax2.set_xticklabels(sorted_mappings, rotation=45, ha="right", fontsize=11)
    ax2.tick_params(axis="y", labelsize=11)

    # Add value labels on bars
    for i, freq in enumerate(ai_freqs):
        if freq > 0:
            ax2.text(i, freq + max(ai_freqs) * 0.01, str(freq), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Adjust layout and save
    plt.tight_layout()

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save the plot with high quality
    output_path = results_dir / "goal_mapping_frequency_comparison.png"
    plt.savefig(output_path, dpi=420, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Plot saved to: {output_path}")

    # Also save as PDF for publication quality
    pdf_path = results_dir / "goal_mapping_frequency_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"PDF version saved to: {pdf_path}")

    # Display summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total Human Goal Mappings: {len(human_mappings)}")
    print(f"Total AI Goal Mappings: {len(ai_mappings)}")
    print(f"Unique Goal Mapping Categories: {len(all_mappings)}")

    print(f"\nTop 5 Human Goal Mappings:")
    for mapping, count in human_counts.most_common(5):
        print(f"  {mapping}: {count}")

    print(f"\nTop 5 AI Goal Mappings:")
    for mapping, count in ai_counts.most_common(5):
        print(f"  {mapping}: {count}")


if __name__ == "__main__":
    create_goal_mapping_plots()
