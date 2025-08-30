#!/Users/donyin/miniconda3/bin/python

"""
Probing Type Effectiveness Analysis

This script analyzes which types of probing lead to deeper responses in human and AI conversations. It provides comprehensive statistical analysis and visualizations to answer Research Question 2.

Main functionality:
- load and process conversation data
- analyze probing effectiveness metrics
- run statistical tests (ANOVA, mixed effects models)
- generate comprehensive visualizations
- provide insights and recommendations

Usage:
    python probing_type_analysis.py
"""

import warnings
from rich import print

warnings.filterwarnings("ignore")

# Import analysis modules via package imports
from src.analysis.utils import load_all_conversations, create_probing_response_pairs, build_interviewee_depths_dataframe
from src.analysis.analysis import analyze_probing_effectiveness, generate_insights
from src.analysis.statistics import run_statistical_analysis
from src.analysis.visualization import create_comprehensive_visualization
from donware import banner


def main():
    """Main analysis function."""
    print("PROBING TYPE EFFECTIVENESS ANALYSIS")
    print("Research Question 2: What Types of Probing Lead to Deeper Responses?")
    banner("-")

    # load data
    print("\nLoading Conversations...")
    conversations = load_all_conversations()
    print(f"[green]Loaded {len(conversations)} conversations total[/green]")
    print("\nCreating Probing-Response Pairs...")
    df = create_probing_response_pairs(conversations)
    depths_df = build_interviewee_depths_dataframe(conversations)

    print(f"[green]Analysis dataset: {len(df)} probing-response pairs[/green]")
    print("\nRunning Analysis Components...")
    effectiveness = analyze_probing_effectiveness(df)
    statistical_results = run_statistical_analysis(df)
    print(f"\nCreating Visualizations...")
    fig = create_comprehensive_visualization(df, effectiveness, depths_df=depths_df)
    insights = generate_insights(df, effectiveness)

    return {"data": df, "effectiveness": effectiveness, "statistical_results": statistical_results, "insights": insights, "figure": fig}


if __name__ == "__main__":
    results = main()
