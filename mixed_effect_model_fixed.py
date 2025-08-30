#!/Users/donyin/miniconda3/bin/python

"""
Fixed Mixed Effects Model Analysis

This version addresses convergence issues by:
1. Properly consolidating sparse categorical variables
2. Implementing robust data quality checks
3. Using appropriate model fitting procedures
4. Removing warning suppression in favor of root cause fixes
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from rich import print
from rich.table import Table
from rich.console import Console

from donware import banner
from src.analysis.utils import load_all_conversations, scale_columns
from src.utils.fix_categorical_variables import prepare_data_for_mixed_effects
import statsmodels.formula.api as smf


@dataclass(frozen=True)
class AnalysisParams:
    numeric_summary_cols: list[str]
    categorical_cols: list[str]
    min_category_count: int
    min_group_size: int
    outlier_cap_cols: list[str]
    outlier_cap_quantile: float
    continuous_vars: list[str]
    model_method: str
    model_reml: bool
    model_maxiter: int
    model_formula: str


PARAMS = AnalysisParams(
    numeric_summary_cols=["depth", "num_goal_mappings", "transcript_length", "session_time_length", "num_themes", "num_probing_types", "num_purposes_if_irrelevant"],
    categorical_cols=["conversation_nature", "goal_relevant", "primary_goal_mapping"],
    min_category_count=10,  # amend later
    min_group_size=10,
    outlier_cap_cols=["session_time_length"],
    outlier_cap_quantile=0.95,
    continuous_vars=["num_goal_mappings", "session_time_length", "num_themes", "num_probing_types", "num_purposes_if_irrelevant"],
    model_method="bfgs",
    model_reml=False,
    model_maxiter=3000,
    model_formula=("depth ~ C(conversation_nature) + C(goal_relevant) + " "C(primary_goal_mapping) + " "session_time_length + num_themes + num_probing_types + num_purposes_if_irrelevant"),
)


def create_analysis_dataframe(conversations):
    """Convert conversations to a dataframe suitable for mixed effects modeling"""
    rows = (
        {
            "conversation_id": conv.id,
            "conversation_nature": conv.nature,
            "speaker": msg.speaker,
            "depth": msg.depth,
            "goal_relevant": msg.goal_relevant,
            # store first/primary goal mapping for categorical analysis
            "primary_goal_mapping": (msg.goal_mapping[0] if isinstance(msg.goal_mapping, list) and len(msg.goal_mapping) > 0 else "None"),
            "num_goal_mappings": len(msg.goal_mapping),
            "transcript_length": len(msg.transcript),
            # Add session-level information
            "session_time_length": conv.session_time_length,
            "mapped_q": msg.mapped_q,
            "primary_theme": (msg.themes[0] if isinstance(msg.themes, list) and len(msg.themes) > 0 else "None"),
            "num_themes": len(msg.themes),
            "primary_probing_type": (msg.probing_type[0] if isinstance(msg.probing_type, list) and len(msg.probing_type) > 0 else "None"),
            "num_probing_types": len(msg.probing_type),
            "primary_purpose_if_irrelevant": (msg.purpose_if_irrelevant[0] if isinstance(msg.purpose_if_irrelevant, list) and len(msg.purpose_if_irrelevant) > 0 else "None"),
            "num_purposes_if_irrelevant": len(msg.purpose_if_irrelevant),
        }
        for conv in conversations
        for msg in conv.rounds
    )
    df = pd.DataFrame(list(rows))
    # align with t-test unit of analysis: interviewee messages, including zeros
    # NOTE if we want 0 exclusion back add it here
    is_interviewee = df["speaker"].astype(str).str.strip().str.lower() == "interviewee"
    df = df[is_interviewee & df["depth"].notna()].copy()
    return df


def run_mixed_effects_model():
    """Run mixed effects model analysis with proper convergence handling"""
    print("FIXED MIXED EFFECTS MODEL ANALYSIS")
    print("Addressing convergence issues through proper data preparation")
    banner("-")

    print("\nLoading Conversations...")
    conversations = load_all_conversations()
    print(f"Loaded {len(conversations)} conversations total")

    print("\nCreating Analysis DataFrame...")
    df = create_analysis_dataframe(conversations)
    print(f"Created dataframe with {len(df)} valid messages")

    # Display basic statistics
    console = Console()
    table = Table(title="Data Summary")
    table.add_column("Variable")
    table.add_column("Mean")
    table.add_column("Std")
    table.add_column("Min")
    table.add_column("Max")

    numeric_cols = PARAMS.numeric_summary_cols
    for col in [c for c in numeric_cols if c in df.columns]:
        table.add_row(col, f"{df[col].mean():.3f}", f"{df[col].std():.3f}", f"{df[col].min():.3f}", f"{df[col].max():.3f}")

    console.print(table)

    print(f"\nDistribution by Conversation Nature:")
    nature_summary = df.groupby("conversation_nature")["depth"].agg(["count", "mean", "std"]).round(3)
    print(nature_summary)

    print(f"\nPreparing Data for Mixed Effects Model...")

    # Handle NaN values in categorical columns before mixed effects preparation
    for col in PARAMS.categorical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"Filling {nan_count} NaN values in '{col}' with 'Unknown'")
                df[col] = df[col].fillna("Unknown")

    # Use the proper data preparation function
    df_clean, processing_info = prepare_data_for_mixed_effects(df, group_col="conversation_id", categorical_cols=PARAMS.categorical_cols, min_category_count=PARAMS.min_category_count, min_group_size=PARAMS.min_group_size)

    # Cap extreme outliers to improve model stability
    for col in set(PARAMS.outlier_cap_cols) & set(df_clean.columns):
        q95 = df_clean[col].quantile(PARAMS.outlier_cap_quantile)
        df_clean[col] = df_clean[col].clip(upper=q95)

    # Scale continuous variables
    df_clean, _ = scale_columns(df_clean, PARAMS.continuous_vars)
    print(f"Estimated Intraclass Correlation Coefficient (ICC): {processing_info.get('estimated_icc', float('nan')):.3f}")

    print(f"\nFitting Mixed Effects Model...")

    # Comprehensive data cleaning for statsmodels compatibility
    # 1. Handle missing values first
    initial_len = len(df_clean)
    df_clean = df_clean.dropna()
    if len(df_clean) < initial_len:
        print(f"Removed {initial_len - len(df_clean)} rows with NaN values")

    # 2. Ensure proper data types
    df_clean["conversation_id"] = df_clean["conversation_id"].astype(str)
    df_clean = df_clean.sort_values("conversation_id").reset_index(drop=True)
    group_sizes = df_clean.groupby("conversation_id").size()
    print(f"Group sizes range: {group_sizes.min()} to {group_sizes.max()}")
    df_model = df_clean.copy()

    print(f"Final data for modeling: {len(df_model)} observations, {len(group_sizes)} groups")
    model_formula = PARAMS.model_formula
    mixed_model = smf.mixedlm(model_formula, df_model, groups=df_model["conversation_id"])
    result = mixed_model.fit(method=PARAMS.model_method, reml=PARAMS.model_reml, maxiter=PARAMS.model_maxiter)
    print(f"Using optimizer=BFGS, reml=False, maxiter={PARAMS.model_maxiter}")
    print(f"Converged: {getattr(result, 'converged', False)}")
    print("\nMixed Effects Model Results:")
    print(result.summary())

    cov_re = getattr(result, "cov_re", None)
    random_effects_var = cov_re.iloc[0, 0] if isinstance(cov_re, pd.DataFrame) and cov_re.size else np.nan
    residual_var = getattr(result, "scale", np.nan)
    icc = random_effects_var / (random_effects_var + residual_var) if (random_effects_var == random_effects_var) and (residual_var == residual_var) else np.nan
    print(f"\nIntraclass Correlation Coefficient (ICC): {icc:.3f}")
    print("This represents the proportion of variance explained by conversation-level differences")

    re_df = pd.DataFrame(getattr(result, "random_effects", {})).T
    random_intercepts = re_df.shape[1] and re_df.iloc[:, 0]
    ri_min = getattr(random_intercepts, "min", lambda: np.nan)()
    ri_max = getattr(random_intercepts, "max", lambda: np.nan)()
    print(f"\nRandom Effects Summary:")
    print(f"Random intercept range: {ri_min:.3f} to {ri_max:.3f}")

    # if random effects variance collapsed or fit did not converge, provide a fixed-effects fallback
    singular_re = not isinstance(cov_re, pd.DataFrame) or (cov_re.size == 0) or (not np.isfinite(random_effects_var)) or (float(random_effects_var) <= 1e-8)
    if singular_re or not bool(getattr(result, "converged", False)):
        print("\nRandom-effects covariance is singular or model did not converge.")
        print("Providing a fixed-effects OLS fit as a fallback for inference on fixed terms.")
        ols_res = smf.ols(model_formula, df_clean).fit()
        print("\nFixed-Effects OLS Results (fallback):")
        print(ols_res.summary())

    return result, df_clean


if __name__ == "__main__":
    result, df = run_mixed_effects_model()
