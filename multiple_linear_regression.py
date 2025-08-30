#!/Users/donyin/miniconda3/bin/python

"""
Multiple Linear Regression Analysis

This script performs multiple linear regression analysis using the same
parameters and data preparation as the mixed effects model, but without
accounting for conversation-level clustering.
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from rich import print
from rich.table import Table
from rich.console import Console
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from donware import banner
from src.analysis.utils import load_all_conversations, scale_columns
from src.utils.fix_categorical_variables import prepare_data_for_mixed_effects


@dataclass(frozen=True)
class AnalysisParams:
    numeric_summary_cols: list[str]
    categorical_cols: list[str]
    min_category_count: int
    min_group_size: int
    outlier_cap_cols: list[str]
    outlier_cap_quantile: float
    continuous_vars: list[str]
    model_formula: str


# Same parameters as mixed effects model
PARAMS = AnalysisParams(
    numeric_summary_cols=["depth", "num_goal_mappings", "transcript_length", "session_time_length", "num_themes", "num_probing_types", "num_purposes_if_irrelevant"],
    categorical_cols=["conversation_nature", "goal_relevant", "primary_goal_mapping"],
    min_category_count=10,
    min_group_size=10,
    outlier_cap_cols=["session_time_length"],
    outlier_cap_quantile=0.95,
    continuous_vars=["num_goal_mappings", "session_time_length", "num_themes", "num_probing_types", "num_purposes_if_irrelevant"],
    model_formula=("depth ~ C(conversation_nature) + C(goal_relevant) + " "C(primary_goal_mapping) + " "session_time_length + num_themes + num_probing_types + num_purposes_if_irrelevant"),
)


def create_analysis_dataframe(conversations):
    """Convert conversations to a dataframe suitable for linear regression modeling"""
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
    is_interviewee = df["speaker"].astype(str).str.strip().str.lower() == "interviewee"
    df = df[is_interviewee & df["depth"].notna()].copy()
    return df


def perform_diagnostic_tests(model_result, df_clean):
    """Perform regression diagnostic tests"""
    diagnostics = {}

    # Get residuals and fitted values
    residuals = model_result.resid
    fitted = model_result.fittedvalues

    # 1. Normality test (Shapiro-Wilk for small samples, Kolmogorov-Smirnov for large)
    if len(residuals) <= 5000:
        normality_stat, normality_p = stats.shapiro(residuals)
        normality_test = "Shapiro-Wilk"
    else:
        normality_stat, normality_p = stats.kstest(residuals, "norm")
        normality_test = "Kolmogorov-Smirnov"

    diagnostics["normality"] = {"test": normality_test, "statistic": normality_stat, "p_value": normality_p, "assumption_met": normality_p > 0.05}

    # 2. Homoscedasticity test (Breusch-Pagan)
    from statsmodels.stats.diagnostic import het_breuschpagan

    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model_result.model.exog)
    diagnostics["homoscedasticity"] = {"test": "Breusch-Pagan", "statistic": bp_stat, "p_value": bp_p, "assumption_met": bp_p > 0.05}

    # 3. Durbin-Watson test for autocorrelation
    from statsmodels.stats.stattools import durbin_watson

    dw_stat = durbin_watson(residuals)
    diagnostics["autocorrelation"] = {"test": "Durbin-Watson", "statistic": dw_stat, "assumption_met": 1.5 < dw_stat < 2.5}  # Rule of thumb

    # 4. Multicollinearity check (VIF)
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Only check VIF for continuous variables
    continuous_mask = []
    exog_df = pd.DataFrame(model_result.model.exog, columns=model_result.model.exog_names)

    vif_data = []
    for i, col in enumerate(exog_df.columns):
        if col != "Intercept":  # Skip intercept
            # Check if column appears to be continuous (not all 0s and 1s)
            unique_vals = exog_df[col].nunique()
            if unique_vals > 2:  # Likely continuous
                try:
                    vif = variance_inflation_factor(model_result.model.exog, i)
                    vif_data.append({"variable": col, "vif": vif})
                except:
                    pass

    diagnostics["multicollinearity"] = {"vif_values": vif_data, "max_vif": max([x["vif"] for x in vif_data]) if vif_data else None, "assumption_met": all(x["vif"] < 10 for x in vif_data) if vif_data else True}

    return diagnostics


def create_diagnostic_plots(model_result, df_clean):
    """Create diagnostic plots for regression analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    residuals = model_result.resid
    fitted = model_result.fittedvalues

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color="red", linestyle="--")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Fitted Values")

    # 2. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot (Normality Check)")

    # 3. Scale-Location Plot
    standardized_residuals = np.sqrt(np.abs(residuals / np.std(residuals)))
    axes[1, 0].scatter(fitted, standardized_residuals, alpha=0.6)
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("√|Standardized Residuals|")
    axes[1, 0].set_title("Scale-Location Plot")

    # 4. Cook's Distance
    from statsmodels.stats.outliers_influence import OLSInfluence

    influence = OLSInfluence(model_result)
    cooks_d = influence.cooks_distance[0]
    axes[1, 1].stem(range(len(cooks_d)), cooks_d, markerfmt=",")
    axes[1, 1].axhline(y=4 / len(cooks_d), color="red", linestyle="--", label="Threshold (4/n)")
    axes[1, 1].set_xlabel("Observation Index")
    axes[1, 1].set_ylabel("Cook's Distance")
    axes[1, 1].set_title("Cook's Distance (Influence)")
    axes[1, 1].legend()

    plt.tight_layout()

    # Save diagnostic plots
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "mlr_diagnostic_plots.png", dpi=300, bbox_inches="tight")
    plt.savefig(results_dir / "mlr_diagnostic_plots.pdf", bbox_inches="tight")

    return fig


def run_multiple_linear_regression():
    """Run multiple linear regression analysis"""
    print("MULTIPLE LINEAR REGRESSION ANALYSIS")
    print("Using same parameters as mixed effects model (without clustering)")
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

    print(f"\nPreparing Data for Multiple Linear Regression...")

    # Handle NaN values in categorical columns
    for col in PARAMS.categorical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"Filling {nan_count} NaN values in '{col}' with 'Unknown'")
                df[col] = df[col].fillna("Unknown")

    # Use the same data preparation function as mixed effects
    df_clean, processing_info = prepare_data_for_mixed_effects(df, group_col="conversation_id", categorical_cols=PARAMS.categorical_cols, min_category_count=PARAMS.min_category_count, min_group_size=PARAMS.min_group_size)

    # Cap extreme outliers to improve model stability
    for col in set(PARAMS.outlier_cap_cols) & set(df_clean.columns):
        q95 = df_clean[col].quantile(PARAMS.outlier_cap_quantile)
        df_clean[col] = df_clean[col].clip(upper=q95)

    # Scale continuous variables
    df_clean, scaling_info = scale_columns(df_clean, PARAMS.continuous_vars)

    print(f"\nFitting Multiple Linear Regression Model...")

    # Data cleaning for statsmodels compatibility
    initial_len = len(df_clean)
    df_clean = df_clean.dropna()
    if len(df_clean) < initial_len:
        print(f"Removed {initial_len - len(df_clean)} rows with NaN values")

    print(f"Final data for modeling: {len(df_clean)} observations")

    # Fit the model
    model = smf.ols(PARAMS.model_formula, df_clean)
    result = model.fit()

    print("\nMultiple Linear Regression Results:")
    print(result.summary())

    # Model performance metrics
    print(f"\nModel Performance Metrics:")
    print(f"R-squared: {result.rsquared:.4f}")
    print(f"Adjusted R-squared: {result.rsquared_adj:.4f}")
    print(f"F-statistic: {result.fvalue:.4f}")
    print(f"F-statistic p-value: {result.f_pvalue:.4e}")
    print(f"AIC: {result.aic:.2f}")
    print(f"BIC: {result.bic:.2f}")

    # Perform diagnostic tests
    print(f"\nPerforming Regression Diagnostics...")
    diagnostics = perform_diagnostic_tests(result, df_clean)

    print(f"\nRegression Assumption Checks:")
    print(f"1. Normality of Residuals ({diagnostics['normality']['test']}):")
    print(f"   Statistic: {diagnostics['normality']['statistic']:.4f}")
    print(f"   P-value: {diagnostics['normality']['p_value']:.4e}")
    print(f"   Assumption met: {diagnostics['normality']['assumption_met']}")

    print(f"\n2. Homoscedasticity ({diagnostics['homoscedasticity']['test']}):")
    print(f"   Statistic: {diagnostics['homoscedasticity']['statistic']:.4f}")
    print(f"   P-value: {diagnostics['homoscedasticity']['p_value']:.4e}")
    print(f"   Assumption met: {diagnostics['homoscedasticity']['assumption_met']}")

    print(f"\n3. No Autocorrelation ({diagnostics['autocorrelation']['test']}):")
    print(f"   Statistic: {diagnostics['autocorrelation']['statistic']:.4f}")
    print(f"   Assumption met: {diagnostics['autocorrelation']['assumption_met']}")

    if diagnostics["multicollinearity"]["vif_values"]:
        print(f"\n4. Multicollinearity (VIF Analysis):")
        for vif_info in diagnostics["multicollinearity"]["vif_values"]:
            print(f"   {vif_info['variable']}: VIF = {vif_info['vif']:.2f}")
        print(f"   Max VIF: {diagnostics['multicollinearity']['max_vif']:.2f}")
        print(f"   Assumption met (all VIF < 10): {diagnostics['multicollinearity']['assumption_met']}")

    # Create diagnostic plots
    print(f"\nCreating Diagnostic Plots...")
    fig = create_diagnostic_plots(result, df_clean)
    print(f"Diagnostic plots saved to: results/mlr_diagnostic_plots.png and .pdf")

    # Effect sizes (Cohen's f²)
    print(f"\nEffect Sizes (Cohen's f²):")
    for param in result.params.index:
        if param != "Intercept":
            # Calculate partial R² for each predictor
            # This is an approximation - true partial R² would require refitting without each variable
            t_val = result.tvalues[param]
            df_resid = result.df_resid
            partial_r2 = t_val**2 / (t_val**2 + df_resid)
            cohens_f2 = partial_r2 / (1 - partial_r2)
            print(f"   {param}: f² = {cohens_f2:.4f}")

    # Prediction accuracy on training data
    predictions = result.fittedvalues
    actual = df_clean["depth"]
    mae = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))

    print(f"\nPrediction Accuracy (Training Data):")
    print(f"   Mean Absolute Error (MAE): {mae:.4f}")
    print(f"   Root Mean Square Error (RMSE): {rmse:.4f}")

    # Comparison with mixed effects model
    print(f"\nComparison Notes:")
    print(f"- This MLR ignores conversation-level clustering")
    print(f"- Standard errors may be underestimated if clustering is important")
    print(f"- Compare with mixed effects ICC to assess clustering importance")

    return result, df_clean, diagnostics


if __name__ == "__main__":
    result, df, diagnostics = run_multiple_linear_regression()
