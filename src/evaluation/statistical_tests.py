"""
Statistical Tests
Significance testing and effect size computation
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List


def format_p_value(p: float) -> str:
    """
    Format p-value for publication according to journal standards.

    Rules:
    - p < 0.001: report as "p < 0.001"
    - p >= 0.001: report exact value with 3 decimal places
    - Never report p = 0.000

    Args:
        p: Raw p-value

    Returns:
        Formatted p-value string
    """
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"


def ttest_independent(group1: np.ndarray,
                     group2: np.ndarray,
                     alternative: str = 'two-sided') -> Dict:
    """
    Independent samples t-test with proper statistical reporting.

    Args:
        group1: First group of scores
        group2: Second group of scores
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dictionary with test results including degrees of freedom
        for proper reporting: t(df) = X.XX, p = Y.YYY
    """
    n1, n2 = len(group1), len(group2)
    t_stat, p_value = stats.ttest_ind(group1, group2, alternative=alternative)

    # Degrees of freedom for independent samples t-test
    # For equal variance assumption: df = n1 + n2 - 2
    df = n1 + n2 - 2

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'p_value_formatted': format_p_value(p_value),
        'df': int(df),  # Degrees of freedom for proper reporting
        'significant': p_value < 0.05,
        'group1_n': int(n1),
        'group2_n': int(n2),
        'group1_mean': float(np.mean(group1)),
        'group2_mean': float(np.mean(group2)),
        'group1_std': float(np.std(group1)),
        'group2_std': float(np.std(group2)),
        # Formatted string for publication: t(df) = X.XX, p = Y.YYY
        'report_string': f"t({df}) = {t_stat:.2f}, {format_p_value(p_value)}",
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size

    Args:
        group1: First group
        group2: Second group

    Returns:
        Cohen's d value (returns 0.0 if computation not possible)

    Interpretation:
        |d| < 0.2: small effect
        |d| < 0.5: medium effect
        |d| >= 0.5: large effect
    """
    n1, n2 = len(group1), len(group2)

    # Guard: Need at least 2 samples in each group for valid variance calculation
    if n1 < 2 or n2 < 2:
        print(f"Warning: Cohen's d requires n >= 2 per group. Got n1={n1}, n2={n2}. Returning 0.0")
        return 0.0

    # Guard: Check for denominator being zero (n1 + n2 - 2 == 0)
    if n1 + n2 - 2 == 0:
        print("Warning: Cohen's d denominator is zero. Returning 0.0")
        return 0.0

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)

    # Guard: Zero pooled std means both groups have zero variance (all identical values)
    if pooled_std == 0 or np.isnan(pooled_std):
        # If means are different but std is 0, effect is theoretically infinite
        # But we return 0 to indicate "cannot compute"
        mean_diff = np.mean(group1) - np.mean(group2)
        if mean_diff == 0:
            return 0.0  # No difference
        else:
            print(f"Warning: pooled_std is zero but means differ by {mean_diff}. Returning 0.0")
            return 0.0

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    return float(d)


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size
    
    Args:
        d: Cohen's d value
    
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def anova_oneway(groups: List[np.ndarray]) -> Dict:
    """
    One-way ANOVA
    
    Args:
        groups: List of groups to compare
    
    Returns:
        Dictionary with ANOVA results
    """
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'num_groups': len(groups),
    }


def compare_all_models(results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compare memorization across all models
    
    Args:
        results_dict: Dictionary mapping model names to results DataFrames
    
    Returns:
        DataFrame with pairwise comparisons
    """
    model_names = list(results_dict.keys())
    comparisons = []
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            scores1 = results_dict[model1]['memorization_score'].values
            scores2 = results_dict[model2]['memorization_score'].values
            
            # T-test
            ttest_result = ttest_independent(scores1, scores2)
            
            # Effect size
            d = cohens_d(scores1, scores2)
            
            comparison = {
                'model1': model1,
                'model2': model2,
                'model1_mean': ttest_result['group1_mean'],
                'model2_mean': ttest_result['group2_mean'],
                'difference': ttest_result['group1_mean'] - ttest_result['group2_mean'],
                't_statistic': ttest_result['t_statistic'],
                'p_value': ttest_result['p_value'],
                'significant': ttest_result['significant'],
                'cohens_d': d,
                'effect_size': interpret_cohens_d(d),
            }
            
            comparisons.append(comparison)
    
    return pd.DataFrame(comparisons)


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Bonferroni correction for multiple comparisons
    
    Args:
        p_values: List of p-values
    
    Returns:
        Corrected p-values
    """
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    return corrected


def compute_correlation(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Compute Pearson correlation
    
    Args:
        x: First variable
        y: Second variable
    
    Returns:
        Dictionary with correlation results
    """
    r, p_value = stats.pearsonr(x, y)
    
    return {
        'correlation': float(r),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
    }


def test_monotonic_trend(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Test for monotonic trend using Spearman correlation
    
    Args:
        x: Independent variable (e.g., epochs)
        y: Dependent variable (e.g., memorization)
    
    Returns:
        Dictionary with trend test results
    """
    rho, p_value = stats.spearmanr(x, y)
    
    return {
        'spearman_rho': float(rho),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'trend': 'increasing' if rho > 0 else 'decreasing',
    }


def generate_statistical_report(results_dict: Dict[str, pd.DataFrame],
                               output_path: str = None) -> str:
    """
    Generate complete statistical report
    
    Args:
        results_dict: Dictionary mapping model names to results DataFrames
        output_path: Path to save report (optional)
    
    Returns:
        Report string
    """
    report = []
    report.append("="*80)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    # Summary statistics per model
    report.append("Summary Statistics:")
    report.append("-"*80)
    for model_name, df in results_dict.items():
        scores = df['memorization_score']
        report.append(f"\n{model_name}:")
        report.append(f"  Mean: {scores.mean():.4f}")
        report.append(f"  Std:  {scores.std():.4f}")
        report.append(f"  Min:  {scores.min():.4f}")
        report.append(f"  Max:  {scores.max():.4f}")
        report.append(f"  N:    {len(scores)}")
    report.append("")
    
    # Pairwise comparisons
    report.append("\nPairwise Comparisons:")
    report.append("-"*80)
    comparisons = compare_all_models(results_dict)
    
    for _, row in comparisons.iterrows():
        report.append(f"\n{row['model1']} vs {row['model2']}:")
        report.append(f"  Difference: {row['difference']:.4f}")
        report.append(f"  t-statistic: {row['t_statistic']:.4f}")
        report.append(f"  p-value: {row['p_value']:.4f} {'*' if row['significant'] else ''}")
        report.append(f"  Cohen's d: {row['cohens_d']:.4f} ({row['effect_size']})")
    
    report.append("")
    report.append("* p < 0.05")
    report.append("="*80)
    
    report_str = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_str)
        print(f"✅ Statistical report saved: {output_path}")
    
    return report_str


# Example usage
if __name__ == "__main__":
    print("Statistical Tests Module")
    print()
    print("Provides:")
    print("  - Independent t-tests")
    print("  - Cohen's d effect sizes")
    print("  - ANOVA")
    print("  - Correlation analysis")
    print("  - Bonferroni correction")
    print("  - Statistical reports")