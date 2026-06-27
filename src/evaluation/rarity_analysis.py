"""
Rarity Analysis Module
Statistical analysis of memorization differences between rare and common disease classes.

This module provides rigorous statistical testing to validate the hypothesis that
rare disease canaries are memorized significantly more than common disease canaries.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings


# =============================================================================
# RARITY TIER DEFINITIONS
# =============================================================================

DEFAULT_RARITY_THRESHOLDS = {
    'very_rare': 0.02,   # < 2% of dataset
    'rare': 0.05,        # 2-5% of dataset
    'moderate': 0.15,    # 5-15% of dataset
    'common': 1.0        # >= 15% of dataset
}


def compute_class_frequencies(df: pd.DataFrame, class_column: str = 'class_name') -> pd.DataFrame:
    """
    Compute frequency statistics for each class.

    Args:
        df: DataFrame with per-sample memorization scores
        class_column: Column containing class labels

    Returns:
        DataFrame with class frequencies and percentages
    """
    class_counts = df[class_column].value_counts()
    total = len(df)

    freq_df = pd.DataFrame({
        'class': class_counts.index,
        'count': class_counts.values,
        'frequency': class_counts.values / total,
        'percentage': (class_counts.values / total) * 100
    })

    return freq_df.sort_values('frequency', ascending=True).reset_index(drop=True)


def assign_rarity_tiers(freq_df: pd.DataFrame,
                        thresholds: Dict[str, float] = None) -> pd.DataFrame:
    """
    Assign rarity tiers to classes based on frequency thresholds.

    Args:
        freq_df: DataFrame from compute_class_frequencies()
        thresholds: Custom thresholds (default: very_rare<2%, rare<5%, moderate<15%, common>=15%)

    Returns:
        DataFrame with rarity_tier column added
    """
    if thresholds is None:
        thresholds = DEFAULT_RARITY_THRESHOLDS

    def get_tier(freq):
        if freq < thresholds['very_rare']:
            return 'very_rare'
        elif freq < thresholds['rare']:
            return 'rare'
        elif freq < thresholds['moderate']:
            return 'moderate'
        else:
            return 'common'

    freq_df = freq_df.copy()
    freq_df['rarity_tier'] = freq_df['frequency'].apply(get_tier)

    return freq_df


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def spearman_correlation(freq_df: pd.DataFrame,
                         scores_df: pd.DataFrame,
                         class_column: str = 'class_name') -> Dict:
    """
    Compute Spearman correlation between class frequency and mean memorization.

    Tests the hypothesis that rarer classes have higher memorization scores.
    Negative correlation indicates rare classes are memorized more.

    Args:
        freq_df: DataFrame with class frequencies
        scores_df: DataFrame with per-sample memorization scores

    Returns:
        Dictionary with correlation results
    """
    # Guard: Check for empty inputs
    if len(freq_df) == 0 or len(scores_df) == 0:
        return {
            'spearman_rho': float('nan'),
            'p_value': float('nan'),
            'significant': False,
            'interpretation': 'Cannot compute - empty input data',
            'n_classes': 0,
            'data': []
        }

    # Compute mean memorization per class
    class_means = scores_df.groupby(class_column)['memorization_score'].mean()

    # Merge with frequency data
    merged = freq_df.merge(
        class_means.reset_index().rename(columns={'memorization_score': 'mean_memorization'}),
        left_on='class', right_on=class_column
    )

    # Guard: Check for empty merged result or insufficient data for correlation
    if len(merged) < 3:
        return {
            'spearman_rho': float('nan'),
            'p_value': float('nan'),
            'significant': False,
            'interpretation': f'Cannot compute - need at least 3 classes, got {len(merged)}',
            'n_classes': int(len(merged)),
            'data': []
        }

    # Spearman correlation
    rho, p_value = stats.spearmanr(merged['frequency'], merged['mean_memorization'])

    # Convert data to JSON-safe format
    data_records = []
    for _, row in merged[['class', 'frequency', 'mean_memorization']].iterrows():
        data_records.append({
            'class': str(row['class']),
            'frequency': float(row['frequency']),
            'mean_memorization': float(row['mean_memorization'])
        })

    return {
        'spearman_rho': float(rho),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'interpretation': 'Rare classes memorized MORE' if rho < 0 else 'Common classes memorized MORE',
        'n_classes': int(len(merged)),
        'data': data_records
    }


def mann_whitney_u_test(scores_df: pd.DataFrame,
                        freq_df: pd.DataFrame,
                        rare_threshold: float = 0.05,
                        class_column: str = 'class_name') -> Dict:
    """
    Mann-Whitney U test comparing memorization of rare vs common classes.

    Non-parametric test suitable for non-normal distributions.
    Tests if rare class samples have significantly higher memorization scores.

    Args:
        scores_df: DataFrame with per-sample memorization scores
        freq_df: DataFrame with class frequencies
        rare_threshold: Classes with frequency < threshold are "rare"

    Returns:
        Dictionary with test results
    """
    # Identify rare and common classes
    rare_classes = freq_df[freq_df['frequency'] < rare_threshold]['class'].tolist()
    common_classes = freq_df[freq_df['frequency'] >= rare_threshold]['class'].tolist()

    # Get scores for each group
    rare_scores = scores_df[scores_df[class_column].isin(rare_classes)]['memorization_score'].values
    common_scores = scores_df[scores_df[class_column].isin(common_classes)]['memorization_score'].values

    if len(rare_scores) == 0 or len(common_scores) == 0:
        return {'error': 'One or both groups have no samples'}

    # Mann-Whitney U test (one-sided: rare > common)
    statistic, p_value_two_sided = stats.mannwhitneyu(rare_scores, common_scores, alternative='two-sided')
    _, p_value_greater = stats.mannwhitneyu(rare_scores, common_scores, alternative='greater')

    return {
        'u_statistic': float(statistic),
        'p_value_two_sided': float(p_value_two_sided),
        'p_value_one_sided': float(p_value_greater),
        'significant_two_sided': bool(p_value_two_sided < 0.05),
        'significant_one_sided': bool(p_value_greater < 0.05),
        'rare_n': int(len(rare_scores)),
        'common_n': int(len(common_scores)),
        'rare_mean': float(np.mean(rare_scores)),
        'common_mean': float(np.mean(common_scores)),
        'rare_median': float(np.median(rare_scores)),
        'common_median': float(np.median(common_scores)),
        'rare_classes': [str(c) for c in rare_classes],
        'common_classes': [str(c) for c in common_classes],
        'threshold_used': float(rare_threshold)
    }


def kruskal_wallis_test(scores_df: pd.DataFrame,
                        freq_df: pd.DataFrame,
                        class_column: str = 'class_name') -> Dict:
    """
    Kruskal-Wallis H test comparing memorization across all rarity tiers.

    Non-parametric alternative to one-way ANOVA.
    Tests if there are significant differences between any rarity tiers.

    Args:
        scores_df: DataFrame with per-sample memorization scores
        freq_df: DataFrame with rarity tiers assigned

    Returns:
        Dictionary with test results
    """
    # Ensure rarity tiers are assigned
    if 'rarity_tier' not in freq_df.columns:
        freq_df = assign_rarity_tiers(freq_df)

    # Merge tier info with scores
    tier_map = dict(zip(freq_df['class'], freq_df['rarity_tier']))
    scores_df = scores_df.copy()
    scores_df['rarity_tier'] = scores_df[class_column].map(tier_map)

    # Group scores by tier
    tiers = scores_df['rarity_tier'].unique()
    tier_scores = [scores_df[scores_df['rarity_tier'] == tier]['memorization_score'].values
                   for tier in tiers]

    # Filter out empty groups
    valid_tiers = [(tier, scores) for tier, scores in zip(tiers, tier_scores) if len(scores) > 0]

    if len(valid_tiers) < 2:
        return {'error': 'Need at least 2 non-empty groups'}

    tiers, tier_scores = zip(*valid_tiers)

    # Kruskal-Wallis test
    h_stat, p_value = stats.kruskal(*tier_scores)

    # Per-tier statistics
    tier_stats = {}
    for tier, scores in zip(tiers, tier_scores):
        tier_stats[str(tier)] = {
            'n': int(len(scores)),
            'mean': float(np.mean(scores)),
            'median': float(np.median(scores)),
            'std': float(np.std(scores))
        }

    return {
        'h_statistic': float(h_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'n_tiers': int(len(tiers)),
        'tier_statistics': tier_stats
    }


def dunn_posthoc_test(scores_df: pd.DataFrame,
                      freq_df: pd.DataFrame,
                      class_column: str = 'class_name') -> pd.DataFrame:
    """
    Dunn's post-hoc test for pairwise comparisons between rarity tiers.

    Used after significant Kruskal-Wallis to identify which pairs differ.
    Applies Bonferroni correction for multiple comparisons.

    Args:
        scores_df: DataFrame with per-sample memorization scores
        freq_df: DataFrame with rarity tiers assigned

    Returns:
        DataFrame with pairwise comparison results
    """
    from itertools import combinations

    # Ensure rarity tiers are assigned
    if 'rarity_tier' not in freq_df.columns:
        freq_df = assign_rarity_tiers(freq_df)

    # Merge tier info
    tier_map = dict(zip(freq_df['class'], freq_df['rarity_tier']))
    scores_df = scores_df.copy()
    scores_df['rarity_tier'] = scores_df[class_column].map(tier_map)

    # Get unique tiers with data
    tiers = [t for t in scores_df['rarity_tier'].unique() if pd.notna(t)]

    comparisons = []
    pairs = list(combinations(tiers, 2))
    n_comparisons = len(pairs)

    for tier1, tier2 in pairs:
        scores1 = scores_df[scores_df['rarity_tier'] == tier1]['memorization_score'].values
        scores2 = scores_df[scores_df['rarity_tier'] == tier2]['memorization_score'].values

        if len(scores1) == 0 or len(scores2) == 0:
            continue

        # Mann-Whitney for pairwise comparison
        stat, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')

        # Bonferroni correction
        p_adjusted = min(p_value * n_comparisons, 1.0)

        comparisons.append({
            'tier1': tier1,
            'tier2': tier2,
            'n1': len(scores1),
            'n2': len(scores2),
            'mean1': float(np.mean(scores1)),
            'mean2': float(np.mean(scores2)),
            'u_statistic': float(stat),
            'p_value': float(p_value),
            'p_adjusted': float(p_adjusted),
            'significant': p_adjusted < 0.05
        })

    return pd.DataFrame(comparisons)


# =============================================================================
# EFFECT SIZES
# =============================================================================

def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
    """
    Compute Cliff's delta effect size (non-parametric).

    Measures the probability that a randomly selected value from group1
    is greater than a randomly selected value from group2.

    Args:
        group1: First group (e.g., rare class scores)
        group2: Second group (e.g., common class scores)

    Returns:
        Tuple of (delta value, interpretation)

    Interpretation:
        |d| < 0.147: negligible
        |d| < 0.33: small
        |d| < 0.474: medium
        |d| >= 0.474: large
    """
    n1, n2 = len(group1), len(group2)

    # Guard: Empty groups
    if n1 == 0 or n2 == 0:
        return 0.0, 'undefined (empty group)'

    # Count dominance - vectorized for efficiency
    # Using broadcasting: group1[:, None] > group2[None, :] creates n1 x n2 comparison matrix
    group1_arr = np.asarray(group1)
    group2_arr = np.asarray(group2)

    greater = np.sum(group1_arr[:, None] > group2_arr[None, :])
    less = np.sum(group1_arr[:, None] < group2_arr[None, :])

    delta = (greater - less) / (n1 * n2)

    # Interpretation
    abs_d = abs(delta)
    if abs_d < 0.147:
        interpretation = 'negligible'
    elif abs_d < 0.33:
        interpretation = 'small'
    elif abs_d < 0.474:
        interpretation = 'medium'
    else:
        interpretation = 'large'

    return float(delta), interpretation


def compute_effect_sizes(scores_df: pd.DataFrame,
                         freq_df: pd.DataFrame,
                         rare_threshold: float = 0.05,
                         class_column: str = 'class_name') -> Dict:
    """
    Compute multiple effect size measures for rare vs common comparison.

    Args:
        scores_df: DataFrame with per-sample memorization scores
        freq_df: DataFrame with class frequencies
        rare_threshold: Classes with frequency < threshold are "rare"

    Returns:
        Dictionary with effect sizes
    """
    # Identify rare and common classes
    rare_classes = freq_df[freq_df['frequency'] < rare_threshold]['class'].tolist()
    common_classes = freq_df[freq_df['frequency'] >= rare_threshold]['class'].tolist()

    # Get scores
    rare_scores = scores_df[scores_df[class_column].isin(rare_classes)]['memorization_score'].values
    common_scores = scores_df[scores_df[class_column].isin(common_classes)]['memorization_score'].values

    if len(rare_scores) == 0 or len(common_scores) == 0:
        return {'error': 'One or both groups have no samples'}

    # Cliff's delta
    delta, delta_interp = cliffs_delta(rare_scores, common_scores)

    # Cohen's d (for comparison, though parametric) with division-by-zero protection
    n_rare, n_common = len(rare_scores), len(common_scores)
    denominator = n_rare + n_common - 2

    if denominator <= 0 or n_rare < 2 or n_common < 2:
        # Cannot compute valid Cohen's d
        cohens_d = 0.0
        d_interp = 'undefined (insufficient samples)'
    else:
        var_rare = np.var(rare_scores, ddof=1)
        var_common = np.var(common_scores, ddof=1)
        pooled_var = ((n_rare - 1) * var_rare + (n_common - 1) * var_common) / denominator
        pooled_std = np.sqrt(pooled_var)

        if pooled_std == 0 or np.isnan(pooled_std):
            cohens_d = 0.0
            d_interp = 'undefined (zero variance)'
        else:
            cohens_d = (np.mean(rare_scores) - np.mean(common_scores)) / pooled_std
            # Interpret Cohen's d
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                d_interp = 'negligible'
            elif abs_d < 0.5:
                d_interp = 'small'
            elif abs_d < 0.8:
                d_interp = 'medium'
            else:
                d_interp = 'large'

    # Common Language Effect Size (probability rare > common)
    # Approximation using normal distribution with division-by-zero protection
    mean_diff = np.mean(rare_scores) - np.mean(common_scores)
    var_rare = np.var(rare_scores)
    var_common = np.var(common_scores)
    pooled_var = (var_rare + var_common) / 2

    if pooled_var <= 0 or np.isnan(pooled_var):
        # Cannot compute CLES with zero/negative variance
        cles = 0.5  # Default to 50% (no difference)
    else:
        denominator_cles = np.sqrt(2 * pooled_var)
        if denominator_cles == 0:
            cles = 0.5
        else:
            cles = stats.norm.cdf(mean_diff / denominator_cles)

    return {
        'cliffs_delta': delta,
        'cliffs_delta_interpretation': delta_interp,
        'cohens_d': float(cohens_d),
        'cohens_d_interpretation': d_interp,
        'common_language_effect_size': float(cles),
        'cles_interpretation': f'{cles*100:.1f}% probability rare > common'
    }


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(data: np.ndarray,
                 statistic_func=np.mean,
                 n_bootstrap: int = 10000,
                 confidence: float = 0.95,
                 random_state: int = 42) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Array of values
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(random_state)

    n = len(data)

    # Guard: Empty data
    if n == 0:
        return float('nan'), float('nan'), float('nan')

    point_estimate = statistic_func(data)

    # Guard: Single sample - CI is just the point estimate
    if n == 1:
        return float(point_estimate), float(point_estimate), float(point_estimate)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)

    return float(point_estimate), float(ci_lower), float(ci_upper)


def bootstrap_difference_ci(group1: np.ndarray,
                            group2: np.ndarray,
                            n_bootstrap: int = 10000,
                            confidence: float = 0.95,
                            random_state: int = 42) -> Dict:
    """
    Bootstrap CI for difference in means between two groups.

    Args:
        group1: First group (rare classes)
        group2: Second group (common classes)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed

    Returns:
        Dictionary with difference statistics and CI
    """
    np.random.seed(random_state)

    n1, n2 = len(group1), len(group2)
    observed_diff = np.mean(group1) - np.mean(group2)

    # Bootstrap
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(group1, size=n1, replace=True)
        sample2 = np.random.choice(group2, size=n2, replace=True)
        bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))

    bootstrap_diffs = np.array(bootstrap_diffs)

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)

    # Check if CI excludes zero (significant difference)
    significant = (ci_lower > 0) or (ci_upper < 0)

    return {
        'observed_difference': float(observed_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'confidence_level': confidence,
        'significant': significant,
        'interpretation': f'Rare classes have {observed_diff:.3f} higher memorization (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])'
    }


# =============================================================================
# COMPREHENSIVE ANALYSIS
# =============================================================================

def run_full_rarity_analysis(scores_path: str,
                             output_dir: str = None,
                             rare_threshold: float = 0.05,
                             class_column: str = 'class_name') -> Dict:
    """
    Run complete statistical analysis of rare vs common disease memorization.

    Args:
        scores_path: Path to memorization scores CSV
        output_dir: Directory to save results (optional)
        rare_threshold: Threshold for defining rare classes
        class_column: Column name for class labels

    Returns:
        Comprehensive results dictionary
    """
    # Load data
    scores_df = pd.read_csv(scores_path)

    # Compute frequencies and assign tiers
    freq_df = compute_class_frequencies(scores_df, class_column)
    freq_df = assign_rarity_tiers(freq_df)

    # Get rare and common scores for effect sizes and bootstrap
    rare_classes = freq_df[freq_df['frequency'] < rare_threshold]['class'].tolist()
    common_classes = freq_df[freq_df['frequency'] >= rare_threshold]['class'].tolist()
    rare_scores = scores_df[scores_df[class_column].isin(rare_classes)]['memorization_score'].values
    common_scores = scores_df[scores_df[class_column].isin(common_classes)]['memorization_score'].values

    # Convert freq_df to JSON-safe format
    class_freq_records = []
    for _, row in freq_df.iterrows():
        class_freq_records.append({
            'class': str(row['class']),
            'count': int(row['count']),
            'frequency': float(row['frequency']),
            'percentage': float(row['percentage']),
            'rarity_tier': str(row['rarity_tier'])
        })

    # Get posthoc comparisons as records
    posthoc_df = dunn_posthoc_test(scores_df, freq_df, class_column)
    posthoc_records = []
    for _, row in posthoc_df.iterrows():
        posthoc_records.append({k: (float(v) if isinstance(v, (np.floating, float)) else
                                    int(v) if isinstance(v, (np.integer, int)) else
                                    bool(v) if isinstance(v, (np.bool_, bool)) else
                                    str(v))
                                for k, v in row.items()})

    results = {
        'dataset_info': {
            'total_samples': int(len(scores_df)),
            'n_classes': int(len(freq_df)),
            'rare_threshold': float(rare_threshold),
            'n_rare_classes': int(len(rare_classes)),
            'n_common_classes': int(len(common_classes)),
            'n_rare_samples': int(len(rare_scores)),
            'n_common_samples': int(len(common_scores))
        },
        'class_frequencies': class_freq_records,
        'spearman_correlation': spearman_correlation(freq_df, scores_df, class_column),
        'mann_whitney_test': mann_whitney_u_test(scores_df, freq_df, rare_threshold, class_column),
        'kruskal_wallis_test': kruskal_wallis_test(scores_df, freq_df, class_column),
        'posthoc_comparisons': posthoc_records,
        'effect_sizes': compute_effect_sizes(scores_df, freq_df, rare_threshold, class_column),
        'bootstrap_ci': bootstrap_difference_ci(rare_scores, common_scores) if len(rare_scores) > 0 and len(common_scores) > 0 else None
    }

    # Save results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        import json

        def convert(obj):
            """Convert numpy/pandas types to Python native types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_list()
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(output_dir / 'rarity_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=convert)

        # Save frequency table
        freq_df.to_csv(output_dir / 'class_frequency_table.csv', index=False)

        print(f"Results saved to {output_dir}")

    return results


def generate_statistical_summary(results: Dict) -> str:
    """
    Generate human-readable statistical summary.

    Args:
        results: Results from run_full_rarity_analysis()

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("RARITY ANALYSIS: STATISTICAL SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Dataset info
    info = results['dataset_info']
    lines.append("DATASET OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total samples: {info['total_samples']}")
    lines.append(f"Number of classes: {info['n_classes']}")
    lines.append(f"Rare threshold: < {info['rare_threshold']*100:.1f}% of dataset")
    lines.append(f"Rare classes: {info['n_rare_classes']} ({info['n_rare_samples']} samples)")
    lines.append(f"Common classes: {info['n_common_classes']} ({info['n_common_samples']} samples)")
    lines.append("")

    # Spearman correlation
    spearman = results['spearman_correlation']
    lines.append("1. SPEARMAN CORRELATION (Frequency vs Memorization)")
    lines.append("-" * 40)
    lines.append(f"   rho = {spearman['spearman_rho']:.4f}")
    lines.append(f"   p-value = {spearman['p_value']:.2e}")
    lines.append(f"   Significant: {'YES' if spearman['significant'] else 'NO'} (alpha=0.05)")
    lines.append(f"   Interpretation: {spearman['interpretation']}")
    lines.append("")

    # Mann-Whitney U
    mw = results['mann_whitney_test']
    if 'error' not in mw:
        lines.append("2. MANN-WHITNEY U TEST (Rare vs Common)")
        lines.append("-" * 40)
        lines.append(f"   U statistic = {mw['u_statistic']:.2f}")
        lines.append(f"   p-value (one-sided) = {mw['p_value_one_sided']:.2e}")
        lines.append(f"   Significant: {'YES' if mw['significant_one_sided'] else 'NO'} (alpha=0.05)")
        lines.append(f"   Rare mean: {mw['rare_mean']:.4f} (n={mw['rare_n']})")
        lines.append(f"   Common mean: {mw['common_mean']:.4f} (n={mw['common_n']})")
        lines.append("")

    # Kruskal-Wallis
    kw = results['kruskal_wallis_test']
    if 'error' not in kw:
        lines.append("3. KRUSKAL-WALLIS H TEST (Across All Tiers)")
        lines.append("-" * 40)
        lines.append(f"   H statistic = {kw['h_statistic']:.4f}")
        lines.append(f"   p-value = {kw['p_value']:.2e}")
        lines.append(f"   Significant: {'YES' if kw['significant'] else 'NO'} (alpha=0.05)")
        lines.append(f"   Number of tiers: {kw['n_tiers']}")
        lines.append("")

    # Effect sizes
    effects = results['effect_sizes']
    if 'error' not in effects:
        lines.append("4. EFFECT SIZES")
        lines.append("-" * 40)
        lines.append(f"   Cliff's delta = {effects['cliffs_delta']:.4f} ({effects['cliffs_delta_interpretation']})")
        lines.append(f"   Cohen's d = {effects['cohens_d']:.4f} ({effects['cohens_d_interpretation']})")
        lines.append(f"   CLES = {effects['common_language_effect_size']:.1%}")
        lines.append(f"   ({effects['cles_interpretation']})")
        lines.append("")

    # Bootstrap CI
    bootstrap = results['bootstrap_ci']
    if bootstrap:
        lines.append("5. BOOTSTRAP CONFIDENCE INTERVAL")
        lines.append("-" * 40)
        lines.append(f"   Mean difference (rare - common): {bootstrap['observed_difference']:.4f}")
        lines.append(f"   95% CI: [{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}]")
        lines.append(f"   CI excludes zero: {'YES' if bootstrap['significant'] else 'NO'}")
        lines.append("")

    # Conclusion
    lines.append("=" * 80)
    lines.append("CONCLUSION")
    lines.append("=" * 80)

    # Determine overall conclusion
    sig_count = 0
    if spearman['significant'] and spearman['spearman_rho'] < 0:
        sig_count += 1
    if 'error' not in mw and mw['significant_one_sided']:
        sig_count += 1
    if bootstrap and bootstrap['significant'] and bootstrap['observed_difference'] > 0:
        sig_count += 1

    if sig_count >= 2:
        lines.append("STRONG EVIDENCE: Rare disease canaries are memorized significantly")
        lines.append("more than common disease canaries.")
    elif sig_count == 1:
        lines.append("MODERATE EVIDENCE: Some tests show significant differences between")
        lines.append("rare and common disease memorization.")
    else:
        lines.append("WEAK/NO EVIDENCE: No significant difference found between rare")
        lines.append("and common disease memorization.")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Rarity Analysis for Memorization')
    parser.add_argument('--scores', type=str, required=True,
                        help='Path to memorization scores CSV')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Rare class threshold (default: 0.05)')

    args = parser.parse_args()

    results = run_full_rarity_analysis(
        scores_path=args.scores,
        output_dir=args.output,
        rare_threshold=args.threshold
    )

    summary = generate_statistical_summary(results)
    print(summary)
