"""
Rarity Visualization Module
Publication-quality plots for rare vs common disease memorization analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif'
})

# Color scheme for rarity tiers
TIER_COLORS = {
    'very_rare': '#d62728',   # Red
    'rare': '#ff7f0e',        # Orange
    'moderate': '#2ca02c',    # Green
    'common': '#1f77b4'       # Blue
}

TIER_ORDER = ['very_rare', 'rare', 'moderate', 'common']


def plot_frequency_vs_memorization(freq_df: pd.DataFrame,
                                   scores_df: pd.DataFrame,
                                   spearman_result: Dict,
                                   output_path: str = None,
                                   figsize: Tuple[int, int] = (10, 7),
                                   class_column: str = 'class_name') -> plt.Figure:
    """
    Scatter plot of class frequency vs mean memorization score.

    Shows negative correlation between frequency and memorization.

    Args:
        freq_df: DataFrame with class frequencies and rarity tiers
        scores_df: DataFrame with per-sample memorization scores
        spearman_result: Result from spearman_correlation()
        output_path: Path to save figure
        figsize: Figure dimensions
    """
    # Compute mean memorization per class
    class_stats = scores_df.groupby(class_column).agg({
        'memorization_score': ['mean', 'std', 'count']
    }).reset_index()
    class_stats.columns = ['class', 'mean_memo', 'std_memo', 'n_samples']

    # Merge with frequency data
    merged = freq_df.merge(class_stats, on='class')

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each class with color by rarity tier
    for tier in TIER_ORDER:
        tier_data = merged[merged['rarity_tier'] == tier]
        if len(tier_data) == 0:
            continue

        ax.scatter(
            tier_data['percentage'],
            tier_data['mean_memo'],
            c=TIER_COLORS[tier],
            s=tier_data['n_samples'] * 0.5,  # Size by sample count
            label=tier.replace('_', ' ').title(),
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5
        )

        # Add class labels
        for _, row in tier_data.iterrows():
            ax.annotate(
                row['class'],
                (row['percentage'], row['mean_memo']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                alpha=0.8
            )

    # Add regression line
    x = merged['percentage'].values
    y = merged['mean_memo'].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), '--', color='gray', linewidth=2, alpha=0.7,
            label=f'Trend (slope={z[0]:.3f})')

    # Add Spearman annotation
    rho = spearman_result['spearman_rho']
    p_val = spearman_result['p_value']
    sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    ax.text(
        0.98, 0.98,
        f"Spearman $\\rho$ = {rho:.3f}{sig_marker}\np = {p_val:.2e}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.set_xlabel('Class Frequency (%)')
    ax.set_ylabel('Mean Memorization Score')
    ax.set_title('Class Frequency vs Memorization Score')
    ax.legend(loc='upper left', title='Rarity Tier')
    ax.grid(True, alpha=0.3)

    # Log scale for x-axis often helps visualization
    ax.set_xscale('log')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig


def plot_rarity_tier_boxplot(scores_df: pd.DataFrame,
                             freq_df: pd.DataFrame,
                             output_path: str = None,
                             figsize: Tuple[int, int] = (10, 7),
                             class_column: str = 'class_name') -> plt.Figure:
    """
    Box plot comparing memorization scores across rarity tiers.

    Args:
        scores_df: DataFrame with per-sample memorization scores
        freq_df: DataFrame with rarity tiers assigned
    """
    # Merge tier info
    tier_map = dict(zip(freq_df['class'], freq_df['rarity_tier']))
    plot_df = scores_df.copy()
    plot_df['rarity_tier'] = plot_df[class_column].map(tier_map)

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for boxplot
    tier_data = []
    tier_labels = []
    tier_colors_list = []

    for tier in TIER_ORDER:
        data = plot_df[plot_df['rarity_tier'] == tier]['memorization_score'].values
        if len(data) > 0:
            tier_data.append(data)
            tier_labels.append(tier.replace('_', ' ').title())
            tier_colors_list.append(TIER_COLORS[tier])

    # Create boxplot
    bp = ax.boxplot(tier_data, labels=tier_labels, patch_artist=True,
                    showfliers=True, flierprops={'marker': 'o', 'alpha': 0.3})

    # Color the boxes
    for patch, color in zip(bp['boxes'], tier_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add sample counts
    for i, (tier, data) in enumerate(zip(tier_labels, tier_data)):
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'n={len(data)}',
                ha='center', fontsize=10)

    # Add means as diamonds
    means = [np.mean(d) for d in tier_data]
    ax.scatter(range(1, len(means) + 1), means, marker='D', color='white',
               edgecolor='black', s=50, zorder=5, label='Mean')

    ax.set_ylabel('Memorization Score')
    ax.set_xlabel('Rarity Tier')
    ax.set_title('Memorization Score Distribution by Rarity Tier')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3, label='High-risk threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig


def plot_rarity_tier_violin(scores_df: pd.DataFrame,
                            freq_df: pd.DataFrame,
                            output_path: str = None,
                            figsize: Tuple[int, int] = (10, 7),
                            class_column: str = 'class_name') -> plt.Figure:
    """
    Violin plot showing full distribution of memorization by rarity tier.
    """
    # Merge tier info
    tier_map = dict(zip(freq_df['class'], freq_df['rarity_tier']))
    plot_df = scores_df.copy()
    plot_df['rarity_tier'] = plot_df[class_column].map(tier_map)

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    tier_data = []
    positions = []
    tier_labels = []

    pos = 1
    for tier in TIER_ORDER:
        data = plot_df[plot_df['rarity_tier'] == tier]['memorization_score'].values
        if len(data) > 0:
            tier_data.append(data)
            positions.append(pos)
            tier_labels.append(tier.replace('_', ' ').title())
            pos += 1

    # Create violin plot
    parts = ax.violinplot(tier_data, positions=positions, showmeans=True,
                          showmedians=True, showextrema=True)

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(TIER_COLORS.values())[i])
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(tier_labels)
    ax.set_ylabel('Memorization Score')
    ax.set_xlabel('Rarity Tier')
    ax.set_title('Memorization Distribution by Rarity Tier (Violin Plot)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig


def plot_risk_proportion_by_tier(scores_df: pd.DataFrame,
                                 freq_df: pd.DataFrame,
                                 output_path: str = None,
                                 figsize: Tuple[int, int] = (10, 6),
                                 class_column: str = 'class_name') -> plt.Figure:
    """
    Stacked bar chart showing proportion of high/moderate/low risk by rarity tier.
    """
    # Merge tier info
    tier_map = dict(zip(freq_df['class'], freq_df['rarity_tier']))
    plot_df = scores_df.copy()
    plot_df['rarity_tier'] = plot_df[class_column].map(tier_map)

    # Categorize risk
    def categorize_risk(score):
        if score > 0.3:
            return 'High'
        elif score > 0.1:
            return 'Moderate'
        else:
            return 'Low'

    plot_df['risk_category'] = plot_df['memorization_score'].apply(categorize_risk)

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate proportions
    tier_risk_counts = plot_df.groupby(['rarity_tier', 'risk_category']).size().unstack(fill_value=0)

    # Normalize to proportions
    tier_risk_props = tier_risk_counts.div(tier_risk_counts.sum(axis=1), axis=0) * 100

    # Ensure order
    risk_order = ['High', 'Moderate', 'Low']
    tier_order_present = [t for t in TIER_ORDER if t in tier_risk_props.index]

    tier_risk_props = tier_risk_props.reindex(tier_order_present)
    tier_risk_props = tier_risk_props.reindex(columns=risk_order, fill_value=0)

    # Stacked bar chart
    x = range(len(tier_order_present))
    width = 0.6

    risk_colors = {'High': '#d62728', 'Moderate': '#ff7f0e', 'Low': '#2ca02c'}

    bottom = np.zeros(len(tier_order_present))
    for risk in risk_order:
        values = tier_risk_props[risk].values
        ax.bar(x, values, width, label=risk, bottom=bottom, color=risk_colors[risk])
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_', ' ').title() for t in tier_order_present])
    ax.set_ylabel('Percentage of Samples')
    ax.set_xlabel('Rarity Tier')
    ax.set_title('Risk Category Distribution by Rarity Tier')
    ax.legend(title='Risk Category')
    ax.set_ylim(0, 100)

    # Add percentage labels
    for i, tier in enumerate(tier_order_present):
        high_pct = tier_risk_props.loc[tier, 'High']
        if high_pct > 5:  # Only label if visible
            ax.text(i, high_pct / 2, f'{high_pct:.0f}%', ha='center', va='center',
                    color='white', fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig


def plot_class_memorization_heatmap(scores_df: pd.DataFrame,
                                    freq_df: pd.DataFrame,
                                    model_scores_dict: Dict[str, pd.DataFrame] = None,
                                    output_path: str = None,
                                    figsize: Tuple[int, int] = (12, 8),
                                    class_column: str = 'class_name') -> plt.Figure:
    """
    Heatmap of memorization scores by class (and optionally by model).

    If model_scores_dict is provided, creates model x class heatmap.
    Otherwise, shows single model with class stats.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if model_scores_dict:
        # Multi-model heatmap
        classes = freq_df.sort_values('frequency')['class'].tolist()
        models = list(model_scores_dict.keys())

        data = np.zeros((len(models), len(classes)))

        for i, model in enumerate(models):
            df = model_scores_dict[model]
            class_means = df.groupby(class_column)['memorization_score'].mean()
            for j, cls in enumerate(classes):
                if cls in class_means.index:
                    data[i, j] = class_means[cls]

        im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto')

        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)

        ax.set_xlabel('Class (sorted by frequency: rare → common)')
        ax.set_ylabel('Model')
        ax.set_title('Mean Memorization Score by Model and Class')

    else:
        # Single model - class-only view
        class_stats = scores_df.groupby(class_column).agg({
            'memorization_score': ['mean', 'std']
        }).reset_index()
        class_stats.columns = ['class', 'mean', 'std']
        class_stats = class_stats.merge(freq_df[['class', 'frequency', 'rarity_tier']], on='class')
        class_stats = class_stats.sort_values('frequency')

        colors = [TIER_COLORS[t] for t in class_stats['rarity_tier']]

        bars = ax.barh(range(len(class_stats)), class_stats['mean'],
                       xerr=class_stats['std'], color=colors, alpha=0.8)

        ax.set_yticks(range(len(class_stats)))
        ax.set_yticklabels(class_stats['class'])
        ax.set_xlabel('Mean Memorization Score')
        ax.set_ylabel('Class (sorted by frequency)')
        ax.set_title('Mean Memorization Score by Class')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.3, label='High-risk')

        # Add legend for tiers
        patches = [mpatches.Patch(color=TIER_COLORS[t], label=t.replace('_', ' ').title())
                   for t in TIER_ORDER if t in class_stats['rarity_tier'].values]
        ax.legend(handles=patches, loc='lower right')

    plt.colorbar(im, ax=ax, label='Memorization Score') if model_scores_dict else None
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig


def plot_effect_size_summary(effect_sizes: Dict,
                             bootstrap_ci: Dict,
                             output_path: str = None,
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Bar chart summarizing effect sizes with confidence intervals.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Effect sizes
    ax1 = axes[0]
    effects = ['Cliff\'s $\\delta$', 'Cohen\'s d']
    values = [effect_sizes['cliffs_delta'], effect_sizes['cohens_d']]
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]

    bars = ax1.barh(effects, values, color=colors, alpha=0.8)
    ax1.axvline(x=0, color='black', linewidth=0.5)

    # Add interpretation labels
    interps = [effect_sizes['cliffs_delta_interpretation'],
               effect_sizes['cohens_d_interpretation']]
    for bar, interp in zip(bars, interps):
        width = bar.get_width()
        ax1.text(width + 0.02 if width > 0 else width - 0.02,
                 bar.get_y() + bar.get_height()/2,
                 f'({interp})',
                 ha='left' if width > 0 else 'right',
                 va='center', fontsize=10)

    ax1.set_xlabel('Effect Size')
    ax1.set_title('Effect Size Comparison')
    ax1.set_xlim(-1, 1.5)

    # Right plot: Bootstrap CI
    ax2 = axes[1]
    diff = bootstrap_ci['observed_difference']
    ci_lower = bootstrap_ci['ci_lower']
    ci_upper = bootstrap_ci['ci_upper']

    ax2.barh(['Rare - Common'], [diff], xerr=[[diff - ci_lower], [ci_upper - diff]],
             color='#1f77b4', alpha=0.8, capsize=5)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)

    ax2.text(0.5, 0.95, f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top')

    sig_text = 'CI excludes 0 (Significant)' if bootstrap_ci['significant'] else 'CI includes 0'
    ax2.text(0.5, 0.85, sig_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', color='green' if bootstrap_ci['significant'] else 'gray')

    ax2.set_xlabel('Mean Difference in Memorization Score')
    ax2.set_title('Bootstrap Confidence Interval')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig


def generate_all_rarity_plots(results: Dict,
                              scores_df: pd.DataFrame,
                              freq_df: pd.DataFrame,
                              output_dir: str,
                              class_column: str = 'class_name') -> List[str]:
    """
    Generate all rarity analysis plots and save to output directory.

    Args:
        results: Results from run_full_rarity_analysis()
        scores_df: Original memorization scores DataFrame
        freq_df: Frequency DataFrame with rarity tiers
        output_dir: Directory to save plots

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # 1. Frequency vs Memorization scatter
    path = output_dir / 'frequency_vs_memorization.png'
    plot_frequency_vs_memorization(
        freq_df, scores_df, results['spearman_correlation'],
        output_path=str(path), class_column=class_column
    )
    saved_files.append(str(path))
    plt.close()

    # 2. Box plot by tier
    path = output_dir / 'rarity_tier_boxplot.png'
    plot_rarity_tier_boxplot(scores_df, freq_df, output_path=str(path), class_column=class_column)
    saved_files.append(str(path))
    plt.close()

    # 3. Violin plot by tier
    path = output_dir / 'rarity_tier_violin.png'
    plot_rarity_tier_violin(scores_df, freq_df, output_path=str(path), class_column=class_column)
    saved_files.append(str(path))
    plt.close()

    # 4. Risk proportion by tier
    path = output_dir / 'risk_proportion_by_tier.png'
    plot_risk_proportion_by_tier(scores_df, freq_df, output_path=str(path), class_column=class_column)
    saved_files.append(str(path))
    plt.close()

    # 5. Class heatmap
    path = output_dir / 'class_memorization_heatmap.png'
    plot_class_memorization_heatmap(scores_df, freq_df, output_path=str(path), class_column=class_column)
    saved_files.append(str(path))
    plt.close()

    # 6. Effect size summary
    if 'error' not in results['effect_sizes'] and results['bootstrap_ci']:
        path = output_dir / 'effect_size_summary.png'
        plot_effect_size_summary(
            results['effect_sizes'], results['bootstrap_ci'],
            output_path=str(path)
        )
        saved_files.append(str(path))
        plt.close()

    print(f"\nGenerated {len(saved_files)} plots in {output_dir}")
    return saved_files


if __name__ == "__main__":
    print("Rarity Plots Module")
    print("Use generate_all_rarity_plots() to create all visualizations")
