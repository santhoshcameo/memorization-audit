"""
Comparison Plots
Visualize memorization comparison across models
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from scipy import stats


def get_significance_stars(p_value: float) -> str:
    """
    Get significance stars for p-value (journal standard).

    Args:
        p_value: Raw p-value from statistical test

    Returns:
        Significance stars: *** (p<0.001), ** (p<0.01), * (p<0.05), ns (not significant)
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def add_significance_annotations(ax, data: List[np.ndarray], labels: List[str],
                                  y_offset: float = 0.05):
    """
    Add significance annotations between groups on a plot.

    Performs pairwise t-tests and adds significance bars with stars.

    Args:
        ax: Matplotlib axis
        data: List of data arrays for each group
        labels: List of group labels
        y_offset: Vertical offset for annotations as fraction of y-range
    """
    if len(data) < 2:
        return

    # Get y-axis limits
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Calculate significance between adjacent pairs
    comparisons = []
    for i in range(len(data) - 1):
        _, p_value = stats.ttest_ind(data[i], data[i + 1])
        stars = get_significance_stars(p_value)
        if stars != 'ns':  # Only show significant differences
            comparisons.append((i, i + 1, stars, p_value))

    # Add annotation bars
    current_height = y_max + y_range * 0.02
    for x1, x2, stars, p_val in comparisons:
        bar_height = current_height + y_range * y_offset
        # Draw bar
        ax.plot([x1 + 1, x1 + 1, x2 + 1, x2 + 1],
                [current_height, bar_height, bar_height, current_height],
                'k-', linewidth=1)
        # Add stars
        ax.text((x1 + x2 + 2) / 2, bar_height + y_range * 0.01, stars,
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        current_height = bar_height + y_range * 0.05

    # Adjust y-axis to fit annotations
    if comparisons:
        ax.set_ylim(y_min, current_height + y_range * 0.05)


def setup_plot_style(style: str = 'seaborn-v0_8-darkgrid', dpi: int = 300):
    """
    Setup matplotlib style for publication-quality figures.

    STANDARDIZED font sizes across all visualization modules:
    - Base font: 12pt
    - Axis labels: 14pt
    - Titles: 16pt
    - Tick labels: 11pt
    - Legend: 11pt

    Args:
        style: Matplotlib style
        dpi: Figure DPI (300 for publication)
    """
    plt.style.use(style)
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
    })


def plot_histogram_comparison(results_dict: Dict[str, pd.DataFrame],
                              output_path: Optional[Path] = None,
                              figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot histogram comparison of memorization scores
    
    Args:
        results_dict: Dictionary mapping model names to results DataFrames
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    colors = sns.color_palette("Set2", len(results_dict))
    
    # Plot histograms
    for (model_name, df), color in zip(results_dict.items(), colors):
        scores = df['memorization_score'].values
        
        ax.hist(scores, bins=30, alpha=0.6, label=model_name.upper(),
               color=color, edgecolor='black', linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('Memorization Score', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Memorization Score Distribution Across Models', 
                fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
    ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate threshold')
    ax.axvline(x=0.3, color='darkred', linestyle='--', linewidth=1, alpha=0.5, label='High risk threshold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_boxplot_comparison(results_dict: Dict[str, pd.DataFrame],
                           output_path: Optional[Path] = None,
                           figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot box plot comparison of memorization scores
    
    Args:
        results_dict: Dictionary mapping model names to results DataFrames
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    data = []
    labels = []
    for model_name, df in results_dict.items():
        data.append(df['memorization_score'].values)
        labels.append(model_name.upper())
    
    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    notch=True, showmeans=True)
    
    # Color boxes
    colors = sns.color_palette("Set2", len(data))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Formatting
    ax.set_ylabel('Memorization Score', fontweight='bold')
    ax.set_title('Memorization Score Comparison (Box Plot)',
                fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.3, color='darkred', linestyle='--', linewidth=1, alpha=0.5)

    # Add significance annotations (*** p<0.001, ** p<0.01, * p<0.05)
    add_significance_annotations(ax, data, labels)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_violin_comparison(results_dict: Dict[str, pd.DataFrame],
                          output_path: Optional[Path] = None,
                          figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot violin plot comparison
    
    Args:
        results_dict: Dictionary mapping model names to results DataFrames
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for seaborn
    plot_data = []
    for model_name, df in results_dict.items():
        for score in df['memorization_score'].values:
            plot_data.append({
                'Model': model_name.upper(),
                'Memorization Score': score
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create violin plot
    sns.violinplot(data=plot_df, x='Model', y='Memorization Score',
                  palette='Set2', ax=ax)
    
    # Add mean markers
    means = plot_df.groupby('Model')['Memorization Score'].mean()
    x_positions = range(len(means))
    ax.scatter(x_positions, means.values, color='red', s=100, 
              zorder=3, marker='D', label='Mean')
    
    # Formatting
    ax.set_ylabel('Memorization Score', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('Memorization Score Distribution (Violin Plot)', 
                fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.3, color='darkred', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_summary_bars(results_dict: Dict[str, pd.DataFrame],
                     output_path: Optional[Path] = None,
                     figsize: tuple = (12, 5)) -> plt.Figure:
    """
    Plot summary statistics as bar charts
    
    Args:
        results_dict: Dictionary mapping model names to results DataFrames
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Figure object
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Prepare data
    model_names = []
    means = []
    standard_errors = []  # Use SE instead of SD for error bars
    sample_counts = []
    high_risk_pcts = []

    for model_name, df in results_dict.items():
        model_names.append(model_name.upper())
        scores = df['memorization_score']
        n = len(scores)
        means.append(scores.mean())
        # Standard Error = SD / sqrt(n) - proper for comparing means
        standard_errors.append(scores.std() / np.sqrt(n))
        sample_counts.append(n)
        high_risk_pcts.append((df['risk_category'] == 'high').mean() * 100)

    colors = sns.color_palette("Set2", len(model_names))
    x = range(len(model_names))

    # Plot 1: Mean memorization with SE error bars (not SD)
    # SE is appropriate for showing uncertainty in the mean estimate
    bars = ax1.bar(x, means, yerr=standard_errors, color=colors, alpha=0.7,
           capsize=5, edgecolor='black', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=0)
    ax1.set_ylabel('Mean Memorization Score (±SE)', fontweight='bold')
    ax1.set_title('Average Memorization by Model', fontweight='bold')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add sample size annotations
    for i, (bar, n) in enumerate(zip(bars, sample_counts)):
        ax1.annotate(f'n={n}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 2: High risk percentage
    ax2.bar(x, high_risk_pcts, color=colors, alpha=0.7,
           edgecolor='black', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=0)
    ax2.set_ylabel('High Risk Samples (%)', fontweight='bold')
    ax2.set_title('Percentage of High-Risk Samples', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_statistical_comparison(comparisons_df: pd.DataFrame,
                                output_path: Optional[Path] = None,
                                figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot statistical comparison results
    
    Args:
        comparisons_df: DataFrame from compare_all_models()
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Figure object
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Create comparison labels
    comparisons_df['comparison'] = (comparisons_df['model1'].str.upper() + 
                                   ' vs\n' + 
                                   comparisons_df['model2'].str.upper())
    
    # Plot 1: Effect sizes
    colors = ['green' if p < 0.05 else 'gray' 
             for p in comparisons_df['p_value']]
    
    ax1.barh(range(len(comparisons_df)), comparisons_df['cohens_d'],
            color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_yticks(range(len(comparisons_df)))
    ax1.set_yticklabels(comparisons_df['comparison'])
    ax1.set_xlabel("Cohen's d (Effect Size)", fontweight='bold')
    ax1.set_title('Pairwise Effect Sizes', fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Small')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Medium')
    ax1.axvline(x=0.8, color='darkred', linestyle='--', alpha=0.5, label='Large')
    ax1.axvline(x=-0.2, color='orange', linestyle='--', alpha=0.5)
    ax1.axvline(x=-0.5, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=-0.8, color='darkred', linestyle='--', alpha=0.5)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: P-values (log scale)
    p_vals = comparisons_df['p_value'].values
    p_vals_log = -np.log10(p_vals)
    
    colors_p = ['green' if p < 0.05 else 'red' for p in p_vals]
    
    ax2.barh(range(len(comparisons_df)), p_vals_log,
            color=colors_p, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_yticks(range(len(comparisons_df)))
    ax2.set_yticklabels(comparisons_df['comparison'])
    ax2.set_xlabel('-log₁₀(p-value)', fontweight='bold')
    ax2.set_title('Statistical Significance', fontweight='bold')
    ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', 
               linewidth=2, label='p=0.05')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def create_all_comparison_plots(results_dict: Dict[str, pd.DataFrame],
                               comparisons_df: pd.DataFrame,
                               output_dir: Path):
    """
    Create all comparison plots
    
    Args:
        results_dict: Dictionary mapping model names to results DataFrames
        comparisons_df: Statistical comparisons DataFrame
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating comparison plots...")
    
    # Histogram
    plot_histogram_comparison(
        results_dict,
        output_path=output_dir / 'comparison_histogram.png'
    )
    
    # Box plot
    plot_boxplot_comparison(
        results_dict,
        output_path=output_dir / 'comparison_boxplot.png'
    )
    
    # Violin plot
    plot_violin_comparison(
        results_dict,
        output_path=output_dir / 'comparison_violin.png'
    )
    
    # Summary bars
    plot_summary_bars(
        results_dict,
        output_path=output_dir / 'comparison_summary.png'
    )
    
    # Statistical comparison
    plot_statistical_comparison(
        comparisons_df,
        output_path=output_dir / 'comparison_statistics.png'
    )
    
    print(f"\n✅ All comparison plots saved to: {output_dir}")
    plt.close('all')


# Example usage
if __name__ == "__main__":
    print("Comparison Plots Module")
    print("Creates model comparison visualizations")