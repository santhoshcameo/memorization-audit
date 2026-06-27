"""
Visualization Module
All plotting functions for paper figures
"""

from .comparison_plots import (
    plot_histogram_comparison,
    plot_boxplot_comparison,
    plot_violin_comparison,
    plot_summary_bars,
    plot_statistical_comparison,
    create_all_comparison_plots
)

from .distribution_plots import (
    plot_disease_comparison,
    plot_disease_heatmap,
    plot_top_memorized_samples,
    plot_distribution_kde,
    create_all_distribution_plots
)

__all__ = [
    # Comparison plots
    'plot_histogram_comparison',
    'plot_boxplot_comparison',
    'plot_violin_comparison',
    'plot_summary_bars',
    'plot_statistical_comparison',
    'create_all_comparison_plots',
    # Distribution plots
    'plot_disease_comparison',
    'plot_disease_heatmap',
    'plot_top_memorized_samples',
    'plot_distribution_kde',
    'create_all_distribution_plots',
]