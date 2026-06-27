"""
Layer-wise Memorization Visualization
Publication-quality plots showing where memorization concentrates in networks.

Provides heatmaps, line profiles, correlation bar charts, cross-model comparisons,
and gradient magnitude comparisons to localize memorization across network depth.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def setup_plot_style(dpi: int = 300):
    """
    Setup matplotlib style for publication-quality figures.

    STANDARDIZED font sizes (consistent with comparison_plots.py):
    - Base font: 12pt
    - Axis labels: 14pt
    - Titles: 16pt
    - Tick labels: 11pt
    - Legend: 11pt

    Args:
        dpi: Figure DPI (300 for publication)
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            plt.style.use('default')

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


def _get_significance_stars(p_value: float) -> str:
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


def _shorten_layer_name(layer_name: str, model_name: str = '') -> str:
    """
    Shorten long layer names for readable axis labels.

    Args:
        layer_name: Full layer name (e.g., 'layer4.2.conv3')
        model_name: Model name for context-specific shortening

    Returns:
        Shortened layer name
    """
    # Remove common prefixes
    name = layer_name
    for prefix in ['module.', 'backbone.', 'encoder.', 'model.']:
        if name.startswith(prefix):
            name = name[len(prefix):]

    # Truncate if still too long
    if len(name) > 20:
        parts = name.split('.')
        if len(parts) > 3:
            name = '.'.join(parts[:2]) + '..' + parts[-1]

    return name


def plot_layer_distance_heatmap(layer_distances_df: pd.DataFrame,
                                memorization_scores_df: pd.DataFrame,
                                model_name: str = 'resnet50',
                                output_path: Optional[str] = None,
                                dpi: int = 300) -> plt.Figure:
    """
    KEY FIGURE: Heatmap where X=layers, Y=samples sorted by M(x), color=feature distance.
    Top-right should be hot (high M(x) + late layers = memorization).

    The heatmap reveals where in the network memorization concentrates: if late layers
    show strong feature divergence for high-M(x) samples, memorization is localized
    in the classification head rather than distributed.

    Args:
        layer_distances_df: DataFrame with columns [image_id, layer_name, feature_distance]
        memorization_scores_df: DataFrame with columns [image_id, memorization_score]
        model_name: Model name for title
        output_path: Where to save figure (PNG or PDF)
        dpi: Figure resolution

    Returns:
        Figure object
    """
    setup_plot_style(dpi=dpi)

    # Merge memorization scores with layer distances
    merged = layer_distances_df.merge(memorization_scores_df[['image_id', 'memorization_score']],
                                       on='image_id', how='inner')

    # Get unique layers in order of appearance
    layer_names = layer_distances_df['layer_name'].unique().tolist()

    # Sort samples by memorization score (ascending, so high M(x) is at top)
    sample_order = (memorization_scores_df
                    .sort_values('memorization_score', ascending=True)['image_id']
                    .tolist())
    # Keep only samples present in both dataframes
    sample_order = [s for s in sample_order if s in merged['image_id'].values]

    # Build the heatmap matrix: rows=samples (sorted by M(x)), cols=layers
    n_samples = len(sample_order)
    n_layers = len(layer_names)
    heatmap_data = np.full((n_samples, n_layers), np.nan)

    # Create lookup for fast indexing
    sample_idx = {sid: i for i, sid in enumerate(sample_order)}

    for _, row in merged.iterrows():
        sid = row['image_id']
        layer = row['layer_name']
        if sid in sample_idx and layer in layer_names:
            r = sample_idx[sid]
            c = layer_names.index(layer)
            heatmap_data[r, c] = row['feature_distance']

    # Determine figure size based on number of layers
    fig_width = max(10, n_layers * 0.8)
    fig_height = max(8, n_samples * 0.05)
    fig_height = min(fig_height, 16)  # Cap height for very large sample sets
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot heatmap with diverging colormap
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r',
                    interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Cosine Distance (candidate vs independent)',
                   rotation=270, labelpad=20, fontweight='bold')

    # X-axis: layer names
    short_names = [_shorten_layer_name(ln, model_name) for ln in layer_names]
    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Network Layer (early $\\rightarrow$ late)', fontweight='bold')

    # Y-axis: sample indices sorted by M(x)
    # Show a subset of tick labels to avoid clutter
    scores_lookup = memorization_scores_df.groupby('image_id')['memorization_score'].mean()
    if n_samples > 30:
        tick_step = max(1, n_samples // 15)
        y_ticks = np.arange(0, n_samples, tick_step)
        y_labels = [f'{scores_lookup.loc[sample_order[i]]:.2f}'
                     if i < n_samples else '' for i in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8)
    else:
        ax.set_yticks(np.arange(n_samples))
        y_labels = [f'{scores_lookup.loc[sid]:.2f}' for sid in sample_order]
        ax.set_yticklabels(y_labels, fontsize=8)

    ax.set_ylabel('Samples (sorted by M(x), low $\\rightarrow$ high)', fontweight='bold')

    ax.set_title(f'Layer-wise Feature Distance Heatmap - {model_name.upper()}\n'
                 f'(Top-right hot = memorization in late layers)',
                 fontweight='bold', pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_layer_distance_profile(layer_summary_df: pd.DataFrame,
                                model_name: str = 'resnet50',
                                output_path: Optional[str] = None,
                                dpi: int = 300) -> plt.Figure:
    """
    Line plot showing mean feature distance at each layer, with separate lines
    for high-memorization vs low-memorization samples (split at median M(x)).
    Error bands show standard deviation.

    Args:
        layer_summary_df: DataFrame with columns [layer_name, group, mean_distance, std_distance]
                          where group is 'high_memorization' or 'low_memorization'
        model_name: Model name for title
        output_path: Where to save figure
        dpi: Figure resolution

    Returns:
        Figure object
    """
    setup_plot_style(dpi=dpi)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Define colors for the two groups
    group_styles = {
        'high_memorization': {'color': '#d62728', 'label': 'High M(x) (above median)',
                              'linestyle': '-', 'marker': 'o'},
        'low_memorization': {'color': '#1f77b4', 'label': 'Low M(x) (below median)',
                             'linestyle': '--', 'marker': 's'},
    }

    # Get layer order (preserve original order from dataframe)
    layer_names = layer_summary_df['layer_name'].unique().tolist()
    x = np.arange(len(layer_names))

    for group_name, style in group_styles.items():
        group_data = layer_summary_df[layer_summary_df['group'] == group_name]
        if group_data.empty:
            continue

        # Align data to layer order
        means = []
        stds = []
        for layer in layer_names:
            row = group_data[group_data['layer_name'] == layer]
            if not row.empty:
                means.append(row['mean_distance'].values[0])
                stds.append(row['std_distance'].values[0])
            else:
                means.append(np.nan)
                stds.append(0.0)

        means = np.array(means)
        stds = np.array(stds)

        # Plot line with error band
        ax.plot(x, means, color=style['color'], linestyle=style['linestyle'],
                marker=style['marker'], markersize=6, linewidth=2,
                label=style['label'], zorder=3)
        ax.fill_between(x, means - stds, means + stds,
                        color=style['color'], alpha=0.15, zorder=1)

    # Formatting
    short_names = [_shorten_layer_name(ln, model_name) for ln in layer_names]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Network Layer (early $\\rightarrow$ late)', fontweight='bold')
    ax.set_ylabel('Mean Cosine Distance', fontweight='bold')
    ax.set_title(f'Layer-wise Cosine Distance Profile - {model_name.upper()}\n'
                 f'(Gap between lines = where memorization concentrates)',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add annotation arrow pointing to the divergence region
    if len(layer_names) > 1:
        # Find layer with maximum gap between high and low groups
        high_data = layer_summary_df[layer_summary_df['group'] == 'high_memorization']
        low_data = layer_summary_df[layer_summary_df['group'] == 'low_memorization']
        if not high_data.empty and not low_data.empty:
            high_means = high_data.set_index('layer_name')['mean_distance']
            low_means = low_data.set_index('layer_name')['mean_distance']
            common_layers = high_means.index.intersection(low_means.index)
            if len(common_layers) > 0:
                gaps = (high_means.loc[common_layers] - low_means.loc[common_layers]).abs()
                max_gap_layer = gaps.idxmax()
                max_gap_idx = layer_names.index(max_gap_layer)
                max_gap_val = gaps[max_gap_layer]

                mid_y = (high_means.loc[max_gap_layer] + low_means.loc[max_gap_layer]) / 2
                ax.annotate(f'Max gap: {max_gap_val:.3f}',
                            xy=(max_gap_idx, mid_y),
                            xytext=(max_gap_idx + 0.5, mid_y + max_gap_val * 0.5),
                            fontsize=10, fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_layer_correlation_bars(correlation_df: pd.DataFrame,
                                model_name: str = 'resnet50',
                                output_path: Optional[str] = None,
                                dpi: int = 300) -> plt.Figure:
    """
    Bar chart showing Spearman correlation between M(x) and feature distance
    at each layer. Highlights layers where correlation is significant (p < 0.05).

    Args:
        correlation_df: DataFrame with columns [layer_name, spearman_rho, p_value]
        model_name: Model name for title
        output_path: Where to save figure
        dpi: Figure resolution

    Returns:
        Figure object
    """
    setup_plot_style(dpi=dpi)

    fig, ax = plt.subplots(figsize=(12, 6))

    layer_names = correlation_df['layer_name'].tolist()
    rhos = correlation_df['spearman_rho'].values
    p_values = correlation_df['p_value'].values

    x = np.arange(len(layer_names))

    # Color bars by significance
    colors = []
    for p in p_values:
        if p < 0.001:
            colors.append('#d62728')   # Dark red for highly significant
        elif p < 0.01:
            colors.append('#ff7f0e')   # Orange for significant
        elif p < 0.05:
            colors.append('#2ca02c')   # Green for marginally significant
        else:
            colors.append('#cccccc')   # Gray for not significant

    bars = ax.bar(x, rhos, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add significance stars above each bar
    for i, (rho, p) in enumerate(zip(rhos, p_values)):
        stars = _get_significance_stars(p)
        y_pos = rho + 0.02 if rho >= 0 else rho - 0.04
        ax.text(i, y_pos, stars, ha='center', va='bottom' if rho >= 0 else 'top',
                fontsize=9, fontweight='bold')

    # Formatting
    short_names = [_shorten_layer_name(ln, model_name) for ln in layer_names]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Network Layer (early $\\rightarrow$ late)', fontweight='bold')
    ax.set_ylabel('Spearman $\\rho$ (M(x) vs Cosine Distance)', fontweight='bold')
    ax.set_title(f'Layer-wise Correlation with Memorization - {model_name.upper()}',
                 fontweight='bold', pad=20)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend for significance colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', label='p < 0.001 (***)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='p < 0.01 (**)'),
        Patch(facecolor='#2ca02c', edgecolor='black', label='p < 0.05 (*)'),
        Patch(facecolor='#cccccc', edgecolor='black', label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9,
              title='Significance')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_cross_model_layer_comparison(all_model_results: Dict[str, pd.DataFrame],
                                       output_path: Optional[str] = None,
                                       dpi: int = 300) -> plt.Figure:
    """
    Multi-panel figure comparing layer distance profiles across models.
    One subplot per model (2x2 grid), showing how memorization concentrates
    differently in ResNet50, ViT, MAE, and MedSAM.

    Args:
        all_model_results: Dictionary mapping model name to layer_summary_df
                           (same format as plot_layer_distance_profile input)
        output_path: Where to save figure
        dpi: Figure resolution

    Returns:
        Figure object
    """
    setup_plot_style(dpi=dpi)

    n_models = len(all_model_results)
    # Use 2x2 grid; if fewer models, leave empty subplots
    n_cols = 2
    n_rows = max(1, (n_models + 1) // 2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows),
                              squeeze=False)

    model_names = list(all_model_results.keys())
    # Model display colors
    model_colors = {
        'resnet50': ('#e74c3c', '#3498db'),   # high=red, low=blue
        'vit': ('#e67e22', '#27ae60'),         # high=orange, low=green
        'mae': ('#9b59b6', '#1abc9c'),         # high=purple, low=teal
        'medsam': ('#e91e63', '#00bcd4'),      # high=pink, low=cyan
    }
    default_colors = ('#d62728', '#1f77b4')

    for idx, model_name in enumerate(model_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        layer_summary_df = all_model_results[model_name]
        layer_names = layer_summary_df['layer_name'].unique().tolist()
        x = np.arange(len(layer_names))

        high_color, low_color = model_colors.get(model_name.lower(), default_colors)

        for group_name, color, ls, marker in [
            ('high_memorization', high_color, '-', 'o'),
            ('low_memorization', low_color, '--', 's'),
        ]:
            group_data = layer_summary_df[layer_summary_df['group'] == group_name]
            if group_data.empty:
                continue

            means = []
            stds = []
            for layer in layer_names:
                row_data = group_data[group_data['layer_name'] == layer]
                if not row_data.empty:
                    means.append(row_data['mean_distance'].values[0])
                    stds.append(row_data['std_distance'].values[0])
                else:
                    means.append(np.nan)
                    stds.append(0.0)

            means = np.array(means)
            stds = np.array(stds)

            label = 'High M(x)' if 'high' in group_name else 'Low M(x)'
            ax.plot(x, means, color=color, linestyle=ls, marker=marker,
                    markersize=4, linewidth=1.8, label=label, zorder=3)
            ax.fill_between(x, means - stds, means + stds,
                            color=color, alpha=0.12, zorder=1)

        # Axis formatting
        short_names = [_shorten_layer_name(ln, model_name) for ln in layer_names]
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Cosine Distance', fontweight='bold')
        ax.set_title(model_name.upper(), fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle('Cross-Model Layer-wise Feature Distance Comparison',
                 fontweight='bold', fontsize=18, y=1.02)
    fig.supxlabel('Network Layer (early $\\rightarrow$ late)', fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_gradient_magnitude_comparison(gradient_df: pd.DataFrame,
                                        model_name: str = 'resnet50',
                                        output_path: Optional[str] = None,
                                        dpi: int = 300) -> plt.Figure:
    """
    Grouped bar chart: per-layer gradient norms for high-mem vs low-mem samples.
    Shows WHERE the model "works hardest" to memorize.

    Args:
        gradient_df: DataFrame with columns [layer_name, group, mean_gradient_norm, std_gradient_norm]
                     where group is 'high_memorization' or 'low_memorization'
        model_name: Model name for title
        output_path: Where to save figure
        dpi: Figure resolution

    Returns:
        Figure object
    """
    setup_plot_style(dpi=dpi)

    fig, ax = plt.subplots(figsize=(14, 6))

    layer_names = gradient_df['layer_name'].unique().tolist()
    x = np.arange(len(layer_names))
    width = 0.35

    group_styles = {
        'high_memorization': {'color': '#d62728', 'label': 'High M(x)', 'offset': -width / 2},
        'low_memorization': {'color': '#1f77b4', 'label': 'Low M(x)', 'offset': width / 2},
    }

    for group_name, style in group_styles.items():
        group_data = gradient_df[gradient_df['group'] == group_name]
        if group_data.empty:
            continue

        means = []
        stds = []
        for layer in layer_names:
            row = group_data[group_data['layer_name'] == layer]
            if not row.empty:
                means.append(row['mean_gradient_norm'].values[0])
                stds.append(row['std_gradient_norm'].values[0])
            else:
                means.append(0.0)
                stds.append(0.0)

        ax.bar(x + style['offset'], means, width, yerr=stds,
               color=style['color'], alpha=0.7, edgecolor='black', linewidth=0.5,
               label=style['label'], capsize=3)

    # Formatting
    short_names = [_shorten_layer_name(ln, model_name) for ln in layer_names]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Network Layer (early $\\rightarrow$ late)', fontweight='bold')
    ax.set_ylabel('Mean Gradient Norm', fontweight='bold')
    ax.set_title(f'Per-Layer Gradient Magnitudes - {model_name.upper()}\n'
                 f'(Larger gap = model works harder to memorize)',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add ratio annotations for layers with large differences
    high_data = gradient_df[gradient_df['group'] == 'high_memorization']
    low_data = gradient_df[gradient_df['group'] == 'low_memorization']
    if not high_data.empty and not low_data.empty:
        high_means = high_data.set_index('layer_name')['mean_gradient_norm']
        low_means = low_data.set_index('layer_name')['mean_gradient_norm']
        common_layers = high_means.index.intersection(low_means.index)

        for layer in common_layers:
            h = high_means.loc[layer]
            l = low_means.loc[layer]
            if l > 0 and h / l > 1.5:
                layer_idx = layer_names.index(layer)
                ratio = h / l
                ax.text(layer_idx, max(h, l) * 1.05, f'{ratio:.1f}x',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color='#d62728')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_probing_accuracy(probing_df: pd.DataFrame,
                          model_name: str = 'resnet50',
                          output_path: Optional[str] = None,
                          dpi: int = 300) -> plt.Figure:
    """
    KEY FIGURE: Bar chart showing linear probe AUROC at each layer.
    The layer where the probe achieves highest AUROC is where memorization
    information is most concentrated (linearly decodable).

    Also overlays Ridge R² as a secondary metric (continuous prediction).

    Args:
        probing_df: DataFrame with columns [layer_name, probe_auroc,
                    probe_auroc_std, ridge_r2, ridge_r2_std]
        model_name: Model name for title
        output_path: Where to save figure
        dpi: Figure resolution

    Returns:
        Figure object
    """
    setup_plot_style(dpi=dpi)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    layer_names = probing_df['layer_name'].tolist()
    x = np.arange(len(layer_names))
    width = 0.35

    # AUROC bars (primary)
    aurocs = probing_df['probe_auroc'].values
    auroc_stds = probing_df.get('probe_auroc_std', pd.Series([0] * len(probing_df))).values

    bars = ax1.bar(x - width / 2, aurocs, width, yerr=auroc_stds,
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5,
                   label='Probe AUROC', capsize=3, zorder=3)

    # Chance line
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5,
                alpha=0.7, label='Chance (0.5)', zorder=1)

    # Highlight the peak AUROC layer
    if len(aurocs) > 0:
        peak_idx = np.argmax(aurocs)
        peak_val = aurocs[peak_idx]
        bars[peak_idx].set_edgecolor('#000000')
        bars[peak_idx].set_linewidth(2.5)
        ax1.annotate(f'Peak: {peak_val:.3f}',
                     xy=(peak_idx - width / 2, peak_val),
                     xytext=(peak_idx + 1, peak_val + 0.03),
                     fontsize=11, fontweight='bold', color='#e74c3c',
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    # Ridge R² bars (secondary)
    if 'ridge_r2' in probing_df.columns:
        r2s = probing_df['ridge_r2'].values
        r2_stds = probing_df.get('ridge_r2_std', pd.Series([0] * len(probing_df))).values

        ax2 = ax1.twinx()
        ax2.bar(x + width / 2, r2s, width, yerr=r2_stds,
                color='#3498db', alpha=0.6, edgecolor='black', linewidth=0.5,
                label='Ridge R²', capsize=3, zorder=2)
        ax2.set_ylabel('Ridge R² (continuous M(x) prediction)', fontweight='bold',
                        color='#3498db')
        ax2.tick_params(axis='y', labelcolor='#3498db')
        ax2.set_ylim(bottom=min(0, min(r2s) - 0.05) if len(r2s) > 0 else -0.1)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                   framealpha=0.9)
    else:
        ax1.legend(loc='upper left', framealpha=0.9)

    # Formatting
    short_names = [_shorten_layer_name(ln, model_name) for ln in layer_names]
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax1.set_xlabel('Network Layer (early $\\rightarrow$ late)', fontweight='bold')
    ax1.set_ylabel('Probe AUROC (memorization decodability)', fontweight='bold',
                    color='#e74c3c')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax1.set_ylim(0.35, max(1.0, max(aurocs) + 0.1) if len(aurocs) > 0 else 1.0)
    ax1.set_title(f'Linear Probe: Where is Memorization Decodable? — {model_name.upper()}\n'
                  f'(Higher AUROC = layer encodes more memorization information)',
                  fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def generate_all_layerwise_plots(results_dict: Dict,
                                  output_dir: str,
                                  model_name: str = 'resnet50',
                                  dpi: int = 300) -> List[str]:
    """
    Generate all layer-wise plots for a given model.

    Args:
        results_dict: Output from LayerMemorizationAnalyzer.analyze(), expected keys:
            - 'layer_distances_df': DataFrame [image_id, layer_name, feature_distance]
            - 'memorization_scores_df': DataFrame [image_id, memorization_score]
            - 'layer_summary_df': DataFrame [layer_name, group, mean_distance, std_distance]
            - 'correlation_df': DataFrame [layer_name, spearman_rho, p_value]
            - 'gradient_df' (optional): DataFrame [layer_name, group, mean_gradient_norm, std_gradient_norm]
        output_dir: Directory to save figures
        model_name: Model name for titles
        dpi: Figure resolution

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    print(f"Generating layer-wise plots for {model_name.upper()}...")

    # 1. Layer distance heatmap (key figure)
    if 'layer_distances_df' in results_dict and 'memorization_scores_df' in results_dict:
        path = output_dir / f'{model_name}_layer_distance_heatmap.png'
        fig = plot_layer_distance_heatmap(
            results_dict['layer_distances_df'],
            results_dict['memorization_scores_df'],
            model_name=model_name,
            output_path=str(path),
            dpi=dpi
        )
        plt.close(fig)
        saved_files.append(str(path))

    # 2. Layer distance profile (high vs low M(x))
    if 'layer_summary_df' in results_dict:
        path = output_dir / f'{model_name}_layer_distance_profile.png'
        fig = plot_layer_distance_profile(
            results_dict['layer_summary_df'],
            model_name=model_name,
            output_path=str(path),
            dpi=dpi
        )
        plt.close(fig)
        saved_files.append(str(path))

    # 3. Layer correlation bar chart
    if 'correlation_df' in results_dict:
        path = output_dir / f'{model_name}_layer_correlation_bars.png'
        fig = plot_layer_correlation_bars(
            results_dict['correlation_df'],
            model_name=model_name,
            output_path=str(path),
            dpi=dpi
        )
        plt.close(fig)
        saved_files.append(str(path))

    # 4. Linear probing accuracy (KEY FIGURE)
    if 'probing_df' in results_dict:
        probing_df = results_dict['probing_df']
        # Plot each feature source separately
        for src in ['diff', 'concat', 'candidate_only']:
            if 'feature_source' in probing_df.columns:
                src_df = probing_df[probing_df['feature_source'] == src].copy()
            else:
                src_df = probing_df.copy()
                src = 'default'
            if len(src_df) == 0:
                continue
            suffix = f'_{src}' if src != 'default' else ''
            path = output_dir / f'{model_name}_probing_accuracy{suffix}.png'
            fig = plot_probing_accuracy(
                src_df,
                model_name=f'{model_name} ({src})',
                output_path=str(path),
                dpi=dpi,
            )
            plt.close(fig)
            saved_files.append(str(path))

    # 5. Gradient magnitude comparison
    if 'gradient_df' in results_dict:
        path = output_dir / f'{model_name}_gradient_magnitude.png'
        fig = plot_gradient_magnitude_comparison(
            results_dict['gradient_df'],
            model_name=model_name,
            output_path=str(path),
            dpi=dpi
        )
        plt.close(fig)
        saved_files.append(str(path))

    print(f"Generated {len(saved_files)} layer-wise plots in {output_dir}")
    plt.close('all')

    return saved_files


# Example usage
if __name__ == "__main__":
    print("Layer-wise Memorization Visualization Module")
    print("Creates publication-quality plots showing where memorization concentrates.")
    print()
    print("Available functions:")
    print("  plot_layer_distance_heatmap()       - Heatmap: layers x samples sorted by M(x)")
    print("  plot_layer_distance_profile()        - Line plot: high vs low M(x) per layer")
    print("  plot_layer_correlation_bars()         - Bar chart: Spearman rho per layer")
    print("  plot_cross_model_layer_comparison()   - 2x2 grid comparing all models")
    print("  plot_gradient_magnitude_comparison()  - Grouped bars: gradient norms per layer")
    print("  generate_all_layerwise_plots()        - Generate all plots for one model")
