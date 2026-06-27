"""
Distribution and Disease-Specific Plots
Analyze memorization by disease class
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def setup_plot_style(style: str = 'seaborn-v0_8-darkgrid', dpi: int = 300):
    """
    Setup matplotlib style for publication-quality figures.

    STANDARDIZED font sizes (consistent with comparison_plots.py):
    - Base font: 12pt
    - Axis labels: 14pt
    - Titles: 16pt
    - Tick labels: 11pt
    - Legend: 11pt
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


def plot_disease_comparison(results_dict: Dict[str, pd.DataFrame],
                           output_path: Optional[Path] = None,
                           figsize: tuple = (14, 6)) -> plt.Figure:
    """
    Plot memorization by disease class for all models
    
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
    disease_data = []
    for model_name, df in results_dict.items():
        for disease in df['class_name'].unique():
            disease_df = df[df['class_name'] == disease]
            disease_data.append({
                'Model': model_name.upper(),
                'Disease': disease.upper(),
                'Mean': disease_df['memorization_score'].mean(),
                'Count': len(disease_df),
                'High_Risk_Pct': (disease_df['risk_category'] == 'high').mean() * 100
            })
    
    disease_summary = pd.DataFrame(disease_data)
    
    # Plot 1: Mean memorization by disease
    diseases = disease_summary['Disease'].unique()
    models = disease_summary['Model'].unique()
    
    x = np.arange(len(diseases))
    width = 0.8 / len(models)
    colors = sns.color_palette("Set2", len(models))
    
    for i, (model, color) in enumerate(zip(models, colors)):
        model_data = disease_summary[disease_summary['Model'] == model]
        means = [model_data[model_data['Disease'] == d]['Mean'].values[0] 
                for d in diseases]
        
        ax1.bar(x + i * width, means, width, label=model, 
               color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Disease Class', fontweight='bold')
    ax1.set_ylabel('Mean Memorization Score', fontweight='bold')
    ax1.set_title('Memorization by Disease Class', fontweight='bold', pad=20)
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(diseases, rotation=45, ha='right')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: High-risk percentage by disease
    for i, (model, color) in enumerate(zip(models, colors)):
        model_data = disease_summary[disease_summary['Model'] == model]
        pcts = [model_data[model_data['Disease'] == d]['High_Risk_Pct'].values[0] 
               for d in diseases]
        
        ax2.bar(x + i * width, pcts, width, label=model,
               color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Disease Class', fontweight='bold')
    ax2.set_ylabel('High-Risk Samples (%)', fontweight='bold')
    ax2.set_title('High-Risk Percentage by Disease', fontweight='bold', pad=20)
    ax2.set_xticks(x + width * (len(models) - 1) / 2)
    ax2.set_xticklabels(diseases, rotation=45, ha='right')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_disease_heatmap(results_dict: Dict[str, pd.DataFrame],
                        output_path: Optional[Path] = None,
                        figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot heatmap of memorization by disease and model
    
    Args:
        results_dict: Dictionary mapping model names to results DataFrames
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for heatmap
    models = list(results_dict.keys())
    diseases = list(results_dict[models[0]]['class_name'].unique())
    
    heatmap_data = np.zeros((len(models), len(diseases)))
    
    for i, model in enumerate(models):
        df = results_dict[model]
        for j, disease in enumerate(diseases):
            disease_scores = df[df['class_name'] == disease]['memorization_score']
            heatmap_data[i, j] = disease_scores.mean()
    
    # Create heatmap
    im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(diseases)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([d.upper() for d in diseases], rotation=45, ha='right')
    ax.set_yticklabels([m.upper() for m in models])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Memorization Score', rotation=270, labelpad=20, fontweight='bold')
    
    # Add values in cells
    for i in range(len(models)):
        for j in range(len(diseases)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                         ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Memorization Heatmap: Model × Disease', fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_top_memorized_samples(results_df: pd.DataFrame,
                               model_name: str,
                               n_samples: int = 20,
                               output_path: Optional[Path] = None,
                               figsize: tuple = (12, 8)) -> plt.Figure:
    """
    Plot top memorized samples
    
    Args:
        results_df: Results DataFrame for one model
        model_name: Model name
        n_samples: Number of top samples to show
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Figure object
    """
    setup_plot_style()
    
    # Get top memorized samples
    top_samples = results_df.nlargest(n_samples, 'memorization_score')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    y_pos = np.arange(len(top_samples))
    scores = top_samples['memorization_score'].values
    colors = ['darkred' if s > 0.3 else 'orange' if s > 0.1 else 'green' 
             for s in scores]
    
    ax.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Set labels
    labels = [f"{row['image_id']} ({row['class_name'].upper()})" 
             for _, row in top_samples.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    
    ax.set_xlabel('Memorization Score', fontweight='bold')
    ax.set_title(f'Top {n_samples} Memorized Samples - {model_name.upper()}',
                fontweight='bold', pad=20)
    ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5, 
              label='Moderate')
    ax.axvline(x=0.3, color='darkred', linestyle='--', linewidth=1, alpha=0.5,
              label='High Risk')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def plot_distribution_kde(results_dict: Dict[str, pd.DataFrame],
                         output_path: Optional[Path] = None,
                         figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot KDE (kernel density estimation) of memorization distributions
    
    Args:
        results_dict: Dictionary mapping model names to results DataFrames
        output_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette("Set2", len(results_dict))
    
    for (model_name, df), color in zip(results_dict.items(), colors):
        scores = df['memorization_score'].values
        
        # Plot KDE
        sns.kdeplot(data=scores, label=model_name.upper(), 
                   color=color, linewidth=2, ax=ax, fill=True, alpha=0.3)
    
    ax.set_xlabel('Memorization Score', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Memorization Score Distribution (KDE)', fontweight='bold', pad=20)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.3, color='darkred', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
    
    return fig


def create_all_distribution_plots(results_dict: Dict[str, pd.DataFrame],
                                  output_dir: Path):
    """
    Create all distribution plots
    
    Args:
        results_dict: Dictionary mapping model names to results DataFrames
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating distribution plots...")
    
    # Disease comparison
    plot_disease_comparison(
        results_dict,
        output_path=output_dir / 'disease_comparison.png'
    )
    
    # Disease heatmap
    plot_disease_heatmap(
        results_dict,
        output_path=output_dir / 'disease_heatmap.png'
    )
    
    # KDE plot
    plot_distribution_kde(
        results_dict,
        output_path=output_dir / 'distribution_kde.png'
    )
    
    # Top memorized for each model
    for model_name, df in results_dict.items():
        plot_top_memorized_samples(
            df,
            model_name,
            n_samples=20,
            output_path=output_dir / f'top_memorized_{model_name}.png'
        )
    
    print(f"\n✅ All distribution plots saved to: {output_dir}")
    plt.close('all')


# Example usage
if __name__ == "__main__":
    print("Distribution Plots Module")
    print("Creates disease-specific and distribution visualizations")