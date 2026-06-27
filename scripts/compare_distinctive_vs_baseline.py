#!/usr/bin/env python3
"""
Compare Distinctive Experiment vs Baseline — "Smoking Gun" Analysis

Quantifies the effect of visual distinctiveness on memorization by comparing
memorization scores between a distinctive experiment and its baseline.

For each class, reports:
  - Baseline mean M(x)
  - Distinctive mean M(x)
  - Fold change (distinctive / baseline)
  - p-value (Mann-Whitney U)
  - Cohen's d effect size

Generates comparison figures:
  1. Side-by-side boxplots: baseline vs distinctive for each class
  2. Fold-change bar chart: M(x) distinctive / M(x) baseline per class
  3. Scatter plot: baseline M(x) vs distinctive M(x) colored by class

Usage:
    python scripts/compare_distinctive_vs_baseline.py \\
        --experiment retinal_oct_distinctive \\
        --baseline retinal_oct_baseline \\
        --models resnet50

    python scripts/compare_distinctive_vs_baseline.py \\
        --experiment ham1000_distinctive \\
        --baseline ham1000_baseline \\
        --models resnet50,vit
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse

PROJECT_ROOT = Path(__file__).parent.parent

# Publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_memorization_scores(results_dir: Path, model: str) -> pd.DataFrame:
    """Load memorization scores CSV for a given model."""
    csv_path = results_dir / 'memorization' / f'{model}_memorization_scores.csv'
    if not csv_path.exists():
        print(f"  [WARNING] Not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} samples from {csv_path.name}")
    return df


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float('nan')
    return (np.mean(group2) - np.mean(group1)) / pooled_std


def compute_per_class_comparison(baseline_df: pd.DataFrame,
                                  distinctive_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-class comparison statistics between baseline and distinctive.

    For each class present in both dataframes, computes:
      - baseline mean/median M(x)
      - distinctive mean/median M(x)
      - fold change
      - Mann-Whitney U test (two-sided)
      - Cohen's d effect size
    """
    # Get classes present in both
    baseline_classes = set(baseline_df['class_name'].unique())
    distinctive_classes = set(distinctive_df['class_name'].unique())
    common_classes = sorted(baseline_classes & distinctive_classes)

    rows = []
    for cls in common_classes:
        b_scores = baseline_df.loc[baseline_df['class_name'] == cls, 'memorization_score'].values
        d_scores = distinctive_df.loc[distinctive_df['class_name'] == cls, 'memorization_score'].values

        b_mean = np.mean(b_scores)
        b_median = np.median(b_scores)
        d_mean = np.mean(d_scores)
        d_median = np.median(d_scores)

        # Fold change: distinctive / baseline (use absolute means if baseline is near zero)
        if abs(b_mean) > 1e-6:
            fold_change = d_mean / b_mean
        else:
            fold_change = float('inf') if d_mean > 0 else (float('-inf') if d_mean < 0 else float('nan'))

        # Mann-Whitney U test (two-sided: is the distribution different?)
        if len(b_scores) >= 3 and len(d_scores) >= 3:
            u_stat, p_value = stats.mannwhitneyu(b_scores, d_scores, alternative='two-sided')
        else:
            u_stat, p_value = float('nan'), float('nan')

        # Cohen's d (positive = distinctive higher than baseline)
        d_effect = cohens_d(b_scores, d_scores)

        rows.append({
            'class': cls,
            'n_baseline': len(b_scores),
            'n_distinctive': len(d_scores),
            'baseline_mean': b_mean,
            'baseline_median': b_median,
            'baseline_std': np.std(b_scores, ddof=1) if len(b_scores) > 1 else 0.0,
            'distinctive_mean': d_mean,
            'distinctive_median': d_median,
            'distinctive_std': np.std(d_scores, ddof=1) if len(d_scores) > 1 else 0.0,
            'fold_change': fold_change,
            'abs_change': d_mean - b_mean,
            'mann_whitney_U': u_stat,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False,
            'cohens_d': d_effect,
        })

    return pd.DataFrame(rows)


def compute_matched_sample_comparison(baseline_df: pd.DataFrame,
                                       distinctive_df: pd.DataFrame) -> dict:
    """
    Compute matched-sample statistics for samples present in both experiments.
    Returns overall comparison metrics on the intersection of image_ids.
    """
    # Merge on image_id to get matched pairs
    merged = pd.merge(
        baseline_df[['image_id', 'class_name', 'memorization_score']],
        distinctive_df[['image_id', 'memorization_score']],
        on='image_id',
        suffixes=('_baseline', '_distinctive'),
        how='inner'
    )

    if len(merged) == 0:
        return {'n_matched': 0, 'warning': 'No matching image_ids between experiments'}

    # Paired Wilcoxon signed-rank test
    diff = merged['memorization_score_distinctive'] - merged['memorization_score_baseline']
    if len(diff) >= 10:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diff)
    else:
        wilcoxon_stat, wilcoxon_p = float('nan'), float('nan')

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(
        merged['memorization_score_distinctive'],
        merged['memorization_score_baseline']
    )

    # Correlation between baseline and distinctive scores
    spearman_rho, spearman_p = stats.spearmanr(
        merged['memorization_score_baseline'],
        merged['memorization_score_distinctive']
    )

    return {
        'n_matched': len(merged),
        'baseline_mean': float(merged['memorization_score_baseline'].mean()),
        'distinctive_mean': float(merged['memorization_score_distinctive'].mean()),
        'mean_difference': float(diff.mean()),
        'median_difference': float(diff.median()),
        'wilcoxon_stat': float(wilcoxon_stat),
        'wilcoxon_p': float(wilcoxon_p),
        'paired_t_stat': float(t_stat),
        'paired_t_p': float(t_p),
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
    }


def print_summary_table(comparison_df: pd.DataFrame, model: str,
                         matched_stats: dict) -> None:
    """Print a formatted summary table to stdout."""
    print(f"\n{'='*90}")
    print(f"  COMPARISON SUMMARY — {model.upper()}")
    print(f"{'='*90}")

    # Header
    print(f"\n{'Class':<12s} {'N_base':>6s} {'N_dist':>6s} "
          f"{'Base M(x)':>10s} {'Dist M(x)':>10s} "
          f"{'Fold':>8s} {'Delta':>8s} "
          f"{'p-value':>10s} {'Sig':>4s} {'Cohen d':>8s}")
    print("-" * 90)

    for _, row in comparison_df.iterrows():
        sig_marker = '***' if row['p_value'] < 0.001 else (
            '**' if row['p_value'] < 0.01 else (
                '*' if row['p_value'] < 0.05 else ''))

        fold_str = f"{row['fold_change']:.1f}x" if np.isfinite(row['fold_change']) else 'INF'

        d_str = f"{row['cohens_d']:.3f}" if np.isfinite(row['cohens_d']) else 'N/A'

        p_str = f"{row['p_value']:.2e}" if not np.isnan(row['p_value']) else 'N/A'

        print(f"{row['class']:<12s} {row['n_baseline']:>6d} {row['n_distinctive']:>6d} "
              f"{row['baseline_mean']:>10.4f} {row['distinctive_mean']:>10.4f} "
              f"{fold_str:>8s} {row['abs_change']:>+8.4f} "
              f"{p_str:>10s} {sig_marker:>4s} {d_str:>8s}")

    print("-" * 90)

    # Overall matched-sample stats
    if matched_stats.get('n_matched', 0) > 0:
        print(f"\n  Matched samples: {matched_stats['n_matched']}")
        print(f"  Overall baseline mean:     {matched_stats['baseline_mean']:.4f}")
        print(f"  Overall distinctive mean:  {matched_stats['distinctive_mean']:.4f}")
        print(f"  Mean difference:           {matched_stats['mean_difference']:+.4f}")
        print(f"  Paired Wilcoxon p:         {matched_stats['wilcoxon_p']:.2e}")
        print(f"  Paired t-test p:           {matched_stats['paired_t_p']:.2e}")
        print(f"  Score correlation (rho):   {matched_stats['spearman_rho']:.3f} "
              f"(p={matched_stats['spearman_p']:.2e})")
    else:
        print(f"\n  [NOTE] No matched image_ids found — experiments may use different canary sets.")

    # Highlight the "smoking gun" findings
    sig_rows = comparison_df[comparison_df['significant'] == True]
    if len(sig_rows) > 0:
        print(f"\n  SMOKING GUN FINDINGS:")
        for _, row in sig_rows.iterrows():
            direction = "INCREASED" if row['abs_change'] > 0 else "DECREASED"
            fold_str = f"{abs(row['fold_change']):.1f}x" if np.isfinite(row['fold_change']) else 'INF'
            print(f"    {row['class']}: M(x) {direction} by {abs(row['abs_change']):.4f} "
                  f"({fold_str} fold change, d={row['cohens_d']:.2f}, p={row['p_value']:.2e})")
    else:
        print(f"\n  No statistically significant differences found.")

    print(f"{'='*90}\n")


def plot_side_by_side_boxplots(baseline_df: pd.DataFrame, distinctive_df: pd.DataFrame,
                                comparison_df: pd.DataFrame, model: str,
                                output_dir: Path, experiment_name: str) -> None:
    """Side-by-side boxplots: baseline vs distinctive for each class."""
    classes = comparison_df['class'].tolist()
    n_classes = len(classes)

    if n_classes == 0:
        print("  [SKIP] No classes to plot for boxplots")
        return

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_classes), 6))

    # Build long-form data for seaborn
    plot_data = []
    for cls in classes:
        b_scores = baseline_df.loc[baseline_df['class_name'] == cls, 'memorization_score']
        for val in b_scores:
            plot_data.append({'class': cls, 'condition': 'Baseline', 'memorization_score': val})
        d_scores = distinctive_df.loc[distinctive_df['class_name'] == cls, 'memorization_score']
        for val in d_scores:
            plot_data.append({'class': cls, 'condition': 'Distinctive', 'memorization_score': val})

    plot_df = pd.DataFrame(plot_data)

    palette = {'Baseline': '#3498DB', 'Distinctive': '#E74C3C'}
    sns.boxplot(data=plot_df, x='class', y='memorization_score', hue='condition',
                palette=palette, ax=ax, showfliers=False, width=0.6)

    # Overlay strip plot for individual points
    sns.stripplot(data=plot_df, x='class', y='memorization_score', hue='condition',
                  palette=palette, ax=ax, dodge=True, alpha=0.3, size=2, legend=False)

    # Add significance annotations
    for i, cls in enumerate(classes):
        row = comparison_df[comparison_df['class'] == cls].iloc[0]
        if row['significant']:
            sig_marker = '***' if row['p_value'] < 0.001 else (
                '**' if row['p_value'] < 0.01 else '*')
            y_max = plot_df[plot_df['class'] == cls]['memorization_score'].quantile(0.95)
            y_pos = y_max + 0.1 * abs(y_max) + 0.05
            ax.annotate(sig_marker, xy=(i, y_pos), ha='center', fontsize=12,
                        fontweight='bold', color='red')

    ax.set_xlabel('Class')
    ax.set_ylabel('Memorization Score M(x)')
    ax.set_title(f'Baseline vs Distinctive Memorization — {model.upper()}\n'
                 f'{experiment_name}', fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.legend(title='Condition', loc='upper right')

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        filepath = output_dir / f'{model}_comparison_boxplots.{fmt}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / f'{model}_comparison_boxplots.png'}")
    plt.close(fig)


def plot_fold_change_bar(comparison_df: pd.DataFrame, model: str,
                          output_dir: Path, experiment_name: str) -> None:
    """Fold-change bar chart: M(x) distinctive / M(x) baseline per class."""
    df = comparison_df.copy()

    # Filter out infinite/nan fold changes for plotting
    df = df[np.isfinite(df['fold_change'])].copy()

    if len(df) == 0:
        print("  [SKIP] No finite fold changes to plot")
        return

    # Sort by fold change for visual clarity
    df = df.sort_values('fold_change', ascending=True)

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(df)), 6))

    x = np.arange(len(df))
    colors = []
    for _, row in df.iterrows():
        if row['significant'] and row['fold_change'] > 1:
            colors.append('#E74C3C')  # Significant increase — red
        elif row['significant'] and row['fold_change'] < 1:
            colors.append('#3498DB')  # Significant decrease — blue
        else:
            colors.append('#BDC3C7')  # Not significant — gray

    bars = ax.bar(x, df['fold_change'].values, color=colors, edgecolor='black',
                  linewidth=0.5, alpha=0.85)

    # Add value labels on bars
    for i, (_, row) in enumerate(df.iterrows()):
        fold_val = row['fold_change']
        sig_marker = ''
        if row['significant']:
            sig_marker = ' ***' if row['p_value'] < 0.001 else (
                ' **' if row['p_value'] < 0.01 else ' *')

        y_offset = 0.05 * max(abs(fold_val), 1)
        va = 'bottom' if fold_val >= 0 else 'top'
        y_pos = fold_val + y_offset if fold_val >= 0 else fold_val - y_offset

        ax.text(i, y_pos, f'{fold_val:.1f}x{sig_marker}', ha='center', va=va,
                fontsize=9, fontweight='bold' if row['significant'] else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels(df['class'].values, rotation=45, ha='right')
    ax.set_ylabel('Fold Change (Distinctive / Baseline)')
    ax.set_title(f'Memorization Fold Change by Class — {model.upper()}\n'
                 f'{experiment_name}', fontweight='bold')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='No change (1.0x)')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', edgecolor='black', label='Significant increase'),
        Patch(facecolor='#3498DB', edgecolor='black', label='Significant decrease'),
        Patch(facecolor='#BDC3C7', edgecolor='black', label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        filepath = output_dir / f'{model}_fold_change_bar.{fmt}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / f'{model}_fold_change_bar.png'}")
    plt.close(fig)


def plot_scatter_baseline_vs_distinctive(baseline_df: pd.DataFrame,
                                          distinctive_df: pd.DataFrame,
                                          model: str, output_dir: Path,
                                          experiment_name: str) -> None:
    """Scatter plot: baseline M(x) vs distinctive M(x) colored by class."""
    # Merge on image_id
    merged = pd.merge(
        baseline_df[['image_id', 'class_name', 'memorization_score']],
        distinctive_df[['image_id', 'memorization_score']],
        on='image_id',
        suffixes=('_baseline', '_distinctive'),
        how='inner'
    )

    if len(merged) == 0:
        print("  [SKIP] No matched samples for scatter plot")
        return

    classes = sorted(merged['class_name'].unique())
    n_classes = len(classes)

    # Use a categorical color palette
    palette = sns.color_palette('husl', n_colors=n_classes)
    color_map = {cls: palette[i] for i, cls in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(8, 8))

    for cls in classes:
        subset = merged[merged['class_name'] == cls]
        ax.scatter(
            subset['memorization_score_baseline'],
            subset['memorization_score_distinctive'],
            c=[color_map[cls]],
            label=f'{cls} (n={len(subset)})',
            alpha=0.5,
            s=20,
            edgecolors='none'
        )

    # Diagonal line (no change)
    all_vals = np.concatenate([
        merged['memorization_score_baseline'].values,
        merged['memorization_score_distinctive'].values
    ])
    lims = [np.percentile(all_vals, 1), np.percentile(all_vals, 99)]
    margin = (lims[1] - lims[0]) * 0.1
    lims = [lims[0] - margin, lims[1] + margin]

    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='No change (y=x)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Overall correlation
    rho, p = stats.spearmanr(
        merged['memorization_score_baseline'],
        merged['memorization_score_distinctive']
    )
    ax.text(0.05, 0.95, f'Spearman rho={rho:.3f}\np={p:.2e}\nn={len(merged)}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Baseline M(x)')
    ax.set_ylabel('Distinctive M(x)')
    ax.set_title(f'Baseline vs Distinctive Per-Sample Memorization — {model.upper()}\n'
                 f'{experiment_name}', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        filepath = output_dir / f'{model}_scatter_baseline_vs_distinctive.{fmt}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_dir / f'{model}_scatter_baseline_vs_distinctive.png'}")
    plt.close(fig)


def run_model_comparison(model: str, baseline_dir: Path, experiment_dir: Path,
                          output_dir: Path, experiment_name: str,
                          baseline_name: str) -> dict:
    """Run full comparison for a single model. Returns results dict."""
    print(f"\n--- Model: {model.upper()} ---")

    baseline_df = load_memorization_scores(baseline_dir, model)
    distinctive_df = load_memorization_scores(experiment_dir, model)

    if baseline_df is None:
        print(f"  [ERROR] No baseline scores for {model}")
        return None
    if distinctive_df is None:
        print(f"  [ERROR] No distinctive scores for {model}")
        return None

    # Per-class comparison
    comparison_df = compute_per_class_comparison(baseline_df, distinctive_df)

    # Matched-sample comparison
    matched_stats = compute_matched_sample_comparison(baseline_df, distinctive_df)

    # Print summary
    print_summary_table(comparison_df, model, matched_stats)

    # Generate figures
    display_name = f'{experiment_name} vs {baseline_name}'
    print(f"  Generating figures...")
    plot_side_by_side_boxplots(baseline_df, distinctive_df, comparison_df,
                                model, output_dir, display_name)
    plot_fold_change_bar(comparison_df, model, output_dir, display_name)
    plot_scatter_baseline_vs_distinctive(baseline_df, distinctive_df,
                                          model, output_dir, display_name)

    # Save comparison CSV
    csv_path = output_dir / f'{model}_per_class_comparison.csv'
    comparison_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Build results dict
    result = {
        'model': model,
        'baseline': baseline_name,
        'experiment': experiment_name,
        'n_baseline_samples': len(baseline_df),
        'n_distinctive_samples': len(distinctive_df),
        'per_class': comparison_df.to_dict(orient='records'),
        'matched_sample_stats': matched_stats,
        'n_significant_classes': int(comparison_df['significant'].sum()),
        'n_total_classes': len(comparison_df),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Compare distinctive experiment vs baseline memorization scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_distinctive_vs_baseline.py \\
      --experiment retinal_oct_distinctive --baseline retinal_oct_baseline

  python scripts/compare_distinctive_vs_baseline.py \\
      --experiment ham1000_distinctive --baseline ham1000_baseline \\
      --models resnet50,vit
        """
    )
    parser.add_argument('--experiment', required=True,
                        help='Distinctive experiment name (e.g., retinal_oct_distinctive)')
    parser.add_argument('--baseline', required=True,
                        help='Baseline experiment name (e.g., retinal_oct_baseline)')
    parser.add_argument('--models', default='resnet50,vit',
                        help='Comma-separated model names (default: resnet50,vit)')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: results/{experiment}/comparison/)')

    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(',')]

    baseline_dir = PROJECT_ROOT / 'results' / args.baseline
    experiment_dir = PROJECT_ROOT / 'results' / args.experiment

    if not baseline_dir.exists():
        print(f"[ERROR] Baseline directory not found: {baseline_dir}")
        sys.exit(1)
    if not experiment_dir.exists():
        print(f"[ERROR] Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else (experiment_dir / 'comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DISTINCTIVE vs BASELINE COMPARISON")
    print("=" * 70)
    print(f"  Baseline:    {args.baseline} ({baseline_dir})")
    print(f"  Experiment:  {args.experiment} ({experiment_dir})")
    print(f"  Models:      {', '.join(models)}")
    print(f"  Output:      {output_dir}")
    print("=" * 70)

    all_results = {}
    for model in models:
        result = run_model_comparison(
            model=model,
            baseline_dir=baseline_dir,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            experiment_name=args.experiment,
            baseline_name=args.baseline,
        )
        if result is not None:
            all_results[model] = result

    if not all_results:
        print("\n[ERROR] No models produced results. Check that memorization scores exist.")
        sys.exit(1)

    # Save combined JSON results
    json_path = output_dir / 'comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to: {json_path}")

    # Print cross-model summary if multiple models
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  CROSS-MODEL SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':<12s} {'Sig Classes':>12s} {'Max Fold':>10s} {'Max |d|':>10s}")
        print("-" * 50)
        for model, result in all_results.items():
            per_class = result['per_class']
            folds = [r['fold_change'] for r in per_class if np.isfinite(r['fold_change'])]
            effects = [abs(r['cohens_d']) for r in per_class if np.isfinite(r['cohens_d'])]
            max_fold = max(folds) if folds else float('nan')
            max_effect = max(effects) if effects else float('nan')
            print(f"{model:<12s} {result['n_significant_classes']:>5d}/{result['n_total_classes']:<5d} "
                  f"{max_fold:>10.1f}x {max_effect:>10.2f}")
        print("-" * 50)

    print(f"\nAll outputs saved to: {output_dir}")
    print("Done.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
