#!/usr/bin/env python3
"""
Visualize Retinal OCT Distinctive Feature Experiment Results

Publication figures:
- Dose-response curve (effect size vs distinctiveness level)
- Per-class comparison across all 4 conditions
- Violin plots for rare vs common at each level

Usage:
    python scripts/visualize_retinal_oct_distinctive.py
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent

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

LEVELS = [
    ('retinal_oct_baseline', 'Baseline\n(Original)', 0),
    ('retinal_oct_distinctive', 'Grayscale\n(Level 1)', 1),
    ('retinal_oct_distinctive_invert', 'Inversion\n(Level 2)', 2),
    ('retinal_oct_distinctive_edge', 'Edge Det.\n(Level 3)', 3),
]

COLORS = {
    'rare': '#E74C3C',
    'common': '#3498DB',
    'effect': '#8E44AD',
    'sig': '#27AE60',
    'nonsig': '#BDC3C7',
}

DISTINCTIVE_CLASSES = ['DRUSEN', 'DME']


def load_all_results():
    results = {}
    for result_dir, label, level in LEVELS:
        rarity_path = (PROJECT_ROOT / 'results' / result_dir /
                       'rarity_analysis' / 'resnet50' / 'rarity_analysis_results.json')
        scores_path = (PROJECT_ROOT / 'results' / result_dir /
                       'memorization' / 'resnet50_memorization_scores.csv')

        entry = {'label': label, 'level': level}
        if rarity_path.exists():
            with open(rarity_path) as f:
                entry['rarity'] = json.load(f)
        if scores_path.exists():
            entry['scores'] = pd.read_csv(scores_path)

        results[result_dir] = entry

    return results


def create_dose_response_figure(results, output_dir):
    """Main publication figure: dose-response curve."""
    print("Creating dose-response figure...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    levels = []
    labels = []
    cliffs_deltas = []
    p_values = []
    rare_means = []
    common_means = []

    for result_dir, label, level in LEVELS:
        entry = results.get(result_dir, {})
        rarity = entry.get('rarity')
        if not rarity:
            continue

        mw = rarity.get('mann_whitney_test', {})
        es = rarity.get('effect_sizes', {})

        levels.append(level)
        labels.append(label)
        cliffs_deltas.append(es.get('cliffs_delta', 0))
        p_val = mw.get('p_value_one_sided', mw.get('p_value_two_sided', 1.0))
        p_values.append(p_val if isinstance(p_val, (int, float)) else 1.0)
        rare_means.append(mw.get('rare_mean', 0))
        common_means.append(mw.get('common_mean', 0))

    if not levels:
        print("[ERROR] No data to plot")
        return

    x = np.array(levels)

    # Panel A: Cliff's delta dose-response
    ax = axes[0]
    colors = [COLORS['sig'] if p < 0.05 else COLORS['nonsig'] for p in p_values]
    ax.bar(x, cliffs_deltas, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    for i, (val, p) in enumerate(zip(cliffs_deltas, p_values)):
        marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax.text(x[i], val + 0.02, f'{val:.2f}\n{marker}', ha='center', fontsize=9)
    ax.axhline(y=0.147, color='green', linestyle='--', alpha=0.4, label='Small (0.147)')
    ax.axhline(y=0.33, color='orange', linestyle='--', alpha=0.4, label='Medium (0.33)')
    ax.axhline(y=0.474, color='red', linestyle='--', alpha=0.4, label='Large (0.474)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cliff's \u03b4 (Rare vs Common)")
    ax.set_title("A. Effect Size by Distinctiveness Level", fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')

    # Panel B: p-value dose-response
    ax = axes[1]
    log_p = [-np.log10(max(p, 1e-10)) for p in p_values]
    colors = [COLORS['sig'] if p < 0.05 else COLORS['nonsig'] for p in p_values]
    ax.bar(x, log_p, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='\u03b1 = 0.05')
    for i, (lp, p) in enumerate(zip(log_p, p_values)):
        ax.text(x[i], lp + 0.1, f'p={p:.4f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('-log\u2081\u2080(p-value)')
    ax.set_title('B. Statistical Significance', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel C: Rare vs common means
    ax = axes[2]
    width = 0.35
    ax.bar(x - width/2, rare_means, width, label='Rare (DRUSEN+DME)', color=COLORS['rare'],
           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, common_means, width, label='Common (CNV+NORMAL)', color=COLORS['common'],
           alpha=0.8, edgecolor='black', linewidth=0.5)
    for i in range(len(x)):
        ax.text(x[i] - width/2, rare_means[i] + 0.02, f'{rare_means[i]:.2f}',
                ha='center', fontsize=8)
        ax.text(x[i] + width/2, common_means[i] + 0.02, f'{common_means[i]:.2f}',
                ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean Memorization M(x)')
    ax.set_title('C. Rare vs Common by Level', fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.legend()

    fig.suptitle('Dose-Response: Visual Distinctiveness \u2192 Memorization of Rare Diseases\n'
                 'Retinal OCT Dataset, ResNet50, Target Classes: DRUSEN + DME',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        filepath = output_dir / f'fig_retinal_oct_dose_response.{fmt}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")
    plt.close(fig)


def create_per_class_figure(results, output_dir):
    """Per-class memorization across all 4 conditions."""
    print("Creating per-class comparison figure...")

    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

    fig, ax = plt.subplots(figsize=(12, 6))

    n_conditions = 0

    for result_dir, label, level in LEVELS:
        entry = results.get(result_dir, {})
        rarity = entry.get('rarity')
        if not rarity:
            continue

        class_data = {d['class']: d['mean_memorization']
                      for d in rarity.get('spearman_correlation', {}).get('data', [])}

        vals = [class_data.get(c, 0) for c in classes if c in class_data]
        present_classes = [c for c in classes if c in class_data]

        if not vals:
            continue

        x = np.arange(len(present_classes))
        offset = (n_conditions - 1.5) * 0.2
        color = plt.cm.viridis(level / 3.0)

        ax.bar(x + offset, vals, 0.18, label=label.replace('\n', ' '),
               color=color, alpha=0.8, edgecolor='black', linewidth=0.3)

        n_conditions += 1

    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Mean Memorization Score M(x)')
    ax.set_title('Per-Class Memorization Across Distinctiveness Levels', fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')

    # Mark distinctive classes
    for i, c in enumerate(classes):
        if c in DISTINCTIVE_CLASSES:
            ax.annotate('\u2020', xy=(i, ax.get_ylim()[1] * 0.95), ha='center',
                        fontsize=14, fontweight='bold', color='red')

    ax.set_xlabel('\u2020 = classes with distinctive transformation applied')

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        filepath = output_dir / f'fig_retinal_oct_per_class.{fmt}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")
    plt.close(fig)


def create_violin_figure(results, output_dir):
    """Violin plots for rare vs common at each level."""
    print("Creating violin comparison figure...")

    available = [(rd, label, level) for rd, label, level in LEVELS
                 if results.get(rd, {}).get('scores') is not None]

    if not available:
        print("[SKIP] No score data for violin plots")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5), squeeze=False)
    axes = axes[0]

    for idx, (result_dir, label, level) in enumerate(available):
        ax = axes[idx]
        df = results[result_dir]['scores']

        rare_scores = df[df['class_name'].isin(DISTINCTIVE_CLASSES)]['memorization_score'].values
        common_scores = df[~df['class_name'].isin(DISTINCTIVE_CLASSES)]['memorization_score'].values

        if len(rare_scores) == 0 or len(common_scores) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label.replace('\n', ' '), fontweight='bold')
            continue

        parts = ax.violinplot([rare_scores, common_scores], positions=[1, 2],
                              showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor([COLORS['rare'], COLORS['common']][i])
            pc.set_alpha(0.7)
        for partname in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
            if partname in parts:
                parts[partname].set_color('black')

        # Significance bracket
        u_stat, p_val = stats.mannwhitneyu(rare_scores, common_scores, alternative='greater')
        sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        y_max = max(rare_scores.max(), common_scores.max()) + 0.5
        ax.plot([1, 1, 2, 2], [y_max, y_max+0.2, y_max+0.2, y_max], 'k-', lw=1)
        ax.text(1.5, y_max+0.3, f'{sig_marker}\np={p_val:.4f}', ha='center', fontsize=8)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Rare\n(DRUSEN+DME)', 'Common\n(CNV+NORMAL)'])
        ax.set_ylabel('M(x)' if idx == 0 else '')
        ax.set_title(label.replace('\n', ' '), fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    fig.suptitle('Rare vs Common Memorization at Each Distinctiveness Level\n'
                 'Retinal OCT, ResNet50',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        filepath = output_dir / f'fig_retinal_oct_violin.{fmt}'
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Retinal OCT Distinctive Feature Visualization")
    print("=" * 60)

    results = load_all_results()

    available = [rd for rd, _, _ in LEVELS if 'rarity' in results.get(rd, {})]
    print(f"\nAvailable conditions: {len(available)}/{len(LEVELS)}")
    for rd, label, level in LEVELS:
        status = "OK" if 'rarity' in results.get(rd, {}) else "MISSING"
        print(f"  {label.replace(chr(10), ' '):<25s} [{status}]")

    if len(available) < 2:
        print("\n[ERROR] Need at least 2 conditions. Run experiments first.")
        return 1

    output_dir = PROJECT_ROOT / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    create_dose_response_figure(results, output_dir)
    create_per_class_figure(results, output_dir)
    create_violin_figure(results, output_dir)

    print(f"\nAll figures saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
