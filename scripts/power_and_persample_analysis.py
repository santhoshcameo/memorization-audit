#!/usr/bin/env python3
"""
Statistical Power Analysis + Per-Sample Correlation Tests

Analyzes all baseline experiments to produce:
1. Statistical power for each dataset x model (explains why some datasets show weak effects)
2. Per-sample Spearman correlation (frequency vs memorization, n=all samples instead of n=classes)
3. Summary tables for publication

Usage:
    python scripts/power_and_persample_analysis.py
    python scripts/power_and_persample_analysis.py --experiments ham1000_baseline,kvasir_baseline
    python scripts/power_and_persample_analysis.py --output results/statistical_analysis/
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# All baseline experiments
ALL_EXPERIMENTS = {
    'ham1000_baseline': 'results/ham1000_baseline',
    'kvasir_baseline': 'results/kvasir_baseline',
    'chestxray_baseline': 'results/chestxray_baseline',
    'odir5k_baseline': 'results/odir5k_baseline',
    'retinal_oct_baseline': 'results/retinal_oct_baseline',
}

MODELS = ['resnet50', 'vit', 'mae', 'medsam']

RARE_THRESHOLD = 0.05  # <5% = rare


def load_memorization_scores(results_dir: Path, model: str) -> pd.DataFrame:
    """Load memorization scores CSV for a model."""
    csv_path = results_dir / 'memorization' / f'{model}_memorization_scores.csv'
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def compute_class_frequencies(df: pd.DataFrame) -> dict:
    """Compute class frequency (proportion) from the data."""
    counts = df['class_name'].value_counts()
    total = len(df)
    return {cls: count / total for cls, count in counts.items()}


def assign_sample_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'class_frequency' column — each sample gets its class's frequency."""
    freq_map = compute_class_frequencies(df)
    df = df.copy()
    df['class_frequency'] = df['class_name'].map(freq_map)
    df['is_rare'] = df['class_frequency'] < RARE_THRESHOLD
    return df


def per_sample_spearman(df: pd.DataFrame) -> dict:
    """
    Per-sample Spearman correlation: class_frequency vs memorization_score.

    This uses n=all_samples (e.g., 1503) instead of n=classes (e.g., 7),
    giving much higher statistical power.
    """
    rho, p_value = stats.spearmanr(df['class_frequency'], df['memorization_score'])
    n = len(df)

    return {
        'spearman_rho': float(rho),
        'p_value': float(p_value),
        'n_samples': n,
        'significant': p_value < 0.05,
        'direction': 'rare memorized MORE' if rho < 0 else 'common memorized MORE',
    }


def per_class_spearman(df: pd.DataFrame) -> dict:
    """
    Per-class Spearman (existing approach): uses class means.
    Included for comparison with per-sample.
    """
    class_stats = df.groupby('class_name').agg(
        mean_mem=('memorization_score', 'mean'),
        frequency=('class_frequency', 'first'),
    )
    n_classes = len(class_stats)

    if n_classes < 3:
        return {'spearman_rho': float('nan'), 'p_value': float('nan'),
                'n_classes': n_classes, 'significant': False, 'direction': 'N/A'}

    rho, p_value = stats.spearmanr(class_stats['frequency'], class_stats['mean_mem'])

    return {
        'spearman_rho': float(rho),
        'p_value': float(p_value),
        'n_classes': n_classes,
        'significant': p_value < 0.05,
        'direction': 'rare memorized MORE' if rho < 0 else 'common memorized MORE',
    }


def mann_whitney_per_sample(df: pd.DataFrame) -> dict:
    """Per-sample Mann-Whitney U: rare group vs common group."""
    rare = df[df['is_rare']]['memorization_score']
    common = df[~df['is_rare']]['memorization_score']

    if len(rare) < 2 or len(common) < 2:
        return {
            'u_statistic': float('nan'), 'p_value': float('nan'),
            'n_rare': len(rare), 'n_common': len(common),
            'significant': False, 'rare_mean': float('nan'), 'common_mean': float('nan'),
        }

    u_stat, p_two = stats.mannwhitneyu(rare, common, alternative='two-sided')
    _, p_greater = stats.mannwhitneyu(rare, common, alternative='greater')

    return {
        'u_statistic': float(u_stat),
        'p_value_two_sided': float(p_two),
        'p_value_one_sided': float(p_greater),
        'n_rare': len(rare),
        'n_common': len(common),
        'rare_mean': float(rare.mean()),
        'common_mean': float(common.mean()),
        'rare_median': float(rare.median()),
        'common_median': float(common.median()),
        'significant': p_two < 0.05,
        'direction': 'rare > common' if rare.mean() > common.mean() else 'common > rare',
    }


def compute_cohens_d(df: pd.DataFrame) -> float:
    """Cohen's d between rare and common groups."""
    rare = df[df['is_rare']]['memorization_score']
    common = df[~df['is_rare']]['memorization_score']

    if len(rare) < 2 or len(common) < 2:
        return float('nan')

    pooled_std = np.sqrt(
        ((len(rare) - 1) * rare.std()**2 + (len(common) - 1) * common.std()**2)
        / (len(rare) + len(common) - 2)
    )
    if pooled_std == 0:
        return float('nan')

    return float((rare.mean() - common.mean()) / pooled_std)


def compute_power(n_rare: int, n_common: int, cohens_d: float,
                  alpha: float = 0.05) -> float:
    """
    Compute statistical power for a two-sample t-test.

    Uses the smaller group size as the limiting factor.
    """
    if np.isnan(cohens_d) or cohens_d == 0 or n_rare < 2 or n_common < 2:
        return float('nan')

    power_analysis = TTestIndPower()
    # Ratio of common to rare group sizes
    ratio = n_common / n_rare

    try:
        power = power_analysis.solve_power(
            effect_size=abs(cohens_d),
            nobs1=n_rare,
            ratio=ratio,
            alpha=alpha,
            alternative='two-sided',
        )
        return float(power)
    except Exception:
        return float('nan')


def compute_min_n_for_power(cohens_d: float, target_power: float = 0.80,
                            ratio: float = 1.0, alpha: float = 0.05) -> int:
    """Compute minimum n_rare needed to detect the given effect at target power."""
    if np.isnan(cohens_d) or cohens_d == 0:
        return -1

    power_analysis = TTestIndPower()
    try:
        n = power_analysis.solve_power(
            effect_size=abs(cohens_d),
            power=target_power,
            ratio=ratio,
            alpha=alpha,
            alternative='two-sided',
        )
        return int(np.ceil(n))
    except Exception:
        return -1


def analyze_experiment(experiment_name: str, results_dir: Path) -> dict:
    """Run full analysis for one experiment."""
    experiment_results = {}

    for model in MODELS:
        df = load_memorization_scores(results_dir, model)
        if df is None:
            print(f"  Skipping {model} — no scores CSV")
            continue

        df = assign_sample_frequency(df)

        n_rare = int(df['is_rare'].sum())
        n_common = int((~df['is_rare']).sum())

        # Per-sample Spearman
        ps_spearman = per_sample_spearman(df)

        # Per-class Spearman (for comparison)
        pc_spearman = per_class_spearman(df)

        # Mann-Whitney U
        mw = mann_whitney_per_sample(df)

        # Effect size
        d = compute_cohens_d(df)

        # Power analysis
        power = compute_power(n_rare, n_common, d)

        # Min N for 80% power
        ratio = n_common / n_rare if n_rare > 0 else 1.0
        min_n = compute_min_n_for_power(d, target_power=0.80, ratio=ratio)

        experiment_results[model] = {
            'n_total': len(df),
            'n_rare': n_rare,
            'n_common': n_common,
            'n_classes': df['class_name'].nunique(),
            'per_sample_spearman': ps_spearman,
            'per_class_spearman': pc_spearman,
            'mann_whitney': mw,
            'cohens_d': d,
            'cohens_d_interpretation': interpret_d(d),
            'power': power,
            'power_interpretation': interpret_power(power),
            'min_n_rare_for_80pct_power': min_n,
            'well_powered': power >= 0.80 if not np.isnan(power) else False,
        }

    return experiment_results


def interpret_d(d: float) -> str:
    if np.isnan(d):
        return 'N/A'
    d = abs(d)
    if d >= 0.8:
        return 'large'
    elif d >= 0.5:
        return 'medium'
    elif d >= 0.2:
        return 'small'
    return 'negligible'


def interpret_power(p: float) -> str:
    if np.isnan(p):
        return 'N/A'
    if p >= 0.80:
        return 'well-powered'
    elif p >= 0.50:
        return 'moderate'
    elif p >= 0.20:
        return 'low'
    return 'very low'


def format_p(p: float) -> str:
    """Format p-value for publication."""
    if np.isnan(p):
        return 'N/A'
    if p < 0.001:
        return f'{p:.2e}'
    elif p < 0.01:
        return f'{p:.4f}'
    else:
        return f'{p:.3f}'


def generate_tables(all_results: dict) -> str:
    """Generate markdown tables for the results."""
    lines = []

    # ========== Table 1: Power Analysis ==========
    lines.append("## Statistical Power Analysis")
    lines.append("")
    lines.append("| Dataset | Model | n_rare | n_common | Cohen's d | Interpretation | Power | Well-Powered? | Min n for 80% |")
    lines.append("|---------|-------|--------|----------|-----------|----------------|-------|---------------|---------------|")

    for exp_name, exp_results in all_results.items():
        dataset = exp_name.replace('_baseline', '').upper()
        for model, r in exp_results.items():
            d_val = r['cohens_d']
            d_str = f"{d_val:.3f}" if not np.isnan(d_val) else "N/A"
            pow_val = r['power']
            pow_str = f"{pow_val:.3f}" if not np.isnan(pow_val) else "N/A"
            min_n = r['min_n_rare_for_80pct_power']
            min_n_str = str(min_n) if min_n > 0 else "N/A"
            well = "YES" if r['well_powered'] else "no"

            lines.append(
                f"| {dataset} | {model} | {r['n_rare']} | {r['n_common']} | "
                f"{d_str} | {r['cohens_d_interpretation']} | {pow_str} | "
                f"**{well}** | {min_n_str} |"
            )

    # ========== Table 2: Per-Sample vs Per-Class Spearman ==========
    lines.append("")
    lines.append("## Per-Sample vs Per-Class Spearman Correlation")
    lines.append("")
    lines.append("| Dataset | Model | Per-Class rho (n=classes) | p-value | Per-Sample rho (n=samples) | p-value | Power Gain |")
    lines.append("|---------|-------|--------------------------|---------|---------------------------|---------|------------|")

    for exp_name, exp_results in all_results.items():
        dataset = exp_name.replace('_baseline', '').upper()
        for model, r in exp_results.items():
            pc = r['per_class_spearman']
            ps = r['per_sample_spearman']

            pc_rho = f"{pc['spearman_rho']:.3f}" if not np.isnan(pc['spearman_rho']) else "N/A"
            ps_rho = f"{ps['spearman_rho']:.3f}"

            pc_p = format_p(pc['p_value'])
            ps_p = format_p(ps['p_value'])

            n_class = pc.get('n_classes', '?')
            n_samp = ps['n_samples']

            lines.append(
                f"| {dataset} | {model} | {pc_rho} (n={n_class}) | {pc_p} | "
                f"{ps_rho} (n={n_samp}) | {ps_p} | {n_samp}x |"
            )

    # ========== Table 3: Mann-Whitney Summary ==========
    lines.append("")
    lines.append("## Per-Sample Mann-Whitney U Test (Rare vs Common)")
    lines.append("")
    lines.append("| Dataset | Model | Rare Mean M(x) | Common Mean M(x) | Difference | U | p-value | Significant? |")
    lines.append("|---------|-------|----------------|------------------|------------|---|---------|--------------|")

    for exp_name, exp_results in all_results.items():
        dataset = exp_name.replace('_baseline', '').upper()
        for model, r in exp_results.items():
            mw = r['mann_whitney']
            if np.isnan(mw.get('rare_mean', float('nan'))):
                continue
            diff = mw['rare_mean'] - mw['common_mean']
            sig = "**YES**" if mw['significant'] else "no"
            lines.append(
                f"| {dataset} | {model} | {mw['rare_mean']:+.3f} | {mw['common_mean']:+.3f} | "
                f"{diff:+.3f} | {mw['u_statistic']:.0f} | {format_p(mw['p_value_two_sided'])} | {sig} |"
            )

    return "\n".join(lines)


def generate_narrative(all_results: dict) -> str:
    """Generate publication-ready narrative summary."""
    lines = []
    lines.append("## Key Findings")
    lines.append("")

    # Count well-powered
    total = 0
    well_powered = 0
    sig_persample = 0
    sig_perclass = 0
    sig_mw = 0

    for exp_results in all_results.values():
        for r in exp_results.values():
            total += 1
            if r['well_powered']:
                well_powered += 1
            if r['per_sample_spearman']['significant']:
                sig_persample += 1
            if r['per_class_spearman']['significant']:
                sig_perclass += 1
            if r['mann_whitney']['significant']:
                sig_mw += 1

    lines.append(f"### Power Analysis ({total} model-dataset pairs)")
    lines.append(f"- **{well_powered}/{total}** pairs are well-powered (>=80%) to detect the observed effect")
    lines.append(f"- Underpowered pairs have small effect sizes or insufficient rare samples")
    lines.append("")

    lines.append(f"### Correlation Tests")
    lines.append(f"- **Per-class Spearman** (n=classes): {sig_perclass}/{total} significant — low power due to small n")
    lines.append(f"- **Per-sample Spearman** (n=samples): {sig_persample}/{total} significant — much higher power")
    lines.append(f"- Per-sample test recovers {sig_persample - sig_perclass} additional significant correlations")
    lines.append("")

    lines.append(f"### Mann-Whitney U (Per-Sample)")
    lines.append(f"- **{sig_mw}/{total}** pairs show significant rare vs common memorization difference")
    lines.append("")

    # Per-dataset summary
    lines.append("### Per-Dataset Power Summary")
    lines.append("")
    for exp_name, exp_results in all_results.items():
        dataset = exp_name.replace('_baseline', '')
        powers = [r['power'] for r in exp_results.values() if not np.isnan(r['power'])]
        n_well = sum(1 for r in exp_results.values() if r['well_powered'])
        n_models = len(exp_results)

        if powers:
            lines.append(f"- **{dataset}**: {n_well}/{n_models} well-powered, mean power={np.mean(powers):.2f}")
        else:
            lines.append(f"- **{dataset}**: insufficient data for power analysis")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Power analysis + per-sample tests')
    parser.add_argument('--experiments', type=str, default=None,
                        help='Comma-separated experiment names (default: all baselines)')
    parser.add_argument('--output', type=str, default='results/statistical_analysis',
                        help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Rare class threshold (default: 0.05)')
    args = parser.parse_args()

    global RARE_THRESHOLD
    RARE_THRESHOLD = args.threshold

    # Select experiments
    if args.experiments:
        experiments = {
            name: ALL_EXPERIMENTS[name]
            for name in args.experiments.split(',')
            if name in ALL_EXPERIMENTS
        }
    else:
        experiments = ALL_EXPERIMENTS

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STATISTICAL POWER ANALYSIS + PER-SAMPLE TESTS")
    print("=" * 80)
    print(f"Experiments: {list(experiments.keys())}")
    print(f"Models: {MODELS}")
    print(f"Rare threshold: {RARE_THRESHOLD}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    all_results = {}

    for exp_name, results_path in experiments.items():
        results_dir = PROJECT_ROOT / results_path
        if not results_dir.exists():
            print(f"\nSkipping {exp_name} — directory not found: {results_dir}")
            continue

        print(f"\n--- {exp_name} ---")
        exp_results = analyze_experiment(exp_name, results_dir)
        all_results[exp_name] = exp_results

        for model, r in exp_results.items():
            d = r['cohens_d']
            p = r['power']
            ps = r['per_sample_spearman']
            print(f"  {model}: d={d:.3f} ({r['cohens_d_interpretation']}), "
                  f"power={p:.3f} ({r['power_interpretation']}), "
                  f"per-sample rho={ps['spearman_rho']:.3f} p={format_p(ps['p_value'])}")

    # Generate outputs
    print("\n" + "=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)

    # 1. Full JSON results
    json_path = output_dir / 'power_and_persample_results.json'

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"  JSON: {json_path}")

    # 2. Markdown tables
    tables = generate_tables(all_results)
    narrative = generate_narrative(all_results)

    md_path = output_dir / 'power_and_persample_summary.md'
    with open(md_path, 'w') as f:
        f.write("# Statistical Power Analysis & Per-Sample Tests\n\n")
        f.write(f"**Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}**\n")
        f.write(f"**Rare threshold: <{RARE_THRESHOLD*100:.0f}%**\n\n")
        f.write("---\n\n")
        f.write(narrative)
        f.write("\n\n---\n\n")
        f.write(tables)
        f.write("\n")
    print(f"  Markdown: {md_path}")

    # 3. CSV summary for easy import
    rows = []
    for exp_name, exp_results in all_results.items():
        for model, r in exp_results.items():
            ps = r['per_sample_spearman']
            pc = r['per_class_spearman']
            mw = r['mann_whitney']
            rows.append({
                'dataset': exp_name.replace('_baseline', ''),
                'model': model,
                'n_total': r['n_total'],
                'n_rare': r['n_rare'],
                'n_common': r['n_common'],
                'n_classes': r['n_classes'],
                'cohens_d': r['cohens_d'],
                'cohens_d_interp': r['cohens_d_interpretation'],
                'power': r['power'],
                'well_powered': r['well_powered'],
                'min_n_for_80pct': r['min_n_rare_for_80pct_power'],
                'persample_rho': ps['spearman_rho'],
                'persample_p': ps['p_value'],
                'persample_sig': ps['significant'],
                'perclass_rho': pc['spearman_rho'],
                'perclass_p': pc['p_value'],
                'perclass_sig': pc['significant'],
                'mw_rare_mean': mw.get('rare_mean', float('nan')),
                'mw_common_mean': mw.get('common_mean', float('nan')),
                'mw_p': mw.get('p_value_two_sided', float('nan')),
                'mw_sig': mw.get('significant', False),
            })

    csv_path = output_dir / 'power_and_persample_summary.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    # Print summary
    print("\n" + narrative)


if __name__ == '__main__':
    main()
