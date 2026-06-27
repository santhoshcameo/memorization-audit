#!/usr/bin/env python3
"""
Membership Inference Attack (MIA) Validation Script

Validates the memorization metric M(x) by demonstrating correlation
with membership inference attack success, particularly for rare classes.

This script:
1. Loads existing memorization measurement results
2. Runs multiple MIA attack methods
3. Analyzes vulnerability by class rarity
4. Generates publication-quality reports and figures


Usage:
    python scripts/run_mia_analysis.py --experiment ham1000_baseline
    python scripts/run_mia_analysis.py --experiment ham1000_baseline --models resnet50,vit
    python scripts/run_mia_analysis.py --all-experiments
"""

import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.membership_inference import (
    MembershipInferenceAttack,
    MIAResult,
    generate_mia_report
)


def load_memorization_results(experiment_dir: Path, model_name: str) -> pd.DataFrame:
    """Load memorization scores for a model"""
    mem_csv = experiment_dir / 'memorization' / f'{model_name}_memorization_scores.csv'
    if not mem_csv.exists():
        raise FileNotFoundError(f"Memorization results not found: {mem_csv}")
    return pd.read_csv(mem_csv)


def load_class_frequencies(experiment_dir: Path, model_name: str) -> dict:
    """Load class frequency data"""
    freq_csv = experiment_dir / 'rarity_analysis' / model_name / 'class_frequency_table.csv'
    if freq_csv.exists():
        df = pd.read_csv(freq_csv)
        return dict(zip(df['class'], df['count']))

    # Fall back to per-class stats
    stats_csv = experiment_dir / 'memorization' / f'{model_name}_per_class_stats.csv'
    if stats_csv.exists():
        df = pd.read_csv(stats_csv)
        if 'class_name' in df.columns and 'count' in df.columns:
            return dict(zip(df['class_name'], df['count']))

    return {}


def load_test_results(experiment_dir: Path, model_name: str) -> pd.DataFrame:
    """
    Load test set results for non-member data.

    CRITICAL: MIA validation requires REAL test data, not synthetic/bootstrap data.
    Using bootstrap creates data leakage and artificially inflates attack success metrics.
    """
    test_csv = experiment_dir / 'memorization' / f'{model_name}_test_scores.csv'
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Test set results not found: {test_csv}\n\n"
            f"MIA validation requires REAL non-member (test set) data.\n"
            f"Using synthetic/bootstrap non-members is DATA LEAKAGE and invalidates results.\n\n"
            f"To fix:\n"
            f"1. Re-run memorization measurement with test set evaluation enabled\n"
            f"2. Ensure memorization.py generates {model_name}_test_scores.csv\n"
            f"3. Test samples must be samples the candidate model NEVER saw during training"
        )
    return pd.read_csv(test_csv)


def create_mia_from_real_data(canary_df: pd.DataFrame,
                               test_df: pd.DataFrame) -> MembershipInferenceAttack:
    """
    Create MIA instance using REAL member (canary) and non-member (test) data.

    CRITICAL: This is the scientifically valid approach.
    - Members: Canary samples (candidate model trained on these)
    - Non-members: Test samples (candidate model NEVER saw these)

    This avoids the data leakage problem of bootstrap/synthetic non-members.

    Args:
        canary_df: DataFrame with canary (member) memorization scores
        test_df: DataFrame with test (non-member) scores

    Returns:
        MembershipInferenceAttack instance
    """
    # Members: Canary samples that candidate model was trained on
    member_candidate_losses = canary_df['loss_candidate'].values
    member_independent_losses = canary_df['loss_independent'].values
    member_classes = canary_df['class_name'].values if 'class_name' in canary_df.columns else None
    n_members = len(canary_df)

    # Non-members: Test samples that candidate model NEVER saw
    non_member_candidate_losses = test_df['loss_candidate'].values
    non_member_independent_losses = test_df['loss_independent'].values
    non_member_classes = test_df['class_name'].values if 'class_name' in test_df.columns else None
    n_non_members = len(test_df)

    print(f"  MIA Setup (REAL DATA - No Bootstrap):")
    print(f"    Members (canary): {n_members} samples")
    print(f"    Non-members (test): {n_non_members} samples")

    # Combine members and non-members
    all_candidate_losses = np.concatenate([member_candidate_losses, non_member_candidate_losses])
    all_independent_losses = np.concatenate([member_independent_losses, non_member_independent_losses])
    all_labels = np.concatenate([np.ones(n_members), np.zeros(n_non_members)])

    # Class labels
    if member_classes is not None and non_member_classes is not None:
        all_classes = np.concatenate([member_classes, non_member_classes])
        unique_classes = list(sorted(set(all_classes)))
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        all_class_indices = np.array([class_to_idx[c] for c in all_classes])
    else:
        all_class_indices = None
        unique_classes = None

    return MembershipInferenceAttack(
        candidate_losses=all_candidate_losses,
        independent_losses=all_independent_losses,
        member_labels=all_labels,
        class_labels=all_class_indices,
        class_names=unique_classes
    )


def analyze_memorization_as_mia_signal(mem_df: pd.DataFrame,
                                        class_frequencies: dict) -> dict:
    """
    Direct analysis of memorization scores as MIA signal.

    This validates that M(x) = L_independent - L_candidate
    can distinguish members from non-members.

    Key insight: For canary samples (members), high M(x) means
    candidate model has much lower loss = strong memorization = privacy risk
    """
    results = {
        'overall': {},
        'per_class': [],
        'rarity_correlation': {}
    }

    # Overall statistics
    mem_scores = mem_df['memorization_score'].values
    risk_cats = mem_df['risk_category'].values

    # What fraction are high/moderate risk?
    results['overall']['total_samples'] = len(mem_df)
    results['overall']['high_risk_pct'] = float(np.mean(risk_cats == 'high') * 100)
    results['overall']['moderate_risk_pct'] = float(np.mean(risk_cats == 'moderate') * 100)
    results['overall']['low_risk_pct'] = float(np.mean(risk_cats == 'low') * 100)
    results['overall']['mean_memorization'] = float(np.mean(mem_scores))
    results['overall']['std_memorization'] = float(np.std(mem_scores))

    # MIA simulation: Can we distinguish high-memorization from low-memorization?
    # This simulates the attack accuracy if we could identify members by their M(x)
    threshold = 0  # M(x) > 0 means candidate model did better = likely member
    predicted_members = mem_scores > threshold

    # For this analysis, ALL samples are members (canary)
    # So accuracy = fraction correctly predicted as members
    results['overall']['mia_accuracy_at_zero_threshold'] = float(np.mean(predicted_members))

    # Per-class analysis
    if 'class_name' in mem_df.columns:
        for class_name in mem_df['class_name'].unique():
            class_mask = mem_df['class_name'] == class_name
            class_df = mem_df[class_mask]

            class_mem_scores = class_df['memorization_score'].values
            class_risks = class_df['risk_category'].values

            freq = class_frequencies.get(class_name, 0)

            class_result = {
                'class': class_name,
                'n_samples': int(len(class_df)),
                'frequency': int(freq),
                'mean_memorization': float(np.mean(class_mem_scores)),
                'std_memorization': float(np.std(class_mem_scores)),
                'high_risk_pct': float(np.mean(class_risks == 'high') * 100),
                'moderate_risk_pct': float(np.mean(class_risks == 'moderate') * 100),
                'mia_accuracy': float(np.mean(class_mem_scores > 0))
            }
            results['per_class'].append(class_result)

        # Sort by frequency (rarest first)
        results['per_class'] = sorted(results['per_class'], key=lambda x: x['frequency'])

        # Rarity correlation
        if len(results['per_class']) >= 3:
            freqs = [c['frequency'] for c in results['per_class']]
            mems = [c['mean_memorization'] for c in results['per_class']]
            risks = [c['high_risk_pct'] for c in results['per_class']]

            # Spearman correlation (frequency vs memorization)
            rho_mem, p_mem = stats.spearmanr(freqs, mems)
            rho_risk, p_risk = stats.spearmanr(freqs, risks)

            results['rarity_correlation']['freq_vs_memorization_rho'] = float(rho_mem)
            results['rarity_correlation']['freq_vs_memorization_p'] = float(p_mem)
            results['rarity_correlation']['freq_vs_highrisk_rho'] = float(rho_risk)
            results['rarity_correlation']['freq_vs_highrisk_p'] = float(p_risk)

            # Rare vs common comparison
            median_freq = np.median(freqs)
            rare_classes = [c for c in results['per_class'] if c['frequency'] < median_freq]
            common_classes = [c for c in results['per_class'] if c['frequency'] >= median_freq]

            if rare_classes and common_classes:
                rare_avg_mem = np.mean([c['mean_memorization'] for c in rare_classes])
                common_avg_mem = np.mean([c['mean_memorization'] for c in common_classes])
                rare_avg_risk = np.mean([c['high_risk_pct'] for c in rare_classes])
                common_avg_risk = np.mean([c['high_risk_pct'] for c in common_classes])

                results['rarity_correlation']['rare_mean_memorization'] = float(rare_avg_mem)
                results['rarity_correlation']['common_mean_memorization'] = float(common_avg_mem)
                results['rarity_correlation']['rare_high_risk_pct'] = float(rare_avg_risk)
                results['rarity_correlation']['common_high_risk_pct'] = float(common_avg_risk)
                results['rarity_correlation']['memorization_difference'] = float(rare_avg_mem - common_avg_mem)
                results['rarity_correlation']['risk_difference'] = float(rare_avg_risk - common_avg_risk)

    return results


def generate_mia_figures(results: dict, output_dir: Path, model_name: str, dataset: str):
    """Generate publication-quality figures for MIA analysis"""

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300
    })

    # Figure 1: Rarity vs MIA Vulnerability
    if 'per_class' in results and len(results['per_class']) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        per_class = results['per_class']
        classes = [c['class'] for c in per_class]
        freqs = [c['frequency'] for c in per_class]
        mems = [c['mean_memorization'] for c in per_class]
        risks = [c['high_risk_pct'] for c in per_class]

        # Scatter: Frequency vs Memorization
        ax1 = axes[0]
        scatter = ax1.scatter(freqs, mems, c=risks, cmap='RdYlGn_r', s=100, alpha=0.8, edgecolors='black')

        # Add class labels
        for i, cls in enumerate(classes):
            ax1.annotate(cls, (freqs[i], mems[i]), xytext=(5, 5),
                        textcoords='offset points', fontsize=9)

        ax1.set_xlabel('Class Frequency (Training Samples)')
        ax1.set_ylabel('Mean Memorization Score M(x)')
        ax1.set_title(f'{model_name.upper()} - {dataset}\nRarity vs Memorization')

        # Add trend line
        z = np.polyfit(freqs, mems, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(freqs), max(freqs), 100)
        ax1.plot(x_line, p(x_line), '--', color='red', alpha=0.7, label='Trend')

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('High Risk %')
        ax1.legend()

        # Bar chart: Rare vs Common
        ax2 = axes[1]
        if 'rarity_correlation' in results:
            rc = results['rarity_correlation']
            categories = ['Rare Classes', 'Common Classes']
            memorization = [rc.get('rare_mean_memorization', 0), rc.get('common_mean_memorization', 0)]
            risk_pcts = [rc.get('rare_high_risk_pct', 0), rc.get('common_high_risk_pct', 0)]

            x = np.arange(len(categories))
            width = 0.35

            bars1 = ax2.bar(x - width/2, memorization, width, label='Mean M(x)', color='steelblue')
            ax2.set_ylabel('Mean Memorization Score')
            ax2.set_title('Rare vs Common Classes')
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories)

            # Add second y-axis for risk
            ax2b = ax2.twinx()
            bars2 = ax2b.bar(x + width/2, risk_pcts, width, label='High Risk %', color='coral')
            ax2b.set_ylabel('High Risk Percentage')

            # Combined legend
            ax2.legend(loc='upper left')
            ax2b.legend(loc='upper right')

            # Add significance annotation
            if 'freq_vs_memorization_p' in rc and rc['freq_vs_memorization_p'] < 0.05:
                ax2.annotate(f"ρ = {rc['freq_vs_memorization_rho']:.3f}*",
                           xy=(0.5, 0.95), xycoords='axes fraction',
                           ha='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        fig.savefig(fig_dir / f'{model_name}_rarity_vulnerability.png', dpi=300, bbox_inches='tight')
        fig.savefig(fig_dir / f'{model_name}_rarity_vulnerability.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {model_name}_rarity_vulnerability.png/pdf")

    # Figure 2: MIA Attack Comparison (if full MIA was run)
    if 'mia_attacks' in results:
        fig, ax = plt.subplots(figsize=(10, 6))

        attacks = results['mia_attacks']
        attack_names = list(attacks.keys())
        aucs = [attacks[a]['auc'] for a in attack_names]
        accuracies = [attacks[a]['accuracy'] for a in attack_names]

        x = np.arange(len(attack_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, aucs, width, label='AUC', color='steelblue')
        bars2 = ax.bar(x + width/2, accuracies, width, label='Accuracy', color='coral')

        ax.axhline(y=0.5, color='gray', linestyle='--', label='Random Baseline')

        ax.set_ylabel('Score')
        ax.set_title(f'{model_name.upper()} - {dataset}\nMIA Attack Performance')
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace('_', '\n') for a in attack_names], rotation=0)
        ax.legend()
        ax.set_ylim(0.3, 1.0)

        plt.tight_layout()
        fig.savefig(fig_dir / f'{model_name}_mia_attacks.png', dpi=300, bbox_inches='tight')
        fig.savefig(fig_dir / f'{model_name}_mia_attacks.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {model_name}_mia_attacks.png/pdf")


def run_mia_analysis_for_model(experiment_dir: Path,
                                model_name: str,
                                run_full_mia: bool = True) -> dict:
    """Run complete MIA analysis for a single model"""

    print(f"\n{'='*60}")
    print(f"MIA ANALYSIS: {model_name.upper()}")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading memorization results...")
    mem_df = load_memorization_results(experiment_dir, model_name)
    class_frequencies = load_class_frequencies(experiment_dir, model_name)

    print(f"  Loaded {len(mem_df)} canary samples")
    print(f"  Class frequencies: {len(class_frequencies)} classes")

    results = {
        'model': model_name,
        'n_samples': len(mem_df),
        'timestamp': datetime.now().isoformat()
    }

    # 1. Direct memorization analysis
    print(f"\n--- Memorization Score Analysis ---")
    mem_analysis = analyze_memorization_as_mia_signal(mem_df, class_frequencies)
    results['memorization_analysis'] = mem_analysis

    # Print key findings
    print(f"\nOverall Statistics:")
    print(f"  Total samples: {mem_analysis['overall']['total_samples']}")
    print(f"  High risk: {mem_analysis['overall']['high_risk_pct']:.1f}%")
    print(f"  Moderate risk: {mem_analysis['overall']['moderate_risk_pct']:.1f}%")
    print(f"  Low risk: {mem_analysis['overall']['low_risk_pct']:.1f}%")
    print(f"  Mean M(x): {mem_analysis['overall']['mean_memorization']:.4f}")

    if 'rarity_correlation' in mem_analysis and mem_analysis['rarity_correlation']:
        rc = mem_analysis['rarity_correlation']
        print(f"\nRarity Correlation:")
        if 'freq_vs_memorization_rho' in rc:
            print(f"  Frequency vs M(x): ρ = {rc['freq_vs_memorization_rho']:.4f} (p = {rc['freq_vs_memorization_p']:.4f})")
        if 'rare_mean_memorization' in rc:
            print(f"  Rare classes mean M(x): {rc['rare_mean_memorization']:.4f}")
            print(f"  Common classes mean M(x): {rc['common_mean_memorization']:.4f}")
            print(f"  Difference: {rc['memorization_difference']:.4f}")
        if 'rare_high_risk_pct' in rc:
            print(f"  Rare classes high risk: {rc['rare_high_risk_pct']:.1f}%")
            print(f"  Common classes high risk: {rc['common_high_risk_pct']:.1f}%")

    # 2. Full MIA analysis (with REAL test set non-members)
    if run_full_mia:
        print(f"\n--- Full MIA Analysis (Real Test Set) ---")

        # CRITICAL: Load real test set data for non-members
        # This avoids the data leakage problem of bootstrap/synthetic non-members
        try:
            test_df = load_test_results(experiment_dir, model_name)
            mia = create_mia_from_real_data(mem_df, test_df)
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            print(f"\n⚠️  Skipping MIA analysis for {model_name} - test data required")
            return results

        attack_results = mia.run_all_attacks()

        # Store attack results
        results['mia_attacks'] = {}
        for name, result in attack_results.items():
            results['mia_attacks'][name] = {
                'accuracy': float(result.accuracy),
                'auc': float(result.auc),
                'precision': float(result.precision),
                'recall': float(result.recall),
                'f1': float(result.f1),
                'tpr_at_low_fpr': {k: float(v) for k, v in result.tpr_at_low_fpr.items()}
            }

        # Per-class calibrated attack
        if mia.class_labels is not None:
            try:
                _, per_class_df = mia.per_class_calibrated_attack()
                results['per_class_mia'] = per_class_df.to_dict('records')
            except Exception as e:
                print(f"  Warning: Per-class attack failed: {e}")

        # Rarity vulnerability analysis
        if mia.class_labels is not None and class_frequencies:
            mem_attack = attack_results['memorization_score']
            rarity_df = mia.analyze_rarity_vulnerability(class_frequencies, mem_attack)

            if len(rarity_df) > 0:
                # Rare vs common
                rare_mask = rarity_df['is_rare']
                if rare_mask.sum() > 0 and (~rare_mask).sum() > 0:
                    rare_auc = rarity_df[rare_mask]['mia_auc'].mean()
                    common_auc = rarity_df[~rare_mask]['mia_auc'].mean()

                    results['rarity_mia'] = {
                        'rare_auc': float(rare_auc),
                        'common_auc': float(common_auc),
                        'auc_difference': float(rare_auc - common_auc)
                    }

                    print(f"\n*** KEY FINDING ***")
                    print(f"  Rare classes MIA AUC: {rare_auc:.4f}")
                    print(f"  Common classes MIA AUC: {common_auc:.4f}")
                    print(f"  Difference: {rare_auc - common_auc:.4f}")

                    if rare_auc > common_auc:
                        print(f"  VALIDATES: Rare classes have higher MIA vulnerability!")

        # Memorization-MIA correlation
        correlation = mia.compute_memorization_mia_correlation()
        results['memorization_mia_correlation'] = {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in correlation.items()
        }

        print(f"\nMemorization-MIA Correlation:")
        print(f"  Correlation: {correlation['correlation']:.4f}")
        print(f"  P-value: {correlation['p_value']:.4f}")
        print(f"  High M(x) MIA accuracy: {correlation['high_memorization_mia_accuracy']:.4f}")
        print(f"  Low M(x) MIA accuracy: {correlation['low_memorization_mia_accuracy']:.4f}")

        if correlation['validates_hypothesis']:
            print(f"  VALIDATES: M(x) significantly predicts MIA vulnerability!")

    return results


def run_experiment_analysis(experiment_name: str,
                            results_dir: Path,
                            models: list = None,
                            run_full_mia: bool = True) -> dict:
    """Run MIA analysis for an experiment"""

    experiment_dir = results_dir / experiment_name
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_dir}")

    # Auto-detect models if not specified
    if models is None:
        mem_dir = experiment_dir / 'memorization'
        if mem_dir.exists():
            mem_files = list(mem_dir.glob('*_memorization_scores.csv'))
            models = [f.stem.replace('_memorization_scores', '') for f in mem_files]

    if not models:
        raise ValueError(f"No models found in {experiment_dir}")

    print(f"\n{'#'*70}")
    print(f"# MIA VALIDATION ANALYSIS")
    print(f"# Experiment: {experiment_name}")
    print(f"# Models: {', '.join(models)}")
    print(f"{'#'*70}")

    # Output directory
    output_dir = experiment_dir / 'mia_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'experiment': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }

    for model in models:
        try:
            model_results = run_mia_analysis_for_model(
                experiment_dir, model, run_full_mia=run_full_mia
            )
            all_results['models'][model] = model_results

            # Generate figures
            print(f"\nGenerating figures for {model}...")
            generate_mia_figures(
                model_results,
                output_dir,
                model,
                experiment_name.replace('_baseline', '')
            )

        except Exception as e:
            print(f"\nError processing {model}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cross-model comparison
    if len(all_results['models']) > 1:
        print(f"\n{'='*60}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*60}")

        comparison = []
        for model, results in all_results['models'].items():
            mem_analysis = results.get('memorization_analysis', {})
            overall = mem_analysis.get('overall', {})
            rarity_corr = mem_analysis.get('rarity_correlation', {})
            mia_attacks = results.get('mia_attacks', {})
            rarity_mia = results.get('rarity_mia', {})

            row = {
                'model': model,
                'high_risk_pct': overall.get('high_risk_pct', 0),
                'mean_memorization': overall.get('mean_memorization', 0),
                'freq_mem_rho': rarity_corr.get('freq_vs_memorization_rho', 0),
                'mem_score_auc': mia_attacks.get('memorization_score', {}).get('auc', 0),
                'rare_mia_auc': rarity_mia.get('rare_auc', 0),
                'common_mia_auc': rarity_mia.get('common_auc', 0),
                'auc_gap': rarity_mia.get('auc_difference', 0)
            }
            comparison.append(row)

        comparison_df = pd.DataFrame(comparison)
        print(comparison_df.to_string(index=False))

        comparison_df.to_csv(output_dir / 'cross_model_mia_comparison.csv', index=False)
        all_results['cross_model_comparison'] = comparison

    # Save full results
    results_file = output_dir / 'mia_validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("RESULTS SAVED")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Results JSON: {results_file}")

    return all_results


def generate_summary_report(all_experiments: dict, output_dir: Path):
    """Generate LaTeX summary table for publication"""

    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{MIA Vulnerability Analysis: Rare vs Common Classes}",
        r"\label{tab:mia_rarity}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Model & High Risk & Rare AUC & Common AUC & Gap \\",
        r"\midrule"
    ]

    for exp_name, exp_results in all_experiments.items():
        dataset = exp_name.replace('_baseline', '').upper()
        for model, model_results in exp_results.get('models', {}).items():
            mem = model_results.get('memorization_analysis', {}).get('overall', {})
            rarity = model_results.get('rarity_mia', {})

            high_risk = mem.get('high_risk_pct', 0)
            rare_auc = rarity.get('rare_auc', 0)
            common_auc = rarity.get('common_auc', 0)
            gap = rarity.get('auc_difference', 0)

            latex_lines.append(
                f"{dataset} & {model.upper()} & {high_risk:.1f}\\% & "
                f"{rare_auc:.3f} & {common_auc:.3f} & {gap:+.3f} \\\\"
            )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    latex_file = output_dir / 'mia_summary_table.tex'
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"\nLaTeX summary table saved: {latex_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run MIA validation analysis on memorization results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--experiment', '-e',
        type=str,
        help='Experiment name (e.g., ham1000_baseline)'
    )
    parser.add_argument(
        '--all-experiments',
        action='store_true',
        help='Run analysis on all experiments'
    )
    parser.add_argument(
        '--models', '-m',
        type=str,
        help='Comma-separated list of models (default: all)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=str(PROJECT_ROOT / 'results'),
        help='Results directory'
    )
    parser.add_argument(
        '--skip-full-mia',
        action='store_true',
        help='Skip full MIA attack analysis (faster)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.models:
        models = [m.strip() for m in args.models.split(',')]
    else:
        models = None

    run_full_mia = not args.skip_full_mia

    all_experiments = {}

    if args.all_experiments:
        # Find all experiment directories
        experiments = [d.name for d in results_dir.iterdir()
                      if d.is_dir() and (d / 'memorization').exists()]
        print(f"Found experiments: {experiments}")

        for exp in experiments:
            try:
                results = run_experiment_analysis(
                    exp, results_dir, models, run_full_mia
                )
                all_experiments[exp] = results
            except Exception as e:
                print(f"Error processing {exp}: {e}")
                continue

        # Generate combined report
        if all_experiments:
            combined_output = results_dir / 'mia_validation_combined'
            combined_output.mkdir(parents=True, exist_ok=True)

            with open(combined_output / 'all_experiments_mia.json', 'w') as f:
                json.dump(all_experiments, f, indent=2, default=str)

            generate_summary_report(all_experiments, combined_output)

    elif args.experiment:
        results = run_experiment_analysis(
            args.experiment, results_dir, models, run_full_mia
        )
        all_experiments[args.experiment] = results

    else:
        parser.print_help()
        print("\nError: Specify --experiment or --all-experiments")
        sys.exit(1)

    print("\n" + "="*70)
    print("MIA VALIDATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
