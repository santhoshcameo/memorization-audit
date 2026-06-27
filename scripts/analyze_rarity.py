#!/usr/bin/env python3
"""
Analyze Rare vs Common Disease Memorization

This script performs rigorous statistical analysis to test whether
rare disease canaries are memorized significantly more than common disease canaries.

Usage:
    python scripts/analyze_rarity.py --experiment ham1000_baseline
    python scripts/analyze_rarity.py --experiment ham1000_baseline --models resnet50,vit
    python scripts/analyze_rarity.py --scores results/ham1000_baseline/memorization/resnet50_memorization_scores.csv
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.evaluation.rarity_analysis import (
    run_full_rarity_analysis,
    generate_statistical_summary,
    compute_class_frequencies,
    assign_rarity_tiers
)
from src.visualization.rarity_plots import generate_all_rarity_plots


def analyze_single_model(scores_path: str,
                         output_dir: str,
                         model_name: str,
                         rare_threshold: float = 0.05) -> dict:
    """
    Run rarity analysis for a single model.

    Args:
        scores_path: Path to memorization scores CSV
        output_dir: Directory to save results
        model_name: Name of the model
        rare_threshold: Threshold for defining rare classes

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")

    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run full statistical analysis
    results = run_full_rarity_analysis(
        scores_path=scores_path,
        output_dir=str(output_dir),
        rare_threshold=rare_threshold,
        class_column='class_name'
    )

    # Generate summary
    summary = generate_statistical_summary(results)
    print(summary)

    # Save summary
    with open(output_dir / 'statistical_summary.txt', 'w') as f:
        f.write(summary)

    # Load data for plotting
    scores_df = pd.read_csv(scores_path)
    freq_df = compute_class_frequencies(scores_df, 'class_name')
    freq_df = assign_rarity_tiers(freq_df)

    # Generate all plots
    generate_all_rarity_plots(
        results=results,
        scores_df=scores_df,
        freq_df=freq_df,
        output_dir=str(output_dir / 'plots'),
        class_column='class_name'
    )

    return results


def analyze_experiment(experiment_name: str,
                       models: list = None,
                       rare_threshold: float = 0.05) -> dict:
    """
    Run rarity analysis for all models in an experiment.

    Args:
        experiment_name: Name of the experiment (e.g., 'ham1000_baseline')
        models: List of model names to analyze (default: all available)
        rare_threshold: Threshold for defining rare classes

    Returns:
        Dictionary of results per model
    """
    results_dir = Path(f'results/{experiment_name}/memorization')
    output_dir = Path(f'results/{experiment_name}/rarity_analysis')

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find all memorization score files
    score_files = list(results_dir.glob('*_memorization_scores.csv'))

    if not score_files:
        raise FileNotFoundError(f"No memorization score files found in {results_dir}")

    # Filter to requested models
    if models:
        score_files = [f for f in score_files
                       if any(m in f.stem for m in models)]

    all_results = {}

    for score_file in score_files:
        model_name = score_file.stem.replace('_memorization_scores', '')

        try:
            results = analyze_single_model(
                scores_path=str(score_file),
                output_dir=str(output_dir),
                model_name=model_name,
                rare_threshold=rare_threshold
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            continue

    # Generate cross-model comparison
    if len(all_results) > 1:
        generate_cross_model_summary(all_results, output_dir)

    return all_results


def generate_cross_model_summary(all_results: dict, output_dir: Path) -> None:
    """
    Generate summary comparing rarity analysis across models.
    """
    output_dir = Path(output_dir)

    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-MODEL RARITY ANALYSIS SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Table header
    lines.append(f"{'Model':<15} {'Spearman rho':<15} {'p-value':<12} {'Cliff delta':<15} {'Significant':<12}")
    lines.append("-" * 80)

    for model, results in all_results.items():
        spearman = results['spearman_correlation']
        effects = results['effect_sizes']

        if 'error' in effects:
            delta = 'N/A'
        else:
            delta = f"{effects['cliffs_delta']:.4f}"

        sig = 'YES' if spearman['significant'] else 'NO'

        lines.append(f"{model:<15} {spearman['spearman_rho']:<15.4f} {spearman['p_value']:<12.2e} {delta:<15} {sig:<12}")

    lines.append("")
    lines.append("=" * 80)

    # Check consistency
    all_significant = all(r['spearman_correlation']['significant'] for r in all_results.values())
    all_negative_rho = all(r['spearman_correlation']['spearman_rho'] < 0 for r in all_results.values())

    if all_significant and all_negative_rho:
        lines.append("CONSISTENT FINDING: All models show significant negative correlation")
        lines.append("between class frequency and memorization (rare classes memorized more)")
    elif all_negative_rho:
        lines.append("TREND: All models show negative correlation, but not all significant")
    else:
        lines.append("MIXED RESULTS: Models show inconsistent patterns")

    lines.append("=" * 80)

    summary = "\n".join(lines)
    print(summary)

    with open(output_dir / 'cross_model_summary.txt', 'w') as f:
        f.write(summary)

    # Also save as CSV for easy comparison
    rows = []
    for model, results in all_results.items():
        spearman = results['spearman_correlation']
        mw = results['mann_whitney_test']
        effects = results['effect_sizes']
        bootstrap = results['bootstrap_ci']

        row = {
            'model': model,
            'spearman_rho': spearman['spearman_rho'],
            'spearman_p': spearman['p_value'],
            'spearman_significant': spearman['significant'],
            'mann_whitney_p': mw.get('p_value_one_sided', None),
            'mw_significant': mw.get('significant_one_sided', None),
            'rare_mean': mw.get('rare_mean', None),
            'common_mean': mw.get('common_mean', None),
            'cliffs_delta': effects.get('cliffs_delta', None),
            'cliffs_interp': effects.get('cliffs_delta_interpretation', None),
            'cohens_d': effects.get('cohens_d', None),
            'bootstrap_diff': bootstrap.get('observed_difference', None) if bootstrap else None,
            'bootstrap_ci_lower': bootstrap.get('ci_lower', None) if bootstrap else None,
            'bootstrap_ci_upper': bootstrap.get('ci_upper', None) if bootstrap else None,
            'bootstrap_significant': bootstrap.get('significant', None) if bootstrap else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'cross_model_comparison.csv', index=False)
    print(f"\nSaved cross-model comparison to {output_dir / 'cross_model_comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze rare vs common disease memorization'
    )
    parser.add_argument('--experiment', type=str,
                        help='Experiment name (e.g., ham1000_baseline)')
    parser.add_argument('--scores', type=str,
                        help='Direct path to memorization scores CSV')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated list of models to analyze')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: results/<experiment>/rarity_analysis)')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Rare class threshold (default: 0.05 = 5%%)')

    args = parser.parse_args()

    if not args.experiment and not args.scores:
        parser.error("Either --experiment or --scores must be provided")

    models = args.models.split(',') if args.models else None

    if args.scores:
        # Single file analysis
        output_dir = args.output or 'results/rarity_analysis'
        model_name = Path(args.scores).stem.replace('_memorization_scores', '')
        analyze_single_model(
            scores_path=args.scores,
            output_dir=output_dir,
            model_name=model_name,
            rare_threshold=args.threshold
        )
    else:
        # Experiment analysis
        analyze_experiment(
            experiment_name=args.experiment,
            models=models,
            rare_threshold=args.threshold
        )


if __name__ == "__main__":
    main()
