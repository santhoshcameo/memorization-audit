#!/usr/bin/env python3
"""
Run Retinal OCT Distinctive Feature Experiment

Runs all 3 distinctiveness levels sequentially and compares across
4 conditions (baseline + 3 levels).

Target classes: DRUSEN and DME (already show higher memorization in baseline)

Levels:
  1. Grayscale (retinal_oct_distinctive)
  2. Color Inversion (retinal_oct_distinctive_invert)
  3. Edge Detection (retinal_oct_distinctive_edge)

Usage:
    # Full pipeline (train + analyze all levels)
    python scripts/run_retinal_oct_distinctive.py

    # Analysis only (results already exist)
    python scripts/run_retinal_oct_distinctive.py --analysis-only

    # Quick test
    python scripts/run_retinal_oct_distinctive.py --epochs 2
"""

import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

LEVELS = [
    {
        'name': 'Baseline (Original)',
        'experiment': 'retinal_oct_baseline',
        'result_dir': 'retinal_oct_baseline',
        'transform': 'none',
        'level': 0,
    },
    {
        'name': 'Level 1: Grayscale',
        'experiment': 'retinal_oct_distinctive',
        'result_dir': 'retinal_oct_distinctive',
        'transform': 'grayscale',
        'level': 1,
    },
    {
        'name': 'Level 2: Color Inversion',
        'experiment': 'retinal_oct_distinctive_invert',
        'result_dir': 'retinal_oct_distinctive_invert',
        'transform': 'invert',
        'level': 2,
    },
    {
        'name': 'Level 3: Edge Detection',
        'experiment': 'retinal_oct_distinctive_edge',
        'result_dir': 'retinal_oct_distinctive_edge',
        'transform': 'edge',
        'level': 3,
    },
]


def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with code {result.returncode}")
    return result.returncode


def has_results(level):
    rarity_path = (PROJECT_ROOT / 'results' / level['result_dir'] /
                   'rarity_analysis' / 'resnet50' / 'rarity_analysis_results.json')
    return rarity_path.exists()


def run_level(level, args):
    """Train and analyze a single level."""
    if has_results(level) and not args.force:
        print(f"\n[SKIP] {level['name']} - results already exist")
        return 0

    # Training
    cmd = [
        sys.executable, 'run_experiment.py',
        '--experiment', level['experiment'],
        '--models', 'resnet50',
    ]
    if args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])

    rc = run_command(cmd, f"Training {level['name']}")
    if rc != 0:
        return rc

    # Rarity analysis
    cmd = [
        sys.executable, 'scripts/analyze_rarity.py',
        '--experiment', level['experiment'],
        '--models', 'resnet50',
        '--threshold', '0.20',
    ]
    return run_command(cmd, f"Rarity analysis for {level['name']}")


def run_comparison():
    """Compare all 4 conditions and print dose-response table."""
    print(f"\n{'='*70}")
    print("RETINAL OCT MULTI-LEVEL DISTINCTIVE FEATURE COMPARISON")
    print(f"{'='*70}\n")

    results = {}
    for level in LEVELS:
        rarity_path = (PROJECT_ROOT / 'results' / level['result_dir'] /
                       'rarity_analysis' / 'resnet50' / 'rarity_analysis_results.json')
        if rarity_path.exists():
            with open(rarity_path) as f:
                results[level['result_dir']] = json.load(f)
        else:
            print(f"[WARNING] Missing results for {level['name']}: {rarity_path}")

    if len(results) < 2:
        print("[ERROR] Need at least 2 conditions to compare")
        return 1

    # Print dose-response table
    print(f"\n{'Condition':<30s} {'Rare M(x)':>10s} {'Common M(x)':>12s} {'Cliff d':>10s} {'p-value':>10s} {'Significant':>12s}")
    print("-" * 90)

    summary = []
    for level in LEVELS:
        data = results.get(level['result_dir'])
        if not data:
            print(f"{level['name']:<30s} {'N/A':>10s}")
            continue

        mw = data.get('mann_whitney_test', {})
        es = data.get('effect_sizes', {})

        rare_mean = mw.get('rare_mean', 0)
        common_mean = mw.get('common_mean', 0)
        cliffs_d = es.get('cliffs_delta', 0)
        p_val = mw.get('p_value_one_sided', mw.get('p_value_two_sided', 1.0))
        sig = mw.get('significant_one_sided', mw.get('significant_two_sided', False))

        print(f"{level['name']:<30s} {rare_mean:>10.3f} {common_mean:>12.3f} {cliffs_d:>10.3f} {p_val:>10.4f} {'YES ***' if sig else 'NO':>12s}")

        summary.append({
            'level': level['level'],
            'name': level['name'],
            'transform': level['transform'],
            'rare_mean': rare_mean,
            'common_mean': common_mean,
            'cliffs_delta': cliffs_d,
            'p_value': p_val,
            'significant': sig,
        })

    # Save summary
    output_path = PROJECT_ROOT / 'results' / 'retinal_oct_multilevel_distinctive_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {output_path}")

    # Dose-response interpretation
    print(f"\n{'='*70}")
    print("DOSE-RESPONSE INTERPRETATION")
    print(f"{'='*70}")

    sig_levels = [s for s in summary if s['significant'] and s.get('level', 0) > 0]
    if len(sig_levels) == 0:
        print("No distinctive level reached significance.")
    elif len(sig_levels) == len(summary) - 1:  # all except baseline
        print("ALL distinctiveness levels show significant effect!")
    else:
        sig_names = [s['name'] for s in sig_levels]
        print(f"Significant at: {', '.join(sig_names)}")
        if any(s['level'] == 3 for s in sig_levels) and not any(s['level'] == 1 for s in sig_levels):
            print("Dose-response confirmed: stronger distinctiveness -> stronger effect")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Run Retinal OCT Distinctive Feature Experiment')
    parser.add_argument('--analysis-only', action='store_true',
                        help='Only run comparison analysis')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if results exist')
    args = parser.parse_args()

    print("=" * 60)
    print("RETINAL OCT DISTINCTIVE FEATURE EXPERIMENT")
    print("=" * 60)
    print("\nTarget classes: DRUSEN, DME")
    print("Hypothesis: distinctive transforms amplify memorization")
    print("\nLevels:")
    for level in LEVELS:
        status = "EXISTS" if has_results(level) else "PENDING"
        print(f"  {level['name']:<30s} [{status}]")

    if not args.analysis_only:
        # Run levels 1-3 (skip baseline, it should already exist)
        for level in LEVELS:
            if level['level'] == 0:
                if not has_results(level):
                    print(f"\n[WARNING] Baseline results missing. Run baseline first.")
                continue
            rc = run_level(level, args)
            if rc != 0:
                print(f"\n[ERROR] {level['name']} failed. Continuing...")

    rc = run_comparison()

    print(f"\n{'='*60}")
    print("RETINAL OCT DISTINCTIVE EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print("\nTo generate publication figures:")
    print("  python scripts/visualize_retinal_oct_distinctive.py")

    return rc


if __name__ == '__main__':
    sys.exit(main())
