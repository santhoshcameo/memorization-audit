#!/usr/bin/env python3
"""
Unified Pipeline Runner — single entry point for ALL experiments.

Usage:
    # Run a single experiment
    python run_pipeline.py --experiment ham1000_baseline
    python run_pipeline.py --experiment chestxray_baseline --models resnet50,vit --epochs 10

    # Run ALL baseline experiments (5 datasets × 4 models)
    python run_pipeline.py --all

    # Run only experiments with missing/failed results
    python run_pipeline.py --pending

    # Quick 1-epoch test (validates pipeline works)
    python run_pipeline.py --test
    python run_pipeline.py --test --experiment chestxray_baseline   # test single

    # Include MIA + rarity analysis after training
    python run_pipeline.py --pending --with-mia --with-rarity

    # Skip phases
    python run_pipeline.py --experiment ham1000_baseline --skip-training
    python run_pipeline.py --all --skip-visualization

    # Production run (30 epochs, all analysis)
    nohup .venv/bin/python run_pipeline.py --pending --with-mia --with-rarity > pipeline.log 2>&1 &
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent

# All baseline experiments and their result directories
ALL_BASELINES = {
    'ham1000_baseline':     'results/ham1000_baseline',
    'kvasir_baseline':      'results/kvasir_baseline',
    'chestxray_baseline':   'results/chestxray_baseline',
    'odir5k_baseline':      'results/odir5k_baseline',
    'retinal_oct_baseline': 'results/retinal_oct_baseline',
}

# Distinctive experiments (single-model ablations)
DISTINCTIVE_EXPERIMENTS = {
    'retinal_oct_distinctive':        'results/retinal_oct_distinctive',
    'retinal_oct_distinctive_edge':   'results/retinal_oct_distinctive_edge',
    'retinal_oct_distinctive_invert': 'results/retinal_oct_distinctive_invert',
}


def get_python():
    """Get the right Python executable."""
    venv_python = PROJECT_ROOT / '.venv' / 'bin' / 'python'
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def is_experiment_complete(result_dir: str) -> bool:
    """Check if an experiment has valid (non-empty) results."""
    eval_file = Path(result_dir) / 'evaluation_summary.json'
    if not eval_file.exists():
        return False
    try:
        with open(eval_file) as f:
            data = json.load(f)
        return bool(data)  # {} is falsy
    except (json.JSONDecodeError, IOError):
        return False


def get_experiment_status(experiments: dict) -> tuple:
    """Return (complete, pending) experiment lists."""
    complete = []
    pending = []
    for exp, result_dir in experiments.items():
        if is_experiment_complete(result_dir):
            complete.append(exp)
        else:
            pending.append(exp)
    return complete, pending


def run_single_experiment(experiment: str, args, test_mode: bool = False) -> int:
    """Run a single experiment using run_experiment.py."""
    python = get_python()
    cmd = [python, 'run_experiment.py', '--experiment', experiment]

    if args.models:
        cmd.extend(['--models', args.models])
    if args.epochs is not None:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.skip_training:
        cmd.append('--skip-training')
    if args.skip_evaluation:
        cmd.append('--skip-evaluation')
    if args.skip_visualization:
        cmd.append('--skip-visualization')
    if args.batch_size:
        cmd.extend(['--batch-size', str(args.batch_size)])
    if args.num_workers is not None:
        cmd.extend(['--num-workers', str(args.num_workers)])
    if args.output:
        cmd.extend(['--output', args.output])
    elif test_mode:
        # Test mode: write to temp dir so real results are never overwritten
        cmd.extend(['--output', f'results/_test_1ep_{experiment}'])
    if args.seed is not None:
        cmd.extend(['--seed', str(args.seed)])

    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    # Clean up test output
    if test_mode and not args.output:
        import shutil
        test_dir = PROJECT_ROOT / f'results/_test_1ep_{experiment}'
        if test_dir.exists():
            shutil.rmtree(test_dir)

    return result.returncode


def run_mia(experiment: str) -> int:
    """Run MIA analysis for an experiment."""
    python = get_python()
    cmd = [python, 'scripts/run_mia_analysis.py', '--experiment', experiment]
    print(f"  MIA: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def run_rarity(experiment: str) -> int:
    """Run rarity analysis for an experiment."""
    python = get_python()
    cmd = [python, 'scripts/analyze_rarity.py', '--experiment', experiment]
    print(f"  RARITY: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def print_status():
    """Print current status of all experiments."""
    print("\n" + "=" * 70)
    print("EXPERIMENT STATUS")
    print("=" * 70)

    all_exps = {**ALL_BASELINES, **DISTINCTIVE_EXPERIMENTS}
    for exp, result_dir in all_exps.items():
        status = "COMPLETE" if is_experiment_complete(result_dir) else "PENDING"
        marker = "+" if status == "COMPLETE" else "-"

        # Get summary info if complete
        detail = ""
        if status == "COMPLETE":
            try:
                with open(Path(result_dir) / 'evaluation_summary.json') as f:
                    data = json.load(f)
                models = list(data.keys())
                detail = f"  [{len(models)} models: {', '.join(models)}]"
            except Exception:
                pass

        print(f"  [{marker}] {exp:40s} {status}{detail}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Unified pipeline runner for all memorization experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment
  %(prog)s --experiment ham1000_baseline
  %(prog)s --experiment chestxray_baseline --models resnet50,vit --epochs 10

  # Batch modes
  %(prog)s --all                         # Run ALL 5 baseline experiments
  %(prog)s --pending                     # Run only failed/missing experiments
  %(prog)s --test                        # 1-epoch pipeline validation

  # With post-training analysis
  %(prog)s --pending --with-mia --with-rarity

  # Production (background)
  nohup .venv/bin/python %(prog)s --pending --with-mia --with-rarity > pipeline.log 2>&1 &

  # Status check (no experiments run)
  %(prog)s --status
        """
    )

    # Mode selection (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--experiment', '-e', type=str,
                      help='Run single experiment by config name')
    mode.add_argument('--all', action='store_true',
                      help='Run ALL baseline experiments')
    mode.add_argument('--pending', action='store_true',
                      help='Run only experiments with missing/failed results')
    mode.add_argument('--test', action='store_true',
                      help='Quick 1-epoch test of pending experiments')
    mode.add_argument('--status', action='store_true',
                      help='Show experiment status (no experiments run)')

    # Training options
    parser.add_argument('--models', '-m', type=str, default=None,
                        help='Comma-separated model names (default: all from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epoch count (default: from config, usually 30)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override data loader workers')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Override output directory (single experiment only)')

    # Phase skipping
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training (use existing checkpoints)')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip memorization evaluation')
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip figure generation')

    # Post-training analysis
    parser.add_argument('--with-mia', action='store_true',
                        help='Run MIA validation after training')
    parser.add_argument('--with-rarity', action='store_true',
                        help='Run rarity analysis after training')

    args = parser.parse_args()

    # Default: show help
    if not any([args.experiment, args.all, args.pending, args.test, args.status]):
        parser.print_help()
        print_status()
        return 0

    # Status mode
    if args.status:
        print_status()
        return 0

    # Test mode: override epochs to 1
    if args.test:
        args.epochs = 1
        args.skip_visualization = True

    # Determine which experiments to run
    if args.experiment:
        experiments = [args.experiment]
    elif args.all:
        experiments = list(ALL_BASELINES.keys())
    elif args.pending or args.test:
        complete, pending = get_experiment_status(ALL_BASELINES)
        experiments = pending
        if not experiments:
            print("\nAll experiments are complete. Nothing to run.")
            print_status()
            return 0
        print(f"\nSkipping (complete): {', '.join(complete) if complete else 'none'}")
        print(f"Will run (pending):  {', '.join(experiments)}")
    else:
        experiments = []

    # Run experiments
    start_time = datetime.now()
    total = len(experiments)
    succeeded = 0
    failed_list = []

    print("\n" + "=" * 70)
    print(f"PIPELINE: {total} experiment(s) to run")
    print(f"Epochs: {args.epochs if args.epochs else 'config default (30)'}")
    print(f"Models: {args.models if args.models else 'all (from config)'}")
    print(f"MIA: {'yes' if args.with_mia else 'no'}")
    print(f"Rarity: {'yes' if args.with_rarity else 'no'}")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for i, exp in enumerate(experiments, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{total}] {exp}")
        print(f"{'=' * 70}")
        print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

        rc = run_single_experiment(exp, args, test_mode=args.test)

        if rc == 0:
            succeeded += 1
            print(f"[OK] {exp}")

            # Post-training analysis (only if training succeeded)
            result_dir = ALL_BASELINES.get(exp, DISTINCTIVE_EXPERIMENTS.get(exp))
            if result_dir and Path(result_dir).exists():
                if args.with_mia:
                    mem_dir = Path(result_dir) / 'memorization'
                    if mem_dir.exists() and list(mem_dir.glob('*_memorization_scores.csv')):
                        mia_rc = run_mia(exp)
                        if mia_rc != 0:
                            print(f"  [WARN] MIA failed for {exp}")
                    else:
                        print(f"  [SKIP] MIA — no memorization scores for {exp}")

                if args.with_rarity:
                    mem_dir = Path(result_dir) / 'memorization'
                    if mem_dir.exists() and list(mem_dir.glob('*_memorization_scores.csv')):
                        rar_rc = run_rarity(exp)
                        if rar_rc != 0:
                            print(f"  [WARN] Rarity analysis failed for {exp}")
                    else:
                        print(f"  [SKIP] Rarity — no memorization scores for {exp}")
        else:
            failed_list.append(exp)
            print(f"[FAIL] {exp} (exit code {rc})")

    # Summary
    elapsed = datetime.now() - start_time
    hours = int(elapsed.total_seconds() // 3600)
    minutes = int((elapsed.total_seconds() % 3600) // 60)

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"Duration:  {hours}h {minutes}m")
    print(f"Succeeded: {succeeded}/{total}")
    print(f"Failed:    {len(failed_list)}/{total}")
    if failed_list:
        print(f"  Failed experiments: {', '.join(failed_list)}")
        print(f"\n  To rerun failed: .venv/bin/python run_pipeline.py --pending --with-mia --with-rarity")
    print(f"{'=' * 70}")

    print_status()

    return 1 if failed_list else 0


if __name__ == '__main__':
    sys.exit(main())
