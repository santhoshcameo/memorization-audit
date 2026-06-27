#!/usr/bin/env python3
"""
Evaluate Script
Measure memorization for trained models
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import measure_memorization, compare_all_models, generate_statistical_report
from src.utils.config_loader import load_experiment_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate memorization for trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model
  python scripts/evaluate.py --model resnet50 --experiment baseline
  
  # Evaluate all models
  python scripts/evaluate.py --model all --experiment baseline
  
  # Custom checkpoint paths
  python scripts/evaluate.py --model resnet50 \\
      --candidate results/resnet50/candidate/best.pth \\
      --independent results/resnet50/independent/best.pth
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='all',
        help='Model to evaluate (or "all" for all models)'
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        required=True,
        help='Experiment name (e.g., baseline)'
    )
    
    parser.add_argument(
        '--candidate',
        type=str,
        default=None,
        help='Path to candidate model checkpoint'
    )
    
    parser.add_argument(
        '--independent',
        type=str,
        default=None,
        help='Path to independent model checkpoint'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: results/<experiment>/memorization)'
    )
    
    parser.add_argument(
        '--num-augmentations', '-n',
        type=int,
        default=5,
        help='Number of augmentations per sample (default: 5)'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()
    
    print("="*80)
    print("MEDICAL MEMORIZATION STUDY - EVALUATION")
    print("="*80)
    print()
    
    # Load configuration
    print(f"Loading experiment: {args.experiment}")
    try:
        config = load_experiment_config(args.experiment)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)
    
    print(f"✅ Configuration loaded")
    print()
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path('results') / args.experiment / 'memorization'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Determine which models to evaluate
    if args.model == 'all':
        models_to_eval = config.get('models', ['mae', 'resnet50', 'vit', 'medsam'])
        if isinstance(models_to_eval, str):
            models_to_eval = [models_to_eval]
    else:
        models_to_eval = [args.model]
    
    print(f"Models to evaluate: {models_to_eval}")
    print()
    
    # Store results for comparison
    all_results = {}
    all_stats = {}
    
    # Evaluate each model
    for i, model_name in enumerate(models_to_eval, 1):
        print("="*80)
        print(f"EVALUATING MODEL {i}/{len(models_to_eval)}: {model_name.upper()}")
        print("="*80)
        print()
        
        try:
            # Determine checkpoint paths
            if args.candidate and args.independent:
                candidate_ckpt = Path(args.candidate)
                independent_ckpt = Path(args.independent)
            else:
                base_dir = Path('results') / args.experiment / model_name
                candidate_ckpt = base_dir / 'candidate' / 'checkpoints' / f'{model_name}_candidate_best.pth'
                independent_ckpt = base_dir / 'independent' / 'checkpoints' / f'{model_name}_independent_best.pth'
            
            # Check if checkpoints exist
            if not candidate_ckpt.exists():
                print(f"⚠️ Candidate checkpoint not found: {candidate_ckpt}")
                print(f"   Skipping {model_name}")
                continue
            
            if not independent_ckpt.exists():
                print(f"⚠️ Independent checkpoint not found: {independent_ckpt}")
                print(f"   Skipping {model_name}")
                continue
            
            # Measure memorization
            results_df, summary_stats = measure_memorization(
                candidate_checkpoint=candidate_ckpt,
                independent_checkpoint=independent_ckpt,
                config=config,
                output_dir=output_dir,
                num_augmentations=args.num_augmentations
            )
            
            all_results[model_name] = results_df
            all_stats[model_name] = summary_stats
            
            print()
            print(f"✅ {model_name.upper()} evaluation complete!")
            print()
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comparison if multiple models evaluated
    if len(all_results) > 1:
        print("="*80)
        print("STATISTICAL COMPARISON")
        print("="*80)
        print()
        
        try:
            # Pairwise comparisons
            comparisons = compare_all_models(all_results)
            comparisons_path = output_dir / 'pairwise_comparisons.csv'
            comparisons.to_csv(comparisons_path, index=False)
            print(f"✅ Pairwise comparisons saved: {comparisons_path}")
            
            # Statistical report
            report = generate_statistical_report(
                all_results,
                output_path=str(output_dir / 'statistical_report.txt')
            )
            print(report)
            
        except Exception as e:
            print(f"⚠️ Error in statistical comparison: {e}")
    
    print()
    print("="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print()
    print(f"Results saved to: {output_dir}")
    print()
    print("Summary:")
    for model_name, stats in all_stats.items():
        print(f"  {model_name.upper():10s}: Mean={stats['mean']:7.4f}, "
              f"High-risk={stats['high_risk_percentage']:5.1f}%")
    print()
    print("Next step:")
    print("  Generate figures: python scripts/visualize.py")
    print()


if __name__ == "__main__":
    main()