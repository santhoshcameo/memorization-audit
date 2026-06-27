#!/usr/bin/env python3
"""
Train Script
Train a single model or all models
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training import train_differential_model
from src.utils.config_loader import load_experiment_config
from src.utils.reproducibility import setup_reproducibility


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train medical memorization models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ResNet50
  python scripts/train.py --model resnet50 --config config/experiments/baseline.yaml
  
  # Train all models
  python scripts/train.py --model all --config config/experiments/baseline.yaml
  
  # Train with custom output directory
  python scripts/train.py --model vit --config baseline --output results/my_experiment
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=['mae', 'resnet50', 'vit', 'medsam', 'all'],
        help='Model to train (or "all" for all models)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to experiment config (YAML) or experiment name'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: results/<experiment_name>)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    parser.add_argument(
        '--gpu', '-g',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    print("="*80)
    print("MEDICAL MEMORIZATION STUDY - TRAINING")
    print("="*80)
    print()
    
    # Load configuration
    print(f"Loading configuration: {args.config}")
    try:
        if args.config.endswith('.yaml'):
            config = load_experiment_config(Path(args.config).stem)
        else:
            config = load_experiment_config(args.config)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)
    
    print(f"✅ Configuration loaded: {config.get('experiment_name', 'unknown')}")
    print()
    
    # Setup reproducibility
    seed = args.seed if args.seed is not None else config.get('training', {}).get('seed', 42)
    setup_reproducibility(seed=seed, deterministic=True)
    print()
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        exp_name = config.get('experiment_name', 'experiment')
        output_dir = Path('results') / exp_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Determine which models to train
    if args.model == 'all':
        models_to_train = config.get('models', ['mae', 'resnet50', 'vit', 'medsam'])
        if isinstance(models_to_train, str):
            models_to_train = [models_to_train]
    else:
        models_to_train = [args.model]
    
    print(f"Models to train: {models_to_train}")
    print()
    
    # Train each model
    for i, model_name in enumerate(models_to_train, 1):
        print("="*80)
        print(f"TRAINING MODEL {i}/{len(models_to_train)}: {model_name.upper()}")
        print("="*80)
        print()
        
        try:
            # Train differential models (candidate + independent)
            histories = train_differential_model(
                model_name=model_name,
                config=config,
                output_dir=output_dir
            )
            
            print()
            print(f"✅ {model_name.upper()} training complete!")
            print(f"   Candidate best acc: {max(histories['candidate']['train_acc']):.2f}%")
            print(f"   Independent best acc: {max(histories['independent']['train_acc']):.2f}%")
            print()
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print()
    print(f"Results saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Evaluate models: python scripts/evaluate.py")
    print("  2. Generate figures: python scripts/visualize.py")
    print()


if __name__ == "__main__":
    main()