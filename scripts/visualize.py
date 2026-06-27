#!/usr/bin/env python3
"""
Visualize Script
Generate all figures for paper
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visualization import (
    create_all_comparison_plots,
    create_all_distribution_plots
)
from src.evaluation import compare_all_models


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate all visualization figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all figures from baseline experiment
  python scripts/visualize.py --experiment baseline
  
  # Generate figures with custom input/output
  python scripts/visualize.py \\
      --input results/baseline/memorization \\
      --output paper/figures
  
  # Specify DPI
  python scripts/visualize.py --experiment baseline --dpi 600
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        help='Experiment name (e.g., baseline)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Input directory with memorization results'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for figures'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure DPI (default: 300)'
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=None,
        help='Models to include (default: all available)'
    )
    
    return parser.parse_args()


def load_results(input_dir: Path, models: list = None) -> dict:
    """
    Load memorization results
    
    Args:
        input_dir: Directory with CSV files
        models: List of model names (None = all available)
    
    Returns:
        Dictionary mapping model names to DataFrames
    """
    input_dir = Path(input_dir)
    results = {}
    
    # Find all memorization CSV files
    csv_files = list(input_dir.glob('*_memorization_scores.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No memorization results found in {input_dir}")
    
    for csv_file in csv_files:
        # Extract model name
        model_name = csv_file.stem.replace('_memorization_scores', '')
        
        # Skip if not in requested models
        if models and model_name not in models:
            continue
        
        # Load data
        try:
            df = pd.read_csv(csv_file)
            results[model_name] = df
            print(f"✅ Loaded: {model_name} ({len(df)} samples)")
        except Exception as e:
            print(f"⚠️ Error loading {csv_file}: {e}")
    
    if not results:
        raise ValueError("No results loaded!")
    
    return results


def main():
    """Main visualization function"""
    args = parse_args()
    
    print("="*80)
    print("MEDICAL MEMORIZATION STUDY - VISUALIZATION")
    print("="*80)
    print()
    
    # Determine input directory
    if args.input:
        input_dir = Path(args.input)
    elif args.experiment:
        input_dir = Path('results') / args.experiment / 'memorization'
    else:
        print("❌ Error: Must specify --experiment or --input")
        sys.exit(1)
    
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    print(f"Input directory: {input_dir}")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    elif args.experiment:
        output_dir = Path('results') / args.experiment / 'figures'
    else:
        output_dir = input_dir.parent / 'figures'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Load results
    print("Loading memorization results...")
    try:
        results = load_results(input_dir, models=args.models)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        sys.exit(1)
    
    print(f"\nLoaded {len(results)} models: {list(results.keys())}")
    print()
    
    # Generate statistical comparisons
    if len(results) > 1:
        print("Computing statistical comparisons...")
        try:
            comparisons = compare_all_models(results)
            print(f"✅ Generated {len(comparisons)} pairwise comparisons")
        except Exception as e:
            print(f"⚠️ Error in comparisons: {e}")
            comparisons = None
    else:
        comparisons = None
        print("⚠️ Only one model loaded, skipping comparisons")
    
    print()
    
    # Generate comparison plots
    if comparisons is not None:
        print("="*80)
        print("GENERATING COMPARISON PLOTS")
        print("="*80)
        print()
        
        try:
            create_all_comparison_plots(
                results_dict=results,
                comparisons_df=comparisons,
                output_dir=output_dir
            )
            print()
        except Exception as e:
            print(f"❌ Error creating comparison plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate distribution plots
    print("="*80)
    print("GENERATING DISTRIBUTION PLOTS")
    print("="*80)
    print()
    
    try:
        create_all_distribution_plots(
            results_dict=results,
            output_dir=output_dir
        )
        print()
    except Exception as e:
        print(f"❌ Error creating distribution plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print()
    print(f"Figures saved to: {output_dir}")
    print()
    print("Generated figures:")
    figures = sorted(output_dir.glob('*.png'))
    for fig in figures:
        size_mb = fig.stat().st_size / (1024 * 1024)
        print(f"  - {fig.name} ({size_mb:.1f} MB)")
    print()
    print(f"Total: {len(figures)} figures")
    print()


if __name__ == "__main__":
    main()