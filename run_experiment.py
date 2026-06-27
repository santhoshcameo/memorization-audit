#!/usr/bin/env python3
"""
Master Experiment Runner
Single entry point for all memorization experiments

Usage:
    python run_experiment.py --experiment baseline
    python run_experiment.py --experiment ablation_epochs --models resnet50,vit
    python run_experiment.py --dataset ham10000 --models all --output results/custom
"""

import argparse
import sys
from pathlib import Path
import yaml
import json
from datetime import datetime
import random
import numpy as np
import torch

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class ExperimentRunner:
    """Orchestrates complete memorization experiment pipeline"""
    
    def __init__(self, experiment_config: dict, args: argparse.Namespace):
        self.config = experiment_config
        self.args = args
        self.start_time = datetime.now()
        
        # Setup paths
        self.output_dir = Path(args.output) if args.output else Path('results') / self.config['name']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self._save_config()
        
        # Setup logging
        self._setup_logging()
        
    def _save_config(self):
        """Save experiment configuration for reproducibility"""
        config_file = self.output_dir / 'experiment_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        # Save command line args
        args_file = self.output_dir / 'args.json'
        with open(args_file, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
    
    def _setup_logging(self):
        """Setup experiment logging"""
        import logging
        
        log_file = self.output_dir / 'experiment.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_reproducibility(self):
        """
        Setup reproducibility for deterministic experiments.

        CRITICAL: This must be called BEFORE any model creation or training.
        Ensures identical results across runs with the same seed.
        """
        seed = self.config.get('training', {}).get('seed', 42)

        self.logger.info("="*80)
        self.logger.info("REPRODUCIBILITY SETUP")
        self.logger.info("="*80)

        # Set Python random seed
        random.seed(seed)
        self.logger.info(f"  ✓ Python random seed: {seed}")

        # Set NumPy random seed
        np.random.seed(seed)
        self.logger.info(f"  ✓ NumPy random seed: {seed}")

        # Set PyTorch random seed
        torch.manual_seed(seed)
        self.logger.info(f"  ✓ PyTorch CPU seed: {seed}")

        # Set CUDA random seed if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.logger.info(f"  ✓ PyTorch CUDA seed: {seed}")

        # CRITICAL: Enable deterministic mode
        # This ensures identical results but may be slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.logger.info(f"  ✓ CuDNN deterministic: True")
        self.logger.info(f"  ✓ CuDNN benchmark: False (disabled for reproducibility)")

        # Set environment variable for additional determinism
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
        self.logger.info(f"  ✓ PYTHONHASHSEED: {seed}")

        self.logger.info("")
        self.logger.info(f"🔒 Reproducibility configured with seed={seed}")
        self.logger.info("   All random operations will produce identical results")
        self.logger.info("="*80)

    def validate_setup(self):
        """Validate experiment setup before running"""
        self.logger.info("="*80)
        self.logger.info("EXPERIMENT VALIDATION")
        self.logger.info("="*80)
        
        # Check dataset
        dataset_name = self.config['dataset']
        dataset_config_path = PROJECT_ROOT / 'config' / 'datasets' / f'{dataset_name}.yaml'
        
        if not dataset_config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
        
        with open(dataset_config_path) as f:
            dataset_config = yaml.safe_load(f)
        
        # Check data exists (resolve relative paths)
        from src.data import resolve_path
        data_root = resolve_path(dataset_config['paths']['images'])
        metadata_path = resolve_path(dataset_config['paths']['metadata'])
        
        if not data_root.exists():
            raise FileNotFoundError(f"Dataset images not found: {data_root}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        # Validate metadata has required columns
        import pandas as pd
        metadata = pd.read_csv(metadata_path)
        required_cols = ['image_id', 'dx', 'split']
        
        missing_cols = [col for col in required_cols if col not in metadata.columns]
        if missing_cols:
            raise ValueError(
                f"Metadata missing required columns: {missing_cols}\n"
                f"Required: {required_cols}\n"
                f"Found: {list(metadata.columns)}\n"
                f"Run: python scripts/preprocess_data.py --dataset {dataset_name}"
            )
        
        # Validate splits exist
        splits = metadata['split'].unique()
        required_splits = ['train', 'canary', 'test']
        missing_splits = [s for s in required_splits if s not in splits]
        if missing_splits:
            raise ValueError(f"Metadata missing required splits: {missing_splits}")
        
        # Check models
        models = self.config['models']
        for model_name in models:
            model_config_path = PROJECT_ROOT / 'config' / 'models' / f'{model_name}.yaml'
            if not model_config_path.exists():
                raise FileNotFoundError(f"Model config not found: {model_config_path}")
        
        # Check CUDA
        if self.config.get('training', {}).get('device') == 'cuda':
            if not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available! Falling back to CPU")
                self.config['training']['device'] = 'cpu'
        
        self.logger.info("✅ Validation passed")
        self.logger.info(f"  Dataset: {dataset_name}")
        self.logger.info(f"  Models: {models}")
        self.logger.info(f"  Device: {self.config['training']['device']}")
        self.logger.info(f"  Output: {self.output_dir}")
        self.logger.info("")
        
    def run_training_phase(self):
        """Phase 1: Train candidate and independent models"""
        if self.args.skip_training:
            self.logger.info("Skipping training phase")
            return
        
        self.logger.info("="*80)
        self.logger.info("PHASE 1: DIFFERENTIAL TRAINING")
        self.logger.info("="*80)
        
        from src.training.differential_trainer import train_differential_model
        
        models = self.config['models']
        results = {}
        
        for i, model_name in enumerate(models, 1):
            self.logger.info(f"\n[{i}/{len(models)}] Training {model_name.upper()}")
            self.logger.info("-"*80)

            # Clear GPU cache before each model to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.info(f"GPU memory cleared before {model_name}")

            try:
                history = train_differential_model(
                    model_name=model_name,
                    config=self._build_model_config(model_name),
                    output_dir=self.output_dir / model_name
                )

                results[model_name] = {
                    'candidate_acc': max(history['candidate']['train_acc']),
                    'independent_acc': max(history['independent']['train_acc'])
                }

                self.logger.info(f"✅ {model_name} complete")
                self.logger.info(f"   Candidate: {results[model_name]['candidate_acc']:.2f}%")
                self.logger.info(f"   Independent: {results[model_name]['independent_acc']:.2f}%")

            except Exception as e:
                self.logger.error(f"❌ Training failed for {model_name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # Clear GPU cache after failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if self.args.strict:
                    raise
                continue

            # Clear GPU cache after each model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save training summary
        summary_file = self.output_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Track failed models for exit code
        failed = len(models) - len(results)
        if failed > 0:
            self.logger.warning(f"⚠️ {failed}/{len(models)} models failed training")
            self._training_failures = failed

        return results
    
    def run_evaluation_phase(self):
        """Phase 2: Measure memorization"""
        if self.args.skip_evaluation:
            self.logger.info("Skipping evaluation phase")
            return
        
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 2: MEMORIZATION MEASUREMENT")
        self.logger.info("="*80)
        
        from src.evaluation.memorization import measure_memorization
        
        models = self.config['models']
        mem_dir = self.output_dir / 'memorization'
        mem_dir.mkdir(exist_ok=True)
        
        results = {}
        
        for i, model_name in enumerate(models, 1):
            self.logger.info(f"\n[{i}/{len(models)}] Evaluating {model_name.upper()}")
            self.logger.info("-"*80)
            
            # Find checkpoints - try multiple naming patterns for flexibility
            candidate_dir = self.output_dir / model_name / 'candidate' / 'checkpoints'
            independent_dir = self.output_dir / model_name / 'independent' / 'checkpoints'

            # Checkpoint naming patterns to try (in order of preference)
            checkpoint_patterns = [
                f'{model_name}_candidate_best.pth',  # Standard pattern
                'best_model.pth',                     # Alternative pattern
                'best.pth',                           # Short pattern
                f'{model_name}_best.pth',             # Model-prefixed
            ]

            candidate_ckpt = None
            independent_ckpt = None

            # Find candidate checkpoint
            for pattern in checkpoint_patterns:
                ckpt_path = candidate_dir / pattern
                if ckpt_path.exists():
                    candidate_ckpt = ckpt_path
                    break

            # Find independent checkpoint (use same pattern index if possible)
            for pattern in checkpoint_patterns:
                # Adjust pattern for independent model
                ind_pattern = pattern.replace('candidate', 'independent')
                ckpt_path = independent_dir / ind_pattern
                if ckpt_path.exists():
                    independent_ckpt = ckpt_path
                    break

            if candidate_ckpt is None or independent_ckpt is None:
                self.logger.warning(f"⚠️ Checkpoints not found for {model_name}")
                self.logger.warning(f"   Searched in: {candidate_dir}")
                self.logger.warning(f"   Patterns tried: {checkpoint_patterns}")
                self.logger.warning(f"   Candidate found: {candidate_ckpt}")
                self.logger.warning(f"   Independent found: {independent_ckpt}")
                continue
            
            try:
                results_df, stats = measure_memorization(
                    candidate_checkpoint=candidate_ckpt,
                    independent_checkpoint=independent_ckpt,
                    config=self._build_model_config(model_name),
                    output_dir=mem_dir,
                    num_augmentations=self.config.get('evaluation', {}).get('num_augmentations', 5)
                )
                
                # Convert numpy types to Python types for JSON serialization
                results[model_name] = {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'high_risk_pct': float(stats['high_risk_percentage'])
                }
                
                self.logger.info(f"✅ {model_name} evaluated")
                self.logger.info(f"   Mean: {stats['mean']:.4f}")
                self.logger.info(f"   High-risk: {stats['high_risk_percentage']:.1f}%")
                
            except Exception as e:
                self.logger.error(f"❌ Evaluation failed for {model_name}: {e}")
                if self.args.strict:
                    raise
                continue
        
        # Statistical comparison
        if len(results) > 1:
            self._run_statistical_analysis(mem_dir)
        
        # Save evaluation summary
        summary_file = self.output_dir / 'evaluation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_visualization_phase(self):
        """Phase 3: Generate figures"""
        if self.args.skip_visualization:
            self.logger.info("Skipping visualization phase")
            return
        
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 3: VISUALIZATION")
        self.logger.info("="*80)
        
        import pandas as pd
        from src.visualization import create_all_comparison_plots, create_all_distribution_plots
        from src.evaluation import compare_all_models
        
        mem_dir = self.output_dir / 'memorization'
        fig_dir = self.output_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)
        
        # Load results
        results = {}
        for csv_file in mem_dir.glob('*_memorization_scores.csv'):
            model_name = csv_file.stem.replace('_memorization_scores', '')
            results[model_name] = pd.read_csv(csv_file)
        
        if not results:
            self.logger.warning("No memorization results found")
            return
        
        self.logger.info(f"Generating figures for {len(results)} models...")
        
        try:
            # Comparison plots
            if len(results) > 1:
                comparisons = compare_all_models(results)
                create_all_comparison_plots(results, comparisons, fig_dir)
            
            # Distribution plots
            create_all_distribution_plots(results, fig_dir)
            
            # Count figures
            figures = list(fig_dir.glob('*.png'))
            self.logger.info(f"✅ Generated {len(figures)} figures")
            
        except Exception as e:
            self.logger.error(f"❌ Visualization failed: {e}")
            if self.args.strict:
                raise
    
    def _run_statistical_analysis(self, mem_dir: Path):
        """Run statistical tests and generate report"""
        import pandas as pd
        from src.evaluation import compare_all_models, generate_statistical_report
        
        # Load all results
        results = {}
        for csv_file in mem_dir.glob('*_memorization_scores.csv'):
            model_name = csv_file.stem.replace('_memorization_scores', '')
            results[model_name] = pd.read_csv(csv_file)
        
        if len(results) < 2:
            return
        
        self.logger.info("\nRunning statistical analysis...")
        
        # Pairwise comparisons
        comparisons = compare_all_models(results)
        comp_file = mem_dir / 'pairwise_comparisons.csv'
        comparisons.to_csv(comp_file, index=False)
        self.logger.info(f"  Saved: {comp_file.name}")
        
        # Statistical report
        report = generate_statistical_report(
            results,
            output_path=str(mem_dir / 'statistical_report.txt')
        )
        self.logger.info(f"  Saved: statistical_report.txt")
    
    def _build_model_config(self, model_name: str) -> dict:
        """Build complete config for a model"""
        # Load base config
        config = self.config.copy()
        
        # Load dataset config
        dataset_name = config['dataset']
        dataset_config_path = PROJECT_ROOT / 'config' / 'datasets' / f'{dataset_name}.yaml'
        with open(dataset_config_path) as f:
            config['dataset_config'] = yaml.safe_load(f)
        
        # Load model config
        model_config_path = PROJECT_ROOT / 'config' / 'models' / f'{model_name}.yaml'
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)
        
        config['model_configs'] = {model_name: model_config}
        config['model'] = model_name
        
        return config
    
    def generate_report(self):
        """Generate final experiment report"""
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT COMPLETE")
        self.logger.info("="*80)
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        self.logger.info(f"\nExperiment: {self.config['name']}")
        self.logger.info(f"Duration: {hours}h {minutes}m")
        self.logger.info(f"Output: {self.output_dir}")
        
        # Check what was generated
        checkpoints = list(self.output_dir.glob('*/*/checkpoints/*.pth'))
        mem_scores = list((self.output_dir / 'memorization').glob('*_memorization_scores.csv'))
        figures = list((self.output_dir / 'figures').glob('*.png'))
        
        self.logger.info(f"\nGenerated:")
        self.logger.info(f"  ✓ {len(checkpoints)} model checkpoints")
        self.logger.info(f"  ✓ {len(mem_scores)} memorization score files")
        self.logger.info(f"  ✓ {len(figures)} figures")
        
        self.logger.info("\n" + "="*80)
    
    def run(self):
        """Run complete experiment pipeline"""
        self._training_failures = 0

        try:
            # CRITICAL: Setup reproducibility FIRST
            # This ensures deterministic behavior across all operations
            self._setup_reproducibility()

            # Validate
            self.validate_setup()

            # Run phases
            self.run_training_phase()
            self.run_evaluation_phase()
            self.run_visualization_phase()

            # Report
            self.generate_report()

            # Return non-zero if ALL models failed (partial success = 0)
            total_models = len(self.config['models'])
            if self._training_failures >= total_models and not self.args.skip_training:
                self.logger.error(f"❌ ALL {total_models} models failed training")
                return 1

            return 0

        except Exception as e:
            self.logger.error(f"\n❌ EXPERIMENT FAILED: {e}")
            import traceback
            traceback.print_exc()
            return 1


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run memorization experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline experiment (all models)
  python run_experiment.py --experiment baseline
  
  # Run with specific models only
  python run_experiment.py --experiment baseline --models resnet50,vit
  
  # Skip training (use existing checkpoints)
  python run_experiment.py --experiment baseline --skip-training
  
  # Custom output directory
  python run_experiment.py --experiment baseline --output results/my_run
  
  # Strict mode (fail on any error)
  python run_experiment.py --experiment baseline --strict
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        required=True,
        help='Experiment config name (e.g., baseline, ablation_epochs)'
    )
    
    parser.add_argument(
        '--models', '-m',
        type=str,
        default=None,
        help='Comma-separated model names (default: all in config)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: results/<experiment>)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training phase'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation phase'
    )
    
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization phase'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail on any error (default: continue)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config, set by GPU profile)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of data loader workers (overrides config)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config, for quick testing)'
    )

    return parser.parse_args()


def load_experiment_config(experiment_name: str) -> dict:
    """Load experiment configuration"""
    config_path = PROJECT_ROOT / 'config' / 'experiments' / f'{experiment_name}.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Add experiment name
    config['name'] = experiment_name
    
    return config


def main():
    """Main entry point"""
    args = parse_args()
    
    # Load config
    config = load_experiment_config(args.experiment)
    
    # Override models if specified - strip whitespace from each model name
    if args.models:
        config['models'] = [m.strip() for m in args.models.split(',')]
    
    # Override seed if specified
    if args.seed is not None:
        config.setdefault('training', {})['seed'] = args.seed

    # Override batch_size if specified (from GPU profile)
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
        print(f"Using batch size: {args.batch_size}")

    # Override num_workers if specified (from GPU profile)
    if args.num_workers is not None:
        config.setdefault('training', {})['num_workers'] = args.num_workers
        print(f"Using num_workers: {args.num_workers}")

    # Override epochs if specified (for test mode)
    if args.epochs is not None:
        config.setdefault('training', {})['epochs'] = args.epochs
        print(f"Using epochs: {args.epochs} (TEST MODE)")

    # Create runner
    runner = ExperimentRunner(config, args)
    
    # Run experiment
    exit_code = runner.run()
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()