#!/usr/bin/env python3
"""
Layer-Wise Memorization Analysis — Orchestrator Script

Loads trained model checkpoints, runs layer-wise feature distance analysis,
and generates publication-quality figures showing WHERE memorization concentrates
in each architecture.

Usage:
    # Single model
    python scripts/run_layerwise_analysis.py --experiment ham1000_baseline --models resnet50

    # All four models
    python scripts/run_layerwise_analysis.py --experiment ham1000_baseline --models resnet50,vit,mae,medsam

    # Distinctive experiment
    python scripts/run_layerwise_analysis.py --experiment ham1000_distinctive --models resnet50,vit

    # Quick test (limit samples, skip gradients)
    python scripts/run_layerwise_analysis.py --experiment ham1000_baseline --models resnet50 --max-samples 100 --skip-gradient
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import importlib
import traceback

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.layerwise_analysis import LayerMemorizationAnalyzer
from src.visualization.layerwise_plots import (
    generate_all_layerwise_plots,
    plot_cross_model_layer_comparison,
)
from src.models.base_model import ModelFactory

# Dataset name -> (module_path, factory_function_name)
DATASET_LOADERS = {
    'ham10000': ('src.data.ham10000', 'create_ham10000_dataloader'),
    'chestxray': ('src.data.chestxray', 'create_chestxray_dataloader'),
    'odir5k': ('src.data.odir5k', 'create_odir5k_dataloader'),
    'retinal_oct': ('src.data.retinal_oct', 'create_retinal_oct_dataloader'),
    'kvasir_capsule': ('src.data.kvasir_capsule', 'create_kvasir_capsule_dataloader'),
}


def load_configs(experiment_name: str):
    """Load experiment and dataset configs. Returns (exp_config, dataset_config, dataset_name)."""
    exp_path = PROJECT_ROOT / 'config' / 'experiments' / f'{experiment_name}.yaml'
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")
    with open(exp_path) as f:
        exp_config = yaml.safe_load(f)

    dataset_name = exp_config['dataset']
    ds_path = PROJECT_ROOT / 'config' / 'datasets' / f'{dataset_name}.yaml'
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {ds_path}")
    with open(ds_path) as f:
        dataset_config = yaml.safe_load(f)

    return exp_config, dataset_config, dataset_name


def build_full_config(exp_config: dict, dataset_config: dict, model_name: str) -> dict:
    """Build the merged config dict expected by dataloaders and model factory."""
    config = exp_config.copy()
    config['dataset_config'] = dataset_config

    model_config_path = PROJECT_ROOT / 'config' / 'models' / f'{model_name}.yaml'
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    config['model_configs'] = {model_name: model_config}
    config['model'] = model_name
    return config


def create_canary_dataloader(dataset_name: str, config: dict, batch_size: int = 64):
    """Create a canary-split dataloader with eval transforms (no augmentation)."""
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Known: {list(DATASET_LOADERS)}")

    module_path, func_name = DATASET_LOADERS[dataset_name]
    module = importlib.import_module(module_path)
    create_fn = getattr(module, func_name)

    # split='canary' with no mode -> eval transforms, no augmentation
    return create_fn(
        config,
        split='canary',
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=False,
    )


def find_checkpoint(results_dir: Path, model_name: str, role: str) -> Path:
    """Find best checkpoint with flexible naming patterns."""
    base = results_dir / model_name / role / 'checkpoints'
    patterns = [
        f'{model_name}_{role}_best.pth',
        'best_model.pth',
        'best.pth',
        f'{model_name}_best.pth',
    ]
    for pat in patterns:
        p = base / pat
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No checkpoint for {model_name}/{role} in {base}. Tried: {patterns}"
    )


def reshape_summary_for_profile_plot(layer_distances_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape raw layer_distances into the long-format summary expected by
    plot_layer_distance_profile: [layer_name, group, mean_distance, std_distance].
    """
    rows = []
    for layer_name in layer_distances_df['layer_name'].unique():
        ldf = layer_distances_df[layer_distances_df['layer_name'] == layer_name]
        for mem_group, plot_group in [
            ('high_mem', 'high_memorization'),
            ('low_mem', 'low_memorization'),
        ]:
            gdata = ldf[ldf['mem_group'] == mem_group]['feature_distance']
            if len(gdata) > 0:
                rows.append({
                    'layer_name': layer_name,
                    'group': plot_group,
                    'mean_distance': float(gdata.mean()),
                    'std_distance': float(gdata.std()),
                })
    return pd.DataFrame(rows)


def aggregate_gradients_for_plot(
    gradient_df: pd.DataFrame,
    memorization_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate raw per-sample gradient data into the format expected by
    plot_gradient_magnitude_comparison:
    [layer_name, group, mean_gradient_norm, std_gradient_norm].
    """
    mem_lookup = memorization_scores_df.groupby('image_id')['memorization_score'].mean()
    gdf = gradient_df.copy()
    gdf['memorization_score'] = gdf['image_id'].map(mem_lookup)
    gdf = gdf.dropna(subset=['memorization_score'])
    if len(gdf) == 0:
        return pd.DataFrame()

    median_score = gdf['memorization_score'].median()
    gdf['group'] = np.where(
        gdf['memorization_score'] >= median_score,
        'high_memorization', 'low_memorization',
    )

    agg = (
        gdf.groupby(['layer_name', 'group'])['grad_norm']
        .agg(mean_gradient_norm='mean', std_gradient_norm='std')
        .reset_index()
    )
    return agg


def run_single_model(
    model_name: str,
    exp_config: dict,
    dataset_config: dict,
    dataset_name: str,
    results_dir: Path,
    output_dir: Path,
    device: str,
    max_samples: int | None,
    skip_gradient: bool,
):
    """Run layer-wise analysis for one model. Returns summary dict or None on failure."""
    print(f"\n{'='*80}")
    print(f"  {model_name.upper()} — Layer-Wise Memorization Analysis")
    print(f"{'='*80}")

    # Build merged config
    config = build_full_config(exp_config, dataset_config, model_name)
    num_classes = (
        dataset_config.get('num_classes')
        or dataset_config.get('info', {}).get('num_classes')
        or len(dataset_config.get('classes', {}).get('names', []))
        or 7
    )

    # Load model-specific config for factory
    model_config_path = PROJECT_ROOT / 'config' / 'models' / f'{model_name}.yaml'
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Find checkpoints
    print("Loading checkpoints...")
    candidate_path = find_checkpoint(results_dir, model_name, 'candidate')
    independent_path = find_checkpoint(results_dir, model_name, 'independent')
    print(f"  Candidate:   {candidate_path}")
    print(f"  Independent: {independent_path}")

    # Create and load models
    candidate_model = ModelFactory.create(model_name, model_config, num_classes=num_classes)
    candidate_model.load_checkpoint(candidate_path)
    candidate_model.to(device).eval()

    independent_model = ModelFactory.create(model_name, model_config, num_classes=num_classes)
    independent_model.load_checkpoint(independent_path)
    independent_model.to(device).eval()

    # Create canary dataloader
    print("Creating canary dataloader...")
    dataloader = create_canary_dataloader(dataset_name, config, batch_size=64)
    print(f"  Samples: {len(dataloader.dataset)}")

    # Load memorization scores
    mem_csv = results_dir / 'memorization' / f'{model_name}_memorization_scores.csv'
    if not mem_csv.exists():
        print(f"  WARNING: Memorization scores not found: {mem_csv}")
        return None
    memorization_scores_df = pd.read_csv(mem_csv)
    print(f"  Memorization scores: {len(memorization_scores_df)} samples")

    # Create analyzer
    analyzer = LayerMemorizationAnalyzer(
        candidate_model, independent_model,
        device=device, model_name=model_name,
    )

    # Run feature distance analysis
    print("\nRunning layer-wise feature distance analysis...")
    results = analyzer.analyze(dataloader, memorization_scores_df, max_samples=max_samples)

    if len(results.get('layer_distances', pd.DataFrame())) == 0:
        print("  WARNING: No layer distances computed (sample mismatch?)")
        return None

    # Linear probing (KEY analysis — measures information content per layer)
    print("\nRunning linear probing analysis...")
    probing_df = analyzer.compute_linear_probing(
        dataloader, memorization_scores_df, max_samples=max_samples,
    )
    if len(probing_df) > 0:
        results['probing'] = probing_df

    # Optional gradient analysis
    gradient_df = None
    if not skip_gradient:
        print("\nRunning gradient magnitude analysis...")
        gradient_raw = analyzer.compute_gradient_magnitudes(
            dataloader, max_samples=max_samples,
        )
        if len(gradient_raw) > 0:
            gradient_df = aggregate_gradients_for_plot(gradient_raw, memorization_scores_df)

    # Save raw results (CSVs + JSON)
    analyzer.save_results(results, output_dir)

    # Reshape summary for profile plot
    layer_summary_long = reshape_summary_for_profile_plot(results['layer_distances'])

    # Build the dict expected by generate_all_layerwise_plots
    plot_dict = {
        'layer_distances_df': results['layer_distances'],
        'memorization_scores_df': memorization_scores_df,
        'layer_summary_df': layer_summary_long,
        'correlation_df': results['correlation'],
    }
    if len(probing_df) > 0:
        plot_dict['probing_df'] = probing_df
    if gradient_df is not None and len(gradient_df) > 0:
        plot_dict['gradient_df'] = gradient_df

    # Generate figures
    figures_dir = output_dir / 'figures'
    saved_files = generate_all_layerwise_plots(
        plot_dict, str(figures_dir), model_name=model_name,
    )

    # Print summary statistics
    print(f"\n  === RESULTS ===")

    # Probing results (the important one)
    if len(probing_df) > 0 and 'probe_auroc' in probing_df.columns:
        print(f"  PROBING (memorization decodability):")
        for src in ['diff', 'concat', 'candidate_only']:
            src_df = probing_df[probing_df['feature_source'] == src]
            if len(src_df) == 0:
                continue
            src_df_auroc = src_df.dropna(subset=['probe_auroc'])
            if len(src_df_auroc) == 0:
                continue
            peak_lin = src_df_auroc.loc[src_df_auroc['probe_auroc'].idxmax()]
            lin_str = (f"Linear={peak_lin['probe_auroc']:.3f} "
                       f"at {peak_lin['layer_name']}")
            mlp_str = ''
            if 'mlp_auroc' in src_df.columns:
                src_mlp = src_df.dropna(subset=['mlp_auroc'])
                if len(src_mlp) > 0:
                    peak_mlp = src_mlp.loc[src_mlp['mlp_auroc'].idxmax()]
                    mlp_str = (f", MLP={peak_mlp['mlp_auroc']:.3f} "
                               f"at {peak_mlp['layer_name']}")
            print(f"    [{src:15s}] {lin_str}{mlp_str}")

    corr = results.get('correlation')
    if corr is not None and len(corr) > 0:
        peak = corr.loc[corr['spearman_rho'].abs().idxmax()]
        print(f"  Cosine distance correlation:")
        print(f"    Peak rho={peak['spearman_rho']:.3f}, p={peak['p_value']:.2e} "
              f"at layer={peak['layer_name']}")

    print(f"  Figures saved: {len(saved_files)}")

    # Free GPU memory
    del candidate_model, independent_model, analyzer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'layer_summary_df': layer_summary_long,
        'correlation': results['correlation'],
        'per_class_correlation': results.get('per_class_correlation'),
        'probing': probing_df,
        'layer_summary_raw': results['layer_summary'],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run layer-wise memorization analysis on trained checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_layerwise_analysis.py --experiment ham1000_baseline --models resnet50
  python scripts/run_layerwise_analysis.py --experiment ham1000_baseline --models resnet50,vit,mae,medsam
  python scripts/run_layerwise_analysis.py --experiment ham1000_distinctive --models resnet50,vit
  python scripts/run_layerwise_analysis.py --experiment ham1000_baseline --models resnet50 --max-samples 100 --skip-gradient
        """,
    )
    parser.add_argument('--experiment', '-e', required=True,
                        help='Experiment name (e.g., ham1000_baseline)')
    parser.add_argument('--models', '-m', default='resnet50,vit,mae,medsam',
                        help='Comma-separated model list (default: resnet50,vit,mae,medsam)')
    parser.add_argument('--device', default=None,
                        help='Device: cuda or cpu (default: auto-detect)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of samples for speed (default: all)')
    parser.add_argument('--skip-gradient', action='store_true',
                        help='Skip gradient magnitude analysis (faster)')
    parser.add_argument('--output', default=None,
                        help='Override output directory')
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = [m.strip() for m in args.models.split(',')]

    # Load configs
    exp_config, dataset_config, dataset_name = load_configs(args.experiment)

    # Determine results directory (where checkpoints and memorization scores live)
    results_dir = Path(
        exp_config.get('output', {}).get('base_dir', f'results/{args.experiment}')
    )
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    # Output directory for layer-wise analysis
    output_dir = Path(args.output) if args.output else results_dir / 'layerwise_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Layer-Wise Memorization Analysis")
    print(f"  Experiment: {args.experiment}")
    print(f"  Dataset:    {dataset_name}")
    print(f"  Models:     {models}")
    print(f"  Device:     {args.device}")
    print(f"  Results:    {results_dir}")
    print(f"  Output:     {output_dir}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")
    if args.skip_gradient:
        print(f"  Gradient analysis: SKIPPED")

    # Process each model
    all_summaries = {}
    start_time = datetime.now()

    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Processing {model_name}...")
        try:
            result = run_single_model(
                model_name=model_name,
                exp_config=exp_config,
                dataset_config=dataset_config,
                dataset_name=dataset_name,
                results_dir=results_dir,
                output_dir=output_dir,
                device=args.device,
                max_samples=args.max_samples,
                skip_gradient=args.skip_gradient,
            )
            if result is not None:
                all_summaries[model_name] = result['layer_summary_df']
        except Exception as e:
            print(f"  ERROR: {model_name} failed: {e}")
            traceback.print_exc()
            continue

    # Cross-model comparison plot
    if len(all_summaries) > 1:
        print(f"\nGenerating cross-model comparison plot...")
        comparison_path = output_dir / 'figures' / 'cross_model_layer_comparison.png'
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        plot_cross_model_layer_comparison(
            all_summaries, output_path=str(comparison_path),
        )

    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n{'='*80}")
    print(f"  COMPLETE — {len(all_summaries)}/{len(models)} models analyzed")
    print(f"  Time: {minutes}m {seconds}s")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
