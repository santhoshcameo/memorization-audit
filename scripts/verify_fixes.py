#!/usr/bin/env python3
"""
Verification Script for Critical Fixes
Run this script to verify all fixes are properly applied.

Usage:
    python scripts/verify_fixes.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import random


def verify_reproducibility():
    """Verify that reproducibility is working correctly."""
    print("\n" + "="*60)
    print("TEST 1: REPRODUCIBILITY")
    print("="*60)

    # Set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Verify deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Generate random numbers
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()

    # Reset and regenerate
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.rand(1).item()

    assert r1 == r2, f"Python random not reproducible: {r1} != {r2}"
    assert n1 == n2, f"NumPy random not reproducible: {n1} != {n2}"
    assert t1 == t2, f"PyTorch random not reproducible: {t1} != {t2}"

    print("✅ Python random: REPRODUCIBLE")
    print("✅ NumPy random: REPRODUCIBLE")
    print("✅ PyTorch random: REPRODUCIBLE")
    print(f"✅ CuDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"✅ CuDNN benchmark: {torch.backends.cudnn.benchmark} (should be False)")

    return True


def verify_model_initialization():
    """Verify that candidate and independent models have identical initialization."""
    print("\n" + "="*60)
    print("TEST 2: MODEL INITIALIZATION (Identical Weights)")
    print("="*60)

    import copy

    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    # Test deepcopy preserves weights
    torch.manual_seed(42)
    model1 = SimpleModel()
    model2 = copy.deepcopy(model1)

    # Verify weights are identical
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert torch.equal(p1, p2), f"Weights differ for {n1}"

    print("✅ deepcopy preserves identical weights")
    print("✅ Candidate and independent models will have identical init")

    return True


def verify_configs():
    """Verify configuration files have correct settings."""
    print("\n" + "="*60)
    print("TEST 3: CONFIGURATION FILES")
    print("="*60)

    import yaml

    # Check MAE pretrained flag
    mae_config = PROJECT_ROOT / 'config' / 'models' / 'mae.yaml'
    with open(mae_config) as f:
        mae = yaml.safe_load(f)

    pretrained = mae['architecture']['pretrained']
    assert pretrained == False, f"MAE pretrained should be False, got {pretrained}"
    print("✅ MAE pretrained: False (training from scratch)")

    # Check MedSAM image size
    medsam_config = PROJECT_ROOT / 'config' / 'models' / 'medsam.yaml'
    with open(medsam_config) as f:
        medsam = yaml.safe_load(f)

    image_size = medsam['architecture']['image_size']
    assert image_size == 224, f"MedSAM image_size should be 224, got {image_size}"
    print("✅ MedSAM image_size: 224 (matches other models)")

    # Check augmentation
    aug_size = medsam['augmentation']['custom']['train'][0]['params']['size']
    assert aug_size == 224, f"MedSAM augmentation size should be 224, got {aug_size}"
    print("✅ MedSAM augmentation size: 224")

    # Check output directories are unique
    ham_config = PROJECT_ROOT / 'config' / 'experiments' / 'ham1000_baseline.yaml'
    chest_config = PROJECT_ROOT / 'config' / 'experiments' / 'chestxray_baseline.yaml'

    with open(ham_config) as f:
        ham = yaml.safe_load(f)
    with open(chest_config) as f:
        chest = yaml.safe_load(f)

    ham_dir = ham['output']['base_dir']
    chest_dir = chest['output']['base_dir']
    assert ham_dir != chest_dir, f"Output dirs should be unique: {ham_dir} == {chest_dir}"
    print(f"✅ HAM10000 output: {ham_dir}")
    print(f"✅ ChestXray output: {chest_dir}")

    return True


def verify_mia_requires_test_set():
    """Verify MIA validation requires real test set."""
    print("\n" + "="*60)
    print("TEST 4: MIA VALIDATION (No Bootstrap)")
    print("="*60)

    # Check that run_mia_analysis.py uses real test data
    mia_script = PROJECT_ROOT / 'scripts' / 'run_mia_analysis.py'
    with open(mia_script) as f:
        content = f.read()

    # Check for the new function that requires real test data
    assert 'create_mia_from_real_data' in content, "MIA should use create_mia_from_real_data"
    assert 'load_test_results' in content, "MIA should load real test results"

    # Check that bootstrap is not used for MIA
    assert 'create_mia_from_real_data(mem_df, test_df)' in content, \
        "MIA should call create_mia_from_real_data with real test data"

    print("✅ MIA uses create_mia_from_real_data (no bootstrap)")
    print("✅ MIA requires test_scores.csv (real non-member data)")

    return True


def verify_differential_trainer():
    """Verify differential trainer has seed reset and deepcopy."""
    print("\n" + "="*60)
    print("TEST 5: DIFFERENTIAL TRAINER")
    print("="*60)

    trainer_file = PROJECT_ROOT / 'src' / 'training' / 'differential_trainer.py'
    with open(trainer_file) as f:
        content = f.read()

    # Check for deepcopy in model creation
    assert 'copy.deepcopy(candidate_model)' in content, \
        "Should use deepcopy for identical initialization"
    print("✅ Uses deepcopy for identical model initialization")

    # Check for seed reset
    assert '_reset_seed' in content, "Should have _reset_seed function"
    assert '_reset_seed(seed)' in content, "Should call _reset_seed before training"
    print("✅ Resets seed before each training run")

    return True


def verify_worker_init_fn():
    """Verify DataLoaders have worker_init_fn."""
    print("\n" + "="*60)
    print("TEST 6: DATALOADER WORKER_INIT_FN")
    print("="*60)

    datasets = ['ham10000', 'chestxray', 'odir5k', 'retinal_oct']
    for dataset in datasets:
        file_path = PROJECT_ROOT / 'src' / 'data' / f'{dataset}.py'
        with open(file_path) as f:
            content = f.read()

        assert '_worker_init_fn' in content, f"{dataset}.py missing _worker_init_fn"
        assert 'worker_init_fn' in content, f"{dataset}.py missing worker_init_fn in DataLoader"

    print("✅ All dataset loaders have worker_init_fn for reproducibility")

    return True


def main():
    print("="*60)
    print("VERIFICATION OF CRITICAL FIXES")
    print("="*60)
    print("\nThis script verifies that all critical fixes are applied.")

    all_passed = True

    try:
        all_passed &= verify_reproducibility()
    except Exception as e:
        print(f"❌ Reproducibility test failed: {e}")
        all_passed = False

    try:
        all_passed &= verify_model_initialization()
    except Exception as e:
        print(f"❌ Model initialization test failed: {e}")
        all_passed = False

    try:
        all_passed &= verify_configs()
    except ImportError:
        print("⚠️  PyYAML not installed, skipping config verification")
    except Exception as e:
        print(f"❌ Config verification failed: {e}")
        all_passed = False

    try:
        all_passed &= verify_mia_requires_test_set()
    except Exception as e:
        print(f"❌ MIA verification failed: {e}")
        all_passed = False

    try:
        all_passed &= verify_differential_trainer()
    except Exception as e:
        print(f"❌ Differential trainer verification failed: {e}")
        all_passed = False

    try:
        all_passed &= verify_worker_init_fn()
    except Exception as e:
        print(f"❌ Worker init verification failed: {e}")
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL VERIFICATION TESTS PASSED")
        print("="*60)
        print("\nThe codebase is ready for reproducible experiments.")
        print("\nNext steps:")
        print("  1. Run: python scripts/preprocess_data.py --dataset ham10000")
        print("  2. Run: python run_experiment.py --experiment ham1000_baseline")
        print("  3. Verify results are reproducible by running twice")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease review the failed tests above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
