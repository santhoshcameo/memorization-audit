"""
Reproducibility Utilities
Set random seeds and deterministic behavior for reproducible experiments
"""

import random
import numpy as np
import torch
import os
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Print confirmation
    print(f"✅ Random seed set to: {seed}")


def set_deterministic(deterministic: bool = True, benchmark: bool = False):
    """
    Set PyTorch deterministic behavior
    
    Args:
        deterministic: If True, use deterministic algorithms (slower but reproducible)
        benchmark: If True, use cudnn benchmark (faster but non-deterministic)
    
    Note:
        deterministic=True and benchmark=True are contradictory
        If both True, deterministic takes precedence
    """
    if deterministic:
        # Use deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for deterministic behavior
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Use deterministic algorithms (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        
        print("✅ Deterministic mode enabled (slower but reproducible)")
    
    elif benchmark:
        # Use cudnn benchmark for speed (non-deterministic)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        print("✅ Benchmark mode enabled (faster but non-deterministic)")
    
    else:
        # Default PyTorch behavior
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        
        print("✅ Default PyTorch behavior")


def setup_reproducibility(seed: int = 42, 
                          deterministic: bool = True,
                          benchmark: bool = False):
    """
    Complete reproducibility setup
    
    Args:
        seed: Random seed
        deterministic: Use deterministic algorithms
        benchmark: Use cudnn benchmark (ignored if deterministic=True)
    """
    print("\n" + "="*60)
    print("REPRODUCIBILITY SETUP")
    print("="*60)
    
    set_seed(seed)
    set_deterministic(deterministic, benchmark)
    
    print("="*60)


def get_random_states():
    """
    Get current random states for all RNGs
    Useful for saving/loading exact state
    
    Returns:
        Dictionary with all random states
    """
    states = {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        states['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return states


def set_random_states(states: dict):
    """
    Restore random states
    
    Args:
        states: Dictionary of random states (from get_random_states)
    """
    random.setstate(states['python_random'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    
    if 'torch_cuda' in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['torch_cuda'])


def save_random_states(filepath: str):
    """
    Save random states to file
    
    Args:
        filepath: Path to save states
    """
    states = get_random_states()
    torch.save(states, filepath)
    print(f"✅ Random states saved to: {filepath}")


def load_random_states(filepath: str):
    """
    Load and restore random states from file
    
    Args:
        filepath: Path to load states from
    """
    states = torch.load(filepath)
    set_random_states(states)
    print(f"✅ Random states loaded from: {filepath}")


def worker_init_fn(worker_id: int, seed: int = 42):
    """
    Initialize worker seed for DataLoader
    Ensures each worker has different but reproducible seed
    
    Args:
        worker_id: Worker ID
        seed: Base seed
    
    Usage:
        DataLoader(..., worker_init_fn=lambda w: worker_init_fn(w, seed=42))
    """
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_reproducibility_info():
    """
    Get information about current reproducibility settings
    
    Returns:
        Dictionary with reproducibility info
    """
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
    
    return info


def print_reproducibility_info():
    """Print reproducibility information"""
    info = get_reproducibility_info()
    
    print("\n" + "="*60)
    print("REPRODUCIBILITY INFO")
    print("="*60)
    for key, value in info.items():
        print(f"{key:25s}: {value}")
    print("="*60)


# Example usage
if __name__ == "__main__":
    # Basic setup
    setup_reproducibility(seed=42, deterministic=True)
    
    # Print info
    print_reproducibility_info()
    
    # Test reproducibility
    print("\nTesting reproducibility:")
    set_seed(42)
    rand1 = torch.rand(3)
    print(f"Random tensor 1: {rand1}")
    
    set_seed(42)
    rand2 = torch.rand(3)
    print(f"Random tensor 2: {rand2}")
    
    if torch.allclose(rand1, rand2):
        print("✅ Reproducible!")
    else:
        print("❌ Not reproducible!")