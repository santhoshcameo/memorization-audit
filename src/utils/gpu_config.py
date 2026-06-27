"""
GPU Configuration Utilities - OPTIMIZED FOR A100
Handles model-specific batch sizes and DataLoader optimizations

CRITICAL OPTIMIZATIONS:
1. Disabled aggressive cache clearing for A100 (was killing performance)
2. Added CUDA-specific optimizations (cudnn.benchmark, matmul precision)
3. Non-blocking transfers enabled
4. Better memory management
"""

import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Flag to track if CUDA optimizations have been applied
_cuda_optimized = False


def detect_gpu_memory() -> int:
    """Detect GPU memory in GB"""
    if torch.cuda.is_available():
        gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
        return gpu_mem_bytes // (1024 ** 3)
    return 0


def get_gpu_profile_name() -> str:
    """Get GPU profile name based on available memory or environment"""
    # Check if set via environment
    env_profile = os.environ.get('GPU_PROFILE', '').lower()
    if env_profile in ['16gb', '40gb', '80gb']:
        return env_profile

    # Auto-detect based on GPU memory
    gpu_mem = detect_gpu_memory()
    if gpu_mem >= 70:
        return '80gb'
    elif gpu_mem >= 35:
        return '40gb'
    else:
        return '16gb'


def load_gpu_profiles() -> Dict:
    """Load GPU profiles from config file"""
    config_path = PROJECT_ROOT / 'config' / 'gpu_profiles.yaml'
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def get_model_batch_size(model_name: str, gpu_profile: Optional[str] = None) -> int:
    """
    Get optimal batch size for a specific model and GPU

    Args:
        model_name: Model name (resnet50, vit, mae, medsam)
        gpu_profile: GPU profile (16gb, 40gb, 80gb) or None for auto-detect

    Returns:
        Optimal batch size
    """
    if gpu_profile is None:
        gpu_profile = get_gpu_profile_name()

    profiles = load_gpu_profiles()
    profile = profiles.get(gpu_profile, profiles.get('16gb', {}))

    batch_sizes = profile.get('batch_sizes', {})

    # Normalize model name
    model_key = model_name.lower()
    if 'resnet' in model_key:
        model_key = 'resnet50'
    elif 'medsam' in model_key:
        model_key = 'medsam'

    # Default batch sizes if not in config
    default_batch_sizes = {
        'resnet50': 64,
        'vit': 32,
        'mae': 24,
        'medsam': 4
    }

    result = batch_sizes.get(model_key, default_batch_sizes.get(model_key, 32))
    return result


def get_gradient_accumulation(model_name: str, gpu_profile: Optional[str] = None) -> int:
    """Get gradient accumulation steps for effective larger batch"""
    if gpu_profile is None:
        gpu_profile = get_gpu_profile_name()

    profiles = load_gpu_profiles()
    profile = profiles.get(gpu_profile, profiles.get('16gb', {}))

    grad_accum = profile.get('gradient_accumulation', {})
    model_key = model_name.lower()

    return grad_accum.get(model_key, 1)


def get_inference_batch_size(model_name: str, gpu_profile: Optional[str] = None) -> int:
    """
    Get optimal batch size for INFERENCE (no gradients).

    Inference can use 2-3x larger batches than training because:
    - No gradient storage needed
    - No optimizer states
    - Only forward pass activations

    Args:
        model_name: Model name (resnet50, vit, mae, medsam)
        gpu_profile: GPU profile or None for auto-detect

    Returns:
        Optimal inference batch size
    """
    if gpu_profile is None:
        gpu_profile = get_gpu_profile_name()

    profiles = load_gpu_profiles()
    profile = profiles.get(gpu_profile, profiles.get('16gb', {}))

    # Try inference_batch_sizes first, fall back to 3x training batch size
    inference_sizes = profile.get('inference_batch_sizes', {})

    model_key = model_name.lower()
    if 'resnet' in model_key:
        model_key = 'resnet50'
    elif 'medsam' in model_key:
        model_key = 'medsam'

    if model_key in inference_sizes:
        return inference_sizes[model_key]

    # Fallback: 3x the training batch size
    training_batch = get_model_batch_size(model_name, gpu_profile)
    return training_batch * 3


def get_dataloader_config(gpu_profile: Optional[str] = None) -> Dict:
    """
    Get optimal DataLoader configuration

    Returns dict with:
        - num_workers
        - pin_memory
        - persistent_workers
        - prefetch_factor
        - drop_last (for training)
    """
    if gpu_profile is None:
        gpu_profile = get_gpu_profile_name()

    profiles = load_gpu_profiles()
    profile = profiles.get(gpu_profile, profiles.get('16gb', {}))

    dl_config = profile.get('dataloader', {})

    # Optimized defaults for each profile - INCREASED for better GPU utilization
    defaults = {
        '80gb': {
            'num_workers': 16,  # Increased from 12
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4,
            'drop_last': True,  # Important for consistent batch sizes
        },
        '40gb': {
            'num_workers': 12,  # Increased from 8
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4,
            'drop_last': True,
        },
        '16gb': {
            'num_workers': 8,  # Increased from 4
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2,
            'drop_last': True,
        },
    }

    config = defaults.get(gpu_profile, defaults['16gb'])
    config.update(dl_config)

    return config


def get_memory_config(gpu_profile: Optional[str] = None) -> Dict:
    """
    Get memory optimization configuration

    CRITICAL: Cache clearing is DISABLED for A100 GPUs as it causes massive slowdowns.
    Only enable for smaller GPUs where memory fragmentation is a real issue.
    """
    if gpu_profile is None:
        gpu_profile = get_gpu_profile_name()

    profiles = load_gpu_profiles()
    profile = profiles.get(gpu_profile, profiles.get('16gb', {}))

    mem_config = profile.get('memory', {})

    # CRITICAL FIX: Disable cache clearing for A100 GPUs
    # torch.cuda.empty_cache() + synchronize() is EXTREMELY slow (~200-500ms per call)
    # On A100 with 40GB+ memory, fragmentation is not an issue
    defaults = {
        'amp_enabled': True,
        'gradient_checkpointing': gpu_profile == '16gb',
        # DISABLED for A100: empty_cache_freq = 0 means never clear
        'empty_cache_freq': 0 if gpu_profile in ['40gb', '80gb'] else 50,
        'non_blocking': True,  # Use non-blocking transfers
    }

    config = defaults.copy()
    config.update(mem_config)

    return config


def setup_cuda_optimizations():
    """
    Apply CUDA-specific optimizations for maximum GPU performance.

    CRITICAL: This should be called ONCE at the start of training.
    These optimizations can significantly speed up training on A100.
    """
    global _cuda_optimized

    if _cuda_optimized:
        return

    if not torch.cuda.is_available():
        return

    gpu_profile = get_gpu_profile_name()

    # 1. Enable cuDNN benchmark mode
    # This lets cuDNN find the best algorithms for your specific GPU/input sizes
    # CRITICAL for performance - can give 10-30% speedup
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # 2. For A100 GPUs, enable TF32 for faster matmul
    # TF32 uses 19-bit precision which is more than enough for training
    # Can give 2-3x speedup on matmul operations with minimal accuracy loss
    if gpu_profile in ['40gb', '80gb']:
        # TF32 mode for matrix multiplications (significant speedup)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set float32 matmul precision to 'medium' or 'high'
        # 'medium' uses TF32, 'high' uses full FP32
        try:
            torch.set_float32_matmul_precision('medium')
        except AttributeError:
            pass  # Older PyTorch versions don't have this

    # 3. Memory allocator config is already set in setup_gpu_environment()
    # We don't set it here to avoid issues with CUDA initialization order

    _cuda_optimized = True
    print(f"✅ CUDA optimizations enabled for {gpu_profile} profile")
    print(f"   cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    if gpu_profile in ['40gb', '80gb']:
        print(f"   TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")


def setup_gpu_environment():
    """Set optimal environment variables for GPU training"""
    # Prevent thread contention - CRITICAL for multi-worker dataloaders
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    # Better CUDA memory allocation - use compatible setting that works on all PyTorch versions
    # Note: expandable_segments is only supported in PyTorch 2.0+, so we use the safe default
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

    # Async CUDA operations (should already be default, but ensure it)
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')

    # Apply CUDA optimizations (but don't fail on errors)
    # This is deferred to avoid triggering CUDA init during import
    # setup_cuda_optimizations() will be called when training starts


def clear_gpu_cache():
    """
    Clear GPU cache to free memory.

    WARNING: This is an EXPENSIVE operation (~200-500ms on A100).
    Should only be called:
    - Between training different models
    - When OOM errors occur
    - NOT during regular training (use get_memory_config to check empty_cache_freq)
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # NOTE: Removed torch.cuda.synchronize() here - it's even more expensive
        # and empty_cache() doesn't require it


def get_optimal_batch_transfer(device: str = 'cuda') -> Dict:
    """
    Get optimal settings for batch data transfer to GPU.

    Returns:
        Dict with non_blocking setting
    """
    return {
        'non_blocking': True,  # Use async transfers
    }


def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_profile = get_gpu_profile_name()

        # Get current memory usage
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)

        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_mem:.1f} GB total, {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        print(f"Profile: {gpu_profile}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    else:
        print("No GPU available, using CPU")


def estimate_memory_usage(model_name: str, batch_size: int, image_size: int = 224) -> Dict:
    """
    Estimate memory usage for a given model and batch size.

    Returns:
        Dict with estimated memory requirements
    """
    # Rough memory estimates per sample (in MB) with AMP
    memory_per_sample = {
        'resnet50': 50,   # ~50MB per sample at 224x224
        'vit': 100,       # ~100MB per sample at 224x224
        'mae': 150,       # ~150MB per sample at 224x224
        'medsam': 200,    # ~200MB per sample at 224x224 (uses 224 not 1024)
    }

    model_key = model_name.lower()
    if 'resnet' in model_key:
        model_key = 'resnet50'
    elif 'medsam' in model_key:
        model_key = 'medsam'

    per_sample = memory_per_sample.get(model_key, 100)

    # Scale by image size
    size_factor = (image_size / 224) ** 2

    estimated_mb = batch_size * per_sample * size_factor
    estimated_gb = estimated_mb / 1024

    return {
        'estimated_mb': estimated_mb,
        'estimated_gb': estimated_gb,
        'per_sample_mb': per_sample * size_factor,
    }


# Initialize environment on import
setup_gpu_environment()
