"""
Training Module
Training loops and utilities
"""

from .trainer import BaseTrainer
from .differential_trainer import DifferentialTrainer, train_differential_model

__all__ = [
    'BaseTrainer',
    'DifferentialTrainer',
    'train_differential_model',
]
