"""
Metrics Computation
Standard classification metrics and evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(model: nn.Module,
                   dataloader: DataLoader,
                   device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate model on dataset
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Device to use
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Unpack batch
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch
        
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        # Store results
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall_weighted': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
    }
    
    return metrics, all_preds, all_labels, all_probs


def compute_per_class_metrics(labels: np.ndarray,
                              preds: np.ndarray,
                              class_names: list) -> Dict[str, Dict]:
    """
    Compute per-class metrics
    
    Args:
        labels: True labels
        preds: Predicted labels
        class_names: List of class names
    
    Returns:
        Dictionary with per-class metrics
    """
    # Compute metrics for each class
    precision = precision_score(labels, preds, average=None, zero_division=0)
    recall = recall_score(labels, preds, average=None, zero_division=0)
    f1 = f1_score(labels, preds, average=None, zero_division=0)
    
    per_class = {}
    for i, class_name in enumerate(class_names):
        per_class[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(np.sum(labels == i))
        }
    
    return per_class


def compute_confusion_matrix(labels: np.ndarray,
                            preds: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix
    
    Args:
        labels: True labels
        preds: Predicted labels
    
    Returns:
        Confusion matrix
    """
    return confusion_matrix(labels, preds)


def print_classification_report(labels: np.ndarray,
                               preds: np.ndarray,
                               class_names: list):
    """
    Print classification report
    
    Args:
        labels: True labels
        preds: Predicted labels
        class_names: List of class names
    """
    print("\nClassification Report:")
    print("="*80)
    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)


def compute_confidence_intervals(scores: np.ndarray,
                                confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence intervals using bootstrap
    
    Args:
        scores: Array of scores
        confidence: Confidence level
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)
    
    # T-distribution
    ci = stats.t.interval(
        confidence,
        n - 1,
        loc=mean,
        scale=se
    )
    
    return ci


# Example usage
if __name__ == "__main__":
    print("Metrics module")
    print("Provides standard classification metrics")