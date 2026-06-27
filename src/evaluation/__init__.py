"""
Evaluation Module
Memorization measurement, metrics, and statistical tests
"""

from .memorization import MemorizationScorer, measure_memorization
from .membership_inference import (
    MembershipInferenceAttack,
    MIAResult,
    create_mia_from_memorization_results,
    generate_mia_report
)
from .metrics import (
    evaluate_model,
    compute_per_class_metrics,
    compute_confusion_matrix,
    print_classification_report,
    compute_confidence_intervals
)
from .statistical_tests import (
    ttest_independent,
    cohens_d,
    interpret_cohens_d,
    anova_oneway,
    compare_all_models,
    bonferroni_correction,
    compute_correlation,
    test_monotonic_trend,
    generate_statistical_report
)

__all__ = [
    # Memorization
    'MemorizationScorer',
    'measure_memorization',
    # Membership Inference Attack
    'MembershipInferenceAttack',
    'MIAResult',
    'create_mia_from_memorization_results',
    'generate_mia_report',
    # Metrics
    'evaluate_model',
    'compute_per_class_metrics',
    'compute_confusion_matrix',
    'print_classification_report',
    'compute_confidence_intervals',
    # Statistical tests
    'ttest_independent',
    'cohens_d',
    'interpret_cohens_d',
    'anova_oneway',
    'compare_all_models',
    'bonferroni_correction',
    'compute_correlation',
    'test_monotonic_trend',
    'generate_statistical_report',
]
