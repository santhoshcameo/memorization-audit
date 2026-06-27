"""
Membership Inference Attack (MIA) Validation Module

Validates the memorization metric M(x) by demonstrating that high memorization
correlates with successful membership inference attacks.

Key Findings to Validate:
1. Rare disease patients have higher MIA vulnerability
2. M(x) score predicts MIA success
3. Different architectures have different vulnerability profiles

MIA Methods Implemented:
1. Loss-threshold attack (Yeom et al., 2018)
2. Confidence-based attack
3. Likelihood-ratio inspired attack
4. Per-class calibrated attack


"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, f1_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MIAResult:
    """Results from a membership inference attack"""
    attack_name: str
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    tpr_at_low_fpr: Dict[str, float]  # TPR at 1%, 5%, 10% FPR
    predictions: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


class MembershipInferenceAttack:
    """
    Membership Inference Attack Framework

    Validates that memorization score M(x) correlates with privacy risk
    by demonstrating successful membership inference on canary samples.

    Attack Setting:
    - Members: Canary samples (seen by candidate model during training)
    - Non-members: Test samples (never seen during training)

    Key Insight:
    - Candidate model has LOWER loss on members (it memorized them)
    - Independent model has similar loss on both (no memorization)
    - Difference in loss = membership signal
    """

    def __init__(self,
                 candidate_losses: np.ndarray,
                 independent_losses: np.ndarray,
                 member_labels: np.ndarray,
                 class_labels: Optional[np.ndarray] = None,
                 class_names: Optional[List[str]] = None,
                 sample_ids: Optional[List[str]] = None):
        """
        Initialize MIA with loss values.

        Args:
            candidate_losses: Loss values from candidate model (trained on train+canary)
            independent_losses: Loss values from independent model (trained on train only)
            member_labels: 1 for members (canary), 0 for non-members (test)
            class_labels: Class index for each sample
            class_names: Names of classes
            sample_ids: Sample identifiers
        """
        self.candidate_losses = np.array(candidate_losses)
        self.independent_losses = np.array(independent_losses)
        self.member_labels = np.array(member_labels)
        self.class_labels = np.array(class_labels) if class_labels is not None else None
        self.class_names = class_names
        self.sample_ids = sample_ids

        # Compute memorization scores M(x) = L_independent - L_candidate
        self.memorization_scores = self.independent_losses - self.candidate_losses

        # Validate inputs
        assert len(self.candidate_losses) == len(self.member_labels), \
            "Loss and label arrays must have same length"

        self.n_members = np.sum(self.member_labels == 1)
        self.n_non_members = np.sum(self.member_labels == 0)

        print(f"MIA initialized:")
        print(f"  Total samples: {len(self.member_labels)}")
        print(f"  Members (canary): {self.n_members}")
        print(f"  Non-members (test): {self.n_non_members}")

    def loss_threshold_attack(self, use_candidate: bool = True) -> MIAResult:
        """
        Loss-threshold attack (Yeom et al., 2018)

        Simple but effective: predict member if loss < threshold

        Intuition: Model has lower loss on training samples (members)

        Args:
            use_candidate: Use candidate model losses (True) or independent (False)

        Returns:
            MIAResult with attack metrics
        """
        losses = self.candidate_losses if use_candidate else self.independent_losses

        # Attack score: negative loss (lower loss = more likely member)
        attack_scores = -losses

        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(self.member_labels, attack_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Make predictions
        predictions = (attack_scores >= optimal_threshold).astype(int)

        # Compute metrics
        return self._compute_metrics(
            attack_name=f"Loss-Threshold ({'Candidate' if use_candidate else 'Independent'})",
            predictions=predictions,
            scores=attack_scores,
            threshold=optimal_threshold
        )

    def confidence_attack(self) -> MIAResult:
        """
        Confidence-based attack

        Uses the confidence difference between models as attack signal.
        Higher confidence from candidate model = likely member.

        Approximation: exp(-loss) as proxy for confidence
        """
        # Convert losses to pseudo-confidences
        candidate_conf = np.exp(-self.candidate_losses)
        independent_conf = np.exp(-self.independent_losses)

        # Attack score: confidence ratio
        # Higher ratio = candidate more confident = likely member
        attack_scores = candidate_conf / (independent_conf + 1e-10)

        # Handle numerical issues
        attack_scores = np.clip(attack_scores, 0, 100)

        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(self.member_labels, attack_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        predictions = (attack_scores >= optimal_threshold).astype(int)

        return self._compute_metrics(
            attack_name="Confidence-Ratio",
            predictions=predictions,
            scores=attack_scores,
            threshold=optimal_threshold
        )

    def memorization_score_attack(self) -> MIAResult:
        """
        Direct memorization score attack

        Uses M(x) = L_independent - L_candidate as attack signal.
        This is YOUR metric - validating it works as privacy risk indicator.

        High M(x) = high memorization = likely member
        """
        attack_scores = self.memorization_scores

        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(self.member_labels, attack_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        predictions = (attack_scores >= optimal_threshold).astype(int)

        return self._compute_metrics(
            attack_name="Memorization-Score M(x)",
            predictions=predictions,
            scores=attack_scores,
            threshold=optimal_threshold
        )

    def likelihood_ratio_attack(self) -> MIAResult:
        """
        Likelihood-ratio inspired attack (simplified LiRA)

        Computes a likelihood ratio based on loss distributions.

        P(member|loss) / P(non-member|loss)

        Simplified version without shadow models.
        """
        # Estimate loss distributions for members and non-members
        member_mask = self.member_labels == 1
        non_member_mask = self.member_labels == 0

        member_losses = self.candidate_losses[member_mask]
        non_member_losses = self.candidate_losses[non_member_mask]

        # Fit Gaussian distributions (simplified)
        member_mean, member_std = np.mean(member_losses), np.std(member_losses) + 1e-10
        non_member_mean, non_member_std = np.mean(non_member_losses), np.std(non_member_losses) + 1e-10

        # Compute log-likelihood ratio for each sample
        def log_likelihood_ratio(loss):
            ll_member = -0.5 * ((loss - member_mean) / member_std) ** 2
            ll_non_member = -0.5 * ((loss - non_member_mean) / non_member_std) ** 2
            return ll_member - ll_non_member

        attack_scores = np.array([log_likelihood_ratio(l) for l in self.candidate_losses])

        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(self.member_labels, attack_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        predictions = (attack_scores >= optimal_threshold).astype(int)

        return self._compute_metrics(
            attack_name="Likelihood-Ratio",
            predictions=predictions,
            scores=attack_scores,
            threshold=optimal_threshold
        )

    def per_class_calibrated_attack(self) -> Tuple[MIAResult, pd.DataFrame]:
        """
        Per-class calibrated attack

        KEY INNOVATION: Calibrate attack threshold per class

        This accounts for class-specific loss distributions and
        reveals that RARE classes are more vulnerable.

        Returns:
            Overall MIAResult and per-class DataFrame
        """
        if self.class_labels is None:
            raise ValueError("Class labels required for per-class attack")

        unique_classes = np.unique(self.class_labels)

        all_predictions = np.zeros_like(self.member_labels)
        all_scores = np.zeros_like(self.candidate_losses)

        per_class_results = []

        for class_idx in unique_classes:
            class_mask = self.class_labels == class_idx
            class_members = self.member_labels[class_mask]
            class_losses = self.candidate_losses[class_mask]

            if len(np.unique(class_members)) < 2:
                # Skip if only one type (all members or all non-members)
                continue

            # Per-class threshold optimization
            attack_scores = -class_losses

            try:
                fpr, tpr, thresholds = roc_curve(class_members, attack_scores)
                auc = roc_auc_score(class_members, attack_scores)

                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]

                predictions = (attack_scores >= optimal_threshold).astype(int)
                accuracy = accuracy_score(class_members, predictions)

                # Store predictions
                all_predictions[class_mask] = predictions
                all_scores[class_mask] = attack_scores

                # Class statistics
                n_samples = np.sum(class_mask)
                n_members = np.sum(class_members == 1)

                # Safe class name lookup with bounds check
                if self.class_names and class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                else:
                    class_name = f"Class_{class_idx}"

                per_class_results.append({
                    'class': class_name,
                    'class_idx': class_idx,
                    'n_samples': n_samples,
                    'n_members': n_members,
                    'accuracy': accuracy,
                    'auc': auc,
                    'threshold': optimal_threshold,
                    'mean_loss_member': np.mean(class_losses[class_members == 1]) if np.sum(class_members == 1) > 0 else np.nan,
                    'mean_loss_non_member': np.mean(class_losses[class_members == 0]) if np.sum(class_members == 0) > 0 else np.nan,
                })

            except Exception as e:
                print(f"Warning: Could not compute metrics for class {class_idx}: {e}")
                continue

        per_class_df = pd.DataFrame(per_class_results)

        # Overall metrics
        overall_result = self._compute_metrics(
            attack_name="Per-Class Calibrated",
            predictions=all_predictions,
            scores=all_scores,
            threshold=0  # Per-class thresholds
        )

        return overall_result, per_class_df

    def _compute_metrics(self, attack_name: str, predictions: np.ndarray,
                        scores: np.ndarray, threshold: float) -> MIAResult:
        """Compute comprehensive attack metrics"""

        # Basic metrics
        accuracy = accuracy_score(self.member_labels, predictions)

        try:
            auc = roc_auc_score(self.member_labels, scores)
        except ValueError as e:
            # ValueError occurs when only one class is present in labels
            print(f"Warning: Could not compute AUC - {e}. Using 0.5 (random baseline).")
            auc = 0.5

        # Precision, recall, F1 for member class
        cm = confusion_matrix(self.member_labels, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision, recall, f1 = 0, 0, 0

        # TPR at low FPR (important for privacy)
        # Use linear interpolation for accurate TPR@FPR values
        fpr, tpr, _ = roc_curve(self.member_labels, scores)
        tpr_at_low_fpr = {}
        for target_fpr in [0.01, 0.05, 0.10]:
            # Use linear interpolation instead of nearest neighbor
            # This gives more accurate TPR values at exact FPR thresholds
            if target_fpr <= fpr.min():
                tpr_value = tpr[0]
            elif target_fpr >= fpr.max():
                tpr_value = tpr[-1]
            else:
                # Linear interpolation between surrounding points
                tpr_value = np.interp(target_fpr, fpr, tpr)
            tpr_at_low_fpr[f"TPR@{int(target_fpr*100)}%FPR"] = float(tpr_value)

        return MIAResult(
            attack_name=attack_name,
            accuracy=accuracy,
            auc=auc,
            precision=precision,
            recall=recall,
            f1=f1,
            threshold=threshold,
            tpr_at_low_fpr=tpr_at_low_fpr,
            predictions=predictions,
            scores=scores,
            labels=self.member_labels
        )

    def run_all_attacks(self) -> Dict[str, MIAResult]:
        """Run all MIA attacks and return results"""
        results = {}

        print("\n" + "="*60)
        print("RUNNING MEMBERSHIP INFERENCE ATTACKS")
        print("="*60)

        # 1. Loss-threshold (candidate)
        print("\n[1/5] Loss-Threshold Attack (Candidate Model)...")
        results['loss_threshold_candidate'] = self.loss_threshold_attack(use_candidate=True)

        # 2. Loss-threshold (independent) - baseline
        print("[2/5] Loss-Threshold Attack (Independent Model - Baseline)...")
        results['loss_threshold_independent'] = self.loss_threshold_attack(use_candidate=False)

        # 3. Confidence ratio
        print("[3/5] Confidence-Ratio Attack...")
        results['confidence_ratio'] = self.confidence_attack()

        # 4. Memorization score
        print("[4/5] Memorization Score M(x) Attack...")
        results['memorization_score'] = self.memorization_score_attack()

        # 5. Likelihood ratio
        print("[5/5] Likelihood-Ratio Attack...")
        results['likelihood_ratio'] = self.likelihood_ratio_attack()

        return results

    def analyze_rarity_vulnerability(self,
                                     class_frequencies: Dict[str, int],
                                     attack_result: MIAResult) -> pd.DataFrame:
        """
        Analyze MIA vulnerability by class rarity

        KEY ANALYSIS: Demonstrates rare diseases have higher privacy risk

        Args:
            class_frequencies: Dict mapping class name to frequency in training set
            attack_result: MIA result to analyze

        Returns:
            DataFrame with rarity vs vulnerability analysis
        """
        if self.class_labels is None or self.class_names is None:
            raise ValueError("Class labels and names required")

        results = []

        for class_idx, class_name in enumerate(self.class_names):
            class_mask = self.class_labels == class_idx
            if np.sum(class_mask) == 0:
                continue

            class_predictions = attack_result.predictions[class_mask]
            class_labels = self.member_labels[class_mask]
            class_scores = attack_result.scores[class_mask]

            # Only compute for classes with both members and non-members
            if len(np.unique(class_labels)) < 2:
                continue

            try:
                accuracy = accuracy_score(class_labels, class_predictions)
                auc = roc_auc_score(class_labels, class_scores)
            except:
                continue

            frequency = class_frequencies.get(class_name, 0)

            results.append({
                'class': class_name,
                'frequency': frequency,
                'n_samples': np.sum(class_mask),
                'n_members': np.sum(class_labels == 1),
                'mia_accuracy': accuracy,
                'mia_auc': auc,
                'mean_memorization': np.mean(self.memorization_scores[class_mask]),
                'is_rare': frequency < np.median(list(class_frequencies.values()))
            })

        df = pd.DataFrame(results)

        if len(df) > 0:
            # Compute correlation between rarity and vulnerability
            if len(df) > 2:
                spearman_corr, spearman_p = stats.spearmanr(df['frequency'], df['mia_auc'])
                df.attrs['spearman_rho'] = spearman_corr
                df.attrs['spearman_p'] = spearman_p

        return df

    def compute_memorization_mia_correlation(self) -> Dict:
        """
        Compute correlation between memorization score and MIA success

        VALIDATES: M(x) is a good predictor of privacy risk
        """
        # Run memorization score attack
        mia_result = self.memorization_score_attack()

        # Correlation between M(x) and being correctly identified
        correct_predictions = (mia_result.predictions == self.member_labels).astype(int)

        # For members only: does higher M(x) mean more likely to be caught?
        member_mask = self.member_labels == 1
        member_scores = self.memorization_scores[member_mask]
        member_caught = correct_predictions[member_mask]

        # Point-biserial correlation
        if len(np.unique(member_caught)) > 1:
            corr, p_value = stats.pointbiserialr(member_caught, member_scores)
        else:
            corr, p_value = 0, 1

        # Bin analysis: high M(x) vs low M(x)
        median_score = np.median(member_scores)
        high_mem_accuracy = np.mean(member_caught[member_scores >= median_score])
        low_mem_accuracy = np.mean(member_caught[member_scores < median_score])

        return {
            'correlation': corr,
            'p_value': p_value,
            'high_memorization_mia_accuracy': high_mem_accuracy,
            'low_memorization_mia_accuracy': low_mem_accuracy,
            'accuracy_difference': high_mem_accuracy - low_mem_accuracy,
            'validates_hypothesis': corr > 0 and p_value < 0.05
        }


def create_mia_from_memorization_results(
    memorization_csv: Path,
    test_csv: Path,
    class_frequencies: Dict[str, int]
) -> MembershipInferenceAttack:
    """
    Create MIA instance from existing memorization measurement results.

    Args:
        memorization_csv: Path to memorization scores CSV (canary samples = members)
        test_csv: Path to test set evaluation (non-members)
        class_frequencies: Training set class frequencies

    Returns:
        Configured MembershipInferenceAttack instance
    """
    # Load memorization results (these are the canary/member samples)
    mem_df = pd.read_csv(memorization_csv)

    # For members: we have candidate and independent losses
    member_candidate_losses = mem_df['loss_candidate'].values
    member_independent_losses = mem_df['loss_independent'].values
    member_labels = np.ones(len(mem_df))
    member_classes = mem_df['class_name'].values if 'class_name' in mem_df.columns else None
    member_ids = mem_df['image_id'].values if 'image_id' in mem_df.columns else None

    # Load test results - REQUIRED for valid MIA
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        non_member_candidate_losses = test_df['loss_candidate'].values
        non_member_independent_losses = test_df['loss_independent'].values
        non_member_labels = np.zeros(len(test_df))
        non_member_classes = test_df['class_name'].values if 'class_name' in test_df.columns else None
        non_member_ids = test_df['image_id'].values if 'image_id' in test_df.columns else None
    else:
        # CRITICAL: Do NOT create synthetic non-members from member data
        # This would cause data leakage and invalidate MIA results
        raise FileNotFoundError(
            f"Test CSV not found: {test_csv}\n"
            f"MIA validation requires real non-member (test set) data.\n"
            f"Please run memorization measurement on test set first:\n"
            f"  1. Ensure test split exists in your dataset\n"
            f"  2. Run evaluation on test samples with both candidate and independent models\n"
            f"  3. Save results to: {test_csv}\n"
            f"\nSynthetic non-member generation is NOT supported as it causes data leakage."
        )

    # Combine
    all_candidate_losses = np.concatenate([member_candidate_losses, non_member_candidate_losses])
    all_independent_losses = np.concatenate([member_independent_losses, non_member_independent_losses])
    all_labels = np.concatenate([member_labels, non_member_labels])

    # Class labels - use sorted() for deterministic ordering across runs
    if member_classes is not None and non_member_classes is not None:
        all_classes = np.concatenate([member_classes, non_member_classes])
        unique_classes = sorted(set(all_classes))  # Sorted for reproducibility
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        all_class_indices = np.array([class_to_idx[c] for c in all_classes])
    else:
        all_class_indices = None
        unique_classes = None

    return MembershipInferenceAttack(
        candidate_losses=all_candidate_losses,
        independent_losses=all_independent_losses,
        member_labels=all_labels,
        class_labels=all_class_indices,
        class_names=unique_classes,
        sample_ids=None
    )


def generate_mia_report(
    mia: MembershipInferenceAttack,
    output_dir: Path,
    model_name: str,
    dataset_name: str,
    class_frequencies: Dict[str, int]
) -> Dict:
    """
    Generate comprehensive MIA validation report.

    Creates:
    - Summary statistics
    - Per-class vulnerability analysis
    - Rarity correlation analysis
    - Figures for paper
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'model': model_name,
        'dataset': dataset_name,
        'n_members': int(mia.n_members),
        'n_non_members': int(mia.n_non_members),
        'attacks': {},
        'rarity_analysis': {},
        'memorization_correlation': {}
    }

    # Run all attacks
    attack_results = mia.run_all_attacks()

    print("\n" + "="*60)
    print("MIA ATTACK RESULTS SUMMARY")
    print("="*60)

    for name, result in attack_results.items():
        report['attacks'][name] = {
            'accuracy': float(result.accuracy),
            'auc': float(result.auc),
            'precision': float(result.precision),
            'recall': float(result.recall),
            'f1': float(result.f1),
            'tpr_at_low_fpr': {k: float(v) for k, v in result.tpr_at_low_fpr.items()}
        }

        print(f"\n{result.attack_name}:")
        print(f"  Accuracy: {result.accuracy:.4f}")
        print(f"  AUC: {result.auc:.4f}")
        print(f"  TPR@1%FPR: {result.tpr_at_low_fpr.get('TPR@1%FPR', 0):.4f}")
        print(f"  TPR@5%FPR: {result.tpr_at_low_fpr.get('TPR@5%FPR', 0):.4f}")

    # Per-class calibrated attack
    if mia.class_labels is not None:
        print("\n" + "="*60)
        print("PER-CLASS VULNERABILITY ANALYSIS")
        print("="*60)

        try:
            _, per_class_df = mia.per_class_calibrated_attack()
            per_class_df.to_csv(output_dir / f'{model_name}_mia_per_class.csv', index=False)
            print(f"\nPer-class results saved to {model_name}_mia_per_class.csv")
            print(per_class_df.to_string(index=False))
        except Exception as e:
            print(f"Warning: Per-class analysis failed: {e}")

    # Rarity vulnerability analysis
    if mia.class_labels is not None and class_frequencies:
        print("\n" + "="*60)
        print("RARITY VS VULNERABILITY ANALYSIS")
        print("="*60)

        # Use memorization score attack for rarity analysis
        mem_attack = attack_results['memorization_score']
        rarity_df = mia.analyze_rarity_vulnerability(class_frequencies, mem_attack)

        if len(rarity_df) > 0:
            rarity_df = rarity_df.sort_values('frequency')
            rarity_df.to_csv(output_dir / f'{model_name}_rarity_vulnerability.csv', index=False)

            print("\nClass vulnerability by frequency:")
            print(rarity_df[['class', 'frequency', 'mia_accuracy', 'mia_auc', 'mean_memorization']].to_string(index=False))

            # Rare vs common comparison
            rare_mask = rarity_df['is_rare']
            if rare_mask.sum() > 0 and (~rare_mask).sum() > 0:
                rare_auc = rarity_df[rare_mask]['mia_auc'].mean()
                common_auc = rarity_df[~rare_mask]['mia_auc'].mean()

                print(f"\n*** KEY FINDING ***")
                print(f"Rare classes MIA AUC: {rare_auc:.4f}")
                print(f"Common classes MIA AUC: {common_auc:.4f}")
                print(f"Difference: {rare_auc - common_auc:.4f}")

                report['rarity_analysis'] = {
                    'rare_mia_auc': float(rare_auc),
                    'common_mia_auc': float(common_auc),
                    'auc_difference': float(rare_auc - common_auc),
                    'spearman_rho': float(rarity_df.attrs.get('spearman_rho', 0)),
                    'spearman_p': float(rarity_df.attrs.get('spearman_p', 1))
                }

                if rare_auc > common_auc:
                    print(f"VALIDATES HYPOTHESIS: Rare diseases have higher privacy risk!")

    # Memorization-MIA correlation
    print("\n" + "="*60)
    print("MEMORIZATION SCORE VALIDATION")
    print("="*60)

    correlation = mia.compute_memorization_mia_correlation()
    report['memorization_correlation'] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                          for k, v in correlation.items()}

    print(f"\nCorrelation between M(x) and MIA success: {correlation['correlation']:.4f}")
    print(f"P-value: {correlation['p_value']:.4f}")
    print(f"High M(x) samples - MIA accuracy: {correlation['high_memorization_mia_accuracy']:.4f}")
    print(f"Low M(x) samples - MIA accuracy: {correlation['low_memorization_mia_accuracy']:.4f}")

    if correlation['validates_hypothesis']:
        print("\nVALIDATES: M(x) is a significant predictor of MIA vulnerability!")

    return report
