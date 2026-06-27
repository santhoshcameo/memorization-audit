"""
Layer-wise Memorization Analysis

Analyzes WHERE memorization happens in the network by comparing intermediate
layer activations between candidate and independent models.

For each layer, computes:
1. Cosine distance (1 - cosine_similarity) between candidate and independent
   activations — scale-invariant, comparable across layers.
2. Normalized L2 distance (L2 of unit-normed vectors) as a secondary metric.
3. Gradient magnitude: per-layer gradient norms for the candidate model.
4. Spearman correlation: per-layer correlation of M(x) vs feature distance.
5. Per-class analysis: within-class memorization effects to remove the
   rare-vs-common confound.

Supports architectures:
- ResNet50: conv1+bn1, layer1, layer2, layer3, layer4, avgpool
- ViT/MAE: patch_embed, blocks.0-11, norm (CLS token extracted separately)
- MedSAM: encoder blocks.0-11, norm, neck, classifier
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

try:
    from ..utils.gpu_config import get_gpu_profile_name
except Exception:
    def get_gpu_profile_name():
        return 'unknown'


class LayerMemorizationAnalyzer:
    """Analyzes where memorization concentrates in the network."""

    def __init__(
        self,
        candidate_model: nn.Module,
        independent_model: nn.Module,
        device: str = 'cuda',
        model_name: str = 'resnet50',
    ):
        self.candidate_model = candidate_model.to(device)
        self.independent_model = independent_model.to(device)
        self.device = device
        self.model_name = model_name.lower()
        self.use_amp = device == 'cuda'

        # Put models in eval mode
        self.candidate_model.eval()
        self.independent_model.eval()

        # Resolve layer structure
        self._layer_names: List[str] = []
        self._candidate_layers: OrderedDict = OrderedDict()
        self._independent_layers: OrderedDict = OrderedDict()
        self._candidate_layers, self._layer_names = self._get_layer_hooks(
            self.candidate_model, self.model_name
        )
        self._independent_layers, _ = self._get_layer_hooks(
            self.independent_model, self.model_name
        )

        # Activation storage (populated by hooks)
        self._candidate_activations: Dict[str, torch.Tensor] = {}
        self._independent_activations: Dict[str, torch.Tensor] = {}

        # Hook handles for cleanup
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []

        print(f"LayerMemorizationAnalyzer initialized")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Layers ({len(self._layer_names)}): {self._layer_names}")

    # ------------------------------------------------------------------
    # Layer detection
    # ------------------------------------------------------------------

    @staticmethod
    def _get_layer_hooks(
        model: nn.Module, model_name: str
    ) -> Tuple[OrderedDict, List[str]]:
        """
        Return an OrderedDict of {layer_name: module} for the given architecture.
        """
        layers = OrderedDict()

        if model_name == 'resnet50':
            bb = model.backbone
            layers['conv1_bn1'] = bb.bn1   # hook AFTER bn1 (post-normalization)
            layers['layer1'] = bb.layer1
            layers['layer2'] = bb.layer2
            layers['layer3'] = bb.layer3
            layers['layer4'] = bb.layer4
            # Hook avgpool instead of fc — we want representations, not logits
            layers['avgpool'] = bb.avgpool

        elif model_name in ('vit', 'mae'):
            bb = model.backbone
            layers['patch_embed'] = bb.patch_embed
            for i, block in enumerate(bb.blocks):
                layers[f'blocks.{i}'] = block
            layers['norm'] = bb.norm
            # Skip bb.head (logits) — norm output is the last representation

        elif model_name == 'medsam':
            enc = model.encoder
            layers['patch_embed'] = enc.patch_embed
            if hasattr(enc, 'blocks'):
                for i, block in enumerate(enc.blocks):
                    layers[f'blocks.{i}'] = block
            if hasattr(enc, 'norm'):
                layers['norm'] = enc.norm
            layers['neck'] = model.neck
            # Skip model.classifier (logits)

        else:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. "
                "Expected one of: resnet50, vit, mae, medsam."
            )

        layer_names = list(layers.keys())
        return layers, layer_names

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register forward hooks on both models to capture activations."""
        self._remove_hooks()
        self._candidate_activations.clear()
        self._independent_activations.clear()

        def _make_hook(storage: dict, name: str):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    storage[name] = output.detach()
                elif isinstance(output, (tuple, list)):
                    for o in output:
                        if isinstance(o, torch.Tensor):
                            storage[name] = o.detach()
                            break
            return hook_fn

        for name, module in self._candidate_layers.items():
            h = module.register_forward_hook(
                _make_hook(self._candidate_activations, name)
            )
            self._hook_handles.append(h)

        for name, module in self._independent_layers.items():
            h = module.register_forward_hook(
                _make_hook(self._independent_activations, name)
            )
            self._hook_handles.append(h)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------
    # Activation post-processing
    # ------------------------------------------------------------------

    def _pool_activation(self, act: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Reduce an activation tensor to shape (B, C).

        For ViT/MAE transformer blocks and norm: extract CLS token (index 0)
        instead of mean-pooling over all tokens, since CLS carries the
        classification signal where memorization manifests.

        For CNN spatial features: adaptive_avg_pool2d -> (B, C).
        """
        if act.ndim == 4:
            # Convolutional feature map (B, C, H, W)
            return F.adaptive_avg_pool2d(act, 1).flatten(1)
        elif act.ndim == 3:
            # Transformer token sequence (B, seq_len, dim)
            # Use CLS token (index 0) for blocks and norm layers
            if self.model_name in ('vit', 'mae', 'medsam'):
                if layer_name.startswith('blocks.') or layer_name == 'norm':
                    return act[:, 0, :]  # CLS token only
            # For patch_embed or unknown: mean pool
            return act.mean(dim=1)
        elif act.ndim == 2:
            return act
        elif act.ndim == 1:
            return act.unsqueeze(0)
        else:
            return act.flatten(1)

    # ------------------------------------------------------------------
    # Core: per-layer feature distances (cosine + normalized L2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_layer_distances(
        self,
        dataloader,
        max_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        For each sample, compute cosine distance and normalized L2 distance
        between candidate and independent activations at every hooked layer.

        Returns:
            DataFrame with columns:
                [image_id, class_name, layer_name, cosine_distance,
                 normalized_l2, raw_l2]
        """
        self._register_hooks()
        records: List[dict] = []
        samples_seen = 0

        amp_ctx = torch.cuda.amp.autocast() if self.use_amp else _nullcontext()

        for batch in tqdm(dataloader, desc="Layer distances"):
            images, labels, metadata = self._unpack_batch(batch)
            batch_size = images.size(0)

            if max_samples is not None and samples_seen >= max_samples:
                break

            images = images.to(self.device, non_blocking=True)

            with amp_ctx:
                self._candidate_activations.clear()
                _ = self.candidate_model(images)

                self._independent_activations.clear()
                _ = self.independent_model(images)

            for layer_name in self._layer_names:
                act_cand = self._candidate_activations.get(layer_name)
                act_ind = self._independent_activations.get(layer_name)
                if act_cand is None or act_ind is None:
                    continue

                pooled_cand = self._pool_activation(act_cand, layer_name).float()
                pooled_ind = self._pool_activation(act_ind, layer_name).float()

                # 1. Cosine distance (scale-invariant, comparable across layers)
                cos_sim = F.cosine_similarity(pooled_cand, pooled_ind, dim=-1)
                cosine_dist = 1.0 - cos_sim  # (B,)

                # 2. Normalized L2 (L2 of unit-normed vectors)
                normed_cand = F.normalize(pooled_cand, p=2, dim=-1)
                normed_ind = F.normalize(pooled_ind, p=2, dim=-1)
                norm_l2 = torch.norm(normed_cand - normed_ind, dim=-1)  # (B,)

                # 3. Raw L2 (kept for backwards compatibility)
                raw_l2 = torch.norm(pooled_cand - pooled_ind, dim=-1)  # (B,)

                cosine_np = cosine_dist.cpu().numpy()
                norm_l2_np = norm_l2.cpu().numpy()
                raw_l2_np = raw_l2.cpu().numpy()

                for i in range(batch_size):
                    if max_samples is not None and (samples_seen + i) >= max_samples:
                        break
                    img_id, cls_name = self._extract_id_class(metadata, i)
                    records.append({
                        'image_id': img_id,
                        'class_name': cls_name,
                        'layer_name': layer_name,
                        'cosine_distance': float(cosine_np[i]),
                        'normalized_l2': float(norm_l2_np[i]),
                        'raw_l2': float(raw_l2_np[i]),
                    })

            samples_seen += batch_size

        self._remove_hooks()
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Core: per-layer gradient magnitudes
    # ------------------------------------------------------------------

    def compute_gradient_magnitudes(
        self,
        dataloader,
        max_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        For each sample, compute per-layer gradient norms on the candidate model.
        """
        self.candidate_model.eval()
        records: List[dict] = []
        samples_seen = 0

        layer_params: OrderedDict = OrderedDict()
        for layer_name, module in self._candidate_layers.items():
            params = [p for p in module.parameters() if p.requires_grad]
            if params:
                layer_params[layer_name] = params

        for batch in tqdm(dataloader, desc="Gradient magnitudes"):
            images, labels, metadata = self._unpack_batch(batch)
            batch_size = images.size(0)

            if max_samples is not None and samples_seen >= max_samples:
                break

            for i in range(batch_size):
                if max_samples is not None and (samples_seen + i) >= max_samples:
                    break

                img = images[i:i+1].to(self.device, non_blocking=True)
                lbl = labels[i:i+1].to(self.device, non_blocking=True)

                self.candidate_model.zero_grad()

                with torch.enable_grad():
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            logits = self.candidate_model(img)
                            loss = F.cross_entropy(logits, lbl)
                    else:
                        logits = self.candidate_model(img)
                        loss = F.cross_entropy(logits, lbl)
                    loss.backward()

                img_id, cls_name = self._extract_id_class(metadata, i)

                for layer_name, params in layer_params.items():
                    total_norm_sq = 0.0
                    for p in params:
                        if p.grad is not None:
                            total_norm_sq += p.grad.data.float().norm().item() ** 2
                    grad_norm = float(np.sqrt(total_norm_sq))

                    records.append({
                        'image_id': img_id,
                        'class_name': cls_name,
                        'layer_name': layer_name,
                        'grad_norm': grad_norm,
                    })

            samples_seen += batch_size

        self.candidate_model.eval()
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Core: linear probing for memorization localization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_layer_features(
        self,
        dataloader,
        max_samples: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract pooled features from BOTH candidate and independent models
        at every hooked layer.

        Returns:
            dict with keys:
                - 'candidate_features': {layer_name: np.ndarray (N, D)}
                - 'independent_features': {layer_name: np.ndarray (N, D)}
                - 'diff_features': {layer_name: np.ndarray (N, D)}
                    (candidate - independent, for probing)
                - 'image_ids': list of str
                - 'class_names': list of str
        """
        self._register_hooks()
        cand_features = {ln: [] for ln in self._layer_names}
        ind_features = {ln: [] for ln in self._layer_names}
        image_ids = []
        class_names = []
        samples_seen = 0

        amp_ctx = torch.cuda.amp.autocast() if self.use_amp else _nullcontext()

        for batch in tqdm(dataloader, desc="Extracting features"):
            images, labels, metadata = self._unpack_batch(batch)
            batch_size = images.size(0)

            if max_samples is not None and samples_seen >= max_samples:
                break

            images = images.to(self.device, non_blocking=True)

            with amp_ctx:
                self._candidate_activations.clear()
                _ = self.candidate_model(images)

                self._independent_activations.clear()
                _ = self.independent_model(images)

            effective_bs = min(
                batch_size,
                (max_samples - samples_seen) if max_samples else batch_size,
            )

            for layer_name in self._layer_names:
                act_c = self._candidate_activations.get(layer_name)
                act_i = self._independent_activations.get(layer_name)
                if act_c is None or act_i is None:
                    continue
                pooled_c = self._pool_activation(act_c, layer_name).float()
                pooled_i = self._pool_activation(act_i, layer_name).float()
                cand_features[layer_name].append(
                    pooled_c[:effective_bs].cpu().numpy()
                )
                ind_features[layer_name].append(
                    pooled_i[:effective_bs].cpu().numpy()
                )

            for i in range(effective_bs):
                img_id, cls_name = self._extract_id_class(metadata, i)
                image_ids.append(img_id)
                class_names.append(cls_name)

            samples_seen += effective_bs

        self._remove_hooks()

        # Stack and compute diffs
        candidate_out = {}
        independent_out = {}
        diff_out = {}
        for ln in self._layer_names:
            if cand_features[ln] and ind_features[ln]:
                c = np.concatenate(cand_features[ln], axis=0)
                i = np.concatenate(ind_features[ln], axis=0)
                candidate_out[ln] = c
                independent_out[ln] = i
                diff_out[ln] = c - i

        return {
            'candidate_features': candidate_out,
            'independent_features': independent_out,
            'diff_features': diff_out,
            'image_ids': image_ids,
            'class_names': class_names,
        }

    def compute_linear_probing(
        self,
        dataloader,
        memorization_scores_df: pd.DataFrame,
        max_samples: Optional[int] = None,
        n_folds: int = 5,
    ) -> pd.DataFrame:
        """
        Linear probing: for each layer, train logistic regression on frozen
        features to predict high/low memorization (per-class median split).

        Probes THREE feature sources per layer:
          1. candidate_only: candidate model features
          2. diff: (candidate - independent) features — isolates memorization
          3. concat: [candidate; independent] concatenated

        Reports AUROC and R² for each, with stratified K-fold CV.

        Returns:
            DataFrame with columns per feature_source:
                [layer_name, feature_source, feat_dim, probe_auroc,
                 probe_auroc_std, ridge_r2, ridge_r2_std, n_samples]
        """
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score, accuracy_score, r2_score

        print("Extracting frozen features from both models...")
        extracted = self.extract_layer_features(dataloader, max_samples=max_samples)
        cand_feats = extracted['candidate_features']
        ind_feats = extracted['independent_features']
        diff_feats = extracted['diff_features']
        image_ids = extracted['image_ids']
        class_names = extracted['class_names']

        # Match memorization scores
        mem_lookup = memorization_scores_df.groupby('image_id')[
            'memorization_score'
        ].mean().to_dict()
        mem_scores = np.array([mem_lookup.get(iid, np.nan) for iid in image_ids])
        valid_mask = ~np.isnan(mem_scores)

        mem_scores = mem_scores[valid_mask]
        class_arr = np.array(class_names)[valid_mask]
        n_samples = len(mem_scores)

        if n_samples < 20:
            print(f"  WARNING: Only {n_samples} matched samples, skipping probing.")
            return pd.DataFrame()

        # Per-class median split for binary labels
        labels = np.zeros(n_samples, dtype=int)
        for cls in np.unique(class_arr):
            cls_mask = class_arr == cls
            cls_median = np.median(mem_scores[cls_mask])
            labels[cls_mask & (mem_scores >= cls_median)] = 1

        print(f"  Samples: {n_samples}, High-mem: {labels.sum()}, "
              f"Low-mem: {(1 - labels).sum()}")

        # Stratified K-fold using class+label for balanced folds
        strat_key = np.array([f"{c}_{l}" for c, l in zip(class_arr, labels)])
        strat_counts = pd.Series(strat_key).value_counts()
        tiny = set(strat_counts[strat_counts < n_folds].index)
        if tiny:
            strat_key = np.array([
                s if s not in tiny else f"_other_{s.split('_')[-1]}"
                for s in strat_key
            ])

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        results = []
        for layer_name in tqdm(self._layer_names, desc="Linear probing"):
            if layer_name not in cand_feats:
                continue

            X_cand = cand_feats[layer_name][valid_mask]
            X_ind = ind_feats[layer_name][valid_mask]
            X_diff = diff_feats[layer_name][valid_mask]
            X_concat = np.concatenate([X_cand, X_ind], axis=1)

            for source_name, X in [
                ('candidate_only', X_cand),
                ('diff', X_diff),
                ('concat', X_concat),
            ]:
                feat_dim = X.shape[1]
                # Hidden layer size: min(256, feat_dim) to keep MLP tractable
                hidden_size = min(256, feat_dim)

                fold_linear_aurocs = []
                fold_linear_accs = []
                fold_ridge_r2s = []
                fold_mlp_aurocs = []
                fold_mlp_accs = []
                fold_mlp_r2s = []

                for train_idx, test_idx in skf.split(X, strat_key):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_cls_train, y_cls_test = labels[train_idx], labels[test_idx]
                    y_reg_train, y_reg_test = mem_scores[train_idx], mem_scores[test_idx]

                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    # --- Linear probes ---
                    try:
                        clf = LogisticRegression(
                            C=1.0, max_iter=1000, solver='lbfgs',
                            class_weight='balanced', random_state=42,
                        )
                        clf.fit(X_train_s, y_cls_train)
                        y_prob = clf.predict_proba(X_test_s)[:, 1]
                        y_pred = clf.predict(X_test_s)
                        if len(np.unique(y_cls_test)) > 1:
                            fold_linear_aurocs.append(roc_auc_score(y_cls_test, y_prob))
                        fold_linear_accs.append(accuracy_score(y_cls_test, y_pred))
                    except Exception:
                        pass

                    try:
                        reg = Ridge(alpha=1.0)
                        reg.fit(X_train_s, y_reg_train)
                        fold_ridge_r2s.append(r2_score(y_reg_test, reg.predict(X_test_s)))
                    except Exception:
                        pass

                    # --- Non-linear (MLP) probes ---
                    try:
                        mlp_clf = MLPClassifier(
                            hidden_layer_sizes=(hidden_size,),
                            max_iter=500, early_stopping=True,
                            validation_fraction=0.15, random_state=42,
                            learning_rate_init=1e-3,
                        )
                        mlp_clf.fit(X_train_s, y_cls_train)
                        y_mlp_prob = mlp_clf.predict_proba(X_test_s)[:, 1]
                        y_mlp_pred = mlp_clf.predict(X_test_s)
                        if len(np.unique(y_cls_test)) > 1:
                            fold_mlp_aurocs.append(roc_auc_score(y_cls_test, y_mlp_prob))
                        fold_mlp_accs.append(accuracy_score(y_cls_test, y_mlp_pred))
                    except Exception:
                        pass

                    try:
                        mlp_reg = MLPRegressor(
                            hidden_layer_sizes=(hidden_size,),
                            max_iter=500, early_stopping=True,
                            validation_fraction=0.15, random_state=42,
                            learning_rate_init=1e-3,
                        )
                        mlp_reg.fit(X_train_s, y_reg_train)
                        fold_mlp_r2s.append(r2_score(y_reg_test, mlp_reg.predict(X_test_s)))
                    except Exception:
                        pass

                row = {
                    'layer_name': layer_name,
                    'feature_source': source_name,
                    'feat_dim': feat_dim,
                    'n_samples': n_samples,
                }
                # Linear probe metrics
                if fold_linear_aurocs:
                    row['probe_auroc'] = float(np.mean(fold_linear_aurocs))
                    row['probe_auroc_std'] = float(np.std(fold_linear_aurocs))
                if fold_linear_accs:
                    row['probe_accuracy'] = float(np.mean(fold_linear_accs))
                    row['probe_accuracy_std'] = float(np.std(fold_linear_accs))
                if fold_ridge_r2s:
                    row['ridge_r2'] = float(np.mean(fold_ridge_r2s))
                    row['ridge_r2_std'] = float(np.std(fold_ridge_r2s))
                # MLP probe metrics
                if fold_mlp_aurocs:
                    row['mlp_auroc'] = float(np.mean(fold_mlp_aurocs))
                    row['mlp_auroc_std'] = float(np.std(fold_mlp_aurocs))
                if fold_mlp_accs:
                    row['mlp_accuracy'] = float(np.mean(fold_mlp_accs))
                    row['mlp_accuracy_std'] = float(np.std(fold_mlp_accs))
                if fold_mlp_r2s:
                    row['mlp_r2'] = float(np.mean(fold_mlp_r2s))
                    row['mlp_r2_std'] = float(np.std(fold_mlp_r2s))

                results.append(row)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # High-level analysis entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        dataloader,
        memorization_scores_df: pd.DataFrame,
        max_samples: Optional[int] = None,
    ) -> Dict:
        """
        Run full layer-wise analysis with both global and per-class statistics.

        Returns dict with keys:
            - layer_distances: raw per-sample, per-layer distances
            - layer_summary: global high-mem vs low-mem comparison
            - correlation: global per-layer Spearman correlation
            - per_class_correlation: within-class correlations (removes
              rare-vs-common confound)
            - class_summary: per-class, per-layer mean distances
        """
        # --- 1. Feature distances ---
        layer_distances = self.compute_layer_distances(
            dataloader, max_samples=max_samples
        )

        if len(layer_distances) == 0:
            print("WARNING: No layer distances computed.")
            return self._empty_results(layer_distances)

        # --- 2. Merge memorization scores ---
        mem_lookup = memorization_scores_df.groupby('image_id')[
            'memorization_score'
        ].mean().to_dict()
        layer_distances['memorization_score'] = layer_distances['image_id'].map(
            mem_lookup
        )
        layer_distances = layer_distances.dropna(subset=['memorization_score'])

        if len(layer_distances) == 0:
            print("WARNING: No samples matched between dataloader and "
                  "memorization_scores_df.")
            return self._empty_results(layer_distances)

        # Use cosine_distance as the primary metric
        # (also keep feature_distance alias for backward compat with plots)
        layer_distances['feature_distance'] = layer_distances['cosine_distance']

        # --- 3. PER-CLASS median split (fixes rare-vs-common confound) ---
        # Within each class, split into high-mem and low-mem
        def _per_class_group(group_df):
            med = group_df['memorization_score'].median()
            group_df = group_df.copy()
            group_df['mem_group'] = np.where(
                group_df['memorization_score'] >= med,
                'high_mem', 'low_mem'
            )
            return group_df

        # Get unique sample-level class info for the split
        sample_classes = layer_distances.groupby('image_id').first()[
            ['class_name', 'memorization_score']
        ].reset_index()
        class_medians = sample_classes.groupby('class_name')[
            'memorization_score'
        ].median().to_dict()

        layer_distances['mem_group'] = layer_distances.apply(
            lambda row: 'high_mem'
            if row['memorization_score'] >= class_medians.get(row['class_name'], 0)
            else 'low_mem',
            axis=1,
        )

        # --- 4. Layer summary (mean cosine distance by group) ---
        layer_summary = self._compute_layer_summary(layer_distances)

        # --- 5. GLOBAL per-layer Spearman correlation ---
        correlation = self._compute_correlations(layer_distances)

        # --- 6. PER-CLASS per-layer Spearman correlation ---
        per_class_corr = self._compute_per_class_correlations(layer_distances)

        # --- 7. Class summary (per-class, per-layer distances) ---
        class_summary = self._compute_class_summary(layer_distances)

        return {
            'layer_distances': layer_distances,
            'layer_summary': layer_summary,
            'correlation': correlation,
            'per_class_correlation': per_class_corr,
            'class_summary': class_summary,
        }

    def _empty_results(self, layer_distances):
        return {
            'layer_distances': layer_distances,
            'layer_summary': pd.DataFrame(),
            'correlation': pd.DataFrame(),
            'per_class_correlation': pd.DataFrame(),
            'class_summary': pd.DataFrame(),
        }

    def _compute_layer_summary(self, layer_distances: pd.DataFrame) -> pd.DataFrame:
        """Compute mean distance per layer, broken down by high/low mem groups."""
        summary_rows: List[dict] = []
        for layer_name in self._layer_names:
            layer_df = layer_distances[layer_distances['layer_name'] == layer_name]
            if layer_df.empty:
                continue

            high = layer_df[layer_df['mem_group'] == 'high_mem']['cosine_distance']
            low = layer_df[layer_df['mem_group'] == 'low_mem']['cosine_distance']

            row = {
                'layer_name': layer_name,
                'mean_distance_all': layer_df['cosine_distance'].mean(),
                'std_distance_all': layer_df['cosine_distance'].std(),
                'mean_distance_high_mem': high.mean() if len(high) > 0 else np.nan,
                'mean_distance_low_mem': low.mean() if len(low) > 0 else np.nan,
                'n_high_mem': len(high),
                'n_low_mem': len(low),
            }

            if len(high) > 1 and len(low) > 1:
                pooled_std = np.sqrt(
                    ((len(high) - 1) * high.std()**2 +
                     (len(low) - 1) * low.std()**2) /
                    (len(high) + len(low) - 2)
                )
                row['cohens_d'] = (
                    (high.mean() - low.mean()) / pooled_std
                    if pooled_std > 0 else 0.0
                )
            else:
                row['cohens_d'] = np.nan

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def _compute_correlations(self, layer_distances: pd.DataFrame) -> pd.DataFrame:
        """Global per-layer Spearman correlation (M(x) vs cosine distance)."""
        corr_rows: List[dict] = []
        for layer_name in self._layer_names:
            layer_df = layer_distances[layer_distances['layer_name'] == layer_name]
            if len(layer_df) < 5:
                continue

            rho, pvalue = stats.spearmanr(
                layer_df['memorization_score'],
                layer_df['cosine_distance'],
            )
            corr_rows.append({
                'layer_name': layer_name,
                'spearman_rho': rho,
                'p_value': pvalue,
                'n_samples': len(layer_df),
                'significant': pvalue < 0.05,
            })

        return pd.DataFrame(corr_rows)

    def _compute_per_class_correlations(
        self, layer_distances: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Per-class, per-layer Spearman correlation.

        This is the KEY analysis that removes the rare-vs-common confound.
        For each class independently, we ask: "within this class, do samples
        with higher memorization show larger cosine distance between
        candidate and independent models at this layer?"

        Also computes a Fisher's combined p-value across classes.
        """
        rows: List[dict] = []

        for layer_name in self._layer_names:
            layer_df = layer_distances[layer_distances['layer_name'] == layer_name]

            per_class_rhos = []
            per_class_pvals = []
            per_class_ns = []

            for cls_name, cls_df in layer_df.groupby('class_name'):
                if len(cls_df) < 10:
                    continue
                rho, pval = stats.spearmanr(
                    cls_df['memorization_score'],
                    cls_df['cosine_distance'],
                )
                rows.append({
                    'layer_name': layer_name,
                    'class_name': cls_name,
                    'spearman_rho': rho,
                    'p_value': pval,
                    'n_samples': len(cls_df),
                    'significant': pval < 0.05,
                })
                if not np.isnan(rho):
                    per_class_rhos.append(rho)
                    per_class_pvals.append(pval)
                    per_class_ns.append(len(cls_df))

            # Weighted average rho across classes (weighted by sqrt(n))
            if per_class_rhos:
                weights = np.sqrt(per_class_ns)
                weighted_rho = np.average(per_class_rhos, weights=weights)

                # Fisher's method to combine p-values
                valid_pvals = [p for p in per_class_pvals if 0 < p < 1]
                if valid_pvals:
                    chi2_stat = -2 * sum(np.log(p) for p in valid_pvals)
                    combined_p = 1.0 - stats.chi2.cdf(chi2_stat, 2 * len(valid_pvals))
                else:
                    combined_p = np.nan

                rows.append({
                    'layer_name': layer_name,
                    'class_name': '__weighted_avg__',
                    'spearman_rho': weighted_rho,
                    'p_value': combined_p,
                    'n_samples': sum(per_class_ns),
                    'significant': combined_p < 0.05 if not np.isnan(combined_p) else False,
                })

        return pd.DataFrame(rows)

    def _compute_class_summary(self, layer_distances: pd.DataFrame) -> pd.DataFrame:
        """Per-class, per-layer mean cosine distances for high/low mem groups."""
        rows = []
        for (layer_name, cls_name), gdf in layer_distances.groupby(
            ['layer_name', 'class_name']
        ):
            high = gdf[gdf['mem_group'] == 'high_mem']['cosine_distance']
            low = gdf[gdf['mem_group'] == 'low_mem']['cosine_distance']
            rows.append({
                'layer_name': layer_name,
                'class_name': cls_name,
                'n_samples': len(gdf),
                'mean_high_mem': high.mean() if len(high) > 0 else np.nan,
                'mean_low_mem': low.mean() if len(low) > 0 else np.nan,
                'diff': (high.mean() - low.mean()) if len(high) > 0 and len(low) > 0 else np.nan,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save_results(self, results: Dict, output_dir: str | Path) -> None:
        """Save analysis results to CSV/JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for key, filename in [
            ('layer_distances', f'{self.model_name}_layer_distances.csv'),
            ('layer_summary', f'{self.model_name}_layer_summary.csv'),
            ('correlation', f'{self.model_name}_layer_correlation.csv'),
            ('per_class_correlation', f'{self.model_name}_per_class_correlation.csv'),
            ('class_summary', f'{self.model_name}_class_summary.csv'),
            ('probing', f'{self.model_name}_probing.csv'),
        ]:
            df = results.get(key, pd.DataFrame())
            if len(df) > 0:
                path = output_dir / filename
                df.to_csv(path, index=False)
                print(f"  Saved {key}: {path}")

        # JSON summary
        json_summary: Dict = {
            'model_name': self.model_name,
            'layers_analyzed': self._layer_names,
            'distance_metric': 'cosine_distance',
            'mem_split': 'per_class_median',
        }
        if 'layer_summary' in results and len(results['layer_summary']) > 0:
            json_summary['layer_summary'] = (
                results['layer_summary'].to_dict(orient='records')
            )
        if 'correlation' in results and len(results['correlation']) > 0:
            json_summary['layer_correlation'] = (
                results['correlation'].to_dict(orient='records')
            )
        if 'per_class_correlation' in results:
            weighted = results['per_class_correlation']
            weighted = weighted[weighted['class_name'] == '__weighted_avg__']
            if len(weighted) > 0:
                json_summary['per_class_weighted_correlation'] = (
                    weighted.to_dict(orient='records')
                )

        if 'probing' in results and len(results.get('probing', pd.DataFrame())) > 0:
            json_summary['probing'] = (
                results['probing'].to_dict(orient='records')
            )

        json_path = output_dir / f'{self.model_name}_layerwise_analysis.json'
        with open(json_path, 'w') as f:
            json.dump(json_summary, f, indent=2, default=_json_default)
        print(f"  Saved JSON summary: {json_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, object]:
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 3:
                return batch[0], batch[1], batch[2]
            elif len(batch) == 2:
                return batch[0], batch[1], None
        raise ValueError(f"Unexpected batch format: {type(batch)}")

    @staticmethod
    def _extract_id_class(metadata, idx: int) -> Tuple[str, str]:
        if metadata is None:
            return f'sample_{idx}', 'unknown'

        if isinstance(metadata, dict):
            img_id = metadata.get('image_id', [f'sample_{idx}'])
            cls_name = metadata.get('class_name', ['unknown'])
            if isinstance(img_id, (list, torch.Tensor)):
                img_id = img_id[idx] if idx < len(img_id) else f'sample_{idx}'
            if isinstance(cls_name, (list, torch.Tensor)):
                cls_name = cls_name[idx] if idx < len(cls_name) else 'unknown'
            return str(img_id), str(cls_name)

        if isinstance(metadata, (list, tuple)):
            if idx < len(metadata) and isinstance(metadata[idx], dict):
                m = metadata[idx]
                return str(m.get('image_id', f'sample_{idx}')), str(
                    m.get('class_name', 'unknown')
                )
            return f'sample_{idx}', 'unknown'

        return f'sample_{idx}', 'unknown'

    def __del__(self):
        self._remove_hooks()


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

class _nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
