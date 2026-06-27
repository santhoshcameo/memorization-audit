# Memorization Audit Toolkit

Code accompanying the manuscript "Visual distinctiveness drives memorization
beyond rarity in fine-tuned medical imaging models" (submitted to npj Digital
Medicine).

## Repository layout
- `medmem/`   — one-command memorization-audit library (differential training,
               per-sample M(x) scoring, membership inference, statistics).
- `src/`      — experiment source (data loaders, models, training, evaluation,
               visualization).
- `scripts/`  — preprocessing and analysis utilities.
- `config/`   — experiment, dataset, and model configurations.
- `run_*.py`, `run_*.sh` — experiment and pipeline entry points.

## Installation
```bash
pip install -r requirements.txt
pip install -e .
pip install -e medmem/
```
Optional (medical-domain backbone): MedSAM weights are downloaded separately:
```bash
python download_medsam.py
```

## Models
Supported backbones (`--models`): `resnet50`, `vit`, `mae`, `medsam`, `biomedclip`
(comma-separated, e.g. `--models resnet50,vit,mae,medsam,biomedclip`).

---

## How the memorization score M(x) is computed

Memorization is measured with the gold-standard **differential (canary) training**
protocol. For each (dataset, model) pair two models are trained with **identical
initialization** (same architecture, deep-copied starting weights, identical seed
and training schedule); only the training data differs:

- **Candidate** model: trained on `train ∪ canary`.
- **Independent** model: trained on `train` only (never sees the canary samples).

For every canary sample `x`, the score is the drop in loss attributable to having
seen `x`:

```
M(x) = L_independent(x) − L_candidate(x)
```

where `L` is the per-sample cross-entropy. A large positive `M(x)` means the
candidate fits `x` much better than a model that never saw it — i.e. `x` was
memorized. To make the score robust, each sample is evaluated under `N`
random augmentations (default 5; set via `evaluation.memorization.num_augmentations`
in the experiment YAML), with the per-sample loss averaged across augmentations
before differencing. Images are denormalized before augmentation and renormalized
afterwards.

Risk thresholds (configurable): `M(x) > 0.3` high, `0.1–0.3` moderate, `≤ 0.1` low.

Implementation: `src/training/differential_trainer.py` (identical-init candidate /
independent training) and `src/evaluation/memorization.py`
(`MemorizationScorer`, augmentation batching, per-sample scoring). The same logic
is packaged in `medmem/engine.py`.

## Configuration (which YAML to pass)

You do **not** pass a YAML path directly — you pass an experiment **name** with
`--experiment <name>`, which loads `config/experiments/<name>.yaml`. Configuration
is assembled by merging four layers (later layers override earlier ones):

```
config/default.yaml            # global defaults (training, memorization,
                               #   reproducibility, paths, hardware, logging)
  + config/experiments/<name>.yaml   # the experiment (overrides everything)
  + config/datasets/<dataset>.yaml   # data paths, classes, split ratios
  + config/models/<model>.yaml       # per-backbone settings
```

(See `src/utils/config_loader.py`, `load_full_config()`.)

- `config/datasets/` — `ham10000`, `chestxray`, `odir5k`, `retinal_oct`,
  `kvasir_capsule`. Set each dataset's local `images:`/`metadata:` paths here.
- `config/models/` — `resnet50`, `vit`, `mae`, `medsam`.
- `config/experiments/` — one file per experiment (the value you give to
  `--experiment`); ties together `dataset`, `models`, `training.epochs`,
  `evaluation.memorization.*`, and (for distinctiveness runs) `distinctive_classes`.
- `config/gpu_profiles.yaml` — batch sizes per GPU memory tier (16/40/80 GB).

Example experiment YAML (`config/experiments/ham1000_distinctive.yaml`):
```yaml
name: ham1000_distinctive
dataset: ham10000             # -> config/datasets/ham10000.yaml
models: [resnet50, vit]       # -> config/models/{resnet50,vit}.yaml
distinctive_classes: [df, akiec]   # rare classes converted to grayscale
training:
  epochs: 30
  seed: 42
evaluation:
  memorization:
    enabled: true
    num_augmentations: 10     # N in M(x) averaging
    high_risk_threshold: 0.3
    moderate_risk_threshold: 0.1
```
Most parameters can also be overridden on the command line, e.g.
`--epochs`, `--models`, `--seed`, `--batch-size`, `--output` (see `--help`).

---

## 1. Data preparation (run first)
Each dataset must be split into train / canary / test before training:
```bash
python scripts/preprocess_data.py --dataset ham10000
python scripts/preprocess_data.py --dataset chestxray
python scripts/preprocess_data.py --dataset odir5k
python scripts/preprocess_data.py --dataset retinal_oct
python scripts/preprocess_data.py --dataset kvasir_capsule
# options: --seed 42 --split-ratio 0.7,0.15,0.15
```
The five datasets are public: HAM10000, NIH ChestX-ray, Kvasir-Capsule,
ODIR-5K, and Retinal OCT. Source URLs are listed in `config/datasets/*.yaml`;
set the local `images:`/`metadata:` paths there before preprocessing.

## 2. Run a single experiment
```bash
# Direct runner (one experiment, chosen models)
python run_experiment.py --experiment ham1000_baseline --models resnet50,vit

# Or the unified pipeline (training + evaluation + visualization)
python run_pipeline.py --experiment ham1000_baseline --models resnet50,vit --epochs 30
```
Available experiment configs (`config/experiments/`):
`ham1000_baseline`, `chestxray_baseline`, `odir5k_baseline`,
`retinal_oct_baseline`, `kvasir_baseline`, and their `*_distinctive`
counterparts (plus `retinal_oct_distinctive_edge` / `_invert`).

## 3. Run ALL experiments together
```bash
# Unified pipeline over every experiment (training -> eval -> viz)
python run_pipeline.py --all

# Only experiments not yet completed, including MIA + rarity analysis
python run_pipeline.py --pending --with-mia --with-rarity

# Pipeline status / dry-run test
python run_pipeline.py --status
python run_pipeline.py --test

# End-to-end shell driver (download -> preprocess -> train -> evaluate ->
# MIA -> rarity -> report); auto-selects GPU batch profile
./run_all.sh                          # full run (30 epochs)
./run_all.sh --test                   # quick validation (5 epochs)
./run_all.sh --only ham10000          # single dataset
./run_all.sh --models "resnet50,vit"  # subset of models
./run_all.sh --gpu-profile 40gb       # 16gb / 40gb / 80gb
```

## 4. Sub-experiments and analyses

**Visual-distinctiveness manipulation** (grayscale on rare classes):
```bash
python run_experiment.py --experiment ham1000_distinctive --models resnet50,vit
python scripts/compare_distinctive_vs_baseline.py \
    --experiment ham1000_distinctive --baseline ham1000_baseline --models resnet50,vit
```

**Membership inference (MIA) validation:**
```bash
./run_mia_validation.sh                    # all experiments
./run_mia_validation.sh ham1000_baseline   # one experiment
```

**Layer-wise memorization analysis:**
```bash
python scripts/run_layerwise_analysis.py \
    --experiment ham1000_baseline --models resnet50,vit,mae,medsam
# full sweep across all model-dataset pairs:
./run_layerwise_all.sh
```

**Rarity analysis** (rare vs common class memorization) is produced by the
pipeline with `--with-rarity` (see step 3).

---

## 5. medmem — one-command audit on your own dataset
For auditing an arbitrary image-classification dataset (no config needed):
```bash
medmem --data /path/to/images --name my_dataset             # folder-per-class
medmem --data /path/to/images --csv labels.csv --name my_dataset
medmem --data /path/to/images --name my_dataset --quick     # fast screening
```
Python API:
```python
import medmem
report = medmem.audit("/path/to/images", csv="labels.csv", name="my_dataset")
```
`medmem` runs the same differential-training -> M(x) -> MIA -> statistics
pipeline used throughout the paper and emits a per-class privacy-risk report.

## Outputs
Per experiment, results are written under `results/<experiment>/`:
trained models, per-sample memorization scores (CSV), MIA results (JSON),
rarity/statistical analyses, and figures.

## Acknowledgements
The differential-training and memorization-scoring implementation adapts code
from "Localizing Memorization in SSL Vision Encoders" (Wang, Dziedzic, Backes,
Boenisch, NeurIPS 2024), released at
https://github.com/sprintml/LocalizingMemorizationInSSL.
