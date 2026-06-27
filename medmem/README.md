# medmem

Memorization auditing for medical AI models — gold-standard differential
training pipeline.

`medmem` trains paired candidate/independent models on your image-classification
dataset, computes per-sample memorization scores
`M(x) = L_independent(x) - L_candidate(x)`, runs membership-inference attacks, and
emits a per-class privacy-risk report (Spearman, Mann-Whitney, Cohen's d, MIA AUC).

## Install
```bash
pip install -e .
```

## CLI
```bash
medmem --data /path/to/images --name my_dataset            # folder-per-class
medmem --data /path/to/images --csv labels.csv --name my_dataset
medmem --data /path/to/images --name my_dataset --quick    # fast screening
```

## Python API
```python
import medmem
report = medmem.audit("/path/to/images", csv="labels.csv", name="my_dataset")
```

Part of the code release for "Visual distinctiveness drives memorization beyond
rarity in fine-tuned medical imaging models."
