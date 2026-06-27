"""
medmem — Memorization auditing for medical AI models.

Gold-standard differential training pipeline: trains candidate and independent
models, computes per-sample M(x) = Loss_independent - Loss_candidate, runs
membership inference attacks, and provides statistical evidence for memorization
risk per class.

Quick start:
    import medmem
    report = medmem.audit("/path/to/images", name="skin_lesions")

CLI:
    medmem --data /path/to/images --name skin_lesions
    medmem --data /path/to/images --csv labels.csv --name chest_xray --quick
"""

__version__ = '1.0.0'

from .pipeline import audit, AuditReport
from .data import ingest_folder, ingest_csv, stratified_split, ImageClassificationDataset
from .engine import train_model, compute_memorization, run_mia, compute_statistics
