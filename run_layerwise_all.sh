#!/bin/bash
# Run layer-wise memorization analysis on ALL experiments
# Includes: cosine distance, per-class correlation, linear probing, MLP probing
# Usage: nohup bash run_layerwise_all.sh > layerwise_analysis.log 2>&1 &

set -e
cd .

echo "============================================"
echo "Layer-Wise Memorization Analysis — Full Run"
echo "  Metrics: cosine distance, linear probe, MLP probe"
echo "  Started: $(date)"
echo "============================================"

# 5 baselines (4 models each = 20 runs)
for exp in ham1000_baseline odir5k_baseline kvasir_baseline chestxray_baseline retinal_oct_baseline; do
  echo ""
  echo ">>> $exp (4 models)"
  .venv/bin/python scripts/run_layerwise_analysis.py --experiment $exp --models resnet50,vit,mae,medsam --skip-gradient
done

# 2 distinctive experiments with 2 models each
for exp in ham1000_distinctive odir5k_distinctive; do
  echo ""
  echo ">>> $exp (resnet50,vit)"
  .venv/bin/python scripts/run_layerwise_analysis.py --experiment $exp --models resnet50,vit --skip-gradient
done

# 3 Retinal OCT distinctive experiments — SKIPPED (checkpoints deleted)
# for exp in retinal_oct_distinctive retinal_oct_distinctive_edge retinal_oct_distinctive_invert; do
#   .venv/bin/python scripts/run_layerwise_analysis.py --experiment $exp --models resnet50 --skip-gradient
# done

echo ""
echo "============================================"
echo "ALL DONE: $(date)"
echo "============================================"
