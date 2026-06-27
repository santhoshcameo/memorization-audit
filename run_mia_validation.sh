#!/bin/bash
# =============================================================================
# MIA VALIDATION SCRIPT
# Membership Inference Attack Analysis for Rare Disease Privacy Hypothesis
# =============================================================================
#
# This script runs comprehensive MIA validation to prove:
# 1. Rare disease patients have higher membership inference vulnerability
# 2. Memorization score M(x) predicts MIA success
# 3. Different architectures have different vulnerability profiles
#

#
# Usage:
#   ./run_mia_validation.sh                    # Run all experiments
#   ./run_mia_validation.sh ham1000_baseline   # Run specific experiment
#   ./run_mia_validation.sh --quick            # Skip full MIA (faster)
#
# Requirements:
#   - Completed memorization measurement (run_all.sh must have been run)
#   - Python environment with required packages
#
# =============================================================================

set -e  # Exit on error

# Python executable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
    PYTHON="python"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Timestamp
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

echo -e "${PURPLE}"
echo "============================================================================="
echo "  MEMBERSHIP INFERENCE ATTACK (MIA) VALIDATION"
echo "  Validating Rare Disease Privacy Vulnerability Hypothesis"
echo "============================================================================="
echo -e "${NC}"
echo "Start time: $(date)"
echo "Project root: $PROJECT_ROOT"
echo ""

# Parse arguments
SPECIFIC_EXPERIMENT=""
QUICK_MODE=false
MODELS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        *)
            SPECIFIC_EXPERIMENT="$1"
            shift
            ;;
    esac
done

# Check if memorization results exist
check_results() {
    local exp=$1
    if [ -d "results/${exp}/memorization" ]; then
        local csv_count=$(ls -1 results/${exp}/memorization/*_memorization_scores.csv 2>/dev/null | wc -l)
        if [ "$csv_count" -gt 0 ]; then
            return 0
        fi
    fi
    return 1
}

# Run MIA analysis for an experiment
run_mia_analysis() {
    local experiment=$1

    echo -e "${CYAN}"
    echo "========================================"
    echo "  MIA Analysis: $experiment"
    echo "========================================"
    echo -e "${NC}"

    local cmd="$PYTHON scripts/run_mia_analysis.py --experiment $experiment"

    if [ "$QUICK_MODE" = true ]; then
        cmd="$cmd --skip-full-mia"
    fi

    if [ -n "$MODELS" ]; then
        cmd="$cmd --models $MODELS"
    fi

    echo "Running: $cmd"
    echo ""

    if $cmd; then
        echo -e "${GREEN}[SUCCESS]${NC} MIA analysis completed for $experiment"
        return 0
    else
        echo -e "${RED}[ERROR]${NC} MIA analysis failed for $experiment"
        return 1
    fi
}

# Main execution
main() {
    local failed=0
    local succeeded=0

    if [ -n "$SPECIFIC_EXPERIMENT" ]; then
        # Run specific experiment
        if check_results "$SPECIFIC_EXPERIMENT"; then
            if run_mia_analysis "$SPECIFIC_EXPERIMENT"; then
                succeeded=$((succeeded + 1))
            else
                failed=$((failed + 1))
            fi
        else
            echo -e "${RED}[ERROR]${NC} No memorization results found for: $SPECIFIC_EXPERIMENT"
            echo "Please run: $PYTHON run_experiment.py --experiment $SPECIFIC_EXPERIMENT"
            exit 1
        fi
    else
        # Run all experiments with results
        echo -e "${BLUE}Scanning for experiments with memorization results...${NC}"
        echo ""

        for exp_dir in results/*/memorization; do
            if [ -d "$exp_dir" ]; then
                experiment=$(basename $(dirname "$exp_dir"))
                echo -e "  Found: ${YELLOW}$experiment${NC}"
            fi
        done
        echo ""

        for exp_dir in results/*/memorization; do
            if [ -d "$exp_dir" ]; then
                experiment=$(basename $(dirname "$exp_dir"))

                echo ""
                if run_mia_analysis "$experiment"; then
                    succeeded=$((succeeded + 1))
                else
                    failed=$((failed + 1))
                fi
            fi
        done
    fi

    # Summary
    echo ""
    echo -e "${PURPLE}"
    echo "============================================================================="
    echo "  MIA VALIDATION COMPLETE"
    echo "============================================================================="
    echo -e "${NC}"
    echo "End time: $(date)"
    echo ""
    echo -e "Results: ${GREEN}$succeeded succeeded${NC}, ${RED}$failed failed${NC}"
    echo ""
    echo "Output locations:"

    if [ -n "$SPECIFIC_EXPERIMENT" ]; then
        echo "  results/${SPECIFIC_EXPERIMENT}/mia_validation/"
    else
        for exp_dir in results/*/mia_validation; do
            if [ -d "$exp_dir" ]; then
                echo "  $exp_dir/"
            fi
        done

        if [ -d "results/mia_validation_combined" ]; then
            echo ""
            echo "Combined results:"
            echo "  results/mia_validation_combined/"
        fi
    fi

    echo ""
    echo "Key files:"
    echo "  - mia_validation_results.json : Full analysis results"
    echo "  - cross_model_mia_comparison.csv : Model comparison table"
    echo "  - figures/*.png/pdf : Publication-ready figures"
    echo ""

    if [ "$failed" -gt 0 ]; then
        exit 1
    fi
}

# Print usage
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [OPTIONS] [EXPERIMENT]"
    echo ""
    echo "Run MIA validation analysis on memorization results."
    echo ""
    echo "Arguments:"
    echo "  EXPERIMENT        Specific experiment to analyze (e.g., ham1000_baseline)"
    echo ""
    echo "Options:"
    echo "  --quick           Skip full MIA attack analysis (faster)"
    echo "  --models MODELS   Comma-separated list of models to analyze"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Analyze all experiments"
    echo "  $0 ham1000_baseline          # Analyze specific experiment"
    echo "  $0 --quick                   # Quick analysis (no full MIA)"
    echo "  $0 --models resnet50,vit     # Only specific models"
    echo ""
    exit 0
fi

# Run main
main
