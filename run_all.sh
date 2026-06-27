#!/bin/bash
#===============================================================================
# Medical Memorization Study - Complete Pipeline
# Single entry point for full experiment run
#
# Stages:
#   1. Data download/preparation (ChestX-ray 50K, Retinal OCT)
#   2. Data preprocessing (HAM10000 + ChestX-ray + Retinal OCT splits)
#   3. Model training (4 models × 5 datasets × 2 variants each)
#   4. Memorization evaluation
#   5. MIA validation
#   6. Rarity analysis (rare vs common disease memorization)
#   7. ODIR-5K Distinctive feature experiment (grayscale rare classes)
#   8. Report generation
#
# Usage:
#   ./run_all.sh                      # FULL RUN: 30 epochs (publication-ready)
#   ./run_all.sh --epochs 1           # QUICK TEST: 1 epoch (verify pipeline works)
#   ./run_all.sh --test               # TEST MODE: 5 epochs (validate pipeline)
#   ./run_all.sh --skip-download      # Skip dataset downloads
#   ./run_all.sh --only ham10000      # Only HAM10000 experiment
#   ./run_all.sh --only chestxray     # Only ChestX-ray experiment
#   ./run_all.sh --only retinal_oct   # Only Retinal OCT experiment
#   ./run_all.sh --only odir5k        # Only ODIR-5K experiment
#   ./run_all.sh --only kvasir_capsule # Only Kvasir-Capsule experiment (extreme imbalance)
#   ./run_all.sh --models "resnet50,vit"  # Only specific models
#   ./run_all.sh --resume             # Resume from last checkpoint
#   ./run_all.sh --gpu-profile 16gb   # For 16GB GPUs (e.g., V100, T4)
#   ./run_all.sh --gpu-profile 40gb   # For A100 40GB
#   ./run_all.sh --gpu-profile 80gb   # For A100 80GB
#   ./run_all.sh --gpu-profile auto   # Auto-detect GPU memory (default)
#
# Hardware: Auto-configures batch size based on GPU memory
#
# RECOMMENDED: Run with --epochs 1 first to validate pipeline before long runs!
#
#===============================================================================
# NOHUP COMMANDS (copy-paste these for background execution):
#===============================================================================
#
# FULL PRODUCTION RUN (30 epochs, all datasets, all models):
#   nohup ./run_all.sh --gpu-profile 40gb --epochs 30  > 26janfull30 2>&1 &
#
# QUICK TEST (1 epoch, verify everything works):
#   nohup ./run_all.sh --epochs 1 > nohup_test_1epoch.log 2>&1 &
#
# SINGLE DATASET (HAM10000 only, 30 epochs):
#   nohup ./run_all.sh --only ham10000 > nohup_ham10000.log 2>&1 &
#
# SINGLE DATASET TEST (HAM10000, 1 epoch):
#   nohup ./run_all.sh --only ham10000 --epochs 1 > nohup_ham10000_test.log 2>&1 &
#
# KVASIR-CAPSULE (extreme imbalance test, 30 epochs):
#   nohup ./run_all.sh --only kvasir_capsule --epochs 30 > nohup_kvasir.log 2>&1 &
#
# SPECIFIC MODELS (ResNet50 and ViT only):
#   nohup ./run_all.sh --models "resnet50,vit" > nohup_resnet_vit.log 2>&1 &
#
# CHECK PROGRESS:
#   tail -f nohup_full_run.log
#   ps aux | grep run_all
#
#===============================================================================

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Main log file
MAIN_LOG="${LOG_DIR}/main.log"

# Python executable — use .venv if available, else fall back to system python
if [ -x "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
    PYTHON="python"
fi

# ChestX-ray download settings
CHESTXRAY_SAMPLES="${CHESTXRAY_SAMPLES:-50000}"
CHESTXRAY_MIN_PER_CLASS="${CHESTXRAY_MIN_PER_CLASS:-500}"

# Parse arguments
SKIP_DOWNLOAD=false
ONLY_DATASET=""
RESUME=false
SKIP_TRAINING=false
FORCE=false
GPU_PROFILE="auto"
TEST_MODE=false
CUSTOM_EPOCHS=""
MODELS=""
DEFAULT_EPOCHS=30  # Publication-ready default

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --only)
            ONLY_DATASET="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --gpu-profile)
            GPU_PROFILE="$2"
            shift 2
            ;;
        --test)
            TEST_MODE=true
            CUSTOM_EPOCHS=5
            shift
            ;;
        --epochs)
            CUSTOM_EPOCHS="$2"
            shift 2
            ;;
        -h|--help)
            echo ""
            echo "Medical Memorization Study - Complete Pipeline"
            echo ""
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "EPOCH OPTIONS:"
            echo "  --epochs <n>       Number of epochs (default: 30 for production)"
            echo "  --test             Quick test mode (5 epochs)"
            echo "  --epochs 1         Minimal test (1 epoch) - RECOMMENDED FIRST RUN"
            echo ""
            echo "DATASET OPTIONS:"
            echo "  --only <dataset>   Only run: ham10000, chestxray, retinal_oct, odir5k, or kvasir_capsule"
            echo "  --skip-download    Skip dataset downloads"
            echo "  --force            Force reprocessing of existing data"
            echo ""
            echo "MODEL OPTIONS:"
            echo "  --models <list>    Comma-separated: resnet50,vit,mae,medsam (default: all)"
            echo ""
            echo "GPU OPTIONS:"
            echo "  --gpu-profile <p>  16gb, 40gb, 80gb, or auto (default: auto)"
            echo ""
            echo "OTHER:"
            echo "  --skip-training    Skip model training (run analysis only)"
            echo "  --resume           Resume from last checkpoint"
            echo "  -h, --help         Show this help"
            echo ""
            echo "EXAMPLES:"
            echo "  ./run_all.sh --epochs 1                    # Quick 1-epoch test"
            echo "  ./run_all.sh --only ham10000 --epochs 1    # Test HAM10000 only"
            echo "  ./run_all.sh                               # Full 30-epoch run"
            echo "  ./run_all.sh --models resnet50,vit         # Only 2 models"
            echo ""
            echo "NOHUP (background execution):"
            echo "  nohup ./run_all.sh > nohup.log 2>&1 &"
            echo "  tail -f nohup.log  # monitor progress"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run './run_all.sh --help' for usage information"
            exit 1
            ;;
    esac
done

# Determine final epoch count
if [[ -n "$CUSTOM_EPOCHS" ]]; then
    EPOCHS=$CUSTOM_EPOCHS
else
    EPOCHS=$DEFAULT_EPOCHS
fi

# Build force flag for preprocessing
FORCE_FLAG=""
if [[ "$FORCE" == "true" ]]; then
    FORCE_FLAG="--force"
fi

#-------------------------------------------------------------------------------
# Helper functions for dataset selection (POSITIVE MATCHING)
#-------------------------------------------------------------------------------
should_run_ham10000() {
    [[ -z "$ONLY_DATASET" ]] || [[ "$ONLY_DATASET" == "ham10000" ]]
}

should_run_chestxray() {
    [[ -z "$ONLY_DATASET" ]] || [[ "$ONLY_DATASET" == "chestxray" ]]
}

should_run_retinal_oct() {
    [[ -z "$ONLY_DATASET" ]] || [[ "$ONLY_DATASET" == "retinal_oct" ]]
}

should_run_odir5k() {
    [[ -z "$ONLY_DATASET" ]] || [[ "$ONLY_DATASET" == "odir5k" ]]
}

should_run_kvasir_capsule() {
    [[ -z "$ONLY_DATASET" ]] || [[ "$ONLY_DATASET" == "kvasir_capsule" ]] || [[ "$ONLY_DATASET" == "kvasir" ]]
}

should_run_retinal_oct_distinctive() {
    [[ -z "$ONLY_DATASET" ]] || [[ "$ONLY_DATASET" == "retinal_oct" ]] || [[ "$ONLY_DATASET" == "retinal_oct_distinctive" ]]
}

#-------------------------------------------------------------------------------
# Data existence checks — skip datasets whose data is not downloaded
#-------------------------------------------------------------------------------
has_ham10000_data() {
    [ -d "data/raw/HAM10000" ] && [ "$(ls data/raw/HAM10000/*.jpg 2>/dev/null | wc -l)" -gt 0 ]
}

has_chestxray_data() {
    [ -d "data/chestxray_50k/images" ] && [ "$(ls data/chestxray_50k/images/*.png 2>/dev/null | wc -l)" -gt 0 ]
}

has_retinal_oct_data() {
    [ -d "data/retinal_oct/images" ] && [ "$(ls data/retinal_oct/images/*.jpeg 2>/dev/null | wc -l)" -gt 0 ]
}

has_odir5k_data() {
    [ -d "data/odir5k/images" ] && [ "$(ls data/odir5k/images/*.jpg 2>/dev/null | wc -l)" -gt 0 ]
}

has_kvasir_capsule_data() {
    [ -d "data/kvasir_capsule/images" ] && [ "$(ls data/kvasir_capsule/images/*.jpg 2>/dev/null | wc -l)" -gt 0 ]
}

#-------------------------------------------------------------------------------
# GPU Profile Configuration
#-------------------------------------------------------------------------------
detect_gpu_memory() {
    # Detect GPU memory in GB
    if command -v nvidia-smi &> /dev/null; then
        local mem_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [[ -n "$mem_mb" ]]; then
            echo $((mem_mb / 1024))
            return
        fi
    fi
    echo "0"
}

configure_gpu_profile() {
    local profile="$1"

    # Auto-detect if needed
    if [[ "$profile" == "auto" ]]; then
        local gpu_mem=$(detect_gpu_memory)
        if [[ "$gpu_mem" -ge 70 ]]; then
            profile="80gb"
        elif [[ "$gpu_mem" -ge 35 ]]; then
            profile="40gb"
        elif [[ "$gpu_mem" -ge 12 ]]; then
            profile="16gb"
        else
            profile="16gb"  # Default fallback
        fi
        log_info "Auto-detected GPU memory: ${gpu_mem}GB -> profile: $profile"
    fi

    # Set environment variables for optimal performance
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1

    # Memory allocation config (compatible with PyTorch 1.x and 2.x)
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

    # Configure based on profile
    # NOTE: Model-specific batch sizes are in config/gpu_profiles.yaml
    # These are loaded by run_experiment.py per model
    case $profile in
        80gb)
            # A100 80GB - Maximum throughput settings
            # OPTIMIZATIONS: cuDNN benchmark, TF32, no cache clearing
            export NUM_WORKERS=16
            export PREFETCH_FACTOR=4
            export AMP_ENABLED=true
            log_info "GPU Profile: 80GB (A100-80GB) - OPTIMIZED"
            log_info "  Model-specific batches: ResNet=256, ViT=128, MAE=96, MedSAM=32"
            log_info "  Workers: 16, Prefetch: 4, AMP: enabled"
            log_info "  OPTIMIZATIONS: cuDNN benchmark, TF32 matmul, cache clearing DISABLED"
            ;;
        40gb)
            # A100 40GB - Optimized settings
            # OPTIMIZATIONS: cuDNN benchmark, TF32, no cache clearing
            export NUM_WORKERS=12
            export PREFETCH_FACTOR=4
            export AMP_ENABLED=true
            log_info "GPU Profile: 40GB (A100-40GB) - OPTIMIZED"
            log_info "  Model-specific batches: ResNet=128, ViT=64, MAE=48, MedSAM=16"
            log_info "  Workers: 12, Prefetch: 4, AMP: enabled"
            log_info "  OPTIMIZATIONS: cuDNN benchmark, TF32 matmul, cache clearing DISABLED"
            ;;
        16gb)
            # V100/T4 16GB - Optimized conservative settings
            export NUM_WORKERS=8
            export PREFETCH_FACTOR=2
            export AMP_ENABLED=true
            log_info "GPU Profile: 16GB (V100/T4) - OPTIMIZED"
            log_info "  Model-specific batches: ResNet=64, ViT=32, MAE=24, MedSAM=8"
            log_info "  Workers: 8, Prefetch: 2, AMP: enabled"
            log_info "  OPTIMIZATIONS: cuDNN benchmark, reduced cache clearing"
            ;;
        *)
            log_error "Unknown GPU profile: $profile"
            log_error "Valid profiles: 16gb, 40gb, 80gb, auto"
            exit 1
            ;;
    esac

    GPU_PROFILE="$profile"
    export GPU_PROFILE
}

#-------------------------------------------------------------------------------
# Colors and formatting
#-------------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

#-------------------------------------------------------------------------------
# Logging functions
#-------------------------------------------------------------------------------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$msg" | tee -a "$MAIN_LOG"
}

log_header() {
    echo "" | tee -a "$MAIN_LOG"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$MAIN_LOG"
    echo -e "${BOLD}${CYAN}  $1${NC}" | tee -a "$MAIN_LOG"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$MAIN_LOG"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$MAIN_LOG"
}

log_error() {
    echo -e "${RED}✗ $1${NC}" | tee -a "$MAIN_LOG"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}" | tee -a "$MAIN_LOG"
}

log_info() {
    echo -e "${CYAN}ℹ $1${NC}" | tee -a "$MAIN_LOG"
}

#-------------------------------------------------------------------------------
# Timer functions
#-------------------------------------------------------------------------------
timer_start() {
    TIMER_START=$(date +%s)
}

timer_stop() {
    local end=$(date +%s)
    local duration=$((end - TIMER_START))
    local hours=$((duration / 3600))
    local minutes=$(( (duration % 3600) / 60 ))
    local seconds=$((duration % 60))
    echo "${hours}h ${minutes}m ${seconds}s"
}

#-------------------------------------------------------------------------------
# Experiment timing (per-experiment cumulative timers)
#-------------------------------------------------------------------------------
declare -A EXPERIMENT_TIMES
declare -a EXPERIMENT_ORDER=()

run_experiment_cmd() {
    local experiment_name="$1"
    shift
    # Register experiment if first time seen
    if [[ -z "${EXPERIMENT_TIMES[$experiment_name]+x}" ]]; then
        EXPERIMENT_ORDER+=("$experiment_name")
        EXPERIMENT_TIMES[$experiment_name]=0
    fi
    local _exp_start=$(date +%s)
    run_cmd "$@"
    local _rc=$?
    local _exp_end=$(date +%s)
    EXPERIMENT_TIMES[$experiment_name]=$(( ${EXPERIMENT_TIMES[$experiment_name]} + (_exp_end - _exp_start) ))
    return $_rc
}

format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    printf "%dh %02dm %02ds" "$hours" "$minutes" "$seconds"
}

#-------------------------------------------------------------------------------
# GPU monitoring
#-------------------------------------------------------------------------------
log_gpu_status() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null)
        log_info "GPU: $gpu_info"
    fi
}

#-------------------------------------------------------------------------------
# Verify Python environment
#-------------------------------------------------------------------------------
verify_python() {
    log "Using Python: $PYTHON"
    if ! "$PYTHON" -c "import torch" 2>/dev/null; then
        log_error "PyTorch not found in $PYTHON"
        log_error "Run: $PYTHON -m pip install -r requirements.txt"
        exit 1
    fi
    log_info "Python: $PYTHON"
    log_info "PyTorch: $($PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null)"
}

#-------------------------------------------------------------------------------
# Run command with logging
#-------------------------------------------------------------------------------
run_cmd() {
    local cmd="$1"
    local log_file="$2"
    local description="$3"

    log "Running: $description"
    log_info "Command: $cmd"
    log_info "Log file: $log_file"

    timer_start

    if eval "$cmd" > "$log_file" 2>&1; then
        local duration=$(timer_stop)
        log_success "Completed in $duration"
        return 0
    else
        local exit_code=$?
        log_error "Failed with exit code $exit_code"
        log_error "Check log file: $log_file"
        tail -20 "$log_file" | tee -a "$MAIN_LOG"
        return $exit_code
    fi
}

#===============================================================================
# MAIN PIPELINE
#===============================================================================

TOTAL_START=$(date +%s)

echo ""
echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║                    MEDICAL MEMORIZATION STUDY                                 ║${NC}"
echo -e "${BOLD}${BLUE}║                    Complete Pipeline Execution                                ║${NC}"
echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

log "Pipeline started at $(date)"
log "Log directory: $LOG_DIR"
log_gpu_status

verify_python

# Configure GPU profile (must be after conda activation for log functions)
configure_gpu_profile "$GPU_PROFILE"

#-------------------------------------------------------------------------------
# STAGE 1: Data Download
#-------------------------------------------------------------------------------
log_header "STAGE 1: DOWNLOAD DATASETS"

# HAM10000 download
if should_run_ham10000 && [[ "$SKIP_DOWNLOAD" == "false" ]]; then
    log "Checking HAM10000 data..."

    # Quick check: does the directory exist and have jpg files?
    if [ -d "data/raw/HAM10000" ]; then
        HAM_IMG_COUNT=$(ls data/raw/HAM10000/*.jpg 2>/dev/null | wc -l || echo "0")
        if [ "$HAM_IMG_COUNT" -gt 5000 ]; then
            log_info "HAM10000 data already exists ($HAM_IMG_COUNT images), skipping download"
        else
            log_warning "HAM10000 data incomplete ($HAM_IMG_COUNT images). Downloading..."
            log_info "Note: Requires Kaggle API credentials (kaggle.json)"
            run_cmd "$PYTHON scripts/download_ham10000.py" \
                "${LOG_DIR}/download_ham10000.log" \
                "HAM10000 download"
        fi
    else
        log_warning "HAM10000 data not found. Downloading..."
        log_info "Note: Requires Kaggle API credentials (kaggle.json)"
        run_cmd "$PYTHON scripts/download_ham10000.py" \
            "${LOG_DIR}/download_ham10000.log" \
            "HAM10000 download"
    fi
else
    if ! should_run_ham10000; then
        log_info "Skipping HAM10000 download (not selected)"
    fi
fi

# ChestX-ray download
if should_run_chestxray && [[ "$SKIP_DOWNLOAD" == "false" ]]; then
    log "Downloading ChestX-ray (${CHESTXRAY_SAMPLES} samples)..."

    if [ -d "data/chestxray_50k/images" ] && [ "$(ls data/chestxray_50k/images/*.png 2>/dev/null | wc -l)" -gt 40000 ]; then
        log_info "ChestX-ray data already exists, skipping download"
    else
        run_cmd "$PYTHON scripts/download_chestxray.py \
            --target-samples $CHESTXRAY_SAMPLES \
            --min-per-class $CHESTXRAY_MIN_PER_CLASS" \
            "${LOG_DIR}/download_chestxray.log" \
            "ChestX-ray download and stratified sampling"
    fi
else
    if ! should_run_chestxray; then
        log_info "Skipping ChestX-ray download (not selected)"
    fi
fi

# Retinal OCT download
if should_run_retinal_oct && [[ "$SKIP_DOWNLOAD" == "false" ]]; then
    log "Downloading Retinal OCT..."

    if [ -d "data/retinal_oct/images" ] && [ "$(ls data/retinal_oct/images/*.jpeg 2>/dev/null | wc -l)" -gt 50000 ]; then
        log_info "Retinal OCT data already exists, skipping download"
    else
        run_cmd "$PYTHON scripts/download_retinal_oct.py" \
            "${LOG_DIR}/download_retinal_oct.log" \
            "Retinal OCT download and processing"
    fi
else
    if ! should_run_retinal_oct; then
        log_info "Skipping Retinal OCT download (not selected)"
    fi
fi

# Kvasir-Capsule download
if should_run_kvasir_capsule && [[ "$SKIP_DOWNLOAD" == "false" ]]; then
    log "Downloading Kvasir-Capsule (47K images, 14 classes, extreme imbalance)..."

    if [ -d "data/kvasir_capsule/images" ] && [ "$(ls data/kvasir_capsule/images/*.jpg 2>/dev/null | wc -l)" -gt 45000 ]; then
        log_info "Kvasir-Capsule data already exists, skipping download"
    else
        run_cmd "$PYTHON scripts/download_kvasir_capsule.py" \
            "${LOG_DIR}/download_kvasir_capsule.log" \
            "Kvasir-Capsule download and processing"
    fi
else
    if ! should_run_kvasir_capsule; then
        log_info "Skipping Kvasir-Capsule download (not selected)"
    fi
fi

#-------------------------------------------------------------------------------
# STAGE 2: Data Preprocessing
#-------------------------------------------------------------------------------
log_header "STAGE 2: DATA PREPROCESSING"

# HAM10000
if should_run_ham10000; then
    if ! has_ham10000_data; then
        log_warning "HAM10000 data not found, skipping preprocessing"
    elif [[ "$FORCE" == "false" ]] && \
       ( [ -f "data/processed/ham10000/split_info.yaml" ] || \
         grep -q "split" data/raw/HAM10000/HAM10000_metadata.csv 2>/dev/null ); then
        log_info "HAM10000 splits already exist (use --force to reprocess)"
    else
        run_cmd "$PYTHON scripts/preprocess_data.py --dataset ham10000 $FORCE_FLAG" \
            "${LOG_DIR}/preprocess_ham10000.log" \
            "HAM10000 preprocessing"
    fi
fi

# ChestX-ray
if should_run_chestxray; then
    if ! has_chestxray_data; then
        log_warning "ChestX-ray data not found, skipping preprocessing"
    elif [[ "$FORCE" == "false" ]] && [ -f "data/chestxray_50k/split_info.yaml" ]; then
        log_info "ChestX-ray splits already exist (use --force to reprocess)"
    else
        run_cmd "$PYTHON scripts/preprocess_data.py --dataset chestxray $FORCE_FLAG" \
            "${LOG_DIR}/preprocess_chestxray.log" \
            "ChestX-ray preprocessing"
    fi
fi

# Retinal OCT (splits created by download script)
if should_run_retinal_oct; then
    if ! has_retinal_oct_data; then
        log_warning "Retinal OCT data not found, skipping preprocessing"
    elif [[ "$FORCE" == "false" ]] && [ -f "data/retinal_oct/split_info.yaml" ]; then
        log_info "Retinal OCT splits already exist (use --force to reprocess)"
    else
        run_cmd "$PYTHON scripts/preprocess_data.py --dataset retinal_oct $FORCE_FLAG" \
            "${LOG_DIR}/preprocess_retinal_oct.log" \
            "Retinal OCT preprocessing"
    fi
fi

# Kvasir-Capsule (splits created by download script)
if should_run_kvasir_capsule; then
    if ! has_kvasir_capsule_data; then
        log_warning "Kvasir-Capsule data not found, skipping preprocessing"
    elif [[ "$FORCE" == "false" ]] && [ -f "data/kvasir_capsule/split_info.yaml" ]; then
        log_info "Kvasir-Capsule splits already exist (use --force to reprocess)"
    else
        run_cmd "$PYTHON scripts/preprocess_data.py --dataset kvasir_capsule $FORCE_FLAG" \
            "${LOG_DIR}/preprocess_kvasir_capsule.log" \
            "Kvasir-Capsule preprocessing"
    fi
fi

#-------------------------------------------------------------------------------
# STAGE 3: Model Training
#-------------------------------------------------------------------------------
if [[ "$SKIP_TRAINING" == "false" ]]; then
    log_header "STAGE 3: MODEL TRAINING"

    # Display configuration
    echo ""
    log_info "═══════════════════════════════════════════════════════════"
    log_info "TRAINING CONFIGURATION"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Epochs: $EPOCHS"
    if [[ -n "$MODELS" ]]; then
        log_info "Models: $MODELS"
    else
        log_info "Models: mae, resnet50, vit, medsam (all)"
    fi
    log_info "GPU Profile: $GPU_PROFILE"
    log_info "Mixed precision: ENABLED"
    log_info "═══════════════════════════════════════════════════════════"
    echo ""

    if [[ "$EPOCHS" -eq 1 ]]; then
        log_warning "QUICK TEST MODE: Running with 1 epoch only"
        log_warning "This is for pipeline validation, not real results!"
    elif [[ "$EPOCHS" -lt 30 ]]; then
        log_warning "TEST MODE: Running with $EPOCHS epochs (less than recommended 30)"
    else
        log_success "PRODUCTION MODE: Running with $EPOCHS epochs"
    fi

    log_gpu_status

    # Build common training args
    # Note: Batch size is MODEL-SPECIFIC from config/gpu_profiles.yaml
    # GPU_PROFILE env var is set by configure_gpu_profile() and read by Python
    TRAIN_ARGS="--epochs $EPOCHS"
    if [[ -n "$NUM_WORKERS" ]]; then
        TRAIN_ARGS="$TRAIN_ARGS --num-workers $NUM_WORKERS"
    fi
    if [[ -n "$MODELS" ]]; then
        TRAIN_ARGS="$TRAIN_ARGS --models $MODELS"
    fi

    log_info "Training args: $TRAIN_ARGS"
    log_info "Model-specific batch sizes from GPU profile: $GPU_PROFILE"

    # HAM10000
    if should_run_ham10000; then
        if has_ham10000_data; then
            log_header "Training HAM10000 ($EPOCHS epochs)"
            run_experiment_cmd "HAM10000" "$PYTHON run_experiment.py --experiment ham1000_baseline $TRAIN_ARGS" \
                "${LOG_DIR}/train_ham10000.log" \
                "HAM10000 experiment"
        else
            log_warning "Skipping HAM10000 training: data not found at data/raw/HAM10000/"
        fi
    fi

    # ChestX-ray
    if should_run_chestxray; then
        if has_chestxray_data; then
            log_header "Training ChestX-ray ($EPOCHS epochs)"
            run_experiment_cmd "ChestX-ray" "$PYTHON run_experiment.py --experiment chestxray_baseline $TRAIN_ARGS" \
                "${LOG_DIR}/train_chestxray.log" \
                "ChestX-ray experiment"
        else
            log_warning "Skipping ChestX-ray training: data not found at data/chestxray_50k/images/"
        fi
    fi

    # Retinal OCT
    if should_run_retinal_oct; then
        if has_retinal_oct_data; then
            log_header "Training Retinal OCT ($EPOCHS epochs)"
            run_experiment_cmd "Retinal OCT" "$PYTHON run_experiment.py --experiment retinal_oct_baseline $TRAIN_ARGS" \
                "${LOG_DIR}/train_retinal_oct.log" \
                "Retinal OCT experiment"
        else
            log_warning "Skipping Retinal OCT training: data not found at data/retinal_oct/images/"
        fi
    fi

    # ODIR-5K
    if should_run_odir5k; then
        if has_odir5k_data; then
            log_header "Training ODIR-5K ($EPOCHS epochs)"
            run_experiment_cmd "ODIR-5K" "$PYTHON run_experiment.py --experiment odir5k_baseline $TRAIN_ARGS" \
                "${LOG_DIR}/train_odir5k.log" \
                "ODIR-5K experiment"
        else
            log_warning "Skipping ODIR-5K training: data not found at data/odir5k/images/"
        fi
    fi

    # Kvasir-Capsule (EXTREME IMBALANCE TEST - 3434:1 ratio)
    if should_run_kvasir_capsule; then
        if has_kvasir_capsule_data; then
            log_header "Training Kvasir-Capsule ($EPOCHS epochs) - EXTREME IMBALANCE TEST"
            log_info "14 classes with 3434:1 imbalance ratio - strongest hypothesis test"
            run_experiment_cmd "Kvasir-Capsule" "$PYTHON run_experiment.py --experiment kvasir_baseline $TRAIN_ARGS" \
                "${LOG_DIR}/train_kvasir_capsule.log" \
                "Kvasir-Capsule experiment"
        else
            log_warning "Skipping Kvasir-Capsule training: data not found at data/kvasir_capsule/images/"
        fi
    fi
else
    log_info "Skipping training (--skip-training)"
fi

#-------------------------------------------------------------------------------
# STAGE 4: MIA Validation (Membership Inference Attack)
#-------------------------------------------------------------------------------
log_header "STAGE 4: MIA VALIDATION (Membership Inference Attack)"
log_info "Validates memorization scores predict real privacy vulnerabilities"

# HAM10000
if should_run_ham10000; then
    if [ -d "results/ham1000_baseline/memorization" ]; then
        run_experiment_cmd "HAM10000" "$PYTHON scripts/run_mia_analysis.py --experiment ham1000_baseline" \
            "${LOG_DIR}/mia_ham10000.log" \
            "HAM10000 MIA validation"
    else
        log_warning "HAM10000 memorization results not found, skipping MIA"
    fi
fi

# ChestX-ray
if should_run_chestxray; then
    if [ -d "results/chestxray_baseline/memorization" ]; then
        run_experiment_cmd "ChestX-ray" "$PYTHON scripts/run_mia_analysis.py --experiment chestxray_baseline" \
            "${LOG_DIR}/mia_chestxray.log" \
            "ChestX-ray MIA validation"
    else
        log_warning "ChestX-ray memorization results not found, skipping MIA"
    fi
fi

# Retinal OCT
if should_run_retinal_oct; then
    if [ -d "results/retinal_oct_baseline/memorization" ]; then
        run_experiment_cmd "Retinal OCT" "$PYTHON scripts/run_mia_analysis.py --experiment retinal_oct_baseline" \
            "${LOG_DIR}/mia_retinal_oct.log" \
            "Retinal OCT MIA validation"
    else
        log_warning "Retinal OCT memorization results not found, skipping MIA"
    fi
fi

# ODIR-5K
if should_run_odir5k; then
    if [ -d "results/odir5k_baseline/memorization" ]; then
        run_experiment_cmd "ODIR-5K" "$PYTHON scripts/run_mia_analysis.py --experiment odir5k_baseline" \
            "${LOG_DIR}/mia_odir5k.log" \
            "ODIR-5K MIA validation"
    else
        log_warning "ODIR-5K memorization results not found, skipping MIA"
    fi
fi

# Kvasir-Capsule
if should_run_kvasir_capsule; then
    if [ -d "results/kvasir_baseline/memorization" ]; then
        run_experiment_cmd "Kvasir-Capsule" "$PYTHON scripts/run_mia_analysis.py --experiment kvasir_baseline" \
            "${LOG_DIR}/mia_kvasir_capsule.log" \
            "Kvasir-Capsule MIA validation"
    else
        log_warning "Kvasir-Capsule memorization results not found, skipping MIA"
    fi
fi

#-------------------------------------------------------------------------------
# STAGE 5: Rarity Analysis
#-------------------------------------------------------------------------------
log_header "STAGE 5: RARITY ANALYSIS (Rare vs Common Disease Memorization)"

# HAM10000
if should_run_ham10000; then
    if [ -d "results/ham1000_baseline/memorization" ]; then
        run_experiment_cmd "HAM10000" "$PYTHON scripts/analyze_rarity.py --experiment ham1000_baseline" \
            "${LOG_DIR}/rarity_ham10000.log" \
            "HAM10000 rarity analysis"
    else
        log_warning "HAM10000 memorization results not found, skipping rarity analysis"
    fi
fi

# ChestX-ray
if should_run_chestxray; then
    if [ -d "results/chestxray_baseline/memorization" ]; then
        run_experiment_cmd "ChestX-ray" "$PYTHON scripts/analyze_rarity.py --experiment chestxray_baseline" \
            "${LOG_DIR}/rarity_chestxray.log" \
            "ChestX-ray rarity analysis"
    else
        log_warning "ChestX-ray memorization results not found, skipping rarity analysis"
    fi
fi

# Retinal OCT
if should_run_retinal_oct; then
    if [ -d "results/retinal_oct_baseline/memorization" ]; then
        run_experiment_cmd "Retinal OCT" "$PYTHON scripts/analyze_rarity.py --experiment retinal_oct_baseline" \
            "${LOG_DIR}/rarity_retinal_oct.log" \
            "Retinal OCT rarity analysis"
    else
        log_warning "Retinal OCT memorization results not found, skipping rarity analysis"
    fi
fi

# ODIR-5K
if should_run_odir5k; then
    if [ -d "results/odir5k_baseline/memorization" ]; then
        run_experiment_cmd "ODIR-5K" "$PYTHON scripts/analyze_rarity.py --experiment odir5k_baseline" \
            "${LOG_DIR}/rarity_odir5k.log" \
            "ODIR-5K rarity analysis"
    else
        log_warning "ODIR-5K memorization results not found, skipping rarity analysis"
    fi
fi

# Kvasir-Capsule (CRITICAL - strongest test for rare disease hypothesis)
if should_run_kvasir_capsule; then
    if [ -d "results/kvasir_baseline/memorization" ]; then
        run_experiment_cmd "Kvasir-Capsule" "$PYTHON scripts/analyze_rarity.py --experiment kvasir_baseline" \
            "${LOG_DIR}/rarity_kvasir_capsule.log" \
            "Kvasir-Capsule rarity analysis (extreme imbalance test)"
    else
        log_warning "Kvasir-Capsule memorization results not found, skipping rarity analysis"
    fi
fi

#-------------------------------------------------------------------------------
# STAGE 6: Retinal OCT Distinctive Feature Experiment
#-------------------------------------------------------------------------------
if should_run_retinal_oct_distinctive; then
    log_header "STAGE 6: RETINAL OCT DISTINCTIVE FEATURE EXPERIMENT"
    log_info "Hypothesis: Visual distinctiveness amplifies memorization of DRUSEN + DME"
    log_info "Design: Rare classes (DRUSEN, DME) transformed at 3 levels"

    # --- Level 1: Grayscale ---
    log_header "STAGE 6a: Retinal OCT Distinctive Level 1 (Grayscale)"

    if [[ "$SKIP_TRAINING" == "false" ]] && has_retinal_oct_data; then
        DIST_TRAIN_ARGS="--models resnet50 --epochs $EPOCHS"
        if [[ -n "$NUM_WORKERS" ]]; then
            DIST_TRAIN_ARGS="$DIST_TRAIN_ARGS --num-workers $NUM_WORKERS"
        fi

        run_experiment_cmd "OCT Distinctive" "$PYTHON run_experiment.py --experiment retinal_oct_distinctive $DIST_TRAIN_ARGS" \
            "${LOG_DIR}/train_retinal_oct_distinctive.log" \
            "Retinal OCT Distinctive Grayscale training (ResNet50)"
    fi

    if [ -d "results/retinal_oct_distinctive/memorization" ]; then
        run_experiment_cmd "OCT Distinctive" "$PYTHON scripts/analyze_rarity.py --experiment retinal_oct_distinctive --models resnet50 --threshold 0.20" \
            "${LOG_DIR}/rarity_retinal_oct_distinctive.log" \
            "Retinal OCT Distinctive Grayscale rarity analysis"
    else
        log_warning "Retinal OCT Distinctive memorization results not found, skipping rarity analysis"
    fi

    # --- Level 2: Color Inversion ---
    log_header "STAGE 6b: Retinal OCT Distinctive Level 2 (Color Inversion)"

    if [[ "$SKIP_TRAINING" == "false" ]] && has_retinal_oct_data; then
        DIST_TRAIN_ARGS="--models resnet50 --epochs $EPOCHS"
        if [[ -n "$NUM_WORKERS" ]]; then
            DIST_TRAIN_ARGS="$DIST_TRAIN_ARGS --num-workers $NUM_WORKERS"
        fi

        run_experiment_cmd "OCT Distinctive" "$PYTHON run_experiment.py --experiment retinal_oct_distinctive_invert $DIST_TRAIN_ARGS" \
            "${LOG_DIR}/train_retinal_oct_distinctive_invert.log" \
            "Retinal OCT Distinctive Invert training (ResNet50)"
    fi

    if [ -d "results/retinal_oct_distinctive_invert/memorization" ]; then
        run_experiment_cmd "OCT Distinctive" "$PYTHON scripts/analyze_rarity.py --experiment retinal_oct_distinctive_invert --models resnet50 --threshold 0.20" \
            "${LOG_DIR}/rarity_retinal_oct_distinctive_invert.log" \
            "Retinal OCT Distinctive Invert rarity analysis"
    fi

    # --- Level 3: Edge Detection ---
    log_header "STAGE 6c: Retinal OCT Distinctive Level 3 (Edge Detection)"

    if [[ "$SKIP_TRAINING" == "false" ]] && has_retinal_oct_data; then
        DIST_TRAIN_ARGS="--models resnet50 --epochs $EPOCHS"
        if [[ -n "$NUM_WORKERS" ]]; then
            DIST_TRAIN_ARGS="$DIST_TRAIN_ARGS --num-workers $NUM_WORKERS"
        fi

        run_experiment_cmd "OCT Distinctive" "$PYTHON run_experiment.py --experiment retinal_oct_distinctive_edge $DIST_TRAIN_ARGS" \
            "${LOG_DIR}/train_retinal_oct_distinctive_edge.log" \
            "Retinal OCT Distinctive Edge training (ResNet50)"
    fi

    if [ -d "results/retinal_oct_distinctive_edge/memorization" ]; then
        run_experiment_cmd "OCT Distinctive" "$PYTHON scripts/analyze_rarity.py --experiment retinal_oct_distinctive_edge --models resnet50 --threshold 0.20" \
            "${LOG_DIR}/rarity_retinal_oct_distinctive_edge.log" \
            "Retinal OCT Distinctive Edge rarity analysis"
    fi

    # --- Multi-Level Comparison ---
    log_header "STAGE 6d: RETINAL OCT MULTI-LEVEL DISTINCTIVE COMPARISON"
    run_experiment_cmd "OCT Distinctive" "$PYTHON scripts/run_retinal_oct_distinctive.py --analysis-only" \
        "${LOG_DIR}/retinal_oct_distinctive_comparison.log" \
        "Retinal OCT multi-level distinctive comparison" || true

    # Visualization
    run_experiment_cmd "OCT Distinctive" "$PYTHON scripts/visualize_retinal_oct_distinctive.py" \
        "${LOG_DIR}/visualize_retinal_oct_distinctive.log" \
        "Retinal OCT distinctive visualization" || true
else
    log_info "Skipping Retinal OCT Distinctive experiment (not selected)"
fi

#-------------------------------------------------------------------------------
# STAGE 7: Report Generation
#-------------------------------------------------------------------------------
log_header "STAGE 7: REPORT GENERATION"

run_experiment_cmd "Report" "$PYTHON scripts/generate_report.py" \
    "${LOG_DIR}/generate_report.log" \
    "Summary report generation"

# Compile PDF if pdflatex available
if command -v pdflatex &> /dev/null; then
    log "Compiling PDF report..."
    pdflatex -interaction=nonstopmode summary_report.tex > "${LOG_DIR}/pdflatex.log" 2>&1 || {
        log_warning "PDF compilation had warnings (see ${LOG_DIR}/pdflatex.log)"
    }
    log_success "PDF generated: summary_report.pdf"
else
    log_warning "pdflatex not found. Run manually: pdflatex summary_report.tex"
fi

#===============================================================================
# SUMMARY
#===============================================================================
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(( (TOTAL_DURATION % 3600) / 60 ))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                          PIPELINE COMPLETE                                   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
log "Completed at: $(date)"
log "Total duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo -e "  Epochs:             $EPOCHS"
if [[ -n "$MODELS" ]]; then
    echo -e "  Models:             $MODELS"
else
    echo -e "  Models:             mae, resnet50, vit, medsam"
fi
echo -e "  GPU Profile:        $GPU_PROFILE"
echo ""

#-------------------------------------------------------------------------------
# Per-Experiment Timing Table
#-------------------------------------------------------------------------------
echo -e "${CYAN}${BOLD}Experiment Timing Summary:${NC}"
echo "" | tee -a "$MAIN_LOG"
_TABLE_SEP="+----------------------+----------------+"
echo "$_TABLE_SEP" | tee -a "$MAIN_LOG"
printf "| %-20s | %-14s |\n" "Experiment" "Duration" | tee -a "$MAIN_LOG"
echo "$_TABLE_SEP" | tee -a "$MAIN_LOG"

_TRACKED_TOTAL=0
for _exp_name in "${EXPERIMENT_ORDER[@]}"; do
    _exp_secs=${EXPERIMENT_TIMES[$_exp_name]}
    _TRACKED_TOTAL=$((_TRACKED_TOTAL + _exp_secs))
    printf "| %-20s | %-14s |\n" "$_exp_name" "$(format_duration $_exp_secs)" | tee -a "$MAIN_LOG"
done

echo "$_TABLE_SEP" | tee -a "$MAIN_LOG"
printf "| %-20s | %-14s |\n" "TOTAL (all exps)" "$(format_duration $_TRACKED_TOTAL)" | tee -a "$MAIN_LOG"
printf "| %-20s | %-14s |\n" "TOTAL (wall clock)" "$(format_duration $TOTAL_DURATION)" | tee -a "$MAIN_LOG"
echo "$_TABLE_SEP" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

echo -e "${CYAN}Results:${NC}"
echo -e "  HAM10000:           results/ham1000_baseline/"
echo -e "  ChestX-ray:         results/chestxray_baseline/"
echo -e "  Retinal OCT:        results/retinal_oct_baseline/"
echo -e "  ODIR-5K:            results/odir5k_baseline/"
echo -e "  OCT Distinctive:    results/retinal_oct_distinctive/"
echo -e "  OCT Invert:        results/retinal_oct_distinctive_invert/"
echo -e "  OCT Edge:          results/retinal_oct_distinctive_edge/"
echo -e "  Kvasir-Capsule:     results/kvasir_baseline/"
echo -e "  MIA Validation:     results/*/mia_validation/"
echo -e "  Rarity Analysis:    results/*/rarity_analysis/"
echo -e "  Report:             summary_report.pdf"
echo -e "  Logs:               $LOG_DIR/"
echo ""

# Final GPU status
log_gpu_status

echo -e "${GREEN}${BOLD}Done!${NC}"
