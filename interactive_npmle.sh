#!/bin/bash
# Interactive session for debugging NPMLE experiments
# Usage:
#   ./interactive_npmle.sh          # Just allocate and get a shell
#   ./interactive_npmle.sh --python # Allocate and start Python
#   ./interactive_npmle.sh --run    # Allocate and run the experiment

# Source config for settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/parse_config.sh"

# Resource settings for debugging (lighter weight for faster allocation)
DEBUG_CPUS=4
DEBUG_MEM="8G"
DEBUG_TIME="1:00:00"

# Interactive run mode uses shorter time than batch jobs (partition limit)
INTERACTIVE_RUN_TIME="4:00:00"

# Parse arguments
MODE="shell"
if [ "$1" == "--python" ]; then
    MODE="python"
elif [ "$1" == "--run" ]; then
    MODE="run"
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  (none)     Allocate node and drop into shell"
    echo "  --python   Allocate node and start Python interpreter"
    echo "  --run      Allocate node and run the experiment"
    echo "  --help     Show this help message"
    exit 0
fi

echo "=============================================="
echo "Starting interactive session"
echo "=============================================="
echo "Partition: $PARTITION"
echo "Node:      $NODELIST"
if [ "$MODE" == "run" ]; then
    echo "CPUs:      $CPUS_PER_TASK (full resources for experiment)"
    echo "Memory:    $MEM"
    echo "Time:      $INTERACTIVE_RUN_TIME"
else
    echo "CPUs:      $DEBUG_CPUS (debug mode)"
    echo "Memory:    $DEBUG_MEM"
fi
echo "Mode:      $MODE"
echo ""

if [ "$MODE" == "shell" ]; then
    echo "Once connected, run these commands:"
    echo "  module load miniforge"
    echo "  source $CONDA_PATH/etc/profile.d/conda.sh"
    echo "  conda activate $CONDA_ENV"
    echo "  python"
    echo "=============================================="
    echo ""

    salloc \
        --partition=$PARTITION \
        --nodelist=$NODELIST \
        --ntasks=1 \
        --cpus-per-task=$DEBUG_CPUS \
        --mem=$DEBUG_MEM \
        --gpus-per-node=1 \
        --time=$DEBUG_TIME \
        --job-name=npmle_debug

elif [ "$MODE" == "python" ]; then
    echo "Will start Python after setup..."
    echo "=============================================="
    echo ""

    srun \
        --partition=$PARTITION \
        --nodelist=$NODELIST \
        --ntasks=1 \
        --cpus-per-task=$DEBUG_CPUS \
        --mem=$DEBUG_MEM \
        --gpus-per-node=1 \
        --time=$DEBUG_TIME \
        --job-name=npmle_python \
        --pty \
        bash -c '
            module load miniforge

            # Initialize conda - try multiple methods
            if command -v conda &> /dev/null; then
                echo "Found conda in PATH: $(which conda)"
                eval "$(conda shell.bash hook)"
            elif [ -n "$MAMBA_ROOT_PREFIX" ] && [ -f "$MAMBA_ROOT_PREFIX/etc/profile.d/conda.sh" ]; then
                echo "Found conda via MAMBA_ROOT_PREFIX"
                source "$MAMBA_ROOT_PREFIX/etc/profile.d/conda.sh"
            elif [ -f "'"$CONDA_PATH"'/etc/profile.d/conda.sh" ]; then
                source "'"$CONDA_PATH"'/etc/profile.d/conda.sh"
            elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
                source "$HOME/miniforge3/etc/profile.d/conda.sh"
            else
                echo "ERROR: Could not find conda"
                exit 1
            fi

            conda activate '"$CONDA_ENV"'
            pip install --upgrade '"$PIP_PACKAGES"'
            echo ""
            echo "Python environment ready. Starting Python..."
            echo "Working directory: $(pwd)"
            echo ""
            python
        '

elif [ "$MODE" == "run" ]; then
    # Generate output directory name with timestamp (before srun so we can reference it)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="$BASE_DIR/results_interactive_${TIMESTAMP}"
    LOG_FILE="${OUTPUT_DIR}/interactive_session.log"

    echo "Will run experiment interactively..."
    echo "=============================================="
    echo "Output directory: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "To monitor: tail -f $LOG_FILE"
    echo "=============================================="
    echo ""

    srun \
        --partition=$PARTITION \
        --nodelist=$NODELIST \
        --ntasks=1 \
        --cpus-per-task=$CPUS_PER_TASK \
        --mem=$MEM \
        --gpus-per-node=1 \
        --time=$INTERACTIVE_RUN_TIME \
        --job-name=npmle_interactive \
        --pty \
        bash -c '
            # Create output directory and log file
            mkdir -p '"$OUTPUT_DIR"'
            LOG_FILE='"$LOG_FILE"'

            # Logging function
            log_msg() {
                echo "[$(date '\''+%Y-%m-%d %H:%M:%S'\'')] $1" | tee -a $LOG_FILE
            }

            log_msg "=============================================="
            log_msg "NPMLE Interactive Experiment"
            log_msg "=============================================="
            log_msg "Node: $(hostname)"
            log_msg "Output directory: '"$OUTPUT_DIR"'"
            log_msg ""

            cd '"$SCRIPT_DIR"'

            log_msg "Loading miniforge module..."
            module load miniforge 2>&1 | tee -a $LOG_FILE

            # Initialize conda - try multiple methods for robustness
            log_msg "Initializing conda..."
            CONDA_FOUND=0

            # Method 1: Use conda shell hook (works if conda is in PATH after module load)
            if command -v conda &> /dev/null; then
                log_msg "Found conda in PATH: $(which conda)"
                eval "$(conda shell.bash hook)"
                CONDA_FOUND=1
            # Method 2: Check MAMBA_ROOT_PREFIX (common for miniforge)
            elif [ -n "$MAMBA_ROOT_PREFIX" ] && [ -f "$MAMBA_ROOT_PREFIX/etc/profile.d/conda.sh" ]; then
                log_msg "Found conda via MAMBA_ROOT_PREFIX: $MAMBA_ROOT_PREFIX"
                source "$MAMBA_ROOT_PREFIX/etc/profile.d/conda.sh"
                CONDA_FOUND=1
            # Method 3: Check CONDA_PREFIX
            elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
                log_msg "Found conda via CONDA_PREFIX: $CONDA_PREFIX"
                source "$CONDA_PREFIX/etc/profile.d/conda.sh"
                CONDA_FOUND=1
            # Method 4: Try config path as fallback
            elif [ -f "'"$CONDA_PATH"'/etc/profile.d/conda.sh" ]; then
                log_msg "Found conda at config path: '"$CONDA_PATH"'"
                source "'"$CONDA_PATH"'/etc/profile.d/conda.sh"
                CONDA_FOUND=1
            # Method 5: Common miniforge location
            elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
                log_msg "Found conda at $HOME/miniforge3"
                source "$HOME/miniforge3/etc/profile.d/conda.sh"
                CONDA_FOUND=1
            fi

            if [ $CONDA_FOUND -eq 0 ]; then
                log_msg "ERROR: Could not find conda installation"
                log_msg "MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX"
                log_msg "CONDA_PREFIX=$CONDA_PREFIX"
                log_msg "which conda=$(which conda 2>/dev/null || echo not found)"
                exit 1
            fi

            log_msg "Activating conda environment: '"$CONDA_ENV"'"
            conda activate '"$CONDA_ENV"' || {
                log_msg "ERROR: Failed to activate '"$CONDA_ENV"'"
                log_msg "Available environments:"
                conda env list 2>&1 | tee -a $LOG_FILE
                exit 1
            }

            # Verify activation worked
            if ! command -v python &> /dev/null; then
                log_msg "ERROR: python not found after conda activate"
                log_msg "Checking conda info..."
                conda info 2>&1 | tee -a $LOG_FILE
                exit 1
            fi
            log_msg "Python: $(which python)"

            log_msg "Installing/upgrading packages..."
            pip install --upgrade '"$PIP_PACKAGES"' 2>&1 | tee -a $LOG_FILE

            log_msg ""
            log_msg "Starting experiment..."
            log_msg ""

            python npmle_tests_parallel.py \
                --n-workers '"$N_WORKERS"' \
                --n-sims '"$N_SIMS"' \
                --n-points '"$N_POINTS"' \
                --n-min '"$N_MIN"' \
                --n-max '"$N_MAX"' \
                --output-dir '"$OUTPUT_DIR"' \
                --experiment '"$EXPERIMENT"' \
                --true-mu '"$TRUE_MU"' \
                --sos-si-min '"$SOS_SI_MIN"' \
                --sos-si-max '"$SOS_SI_MAX"' \
                --sos-B '"$SOS_B"' \
                --sos-mu0 '"$SOS_MU0"' \
                --sos-m-power '"$SOS_M_POWER"' \
                --tpt-prior-atoms '"$TPT_PRIOR_ATOMS"' \
                --tpt-prior-weights '"$TPT_PRIOR_WEIGHTS"' \
                --tpt-weight-power '"$TPT_WEIGHT_POWER"' \
                --tpt-si-min '"$TPT_SI_MIN"' \
                --tpt-si-max '"$TPT_SI_MAX"' \
                2>&1 | tee -a $LOG_FILE

            log_msg ""
            log_msg "Experiment completed!"
            log_msg "Results in: '"$OUTPUT_DIR"'"
        '
fi
