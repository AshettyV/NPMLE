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
        bash -c "
            module load miniforge
            source $CONDA_PATH/etc/profile.d/conda.sh
            conda activate $CONDA_ENV
            pip install --upgrade $PIP_PACKAGES
            echo ''
            echo 'Python environment ready. Starting Python...'
            echo 'Working directory: \$(pwd)'
            echo ''
            python
        "

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

            # Initialize conda using the same approach as --python mode
            log_msg "Initializing conda from: '"$CONDA_PATH"'/etc/profile.d/conda.sh"
            source "'"$CONDA_PATH"'/etc/profile.d/conda.sh"

            log_msg "Activating conda environment: '"$CONDA_ENV"'"
            conda activate '"$CONDA_ENV"'

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
