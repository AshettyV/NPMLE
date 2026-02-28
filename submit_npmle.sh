#!/bin/bash
#SBATCH --job-name=npmle_tests
#SBATCH --account torch_pr_732_general
#SBATCH --output=/home/jas10184/NPMLE/results/npmle_tests_%j.out
#SBATCH --error=/home/jas10184/NPMLE/results/npmle_tests_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=256G
#SBATCH --time=24:00:00


# =============================================================================
# NPMLE Tests - SLURM Submission Script
# =============================================================================
# Note: SBATCH directives above cannot use variables (parsed before script runs)
# Update them manually if SLURM settings change in npmle_config.sh
# =============================================================================

# Set working directory
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

# Source shared configuration (parses YAML config)
source "${WORK_DIR}/parse_config.sh"

# Create output and log directories
OUTPUT_DIR="${BASE_DIR}/results_${SLURM_JOB_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Log files
LOG_FILE="${LOG_DIR}/experiment.log"
STATUS_FILE="${LOG_DIR}/status.txt"
RESOURCE_LOG="${LOG_DIR}/resources.log"

# Print log locations immediately (visible in SLURM .out file)
echo "=============================================="
echo "NPMLE Experiment Starting"
echo "=============================================="
echo "Log file:      $LOG_FILE"
echo "Status file:   $STATUS_FILE"
echo "Resource log:  $RESOURCE_LOG"
echo "Output dir:    $OUTPUT_DIR"
echo ""
echo "To monitor: tail -f $LOG_FILE"
echo "=============================================="

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "==============================================================================" | tee -a "$LOG_FILE"
    log "$1"
    echo "==============================================================================" | tee -a "$LOG_FILE"
}

update_status() {
    echo "$1" > "$STATUS_FILE"
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')" >> "$STATUS_FILE"
    echo "Job ID: $SLURM_JOB_ID" >> "$STATUS_FILE"
    echo "Node: $SLURMD_NODENAME" >> "$STATUS_FILE"
    log "STATUS: $1"
}

log_resources() {
    {
        echo "=== Resource snapshot at $(date '+%Y-%m-%d %H:%M:%S') ==="
        echo "Memory usage:"
        free -h
        echo ""
        echo "CPU load:"
        uptime
        echo ""
        echo "GPU status (if nvidia-smi available):"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || echo "nvidia-smi not available"
        echo ""
    } >> "$RESOURCE_LOG"
}

# =============================================================================
# Start Logging
# =============================================================================

log_section "NPMLE Tests - Starting Job"
log "Job ID: $SLURM_JOB_ID"
log "Node: $SLURMD_NODENAME"
log "CPUs: $SLURM_CPUS_PER_TASK"
log "GPUs: $SLURM_GPUS"
log "Working directory: $WORK_DIR"
log "Output directory: $OUTPUT_DIR"
log "Log file: $LOG_FILE"

update_status "INITIALIZING"

# Create symlink to latest experiment for easy access
ln -sfn "$OUTPUT_DIR" "$LATEST_LINK"
log "Created symlink: $LATEST_LINK -> $OUTPUT_DIR"

# Log initial resources
log_resources

# Load required modules
log_section "Loading Modules"
update_status "LOADING_MODULES"
module load anaconda3/2025.06 2>&1 | tee -a "$LOG_FILE"

# Initialize conda for this shell
# (Required because non-interactive shells don't source .bashrc)
log "Searching for conda installation..."
log "  CONDA_PATH (from config)=$CONDA_PATH"

# Initialize conda directly using the conda path from config
log "Initializing conda from: $CONDA_PATH"
eval "$($CONDA_PATH/bin/conda shell.bash hook)"

# Check if the Conda environment exists, and create it if it doesn't
if ! $CONDA_PATH/bin/conda env list | grep -q "^$CONDA_ENV "; then
    log "Conda environment $CONDA_ENV not found. Creating it..."
    
    # Accept Conda ToS for required channels
    log "Accepting Conda Terms of Service..."
    $CONDA_PATH/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>&1 | tee -a "$LOG_FILE"
    $CONDA_PATH/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>&1 | tee -a "$LOG_FILE"
    
    log "Running: $CONDA_PATH/bin/conda create -n $CONDA_ENV -y --override-channels --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r python=3.9 numpy scipy matplotlib cvxpy"
    $CONDA_PATH/bin/conda create -n "$CONDA_ENV" -y --override-channels --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r python=3.9 numpy scipy matplotlib cvxpy 2>&1 | tee -a "$LOG_FILE"
    if [ $? -ne 0 ]; then
        log "ERROR: Failed to create Conda environment $CONDA_ENV"
        log "Please check the conda output above for details"
        exit 1
    fi
fi

# Activate conda environment
log "Activating conda environment: $CONDA_ENV"
$CONDA_PATH/bin/conda activate "$CONDA_ENV" || { log "ERROR: Failed to activate $CONDA_ENV"; exit 1; }

# Verify python is available
if ! command -v python &> /dev/null; then
    log "ERROR: python not found after activating conda environment"
    exit 1
fi

log "Python path: $(which python)"
log "Installing/upgrading packages: $PIP_PACKAGES"
pip install --upgrade $PIP_PACKAGES 2>&1 | tee -a "$LOG_FILE"

# Print Python and package info
log_section "Environment Info"
log "Python version: $(python3 --version 2>&1)"
log "NumPy version: $(python3 -c 'import numpy; print(numpy.__version__)' 2>&1)"

# =============================================================================
# Run the NPMLE tests
# =============================================================================

log_section "Starting NPMLE Experiment"
log "Configuration (from npmle_config.yaml):"
log "  General:"
log "    n-workers: $N_WORKERS"
log "    n-sims: $N_SIMS"
log "    n-points: $N_POINTS"
log "    n-min: $N_MIN (10^$N_MIN = $(python3 -c "print(10**$N_MIN)"))"
log "    n-max: $N_MAX (10^$N_MAX = $(python3 -c "print(10**$N_MAX)"))"
log "    experiment: $EXPERIMENT"
log "    true-mu: $TRUE_MU"
log "  SOS Experiment:"
log "    si-min: $SOS_SI_MIN"
log "    si-max: $SOS_SI_MAX"
log "    B: $SOS_B"
log "    mu0: $SOS_MU0"
log "    m-power: $SOS_M_POWER"
log "  Two-Point Experiment:"
log "    prior-atoms: $TPT_PRIOR_ATOMS"
log "    prior-weights: $TPT_PRIOR_WEIGHTS"
log "    weight-power: $TPT_WEIGHT_POWER"
log "    si-min: $TPT_SI_MIN"
log "    si-max: $TPT_SI_MAX"
log "    B: $TPT_B"

update_status "RUNNING_EXPERIMENT"

# Start background resource monitoring (every 5 minutes)
(
    while true; do
        sleep 300
        log_resources
    done
) &
MONITOR_PID=$!

# Run the experiment
START_TIME=$(date +%s)

python npmle_tests_parallel.py \
    --n-workers $N_WORKERS \
    --n-sims $N_SIMS \
    --n-points $N_POINTS \
    --n-min $N_MIN \
    --n-max $N_MAX \
    --output-dir "$OUTPUT_DIR" \
    --experiment $EXPERIMENT \
    --true-mu $TRUE_MU \
    --sos-si-min $SOS_SI_MIN \
    --sos-si-max $SOS_SI_MAX \
    --sos-B $SOS_B \
    --sos-mu0 $SOS_MU0 \
    --sos-m-power $SOS_M_POWER \
    --tpt-prior-atoms $TPT_PRIOR_ATOMS \
    --tpt-prior-weights $TPT_PRIOR_WEIGHTS \
    --tpt-weight-power $TPT_WEIGHT_POWER \
    --tpt-si-min $TPT_SI_MIN \
    --tpt-si-max $TPT_SI_MAX \
    --tpt-B $TPT_B \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Stop resource monitor
kill $MONITOR_PID 2>/dev/null

# Final resource snapshot
log_resources

# =============================================================================
# Job Completion
# =============================================================================

log_section "Job Completed"
log "Exit code: $EXIT_CODE"
log "Duration: $(printf '%dh %dm %ds' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60)))"

if [ $EXIT_CODE -eq 0 ]; then
    update_status "COMPLETED_SUCCESS"
    log "Experiment completed successfully!"
else
    update_status "COMPLETED_WITH_ERRORS (exit code: $EXIT_CODE)"
    log "Experiment completed with errors. Check logs for details."
fi

# List output files
log ""
log "Output files:"
ls -lh "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

# Print helpful commands for user
log ""
log "=============================================="
log "To monitor this experiment:"
log "  View status:    cat $STATUS_FILE"
log "  Tail logs:      tail -f $LOG_FILE"
log "  View resources: cat $RESOURCE_LOG"
log "  Latest results: ls -la $LATEST_LINK/"
log "=============================================="

# Copy results to a permanent location if needed
# cp -r "$OUTPUT_DIR" /path/to/permanent/storage/

exit $EXIT_CODE
