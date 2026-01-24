#!/bin/bash
# Wrapper script to submit NPMLE job and display log locations

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source shared configuration (parses YAML config)
source "${SCRIPT_DIR}/parse_config.sh"

# Submit the job and capture the job ID
JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/submit_npmle.sh")

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit job"
    exit 1
fi

# Define paths for this job
OUTPUT_DIR="${BASE_DIR}/results_${JOB_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"

echo ""
echo "=============================================="
echo "NPMLE Job Submitted Successfully"
echo "=============================================="
echo "Job ID:        $JOB_ID"
echo ""
echo "Experiment configuration:"
echo "  Workers:     $N_WORKERS"
echo "  Simulations: $N_SIMS per n value"
echo "  N points:    $N_POINTS"
echo "  N range:     10^$N_MIN to 10^$N_MAX ($(printf "%.0f" $(echo "10^$N_MIN" | bc)) to $(printf "%.0f" $(echo "10^$N_MAX" | bc)))"
echo "  Experiment:  $EXPERIMENT"
echo ""
echo "Log locations (once job starts):"
echo "  Experiment log:  ${LOG_DIR}/experiment.log"
echo "  Status file:     ${LOG_DIR}/status.txt"
echo "  Resource log:    ${LOG_DIR}/resources.log"
echo "  SLURM output:    ${SLURM_LOG_DIR}/npmle_tests_${JOB_ID}.out"
echo "  SLURM error:     ${SLURM_LOG_DIR}/npmle_tests_${JOB_ID}.err"
echo ""
echo "Quick commands:"
echo "  Check status:    cat ${LATEST_LINK}/logs/status.txt"
echo "  Tail logs:       tail -f ${LATEST_LINK}/logs/experiment.log"
echo "  Job status:      squeue -j $JOB_ID"
echo "  Cancel job:      scancel $JOB_ID"
echo "=============================================="
