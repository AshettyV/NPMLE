#!/bin/bash
#SBATCH --job-name=npmle_tests
#SBATCH --output=npmle_tests_%j.out
#SBATCH --error=npmle_tests_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --gpus=8
#SBATCH --gres=gpu:H200:8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# =============================================================================
# NPMLE Tests - SLURM Submission Script
# Node specification: 120 CPUs, 8 H200 GPUs
# =============================================================================

echo "=============================================="
echo "NPMLE Tests - Starting Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "=============================================="

# Load required modules (adjust based on your cluster's module system)
# Uncomment and modify as needed:
# module purge
# module load python/3.10
# module load cuda/12.0
# module load anaconda3

# Activate virtual environment if using one
# source /path/to/your/venv/bin/activate
# OR for conda:
# conda activate your_env_name

# Set working directory
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

# Create output directory with job ID
OUTPUT_DIR="./results_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Working directory: $WORK_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Print Python and package info
echo "Python version:"
python3 --version
echo ""
echo "NumPy version:"
python3 -c "import numpy; print(numpy.__version__)"
echo ""

# =============================================================================
# Run the NPMLE tests
# =============================================================================
#
# Configuration options:
#   --n-workers    : Number of parallel workers (default: all CPUs)
#   --n-sims       : Number of simulations per n value (default: 100)
#   --n-points     : Number of n values to test (default: 15)
#   --n-min        : Log10 of minimum n (default: 2, i.e., n=100)
#   --n-max        : Log10 of maximum n (default: 5, i.e., n=100000)
#   --output-dir   : Output directory for results
#   --experiment   : Which experiment to run (sos, 2pt, both)
#
# Note: This code is CPU-bound (uses scipy/numpy). The GPUs on this node
# are available but not utilized by the current implementation.
# For GPU acceleration, consider porting to JAX or PyTorch.
# =============================================================================

echo "Starting NPMLE tests..."
echo ""

python3 npmle_tests_parallel.py \
    --n-workers 120 \
    --n-sims 100 \
    --n-points 15 \
    --n-min 2 \
    --n-max 5 \
    --output-dir "$OUTPUT_DIR" \
    --experiment both

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================="

# Copy results to a permanent location if needed
# cp -r "$OUTPUT_DIR" /path/to/permanent/storage/

exit $EXIT_CODE
