#!/bin/bash
# =============================================================================
# NPMLE Experiment Configuration
# =============================================================================
# This file is sourced by run_npmle.sh, submit_npmle.sh, and interactive_npmle.sh
# Edit settings here to change experiment parameters
# =============================================================================

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

# Number of parallel workers for multiprocessing
# Set to number of CPUs available, or less to leave headroom
N_WORKERS=100

# Number of Monte Carlo simulations per n value
# Higher = more accurate results but longer runtime
N_SIMS=100

# Number of sample size (n) values to test
# Points are logarithmically spaced between N_MIN and N_MAX
N_POINTS=15

# Log10 of minimum sample size
# e.g., N_MIN=2 means n_min = 10^2 = 100
N_MIN=2

# Log10 of maximum sample size
# e.g., N_MAX=5 means n_max = 10^5 = 100,000
N_MAX=5

# Which experiment to run
# Options: "sos" (subset-of-signals), "2pt" (two-point), "both"
EXPERIMENT="both"

# =============================================================================
# PATHS
# =============================================================================

# Base directory for all outputs
BASE_DIR="/home/jas10184/NPMLE"

# Directory for SLURM log files (.out and .err)
SLURM_LOG_DIR="${BASE_DIR}/slurm_logs"

# Symlink to the latest experiment results
LATEST_LINK="${BASE_DIR}/latest"

# =============================================================================
# CONDA ENVIRONMENT
# =============================================================================

# Conda installation path (set this if auto-detection fails)
# This is the root directory containing etc/profile.d/conda.sh
CONDA_PATH="/home/jas10184/.conda"

# Conda environment name to activate
CONDA_ENV="trial_env"

# Python packages to install/upgrade before running
# (space-separated list)
PIP_PACKAGES="numpy scipy matplotlib"

# =============================================================================
# SLURM SETTINGS
# =============================================================================
# Note: SBATCH directives in submit_npmle.sh cannot use variables
# If you change these, also update the #SBATCH lines in submit_npmle.sh

# Specific node to run on (leave empty for any available node)
NODELIST="node4500"

# Number of CPUs to request
CPUS_PER_TASK=100

# Memory to request
MEM="256G"

# Maximum wall time (HH:MM:SS)
TIME="24:00:00"

# Job name prefix
JOB_NAME="npmle_tests"
