#!/bin/bash
# =============================================================================
# NPMLE Experiment Configuration
# =============================================================================
# This file is sourced by both run_npmle.sh and submit_npmle.sh
# Edit settings here to change experiment parameters

# Experiment parameters
N_WORKERS=120
N_SIMS=100
N_POINTS=15
N_MIN=2
N_MAX=5
EXPERIMENT="both"

# Paths
BASE_DIR="/home/shetty/orcd/scratch/npmle"
SLURM_LOG_DIR="${BASE_DIR}/slurm_logs"
LATEST_LINK="${BASE_DIR}/latest"

# SLURM settings
PARTITION="pi_rakhlin"
NODELIST="node4500"
CPUS_PER_TASK=120
MEM="256G"
TIME="24:00:00"
