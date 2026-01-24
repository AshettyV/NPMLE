#!/bin/bash
# Interactive session for debugging NPMLE experiments
# This allocates the node and drops you into a shell

# Source config for settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/npmle_config.sh"

echo "=============================================="
echo "Starting interactive session"
echo "=============================================="
echo "Partition: $PARTITION"
echo "Node:      $NODELIST"
echo "CPUs:      $CPUS_PER_TASK"
echo "Memory:    $MEM"
echo "Time:      $TIME"
echo ""
echo "Once connected, run these commands to test:"
echo "  module load miniforge"
echo "  which conda"
echo "  echo \$MAMBA_ROOT_PREFIX"
echo "  conda activate $CONDA_ENV"
echo "  which python"
echo "=============================================="
echo ""

salloc \
    --partition=$PARTITION \
    --nodelist=$NODELIST \
    --cpus-per-task=$CPUS_PER_TASK \
    --mem=$MEM \
    --time=1:00:00 \
    --job-name=npmle_debug
