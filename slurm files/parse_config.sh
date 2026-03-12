#!/bin/bash
# =============================================================================
# YAML Config Parser for Bash
# =============================================================================
# Parses npmle_config.yaml and exports variables for use in bash scripts
# Usage: source parse_config.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/npmle_config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Function to extract a value from YAML (simple key: value format)
yaml_get() {
    local key="$1"
    grep -E "^\s*${key}:" "$CONFIG_FILE" | head -1 | sed 's/.*:\s*//' | sed 's/"//g' | sed "s/'//g" | xargs
}

# Function to extract array values (returns space-separated)
yaml_get_array() {
    local key="$1"
    local in_array=0
    local values=""
    while IFS= read -r line; do
        if [[ "$line" =~ ^[[:space:]]*${key}: ]]; then
            in_array=1
            continue
        fi
        if [ $in_array -eq 1 ]; then
            if [[ "$line" =~ ^[[:space:]]*-[[:space:]] ]]; then
                val=$(echo "$line" | sed 's/.*-\s*//' | sed 's/"//g' | sed "s/'//g" | xargs)
                values="$values $val"
            elif [[ "$line" =~ ^[[:space:]]*[a-z_]+: ]] || [[ -z "$line" ]]; then
                break
            fi
        fi
    done < "$CONFIG_FILE"
    echo "$values" | xargs
}

# =============================================================================
# Export all config values as shell variables
# =============================================================================

# General experiment parameters
export N_WORKERS=$(yaml_get "n_workers")
export N_SIMS=$(yaml_get "n_sims")
export N_POINTS=$(yaml_get "n_points")
export N_MIN=$(yaml_get "n_min")
export N_MAX=$(yaml_get "n_max")
export EXPERIMENT=$(yaml_get "type")
export TRUE_MU=$(yaml_get "true_mu")

# SOS experiment parameters
export SOS_SI_MIN=$(yaml_get "si_min" | head -1)  # First occurrence is in sos section
export SOS_SI_MAX=$(yaml_get "si_max" | head -1)
export SOS_B=$(yaml_get "B")
export SOS_MU0=$(yaml_get "mu0")
export SOS_M_POWER=$(yaml_get "m_power")

# Two-point experiment parameters
export TPT_PRIOR_ATOMS=$(yaml_get_array "prior_atoms")
export TPT_PRIOR_WEIGHTS=$(yaml_get_array "prior_weights")

# Function to extract values from two_point section (more robust than awk range)
yaml_get_two_point() {
    local key="$1"
    local in_section=0
    while IFS= read -r line; do
        # Check if we're entering two_point section
        if [[ "$line" =~ ^two_point: ]]; then
            in_section=1
            continue
        fi
        # Check if we're leaving section (new top-level key)
        if [ $in_section -eq 1 ] && [[ "$line" =~ ^[a-z] ]]; then
            break
        fi
        # Look for our key within the section
        if [ $in_section -eq 1 ] && [[ "$line" =~ ^[[:space:]]*${key}: ]]; then
            echo "$line" | sed 's/.*:[[:space:]]*//' | sed 's/"//g' | sed "s/'//g" | xargs
            return
        fi
    done < "$CONFIG_FILE"
}

# Get si_min/si_max/weight_power/B from two_point section
TPT_SI_MIN_RAW=$(yaml_get_two_point "si_min")
TPT_SI_MAX_RAW=$(yaml_get_two_point "si_max")
TPT_WEIGHT_POWER_RAW=$(yaml_get_two_point "weight_power")
TPT_B_RAW=$(yaml_get_two_point "B")

# Export with defaults
export TPT_SI_MIN=${TPT_SI_MIN_RAW:-0.5}
export TPT_SI_MAX=${TPT_SI_MAX_RAW:-100.0}
export TPT_WEIGHT_POWER=${TPT_WEIGHT_POWER_RAW:-0}
export TPT_B=${TPT_B_RAW:-10.0}

# Paths
export BASE_DIR=$(yaml_get "base_dir")
export SLURM_LOG_DIR=$(yaml_get "slurm_log_dir")
export LATEST_LINK=$(yaml_get "latest_link")

# Conda
export CONDA_PATH=$(yaml_get "path")
export CONDA_ENV=$(yaml_get "env")
export PIP_PACKAGES=$(yaml_get_array "pip_packages")

# SLURM
export PARTITION=$(yaml_get "partition")
export NODELIST=$(yaml_get "nodelist")
export CPUS_PER_TASK=$(yaml_get "cpus_per_task")
export MEM=$(yaml_get "mem")
export TIME=$(yaml_get "time")
export JOB_NAME=$(yaml_get "job_name")

# =============================================================================
# Print loaded config (optional, for debugging)
# =============================================================================

print_config() {
    echo "=== NPMLE Configuration ==="
    echo ""
    echo "General Experiment:"
    echo "  N_WORKERS=$N_WORKERS"
    echo "  N_SIMS=$N_SIMS"
    echo "  N_POINTS=$N_POINTS"
    echo "  N_MIN=$N_MIN"
    echo "  N_MAX=$N_MAX"
    echo "  EXPERIMENT=$EXPERIMENT"
    echo "  TRUE_MU=$TRUE_MU"
    echo ""
    echo "SOS Experiment:"
    echo "  SOS_SI_MIN=$SOS_SI_MIN"
    echo "  SOS_SI_MAX=$SOS_SI_MAX"
    echo "  SOS_B=$SOS_B"
    echo "  SOS_MU0=$SOS_MU0"
    echo "  SOS_M_POWER=$SOS_M_POWER"
    echo ""
    echo "Two-Point Experiment:"
    echo "  TPT_PRIOR_ATOMS=$TPT_PRIOR_ATOMS"
    echo "  TPT_PRIOR_WEIGHTS=$TPT_PRIOR_WEIGHTS"
    echo "  TPT_SI_MIN=$TPT_SI_MIN"
    echo "  TPT_SI_MAX=$TPT_SI_MAX"
    echo "  TPT_WEIGHT_POWER=$TPT_WEIGHT_POWER"
    echo "  TPT_B=$TPT_B"
    echo ""
    echo "Paths:"
    echo "  BASE_DIR=$BASE_DIR"
    echo "  SLURM_LOG_DIR=$SLURM_LOG_DIR"
    echo "  LATEST_LINK=$LATEST_LINK"
    echo ""
    echo "Conda:"
    echo "  CONDA_PATH=$CONDA_PATH"
    echo "  CONDA_ENV=$CONDA_ENV"
    echo "  PIP_PACKAGES=$PIP_PACKAGES"
    echo ""
    echo "SLURM:"
    echo "  PARTITION=$PARTITION"
    echo "  NODELIST=$NODELIST"
    echo "  CPUS_PER_TASK=$CPUS_PER_TASK"
    echo "  MEM=$MEM"
    echo "  TIME=$TIME"
    echo "  JOB_NAME=$JOB_NAME"
    echo "==========================="
}

# If run directly (not sourced), print the config
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    print_config
fi
