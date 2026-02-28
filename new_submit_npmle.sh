#!/bin/bash
# =============================================================================
# NPMLE - SLURM submit script for NYU Torch HPC (follows HPC doc pattern)
# =============================================================================
# Prereq: Create conda env on scratch (see CONDA_SETUP.md). In npmle_config.yaml
#         set conda.env to the full path (e.g. $SCRATCH/conda_envs/trial_env).
# Edit:   --account to your project; change anaconda3/2024.02 if your cluster uses another version.
# =============================================================================

#SBATCH --job-name=npmle
#SBATCH --account=torch_pr_732_general
#SBATCH --output=results/npmle_%j.out
#SBATCH --error=results/npmle_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=256G
#SBATCH --time=24:00:00

set -e
WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"
mkdir -p results

# So you can see output: one file has both stdout and stderr
echo "=== NPMLE job started $(date) === Job ID: ${SLURM_JOB_ID:-N/A} === CWD: $(pwd) ==="

# Load config (npmle_config.yaml) and set OUTPUT_DIR
source "${WORK_DIR}/parse_config.sh"
# Ensure $SCRATCH is set so conda.env path expands (e.g. on NYU Torch, /scratch/NetID)
export SCRATCH=${SCRATCH:-/scratch/$USER}
OUTPUT_DIR="${BASE_DIR}/results_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Full path to conda env (NYU HPC doc: "replace <full path to penv> with full path to your penv directory")
# Set conda.env in npmle_config.yaml to this path (e.g. $SCRATCH/conda_envs/trial_env or /scratch/NetID/conda_envs/trial_env)
export PATH_TO_ENV=$(eval echo "$CONDA_ENV")
if [[ -z "$PATH_TO_ENV" || "$PATH_TO_ENV" != /* ]]; then
  echo "ERROR: conda.env must be the full path to your environment (e.g. /scratch/NetID/conda_envs/trial_env)." 1>&2
  echo "Current value: '${CONDA_ENV}' -> '${PATH_TO_ENV}'" 1>&2
  echo "When running via sbatch on the cluster, \$SCRATCH is set. When testing locally, use the literal path in npmle_config.yaml." 1>&2
  exit 1
fi

# Match NYU HPC documentation (https://services.rt.nyu.edu/docs/hpc/tools_and_software/conda_environments/)
module purge
module load anaconda3/2024.02
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source "${CONDA_PATH}/etc/profile.d/conda.sh"
source activate "$PATH_TO_ENV"
export PATH=$PATH_TO_ENV/bin:$PATH

# Run the experiment (all parameters from npmle_config.yaml)
echo "=== Starting python npmle_tests_parallel.py at $(date) ==="
python npmle_tests_parallel.py \
  --n-workers "$N_WORKERS" \
  --n-sims "$N_SIMS" \
  --n-points "$N_POINTS" \
  --n-min "$N_MIN" \
  --n-max "$N_MAX" \
  --output-dir "$OUTPUT_DIR" \
  --experiment "$EXPERIMENT" \
  --true-mu "$TRUE_MU" \
  --sos-si-min "$SOS_SI_MIN" \
  --sos-si-max "$SOS_SI_MAX" \
  --sos-B "$SOS_B" \
  --sos-mu0 "$SOS_MU0" \
  --sos-m-power "$SOS_M_POWER" \
  --tpt-prior-atoms $TPT_PRIOR_ATOMS \
  --tpt-prior-weights $TPT_PRIOR_WEIGHTS \
  --tpt-weight-power "$TPT_WEIGHT_POWER" \
  --tpt-si-min "$TPT_SI_MIN" \
  --tpt-si-max "$TPT_SI_MAX" \
  --tpt-B "$TPT_B"
