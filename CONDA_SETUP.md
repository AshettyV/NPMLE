# Step-by-step: Conda env on scratch (avoid "Disk quota exceeded")

Use this on **NYU HPC (Torch)** or any cluster where home has a small quota. Scratch on NYU is `/scratch/$USER` (5 TB); home is 50 GB with 30K inodes—conda in home often hits "Disk quota exceeded" or inode limits.

---

## Step 1: Log in and start an interactive session

```bash
ssh your_netid@torch.hpc.nyu.edu
# If your cluster uses interactive nodes:
srun --pty bash
```

---

## Step 2: Set scratch and (optional) conda cache

```bash
# On NYU HPC, $SCRATCH is usually set to /scratch/$USER. If not:
export SCRATCH=/scratch/$USER

# Optional but recommended: put conda's package cache on scratch so downloads don't fill home
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
mkdir -p $SCRATCH/conda_pkgs
```

---

## Step 3: Load conda and create the env on scratch

```bash
# Load the conda module (adjust name/version if your cluster differs)
module load anaconda3/2025.06

# Create directories and env on scratch (not in home)
mkdir -p $SCRATCH/conda_envs
conda create --prefix $SCRATCH/conda_envs/trial_env -y python=3.9 numpy scipy matplotlib cvxpy
```

Wait until `conda create` finishes (can take 10–20 minutes). When it’s done, you can leave the interactive session (`exit`).

---

## Step 4: Edit `npmle_config.yaml` so jobs use the scratch env

Open `npmle_config.yaml` in the NPMLE repo (e.g. under `~/NPMLE` or `/home/jas10184/NPMLE`).

Find the **conda** section and set **`env`** to the **full path** of the env you created.

**Option A – use `$SCRATCH` (recommended if your jobs have `SCRATCH` set):**

```yaml
conda:
  path: "/share/apps/anaconda3/2025.06"
  env: "$SCRATCH/conda_envs/trial_env"
```

**Option B – use a literal path (if `$SCRATCH` isn’t set in the job):**

Replace `YOUR_NETID` with your NYU NetID (e.g. `jas10184`):

```yaml
conda:
  path: "/share/apps/anaconda3/2025.06"
  env: "/scratch/YOUR_NETID/conda_envs/trial_env"
```

Save the file.

---

## Step 5: Submit the job

From the directory that contains `submit_npmle.sh` and `npmle_config.yaml`:

```bash
cd ~/NPMLE
sbatch submit_npmle.sh
```

The script will activate the env at the path you set (no env creation in the job), then run `npmle_tests_parallel.py`.

---

## Quick reference

| Step | What to do |
|------|------------|
| 1 | SSH + (if needed) `srun --pty bash` |
| 2 | `export SCRATCH=/scratch/$USER` and optionally `CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs` |
| 3 | `module load anaconda3/2025.06` then `conda create --prefix $SCRATCH/conda_envs/trial_env -y python=3.9 numpy scipy matplotlib cvxpy` |
| 4 | In `npmle_config.yaml`, set `conda.env` to `"$SCRATCH/conda_envs/trial_env"` or `"/scratch/YOUR_NETID/conda_envs/trial_env"` |
| 5 | `cd ~/NPMLE` and `sbatch submit_npmle.sh` |

---

## Notes (NYU scratch)

- **No backups** – If the env is lost, run Step 3 again.
- **60-day purging** – Scratch may purge files not accessed for 60+ days. If the env disappears after a long break, recreate it with Step 3 and keep Step 4 as-is.
- Keep your **code and config** (this repo, `npmle_config.yaml`) in **$HOME**; use scratch only for the conda env and large run/result data.
