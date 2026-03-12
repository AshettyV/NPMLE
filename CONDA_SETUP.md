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

---

## "Disk quota exceeded" (e.g. when running `git pull`)

This means your **home directory** is over its 50 GB or 30K-inode limit. Git writes into `.git/` in your repo (in home), so `git pull` fails until you free space.

### 1. Check how much you're using

On NYU HPC:

```bash
myquota
```

Or:

```bash
du -sh $HOME
du -sh $HOME/.conda
```

### 2. Free space in home

**Conda package cache (often the biggest):**

If you ever ran conda from home, the cache is in `~/.conda/pkgs` and can be several GB. Clear it (you can re-download packages when needed).

Bash doesn't know `conda` until you load it or use the full path. Use **one** of these:

```bash
# Option A: Load the module, then run conda (NYU HPC)
module load anaconda3/2025.06
conda clean --all -y
```

```bash
# Option B: Use the full path so you don't need to load conda (works even if "conda: command not found")
/share/apps/anaconda3/2025.06/bin/conda clean --all -y
```

If your cluster uses a different path, replace it with the one in your `npmle_config.yaml` under `conda.path`.

**Old run outputs and logs:**

Move large result/log directories to scratch so home stays under quota:

```bash
# Example: move old result dirs to scratch
mkdir -p $SCRATCH/NPMLE_archive
mv ~/NPMLE/results_* $SCRATCH/NPMLE_archive/
mv ~/NPMLE/results/*.out $SCRATCH/NPMLE_archive/ 2>/dev/null
```

Delete anything you don't need from `~/NPMLE/results/` and `~/NPMLE/results_*/`.

**Other large files:**

```bash
du -sh $HOME/* $HOME/.* 2>/dev/null | sort -h
```

Move or delete large items you don't need (e.g. old envs in `~/.conda/envs` if you're now using the scratch env).

### 3. Retry `git pull`

After freeing enough space:

```bash
cd ~/NPMLE
git pull
```

### 4. Avoid filling home again

- Keep the conda env and package cache on **scratch** (Steps 2–3 above).
- Keep **job outputs** (e.g. `results_*`, large `.out`/`.pkl` files) on scratch or archive them there; only keep the repo and configs in home.
- Periodically run `conda clean --all -y` if you use conda from home at all.
