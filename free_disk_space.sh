#!/bin/bash
# Free home-directory space on NYU HPC (run from login node or interactive session).
# Usage: bash free_disk_space.sh

set -e
echo "=== Checking home usage ==="
du -sh "$HOME" 2>/dev/null || true
echo ""

# 1. Conda package cache (often 1–5+ GB)
if [[ -d "$HOME/.conda/pkgs" ]]; then
  echo "=== Clearing conda package cache (~/.conda/pkgs) ==="
  if command -v conda &>/dev/null; then
    conda clean --all -y
  elif [[ -x /share/apps/anaconda3/2024.02/bin/conda ]]; then
    /share/apps/anaconda3/2024.02/bin/conda clean --all -y
  elif [[ -x /share/apps/anaconda3/2025.06/bin/conda ]]; then
    /share/apps/anaconda3/2025.06/bin/conda clean --all -y
  else
    echo "Conda not found; deleting ~/.conda/pkgs contents manually (safe, packages can be re-downloaded)"
    rm -rf "$HOME/.conda/pkgs"/*
  fi
  echo "Done."
else
  echo "No ~/.conda/pkgs found, skipping."
fi
echo ""

# 2. Move old NPMLE results to scratch (if SCRATCH is set)
export SCRATCH=${SCRATCH:-/scratch/$USER}
if [[ -d "$SCRATCH" ]]; then
  echo "=== Moving old results to $SCRATCH/NPMLE_archive ==="
  mkdir -p "$SCRATCH/NPMLE_archive"
  for d in "$HOME/NPMLE"/results_*; do
    [[ -d "$d" ]] && mv "$d" "$SCRATCH/NPMLE_archive/" && echo "  Moved $d"
  done
  for f in "$HOME/NPMLE/results/"*.out "$HOME/NPMLE/results/"*.err 2>/dev/null; do
    [[ -f "$f" ]] && mv "$f" "$SCRATCH/NPMLE_archive/" && echo "  Moved $f"
  done
  echo "Done."
else
  echo "SCRATCH not found, skipping result move."
fi
echo ""

# 3. Show largest dirs in home (so you can delete more if needed)
echo "=== Largest items in home (consider deleting or moving to scratch) ==="
du -sh "$HOME"/* "$HOME"/.* 2>/dev/null | sort -hr | head -20
echo ""
echo "=== Home usage after cleanup ==="
du -sh "$HOME" 2>/dev/null || true
echo "Run 'myquota' to see quota status."
