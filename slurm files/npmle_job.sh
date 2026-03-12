#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --account torch_pr_732_general

echo -n "this script is running on "
hostname