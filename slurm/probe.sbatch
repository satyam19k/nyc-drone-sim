#!/bin/bash
#SBATCH --job-name=ndsim-probe
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# Submit with: sbatch --account=<account> --export=ALL,LOCATION="120 Broadway" slurm/probe.sbatch
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
REPO=/scratch/sk12075/nyc-drone-sim
source "$REPO/env_sim/bin/activate"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m nyc_drone_sim.sim.probe --config configs/default.yaml \
    --location "${LOCATION:-120 Broadway}" --run "${RUN_ID:-$SLURM_JOB_ID}"
