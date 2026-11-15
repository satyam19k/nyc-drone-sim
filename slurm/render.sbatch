#!/bin/bash
#SBATCH --job-name=ndsim-render
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# Submit with: sbatch --account=<account> --export=ALL,LOCATION="120 Broadway",RUN_ID=<id> slurm/render.sbatch
# Add VARIANT=street to the export list for the long street-following clips.
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
REPO=/scratch/sk12075/nyc-drone-sim
source "$REPO/env_sim/bin/activate"
cd "$REPO"
RUN_ID="${RUN_ID:-$SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
for ALT in ${ALTITUDES:-150 500}; do
    python -m nyc_drone_sim.sim.render --config configs/default.yaml \
        --location "${LOCATION:-120 Broadway}" --altitude "$ALT" --run "$RUN_ID" \
        ${VARIANT:+--variant "$VARIANT"}
done
