#!/bin/bash
#SBATCH --job-name=ndsim-shots
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# Submit with: sbatch --account=<account> --export=ALL,LOCATION="120 Broadway" slurm/populate_probe.sbatch
# Renders the shot lists written by `python -m nyc_drone_sim.shotlist`. Set RENDER_RUN=<id> to
# also render the clips afterwards in the same job.
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
REPO=/scratch/sk12075/nyc-drone-sim
source "$REPO/env_sim/bin/activate"
cd "$REPO"
SLUG=$(python -c "from nyc_drone_sim.common import slugify; print(slugify('${LOCATION:-120 Broadway}'))")
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
for G in ${GROUPS_TO_RUN:-unit scene}; do
    python -m nyc_drone_sim.sim.shots --config configs/default.yaml --shots "runs/scenes/$SLUG/populate_probe/$G/shots.json"
done
if [ -n "${RENDER_RUN:-}" ]; then
    for ALT in ${ALTITUDES:-150 500}; do
        python -m nyc_drone_sim.sim.render --config configs/default.yaml \
            --location "${LOCATION:-120 Broadway}" --altitude "$ALT" --run "$RENDER_RUN" \
            ${VARIANT:+--variant "$VARIANT"}
    done
fi
