#!/bin/bash
#SBATCH --job-name=ndsim-batch
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# One allocation, many steps. The queue wait is paid once.
# Submit with: sbatch --account=<account> --export=ALL,STEPS=<file> slurm/batch.sbatch
# STEPS is a text file with one command per line (blank lines and lines starting with # are
# skipped). Steps run in order. A failed step is recorded and the next one still runs; the job
# exits non-zero if any step failed. A summary is written next to the steps file.
set -uo pipefail
export OMNI_KIT_ACCEPT_EULA=YES
REPO=/scratch/sk12075/nyc-drone-sim
source "$REPO/env_sim/bin/activate"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
SUMMARY="${STEPS%.txt}.summary.txt"
: > "$SUMMARY"
FAILED=0
N=0
while IFS= read -r CMD || [ -n "$CMD" ]; do
    case "$CMD" in ''|'#'*) continue ;; esac
    N=$((N + 1))
    echo "=== step $N: $CMD"
    T0=$(date +%s)
    bash -c "$CMD"
    RC=$?
    T1=$(date +%s)
    echo "step $N exit $RC in $((T1 - T0)) s: $CMD" | tee -a "$SUMMARY"
    [ $RC -ne 0 ] && FAILED=$((FAILED + 1))
done < "$STEPS"
echo "batch done: $N steps, $FAILED failed" | tee -a "$SUMMARY"
[ $FAILED -eq 0 ]
