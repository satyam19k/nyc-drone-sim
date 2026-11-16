#!/bin/bash
#SBATCH --job-name=ndsim-genq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=slurm-%j.out

# Video-model work queue. Runs the items in QUEUE_DIR one at a time, in name order: NNN_name.sh.
# An item is written once and never edited; to add work, add a file with a higher number, also
# while the job is running. NNN_name.done records an item's exit code, so a resubmitted job skips
# what is finished. A failed item is logged and the next one starts.
#
# The GPU must never sit idle: a job on this cluster was cancelled by the system about an hour
# after its GPU went idle. So when no item is left the job waits at most IDLE_WAIT_S for a new
# one and then ends, instead of holding the GPU.
# Submit with: sbatch --account=<account> --partition=<list> --time=<limit> \
#                     --export=ALL,QUEUE_DIR=runs/gen_queue/main slurm/gen_queue.sbatch
set -uo pipefail
REPO=/scratch/sk12075/nyc-drone-sim
export HF_HOME=/scratch/sk12075/.huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1 UV_LINK_MODE=copy
export UV_CACHE_DIR=/scratch/sk12075/cache/uv UV_PYTHON_INSTALL_DIR=/scratch/sk12075/cache/uv-python UV_TOOL_DIR=/scratch/sk12075/cache/uv-tools
export PATH="$HOME/.local/bin:$PATH"
export CUDA_HOME="$REPO/env_gen/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
source "$REPO/env_gen/bin/activate"
export LD_LIBRARY_PATH="/scratch/sk12075/cache/syslibs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MODEL_REPO=/scratch/sk12075/src/cosmos-transfer2.5
cd "$REPO"
Q="$REPO/${QUEUE_DIR:?set QUEUE_DIR}"
IDLE_WAIT_S="${IDLE_WAIT_S:-600}"
LOG="$Q/queue.log"
echo "$(date '+%F %T') job ${SLURM_JOB_ID:-local} started on $(hostname): $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)" >> "$LOG"
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${MEM:-0}" -lt 70000 ]; then
    echo "$(date '+%F %T') this GPU has ${MEM} MiB; the model needs 70 GB. Ending without running anything." >> "$LOG"
    exit 2
fi
WAITED=0
while :; do
    NEXT=""
    for f in "$Q"/[0-9][0-9][0-9]_*.sh; do
        [ -e "$f" ] || continue
        [ -e "${f%.sh}.done" ] && continue
        NEXT="$f"; break
    done
    if [ -z "$NEXT" ]; then
        if [ "$WAITED" -ge "$IDLE_WAIT_S" ]; then
            echo "$(date '+%F %T') no item left after waiting ${WAITED} s: ending so the GPU is not held idle" >> "$LOG"
            break
        fi
        sleep 20; WAITED=$((WAITED + 20)); continue
    fi
    WAITED=0
    echo "$(date '+%F %T') start $(basename "$NEXT")" >> "$LOG"
    bash "$NEXT" > "${NEXT%.sh}.log" 2>&1
    RC=$?
    echo "exit $RC at $(date '+%F %T') in job ${SLURM_JOB_ID:-local}" > "${NEXT%.sh}.done"
    echo "$(date '+%F %T') end   $(basename "$NEXT") exit $RC" >> "$LOG"
done
