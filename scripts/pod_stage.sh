#!/bin/bash
# Runs one paid stage of the video-model job on a GPU pod, then removes the pod.
#
#   bash scripts/pod_stage.sh LIMIT_MINUTES [job arguments]
#   bash scripts/pod_stage.sh 70 --self-test-only
#   bash scripts/pod_stage.sh 75 --skip-self-test --max-frames 93 --batch
#
# It restores what lives on the container disk (pod_setup.sh system), runs the job driver with the
# network switched off for model lookups and under a hard time limit, and when the job ends, for
# any reason, removes the pod it is running on. Results, reports and logs are on the network
# volume, so nothing is lost and nothing bills while waiting for a person. KEEP_POD=1 keeps the pod.
set -uo pipefail

WS=${WS:-/workspace}
REPO=${REPO:-$WS/nyc-drone-sim}
RUN_FOLDER=${RUN_FOLDER:-runs/run_006/hudson-park-run}
LIMIT=${1:?give the time limit in minutes}
shift
LOG="$WS/stage_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
echo "--- pod_stage.sh, limit $LIMIT min, job arguments: $*, started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

finish() {
    echo "--- stage over $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sync
    if [ -n "${KEEP_POD:-}" ]; then
        echo "KEEP_POD is set: the pod stays up and keeps billing"
    elif [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl > /dev/null 2>&1; then
        echo "removing pod $RUNPOD_POD_ID"
        runpodctl remove pod "$RUNPOD_POD_ID"
    else
        echo "WARNING: cannot remove the pod from inside (no pod id or no runpodctl): remove it by hand"
    fi
}
trap finish EXIT

bash "$REPO/scripts/pod_setup.sh" system || { echo "STOP: the system stage failed, the job was not started"; exit 1; }
# shellcheck disable=SC1091
source "$WS/env.sh"
set +u
# shellcheck disable=SC1091
source "$REPO/env_gen/bin/activate"
set -u
# every weight file is already in the cache, pinned by revision: no lookup may reach the network
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1
cd "$REPO" || exit 1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
timeout --signal=TERM --kill-after=60 "${LIMIT}m" \
    python -m nyc_drone_sim.gen.job --config configs/gen_pod.yaml --run-folder "$RUN_FOLDER" "$@"
RC=$?
if [ $RC -eq 124 ] || [ $RC -eq 137 ]; then
    echo "job stopped at the $LIMIT minute limit (exit code $RC): finished runs are kept, a later start skips them"
else
    echo "job exit code $RC"
fi
exit $RC
