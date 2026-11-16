#!/bin/bash
# Sets up the video-model machine on a rented pod whose network volume is mounted at /workspace.
# Unattended and idempotent: after any failure run it again and finished steps are skipped.
#
#   bash scripts/pod_setup.sh            everything; the first pod. Needs HF_TOKEN and the internet.
#   bash scripts/pod_setup.sh system     only what lives on the container disk; every later pod.
#
# Expected before it starts: this repository uploaded to $WS/nyc-drone-sim (with .git), and the
# clip bundle of RUN_FOLDER uploaded to the same relative path under it.
# Everything that must outlive the pod is written under $WS. The container disk is erased when
# the pod goes, so system packages and ~/.bashrc are restored by the "system" stage each time.
#
# Environment: HF_TOKEN (read by the download tool, never printed), GIT_USER_EMAIL (only if the
# uploaded tree carries no user.email), EXPECT_VOLUME_GB (the network volume's size, to prove it
# attached), RUN_FOLDER and ALTITUDES to prepare another bundle.
set -euo pipefail

WS=${WS:-/workspace}
REPO=${REPO:-$WS/nyc-drone-sim}
MODEL_REPO=$WS/src/cosmos-transfer2.5
MODEL_URL=https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
RUN_FOLDER=${RUN_FOLDER:-runs/run_006/hudson-park-run}
ALTITUDES=${ALTITUDES:-150}
STAGE=${1:-all}
HF_TOOL="hf>=1.3.5"

step() { echo; echo "=== [$(date -u +%H:%M:%S)] $*"; }
die() { echo "STOP: $*" >&2; exit 1; }
# Readers below consume all their input (awk, sed -n 1p, grep > /dev/null): under pipefail a
# reader that closes the pipe early, like head -1 or grep -q, fails the pipeline at random.
cfg() { awk -v k="$1:" '$1 == k && !done { v = $2; gsub(/"/, "", v); print v; done = 1 }' "$REPO/configs/gen.yaml"; }
at_least() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | sed -n 1p)" = "$1" ]; }   # at_least MIN HAVE
trap 'echo "STOP: line $LINENO failed: $BASH_COMMAND" >&2' ERR

case "$STAGE" in all|system) ;; *) die "unknown stage '$STAGE': use no argument or 'system'" ;; esac
[ -f "$REPO/configs/gen.yaml" ] || die "repository not found at $REPO: upload it before running this"
mkdir -p "$WS/bin" "$WS/cache" "$WS/src"
exec > >(tee -a "$WS/pod_setup.log") 2>&1
echo "--- pod_setup.sh $STAGE, started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------------------------------------------------------------------------------------
step "0. machine checks"
df -h "$WS" | tail -1
if [ "$(df --output=source "$WS" | tail -1)" = "$(df --output=source / | tail -1)" ]; then
    die "$WS is on the container disk: the network volume did not attach. Terminate this pod, nothing built here would survive."
fi
if [ -n "${EXPECT_VOLUME_GB:-}" ]; then
    HAVE_GB=$(df -BG --output=size "$WS" | tail -1 | tr -dc '0-9')
    if [ "$HAVE_GB" -lt $((EXPECT_VOLUME_GB * 85 / 100)) ] || [ "$HAVE_GB" -gt $((EXPECT_VOLUME_GB * 115 / 100)) ]; then
        die "$WS is ${HAVE_GB} GB, the network volume is ${EXPECT_VOLUME_GB} GB: it did not attach. Terminate this pod, nothing built here would survive."
    fi
    echo "$WS is ${HAVE_GB} GB, as expected for the ${EXPECT_VOLUME_GB} GB network volume"
fi
GLIBC=$(ldd --version 2>&1 | awk 'NR == 1 { print $NF }')
at_least "$(cfg min_glibc)" "$GLIBC" || die "glibc $GLIBC is older than $(cfg min_glibc): pick another image"
echo "glibc $GLIBC (needs $(cfg min_glibc) or newer)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p)
    at_least "$(cfg min_driver)" "$DRIVER" || echo "WARNING: NVIDIA driver $DRIVER is older than $(cfg min_driver). Setup needs no GPU, but the model must not run on this machine."
else
    echo "no GPU on this machine. Setup needs none; the driver is checked by the job on the GPU machine."
fi
if [ "$STAGE" = all ]; then
    [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set in this pod's environment"
    echo "token present"
fi

# ---------------------------------------------------------------------------------------------
step "1. system packages and shell environment (container disk)"
PKGS="git git-lfs curl ca-certificates ffmpeg libgl1 libglib2.0-0 libx11-6 libxext6 rsync tmux"
MISSING=""
for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || MISSING="$MISSING $p"; done
if [ -n "$MISSING" ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    # shellcheck disable=SC2086
    apt-get install -y -qq --no-install-recommends $MISSING
fi
echo "packages present: $PKGS"

# Every cache on the volume. CUDA_HOME: one dependency looks for the CUDA runtime compiler
# library in a system CUDA install; this points it at the copy inside env_gen.
cat > "$WS/env.sh" <<EOF
export WS=$WS
export PATH="$WS/bin:\$PATH"
export UV_CACHE_DIR=$WS/cache/uv
export UV_PYTHON_INSTALL_DIR=$WS/cache/uv-python
export UV_TOOL_DIR=$WS/cache/uv-tools
export UV_LINK_MODE=copy
export HF_HOME=$WS/.huggingface
export CUDA_HOME=$REPO/env_gen/lib/python3.10/site-packages/nvidia/cuda_nvrtc
EOF
grep -qxF "source $WS/env.sh" ~/.bashrc 2>/dev/null || echo "source $WS/env.sh" >> ~/.bashrc
# shellcheck disable=SC1091
source "$WS/env.sh"

git lfs install --skip-repo >/dev/null
for d in "$REPO" "$MODEL_REPO"; do      # uploaded and cloned trees may belong to another user id
    git config --global --get-all safe.directory | grep -xF "$d" > /dev/null || git config --global --add safe.directory "$d"
done

step "2. the uploaded tree as a git repository, and its commit identity"
# an upload through object storage keeps neither empty folders nor file modes
mkdir -p "$REPO/.git/refs/heads" "$REPO/.git/refs/tags" "$REPO/.git/objects/info" "$REPO/.git/objects/pack"
git -C "$REPO" config core.fileMode false
echo "repository at commit $(git -C "$REPO" rev-parse --short HEAD), $(git -C "$REPO" status --porcelain | wc -l) files differ from it"
git -C "$REPO" config user.name satyam19k
if [ -z "$(git -C "$REPO" config user.email || true)" ]; then
    [ -n "${GIT_USER_EMAIL:-}" ] || die "the uploaded tree has no user.email: set GIT_USER_EMAIL in the pod's environment"
    git -C "$REPO" config user.email "$GIT_USER_EMAIL"
fi
echo "user.name $(git -C "$REPO" config user.name), user.email set"

if [ "$STAGE" = system ]; then
    [ -x "$REPO/env_gen/bin/python" ] || die "no environment at $REPO/env_gen: run this script without an argument first"
    echo; echo "system stage complete $(date -u +%H:%M:%S). New shells read $WS/env.sh."
    exit 0
fi

# ---------------------------------------------------------------------------------------------
step "3. uv and the model repository at its pinned commit"
if [ ! -x "$WS/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WS/bin" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv --version
COMMIT=$(cfg commit)
[ ${#COMMIT} -eq 40 ] || die "could not read the model commit from configs/gen.yaml"
if [ "$(git -C "$MODEL_REPO" rev-parse HEAD 2>/dev/null || true)" != "$COMMIT" ]; then
    [ -d "$MODEL_REPO/.git" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q "$MODEL_URL" "$MODEL_REPO"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$MODEL_REPO" checkout -q "$COMMIT"
fi
git -C "$MODEL_REPO" lfs pull
echo "model repository at $(git -C "$MODEL_REPO" rev-parse HEAD)"

step "4. Python environment (3.10: newer versions have no wheels for the model's pins)"
( cd "$MODEL_REPO" && UV_PROJECT_ENVIRONMENT="$REPO/env_gen" uv sync --extra=cu128 --python 3.10 )
"$REPO/env_gen/bin/python" -c "import sys, torch, cv2, yaml; print('python', sys.version.split()[0], 'torch', torch.__version__, 'opencv', cv2.__version__)"
# The environment holds full copies of its packages (UV_LINK_MODE=copy), so the package cache,
# some 30 GB, is dead weight on a volume that must also hold the weights. The download tool's
# own small cache is filled again in step 5.
uv cache clean > /dev/null 2>&1 || true
du -sh "$REPO/env_gen" "$UV_CACHE_DIR" 2>/dev/null || true

# ---------------------------------------------------------------------------------------------
step "5. weights, each pinned by revision"
dl() {      # dl REPOSITORY [download arguments]
    local repo=$1
    if ! uvx "$HF_TOOL" download "$@" --quiet; then
        die "download failed for $repo. If the message above says access denied or requires approval, the licence of $repo has not been accepted by the account that owns HF_TOKEN."
    fi
    echo "ok  $*"
}
T=nvidia/Cosmos-Transfer2.5-2B
dl nvidia/Cosmos-Guardrail1 --repo-type model --revision d6d4bfa899a71454a700907664f3e88f503950cf --include "*"
# all four control checkpoints, even for a run that uses fewer: the multi-control model loads every branch
dl $T --repo-type model --revision b67b64abda3801a9aceddbff2bdb86126c06db74 general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt
dl $T --repo-type model --revision dea7737ca29dd8d9086413c6dc5724b8250a0bb4 general/depth/626e6618-bfcd-4d9a-a077-1409e2ce353f_ema_bf16.pt
dl $T --repo-type model --revision eb5325b77d358944da58a690157dd2b8071bbf85 general/blur/ba2f44f2-c726-4fe7-949f-597069d9b91c_ema_bf16.pt
dl $T --repo-type model --revision 23057a4167b89de89a4a397fdbf3887994d115eb general/seg/5136ef49-6d8d-42e8-8abf-7dac722a304a_ema_bf16.pt
# distilled edge model (4 steps, edge control only, 93 frames at most)
dl $T --repo-type model --revision bbaeedb2b57cc8b14a44653099e2551adb69dcc7 distilled/general/edge/41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt
dl nvidia/Cosmos-Predict2.5-2B --repo-type model --revision f176dc95b4a70f53ce01c4b302851595e7322b00 tokenizer.pth
dl nvidia/Cosmos-Reason1-7B --repo-type model --revision 3210bec0495fdc7a8d3dbb8d58da5711eab4b423 --include "*"
dl Qwen/Qwen3Guard-Gen-0.6B
dl google/siglip-so400m-patch14-384
# The model looks its weights up by running the download tool itself. The uvx inside env_gen is a
# different uv from the one in $WS/bin and keeps its own tool cache, so warm both while online.
"$REPO/env_gen/bin/uvx" "$HF_TOOL" version
uvx "$HF_TOOL" version
du -sh "$HF_HOME/hub"

# ---------------------------------------------------------------------------------------------
step "6. this machine's config: its paths and the short run list"
cd "$REPO"
set +u
# shellcheck disable=SC1091
source env_gen/bin/activate
set -u
MODEL_REPO="$MODEL_REPO" REPO="$REPO" python - <<'EOF'
import os
import yaml

cfg = yaml.safe_load(open("configs/gen.yaml"))
cfg["model"]["repo_dir"] = os.environ["MODEL_REPO"]
cfg["model"]["env_dir"] = os.environ["REPO"] + "/env_gen"
# 150 ft, four weathers. Rendered edges carry sun shadows, which belong only under a clear sky.
cfg["runs"] = [{"altitude": 150, "weather": w, "edge": "rendered" if w == "clear" else "geometry"}
               for w in ("clear", "rain", "snow", "fog")]
yaml.safe_dump(cfg, open("configs/gen_pod.yaml", "w"), sort_keys=False, width=100)
print("configs/gen_pod.yaml:", cfg["model"]["repo_dir"], [f"{r['weather']}_{r['edge']}" for r in cfg["runs"]])
EOF

step "7. offline check: every weight file found with the network switched off"
RC=0
HF_HUB_OFFLINE=1 UV_OFFLINE=1 TRANSFORMERS_OFFLINE=1 COSMOS_EXPERIMENTAL_CHECKPOINTS=1 \
    python -m nyc_drone_sim.gen.offline_check --config configs/gen_pod.yaml > "$WS/offline_check.out" 2>&1 || RC=$?
cat "$WS/offline_check.out"
[ "$RC" -eq 0 ] || die "the offline check exited with code $RC"
grep -qx "all found offline" "$WS/offline_check.out" || die "the offline check did not print 'all found offline'"
grep -q "distilled edge checkpoint" "$WS/offline_check.out" || die "the offline check did not cover the distilled edge checkpoint"

step "8. controls, specs and the plan for $RUN_FOLDER ($ALTITUDES ft)"
for ALT in $ALTITUDES; do
    B="$REPO/$RUN_FOLDER/$ALT"
    [ -f "$B/bundle.json" ] || die "no clip bundle at $B: upload it to that path"
    echo "$ALT ft: $(find "$B/clip_0000/rgb" -name '*.png' | wc -l) colour frames, bundle.json sha256 $(sha256sum "$B/bundle.json" | cut -d' ' -f1)"
    # specs written on another machine hold that machine's absolute paths
    rm -rf "$B/gen/cosmos-transfer2.5-2b/specs"
    if [ -f "$B/gen/inputs/inputs.json" ] && [ -f "$B/gen/inputs/depth_control.mp4" ] && [ -f "$B/gen/inputs/edges_geometry.mp4" ] \
       && grep -q "$(sha256sum "$B/bundle.json" | cut -d' ' -f1)" "$B/gen/inputs/inputs.json"; then
        echo "$ALT ft: control videos already built from this bundle, kept"
    else
        python -m nyc_drone_sim.gen.controls --config configs/gen_pod.yaml --run-folder "$RUN_FOLDER" --altitude "$ALT"
    fi
done
export HF_HUB_OFFLINE=1 UV_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RC=0
PYTHONPATH="$MODEL_REPO" python -m nyc_drone_sim.gen.plan --config configs/gen_pod.yaml --run-folder "$RUN_FOLDER" > "$WS/plan.out" 2>&1 || RC=$?
cat "$WS/plan.out"
[ "$RC" -eq 0 ] || die "the plan lists a run that is not ok, or the planner failed"

step "done"
du -sh "$REPO/env_gen" "$HF_HOME/hub" "$MODEL_REPO" "$REPO/runs" 2>/dev/null
df -h "$WS" | tail -1
echo "pod_setup.sh complete $(date -u +%Y-%m-%dT%H:%M:%SZ): environment, weights, offline check and plan are on the volume."
