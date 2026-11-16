#!/bin/bash
#SBATCH --job-name=ndsim-gen-offline
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm-%j.out

# CPU job on a compute node: is every weight file the video model loads already cached, with the
# network switched off? Also reports whether the node can reach Hugging Face at all.
# Submit with: sbatch --account=<account> slurm/gen_offline_check.sbatch
set -uo pipefail
REPO=/scratch/sk12075/nyc-drone-sim
export HF_HOME=/scratch/sk12075/.huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export UV_CACHE_DIR=/scratch/sk12075/cache/uv
export UV_PYTHON_INSTALL_DIR=/scratch/sk12075/cache/uv-python
export UV_TOOL_DIR=/scratch/sk12075/cache/uv-tools
export PATH="$HOME/.local/bin:$PATH"
export CUDA_HOME="$REPO/env_gen/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
source "$REPO/env_gen/bin/activate"
# Compute nodes have no libGL, which the model's OpenCV build links against (it is never
# called here). Copies of libGL, libGLX and libGLdispatch from the login node are kept on scratch.
export LD_LIBRARY_PATH="/scratch/sk12075/cache/syslibs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO"
echo "node $(hostname), $(ldd --version | head -1)"
if curl -sI -m 10 https://huggingface.co >/dev/null 2>&1; then echo "internet: this node can reach huggingface.co"; else echo "internet: this node cannot reach huggingface.co"; fi
which uvx
python -m nyc_drone_sim.gen.offline_check
