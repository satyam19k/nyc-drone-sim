#!/bin/bash
#SBATCH --job-name=ndsim-gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=slurm-%j.out

# One unattended job for the video model: node check, self-test, then the run list in
# configs/gen.yaml. Resumable: submit the same command again and finished runs are skipped.
# Submit with:
#   sbatch --account=<account> --partition=<list> --time=<limit> \
#          --export=ALL,RUN_FOLDER=runs/run_006/hudson-park-run,THEN=runs/run_008/120-broadway slurm/gen_job.sbatch
# Add SELF_TEST_ONLY=1 to the export list for a short job that stops after the self-test.
# Compute nodes are assumed to have no internet: every weight file must already be in the
# Hugging Face cache, and nothing here may try to download.
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
# One of the model's dependencies looks for the CUDA runtime compiler library in a system CUDA
# install; point it at the copy shipped inside env_gen so no system CUDA toolkit is needed.
export CUDA_HOME="$REPO/env_gen/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
source "$REPO/env_gen/bin/activate"
# Compute nodes have no libGL, which the model's OpenCV build links against (it is never
# called here). Copies of libGL, libGLX and libGLdispatch from the login node are kept on scratch.
export LD_LIBRARY_PATH="/scratch/sk12075/cache/syslibs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m nyc_drone_sim.gen.job --run-folder "${RUN_FOLDER:?set RUN_FOLDER}" ${THEN:+--then "$THEN"} ${SKIP_SELF_TEST:+--skip-self-test} ${SELF_TEST_ONLY:+--self-test-only}
