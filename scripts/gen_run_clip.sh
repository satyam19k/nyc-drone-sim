#!/bin/bash
# Generate one clip from one spec with the video model, inside a GPU allocation.
#   gen_run_clip.sh <spec.json> <final output folder>
# The model writes into a temporary folder beside the final one; the folder is renamed only when
# the run succeeded and produced out.mp4, so a killed or failed run never looks finished. An
# existing final folder is never overwritten: the new result gets _r2, _r3, ...
# Needs MODEL_REPO in the environment and the model's Python environment active.
set -uo pipefail
SPEC="$1"; FINAL="$2"
REPO=/scratch/sk12075/nyc-drone-sim
TMP="${FINAL}.tmp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TMP"
T0=$(date +%s)
( nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$TMP/vram_samples.txt" 2>/dev/null ) &
SAMPLER=$!
( cd "$MODEL_REPO" && python examples/inference.py -i "$SPEC" -o "$TMP" ) > "$TMP/run.log" 2>&1
RC=$?
kill "$SAMPLER" 2>/dev/null
T1=$(date +%s)
if [ $RC -ne 0 ] || [ ! -s "$TMP/out.mp4" ]; then
    echo "FAILED (exit $RC, out.mp4 $( [ -s "$TMP/out.mp4" ] && echo present || echo missing )); kept in $TMP"
    tail -c 2000 "$TMP/run.log" | tr '\r' '\n' | tail -15
    exit 1
fi
cp "$SPEC" "$TMP/spec.json"
python - "$TMP" "$SPEC" "$((T1 - T0))" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
tmp, spec, seconds = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), int(sys.argv[3])
bundle = Path(spec["video_path"]).parents[1]
(tmp / "prompt.txt").write_text(spec["prompt"] + "\n")
(tmp / "seed.txt").write_text(f"{spec['seed']}\n")
(tmp / "bundle_hash.txt").write_text(hashlib.sha256((bundle / "bundle.json").read_bytes()).hexdigest() + "\n")
log = (tmp / "run.log").read_text(errors="replace")
per = re.findall(r"Average time per chunk: ([0-9.]+)", log)
chunks = re.findall(r"Generating chunk \d+/(\d+)", log)
vram = [int(x) for x in (tmp / "vram_samples.txt").read_text().split() if x.isdigit()]
(tmp / "timing.json").write_text(json.dumps({"seconds": seconds, "seconds_per_chunk": float(per[-1]) if per else None,
    "chunks": int(chunks[-1]) if chunks else None, "peak_vram_mib": max(vram) if vram else None}, indent=1) + "\n")
PY
git -C "$MODEL_REPO" rev-parse HEAD > "$TMP/model_commit.txt"
OUT="$FINAL"; N=2
while [ -e "$OUT" ]; do OUT="${FINAL}_r$N"; N=$((N + 1)); done
mv "$TMP" "$OUT"
echo "done in $((T1 - T0)) s -> $OUT"
cat "$OUT/timing.json"
