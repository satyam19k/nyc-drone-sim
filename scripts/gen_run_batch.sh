#!/bin/bash
# Generate several short clips with one model load, inside a GPU allocation.
#   gen_run_batch.sh <output folder> <spec.json> [<spec.json> ...]
# Each spec's "name" becomes <output folder>/<name>.mp4. A spec whose output already exists is
# skipped, so a batch that was cut off can be run again and only does what is missing.
# Needs MODEL_REPO in the environment and the model's Python environment active.
set -uo pipefail
OUT="$1"; shift
mkdir -p "$OUT"
TODO=()
for SPEC in "$@"; do
    NAME=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$SPEC")
    if [ -s "$OUT/$NAME.mp4" ]; then echo "$NAME: already generated, skipped"; else TODO+=("$SPEC"); fi
done
if [ ${#TODO[@]} -eq 0 ]; then echo "nothing to do"; exit 0; fi
echo "$(date '+%F %T') generating ${#TODO[@]} clip(s)"
( cd "$MODEL_REPO" && python examples/inference.py -i "${TODO[@]}" -o "$OUT" )
RC=$?
echo "$(date '+%F %T') exit $RC"
exit $RC
