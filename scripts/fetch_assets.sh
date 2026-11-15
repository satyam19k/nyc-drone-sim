#!/bin/bash
# Download the model sets used to populate the scene, then convert them to the USD library.
# Models come from the Valdrada dataset on Hugging Face, pinned to one tag. Only the vehicle,
# tree and Rocketbox pedestrian folders are fetched (about 0.6 GB). Nothing here is committed.
#
# The pedestrian folder peds24/bodies is deliberately not downloaded: its licence notice forbids
# using those models to build datasets or to train or test models.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${VALDRADA_TAG:-v0.7.1}"
BASE="https://huggingface.co/datasets/mehmetkeremturkcan/valdrada"
DEST="$REPO/data/assets/valdrada"
mkdir -p "$DEST"
curl -sfL "$BASE/resolve/$TAG/LICENSE.md" -o "$DEST/LICENSE.md"

list_dir() {   # print every file path under one repository folder
    python3 - "$1" "$2" <<'PY'
import json, sys, urllib.request
url = "https://huggingface.co/api/datasets/mehmetkeremturkcan/valdrada/tree/%s/%s?recursive=true&limit=1000" % (sys.argv[2], sys.argv[1])
while url:
    with urllib.request.urlopen(url) as r:
        for f in json.load(r):
            if f["type"] == "file":
                print(f["path"])
        link = r.headers.get("Link", "")
        url = link[link.find("<") + 1:link.find(">")] if 'rel="next"' in link else None
PY
}

for DIR in Content/models/fleet24 Content/models/bxtrees Content/models/peds24/rb27; do
    list_dir "$DIR" "$TAG" | while read -r F; do
        mkdir -p "$DEST/$(dirname "$F")"
        [ -s "$DEST/$F" ] || curl -sfL --retry 3 "$BASE/resolve/$TAG/$F" -o "$DEST/$F"
    done
done
for F in Content/models/peds24/manifest.json Content/models/peds24/NOTICE.md; do
    mkdir -p "$DEST/$(dirname "$F")"
    curl -sfL "$BASE/resolve/$TAG/$F" -o "$DEST/$F"
done
echo "downloaded to $DEST"
if [ -z "${SKIP_CONVERT:-}" ]; then
    cd "$REPO" && python -m nyc_drone_sim.assets
fi
