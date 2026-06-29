#!/usr/bin/env bash
#
# Assemble the Hugging Face Space (backend only) into a cloned Space repo.
#
# Usage:
#   1. Create a Docker Space on https://huggingface.co/new-space (SDK: Docker).
#   2. Clone it:   git clone https://huggingface.co/spaces/<user>/<space>  hf-space
#   3. Run:        deploy/prepare_hf_space.sh hf-space
#   4. Push:       cd hf-space && git add -A && git commit -m "NIDS backend" && git push
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:?Usage: deploy/prepare_hf_space.sh <cloned-space-dir>}"

if [ ! -d "$DEST/.git" ]; then
  echo "error: '$DEST' is not a git repo. Clone your HF Space there first." >&2
  exit 1
fi

echo "Assembling backend into $DEST ..."

# copy only what the API needs to run
cp "$ROOT/Dockerfile"          "$DEST/Dockerfile"
cp "$ROOT/requirements.txt"    "$DEST/requirements.txt"
cp "$ROOT/.dockerignore"       "$DEST/.dockerignore"
cp "$ROOT/deploy/space-README.md" "$DEST/README.md"   # HF reads the YAML front-matter

mkdir -p "$DEST/scripts" "$DEST/models" "$DEST/data"
cp "$ROOT/scripts/api.py" "$DEST/scripts/api.py"
rm -rf "$DEST/scripts/app"
cp -r "$ROOT/scripts/app" "$DEST/scripts/app"
rm -rf "$DEST/models/new"
cp -r "$ROOT/models/new" "$DEST/models/new"
cp "$ROOT/data/demo_flows.npy" "$DEST/data/demo_flows.npy"

# drop python caches
find "$DEST/scripts" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

# Git LFS for the model files; HF rejects non-LFS files over 10 MB
cd "$DEST"
git lfs install --local
git lfs track "*.h5" "*.tflite" >/dev/null

echo
echo "Done. Files staged in $DEST:"
echo "  Dockerfile, requirements.txt, README.md, scripts/, models/new/, data/demo_flows.npy"
echo
echo "Next:"
echo "  cd $DEST"
echo "  git add -A && git commit -m 'NIDS backend' && git push"
