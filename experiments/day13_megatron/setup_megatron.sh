#!/usr/bin/env bash
# Day 13: put Megatron-LM core_v0.19.2 at /workspace/Megatron-LM (source checkout, used via PYTHONPATH; no pip
# install of Megatron, no Transformer Engine, no apex) and install its one missing pure-Python dependency (einops).
# Idempotent: re-running only checks. Does not touch torch.
#   bash experiments/day13_megatron/setup_megatron.sh [PATH]
set -Eeuo pipefail
DEST=${1:-/workspace/Megatron-LM}
TAG=core_v0.19.2
COMMIT=4b4acac9a1d28ea6829c8d4f566d75698a21249d
if [[ ! -d "$DEST/megatron/core" ]]; then
    git clone -q --depth 1 --branch "$TAG" https://github.com/NVIDIA/Megatron-LM.git "$DEST"
fi
HEAD=$(git -C "$DEST" rev-parse HEAD)
[[ "$HEAD" == "$COMMIT" ]] || { echo "Megatron-LM at $DEST is $HEAD, expected $TAG = $COMMIT" >&2; exit 2; }
PY=${PYTHON:-/venv/main/bin/python}
"$PY" -c 'import einops' 2>/dev/null || "$PY" -m pip install -q einops
"$PY" -c 'import torch, einops; print("torch", torch.__version__, "| einops", einops.__version__)'
echo "Megatron-LM $TAG ($HEAD) ready at $DEST"
