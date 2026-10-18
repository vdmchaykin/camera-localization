#!/bin/bash
# Creates .venv with the pinned dependencies. Needs python3.12 and an NVIDIA driver for CUDA 12.8.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3.12}
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
