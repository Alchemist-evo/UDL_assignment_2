#!/usr/bin/env bash
# Creates .venv and installs everything. Tested on Fedora / Ubuntu.
# usage: bash setup_env.sh            (auto-detects NVIDIA / Intel Arc / CPU)
#        TORCH_BACKEND=cpu bash setup_env.sh   (force one: cuda | xpu | cpu)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10"'

if [ ! -d .venv ]; then
    "$PY" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel

BACKEND=${TORCH_BACKEND:-auto}
if [ "$BACKEND" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        BACKEND=cuda
    elif lspci 2>/dev/null | grep -Eiq 'VGA.*Intel.*(Arc|DG2)|Display.*Intel.*(Arc|DG2)'; then
        BACKEND=xpu
    else
        BACKEND=cpu
    fi
fi
echo ">> installing PyTorch for backend: $BACKEND"
case "$BACKEND" in
    cuda) python -m pip install torch torchvision ;;                                     # PyPI Linux wheels ship CUDA
    xpu)  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu ;;
    cpu)  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu ;;
    *) echo "unknown TORCH_BACKEND=$BACKEND"; exit 1 ;;
esac
python -m pip install -r requirements.txt
python -m ipykernel install --user --name udl-a2 --display-name "Python (udl-a2)"

python - <<'EOF'
import torch
if torch.cuda.is_available():
    print("OK: CUDA ->", torch.cuda.get_device_name(0))
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    print("OK: Intel XPU ->", torch.xpu.get_device_name(0))
else:
    print("WARNING: no GPU visible to PyTorch, running on CPU")
EOF
echo ">> done. activate with:  source .venv/bin/activate"
