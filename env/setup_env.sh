#!/usr/bin/env bash
# Creates the exact environment of the France experiment in a separate Python 3.11 venv: PyTorch 2.1.0+cu121,
# MMCV 2.1.0 (official OpenMMLab build), MMEngine 0.10.7, MMSegmentation 1.2.2 and every other package pinned in
# env/requirements.lock (installed without dependency resolution, so versions cannot drift).
# Works on Google Colab (any Python in the notebook kernel) and on any Linux x86_64 (an NVIDIA GPU with driver >= 525
# is recommended; without it everything runs on the CPU, much slower).
# Usage: bash env/setup_env.sh [env_dir=/content/py311]   (idempotent: does nothing if the env already passes the check)
set -euo pipefail
export MPLBACKEND=Agg   # notebook kernels export an inline matplotlib backend unknown to this environment
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${1:-/content/py311}"
PY="$ENV/bin/python"
if [ -x "$PY" ] && "$PY" "$BUNDLE/env/check_env.py" --quiet; then
  echo "environment is ready: $ENV"; exit 0
fi
# uv (Python and package installer): an installed one, else from pip, else the official standalone installer into
# .uv/ of the repository (plain Linux without pip, e.g. a fresh Ubuntu 24.04)
if command -v uv >/dev/null 2>&1; then UV=uv
elif python3 -m uv --version >/dev/null 2>&1; then UV="python3 -m uv"
else
  echo "== installing uv (Python and package installer)"
  if python3 -m pip install -q "uv>=0.4" 2>/dev/null || python3 -m pip install -q --user --break-system-packages "uv>=0.4" 2>/dev/null; then
    UV="python3 -m uv"
  else
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$BUNDLE/.uv" INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null
    UV="$BUNDLE/.uv/uv"
  fi
fi
echo "== creating Python 3.11 environment in $ENV"
$UV venv --python 3.11 --seed "$ENV"
echo "== installing pinned packages (PyTorch 2.1.0 cu121, MMEngine, MMSegmentation, ...): ~2-3 min"
$UV pip install --python "$PY" --no-deps --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 -r "$BUNDLE/env/requirements.lock"
echo "== installing MMCV 2.1.0 (prebuilt for torch 2.1 / CUDA 12.1)"
WHEEL="$BUNDLE/wheels/mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl"
[ -f "$WHEEL" ] || python3 "$BUNDLE/scripts/fetch.py" mmcv --out "$WHEEL"   # project release, else OpenMMLab
$UV pip install --python "$PY" --no-deps "$WHEEL"
"$PY" "$BUNDLE/env/check_env.py"
