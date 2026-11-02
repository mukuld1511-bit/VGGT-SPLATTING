#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# DGX Spark — Video-to-3D Reconstruction System Setup
# Target: NVIDIA GB10 (Blackwell, sm_121), 128 GB unified memory, aarch64 Linux
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="vggt-splat"
PYTHON_VERSION="3.11"

echo "════════════════════════════════════════════════════════════════════"
echo "  DGX Spark · Video-to-3D Setup"
echo "════════════════════════════════════════════════════════════════════"

# ── 1. Create / activate conda environment ────────────────────────────────────
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "[1/7] Conda environment '${ENV_NAME}' already exists — reusing."
else
    echo "[1/7] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})…"
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

# Activate inside the script (works with conda init'd shells)
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ── 2. Install PyTorch nightly for Blackwell (sm_121 / cu128) ─────────────────
echo "[2/7] Installing PyTorch nightly (cu128 — Blackwell support)…"
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Quick sanity check
python -c "
import torch
print(f'PyTorch {torch.__version__}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f'GPU: {torch.cuda.get_device_name()} — SM {cap[0]}.{cap[1]}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('WARNING: CUDA not available — check your driver / toolkit')
"

# ── 3. Clone and install VGGT ─────────────────────────────────────────────────
echo "[3/7] Setting up VGGT…"
mkdir -p "${SCRIPT_DIR}/external"
if [ -d "${SCRIPT_DIR}/external/vggt" ]; then
    echo "  → VGGT already cloned, pulling latest…"
    git -C "${SCRIPT_DIR}/external/vggt" pull --ff-only || true
else
    git clone https://github.com/facebookresearch/vggt.git "${SCRIPT_DIR}/external/vggt"
fi
pip install -e "${SCRIPT_DIR}/external/vggt"

# ── 4. Install gsplat ────────────────────────────────────────────────────────
echo "[4/7] Installing gsplat…"
pip install "gsplat>=1.3.0"

# ── 5. Install remaining Python dependencies ─────────────────────────────────
echo "[5/7] Installing project dependencies…"
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ── 6. Install optional VGGT demo dependencies (pycolmap, LightGlue) ─────────
echo "[6/7] Installing VGGT demo extras (COLMAP bridge)…"
pip install -r "${SCRIPT_DIR}/external/vggt/requirements_demo.txt" || {
    echo "  ⚠ Some demo dependencies failed — non-BA mode will still work."
}

# ── 7. Pre-download VGGT model weights ────────────────────────────────────────
echo "[7/7] Pre-downloading VGGT-1B model weights (~4 GB)…"
python -c "
from vggt.models.vggt import VGGT
model = VGGT.from_pretrained('facebook/VGGT-1B')
print('✓ VGGT-1B model weights cached successfully')
del model
"

# ── 8. Environment variables ─────────────────────────────────────────────────
echo ""
echo "Add the following to your shell profile (e.g. ~/.bashrc):"
echo ""
echo "  export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas"
echo "  export TORCH_FLOAT32_MATMUL_PRECISION=high"
echo "  conda activate ${ENV_NAME}"
echo ""

# ── Create data directories ──────────────────────────────────────────────────
mkdir -p "${SCRIPT_DIR}/data" "${SCRIPT_DIR}/results"

echo "════════════════════════════════════════════════════════════════════"
echo "  ✓ Setup complete!  Run:  python app.py"
echo "════════════════════════════════════════════════════════════════════"
