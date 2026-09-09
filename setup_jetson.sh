#!/usr/bin/env bash
# =============================================================================
# setup_jetson.sh — Jetson Orin Nano 8GB environment setup
# Tested on JetPack 6.0 / Ubuntu 22.04 / CUDA 12.2
# Run once: bash setup_jetson.sh
# =============================================================================
set -euo pipefail

# ─── Colours ─────────────────────────────────────────────────────────────────
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; NC="\033[0m"
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_DIR="${HOME}/avera_models"
PIPER_DIR="${MODEL_DIR}/piper"
VENV_DIR="${HOME}/avera_venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

# Model URLs — pin exact versions for reproducibility
QWEN_GGUF_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
PIPER_VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
PIPER_CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
WHISPER_MODEL="small.en"  # downloaded automatically by faster-whisper
# NOTE: Bhashini TTS uses an online API — no model download needed.
# Get your API key at: https://bhashini.gov.in/ulca/user/register
# Then set BHASHINI_API_KEY in your shell or in config.yaml.

# ─── 1. Verify JetPack ───────────────────────────────────────────────────────
info "Checking JetPack version..."
if [ -f /etc/nv_tegra_release ]; then
    cat /etc/nv_tegra_release
else
    warn "Could not find /etc/nv_tegra_release — ensure you are on a Jetson device."
fi

# ─── 2. System packages ──────────────────────────────────────────────────────
info "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-venv python3-dev \
    tesseract-ocr tesseract-ocr-eng \
    portaudio19-dev libsndfile1 \
    libopenblas-dev \
    cmake ninja-build \
    curl wget git \
    htop nvtop  # for memory monitoring

# ─── 3. Python virtual environment ───────────────────────────────────────────
info "Creating Python virtual environment at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools

# ─── 4. PyTorch for Jetson ───────────────────────────────────────────────────
# NVIDIA provides Jetson-specific PyTorch wheels — do NOT use PyPI torch
info "Installing PyTorch for Jetson (CUDA 12.2)..."
TORCH_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0+nv24.4-cp310-cp310-linux_aarch64.whl"
pip install "${TORCH_WHEEL}" || warn "PyTorch wheel failed — install manually from NVIDIA Developer site."

# ─── 5. llama-cpp-python (CUDA build) ────────────────────────────────────────
info "Building llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --upgrade --no-binary llama-cpp-python

# ─── 6. Python requirements (without llama-cpp-python, already installed) ────
info "Installing Python requirements..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    || error "pip install failed"

# ─── 7. Model directory ──────────────────────────────────────────────────────
info "Creating model directory: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}" "${PIPER_DIR}"

# ─── 8. Download Qwen2.5-1.5B GGUF ──────────────────────────────────────────
QWEN_DEST="${MODEL_DIR}/qwen2.5-1.5b-instruct-q4_k_m.gguf"
if [ ! -f "${QWEN_DEST}" ]; then
    info "Downloading Qwen2.5-1.5B-Instruct Q4_K_M GGUF (~1.1 GB)..."
    wget --show-progress -O "${QWEN_DEST}" "${QWEN_GGUF_URL}"
else
    info "Qwen2.5-1.5B GGUF already present — skipping."
fi

# ─── 9. Pre-download AI4Bharat Indic TTS models ──────────────────────────────
INDIC_TTS_DIR="${MODEL_DIR}/indic_tts"
mkdir -p "${INDIC_TTS_DIR}"

info "Pre-downloading AI4Bharat Indic TTS models (Indo-Aryan + Dravidian)..."
info "This is a one-time download (~400-800 MB total). Will be cached offline."

python3 -c "
from huggingface_hub import snapshot_download
import os

groups = {
    'indo_aryan': 'ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4',
    'dravidian':  'ai4bharat/indic-tts-coqui-dravidian-gpu--t4',
}
cache_dir = os.path.expanduser('${INDIC_TTS_DIR}')
for name, hf_id in groups.items():
    dest = os.path.join(cache_dir, name)
    if os.path.exists(os.path.join(dest, 'model_file.pth')):
        print(f'[SKIP] {name} already cached at {dest}')
        continue
    print(f'[DOWNLOAD] {hf_id} -> {dest}')
    snapshot_download(
        repo_id=hf_id,
        local_dir=dest,
        ignore_patterns=['*.msgpack', 'flax_model.*', 'tf_model.*'],
    )
    print(f'[OK] {name} cached.')
"

if [ $? -eq 0 ]; then
    info "AI4Bharat Indic TTS models ready ✓"
else
    warn "Indic TTS model download failed. They will download automatically on first use."
fi

# ─── 10. Pre-download faster-whisper model ───────────────────────────────────
info "Pre-downloading faster-whisper ${WHISPER_MODEL} model..."
"${PYTHON}" -c "
from faster_whisper import WhisperModel
print('Downloading faster-whisper ${WHISPER_MODEL}...')
WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')
print('Done.')
"

# ─── 11. Pre-download HuggingFace models ─────────────────────────────────────
info "Pre-downloading embedding & reranker models..."
"${PYTHON}" -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
print('Downloading bge-small-en-v1.5...')
SentenceTransformer('BAAI/bge-small-en-v1.5')
print('Downloading MiniLM reranker...')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('Done.')
"

info "Downloading Qwen2-VL-2B-Instruct (HuggingFace, ~4.5 GB)..."
"${PYTHON}" -c "
from transformers import AutoProcessor
AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
print('Qwen2-VL processor downloaded.')
"
# Model weights are downloaded on first VLM call

# ─── 12. Verify CUDA access ──────────────────────────────────────────────────
info "Verifying CUDA..."
"${PYTHON}" -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB')
"

# ─── 13. systemd service (optional) ──────────────────────────────────────────
SERVICE_FILE="/etc/systemd/system/avera-rag.service"
if [ ! -f "${SERVICE_FILE}" ]; then
    info "Creating systemd service at ${SERVICE_FILE}..."
    sudo tee "${SERVICE_FILE}" > /dev/null << EOF
[Unit]
Description=Avera Medical RAG API
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${PYTHON} ${SCRIPT_DIR}/main.py --mode serve
Restart=on-failure
RestartSec=10
Environment="PATH=${VENV_DIR}/bin:/usr/local/cuda/bin:/usr/bin:/bin"

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    info "Service installed. Enable with: sudo systemctl enable avera-rag && sudo systemctl start avera-rag"
fi

info "=== Setup complete! ==="
info "Activate venv: source ${VENV_DIR}/bin/activate"
info "Run ingest:    python main.py --mode ingest --path /path/to/docs"
info "Run server:    python main.py --mode serve"
info "Run eval:      python main.py --mode eval"
