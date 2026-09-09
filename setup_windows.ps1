#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Avera RAG — Windows setup script
    Sets up a virtual environment, installs all dependencies,
    and downloads required models for local Windows validation.

.USAGE
    .\setup_windows.ps1

.NOTES
    Run from inside the avera_rag\ directory.
    Requires: Python 3.10+, internet connection (one-time download)
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colors ─────────────────────────────────────────────────────────────────────
function info  { Write-Host "[INFO]  $args" -ForegroundColor Cyan }
function ok    { Write-Host "[OK]    $args" -ForegroundColor Green }
function warn  { Write-Host "[WARN]  $args" -ForegroundColor Yellow }
function err   { Write-Host "[ERROR] $args" -ForegroundColor Red; exit 1 }

$VENV_DIR   = "venv_windows"
$MODEL_DIR  = "d:\Deva\Avera!\avera_models"
$LLM_MODEL  = "$MODEL_DIR\qwen2.5-1.5b-instruct-q4_k_m.gguf"
$LLM_HF_ID  = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
$LLM_FILE   = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

info "Avera RAG — Windows Setup"
info "Working directory: $(Get-Location)"

# ── 1. Check Python ─────────────────────────────────────────────────────────────
info "Checking Python..."
try {
    $pyVersion = python --version 2>&1
    info "Found: $pyVersion"
} catch {
    err "Python not found. Install Python 3.10+ from https://python.org"
}

# ── 2. Create virtual environment ───────────────────────────────────────────────
if (-Not (Test-Path "$VENV_DIR\Scripts\activate.ps1")) {
    info "Creating virtual environment ($VENV_DIR)..."
    python -m venv $VENV_DIR
    ok "Virtual environment created."
} else {
    info "Virtual environment already exists."
}

# Activate
& ".\$VENV_DIR\Scripts\Activate.ps1"
info "Virtual environment activated."

# ── 3. Upgrade pip ──────────────────────────────────────────────────────────────
info "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# ── 4. Install dependencies ─────────────────────────────────────────────────────
info "Installing Windows requirements (this may take 5-10 minutes)..."
pip install -r requirements_windows.txt
ok "Dependencies installed."

# ── 5. Create model directory ───────────────────────────────────────────────────
if (-Not (Test-Path $MODEL_DIR)) {
    New-Item -ItemType Directory -Path $MODEL_DIR -Force | Out-Null
    ok "Created model directory: $MODEL_DIR"
}

# ── 6. Download LLM GGUF (Qwen2.5-1.5B Q4_K_M) ─────────────────────────────────
info "Checking for LLM model..."
if (-Not (Test-Path $LLM_MODEL)) {
    info "Downloading Qwen2.5-1.5B-Instruct Q4_K_M GGUF (~1.1 GB)..."
    info "Source: HuggingFace — $LLM_HF_ID"
    python -c @"
from huggingface_hub import hf_hub_download
import shutil, os

dest = r'$LLM_MODEL'
print(f'Downloading to: {dest}')
tmp = hf_hub_download(
    repo_id='$LLM_HF_ID',
    filename='$LLM_FILE',
    local_dir=r'$MODEL_DIR',
)
print(f'Downloaded: {tmp}')
"@
    if ($LASTEXITCODE -eq 0) {
        ok "LLM model downloaded."
    } else {
        warn "LLM model download failed. Download manually:"
        warn "  URL: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        warn "  Save to: $LLM_MODEL"
    }
} else {
    info "LLM model already present — skipping."
}

# ── 7. Pre-download embedder + reranker ─────────────────────────────────────────
info "Pre-downloading embedder and reranker models (~220 MB)..."
python -c @"
from sentence_transformers import SentenceTransformer, CrossEncoder
print('Downloading BGE embedder...')
SentenceTransformer('BAAI/bge-small-en-v1.5')
print('Downloading MiniLM reranker...')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('Done.')
"@
ok "Embedder and reranker cached."

# ── 8. Pre-download AI4Bharat Indic TTS models ─────────────────────────────────
info "Pre-downloading AI4Bharat Indic TTS models (~400-800 MB)..."
info "This is a one-time download. Subsequent runs are fully offline."
$indicTTSDir = "$MODEL_DIR\indic_tts"
python -c @"
from huggingface_hub import snapshot_download
import os

groups = {
    'indo_aryan': 'ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4',
    'dravidian':  'ai4bharat/indic-tts-coqui-dravidian-gpu--t4',
}
cache_dir = r'$indicTTSDir'
os.makedirs(cache_dir, exist_ok=True)
for name, hf_id in groups.items():
    dest = os.path.join(cache_dir, name)
    checkpoint = os.path.join(dest, 'model_file.pth')
    if os.path.exists(checkpoint):
        print(f'[SKIP] {name} already cached.')
        continue
    print(f'[DOWNLOAD] {name}...')
    snapshot_download(
        repo_id=hf_id,
        local_dir=dest,
        ignore_patterns=['*.msgpack', 'flax_model.*', 'tf_model.*'],
    )
    print(f'[OK] {name} cached at {dest}')
"@
if ($LASTEXITCODE -eq 0) {
    ok "AI4Bharat Indic TTS models ready."
} else {
    warn "TTS model download failed. They will download automatically on first voice use."
}

# ── 9. Summary ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host " Avera RAG — Windows Setup Complete!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate env:  .\$VENV_DIR\Scripts\Activate.ps1"
Write-Host "  2. Ingest data:   python main.py --config config_windows.yaml --mode ingest --path ..\imci_chart_booklet.jsonl"
Write-Host "  3. Ingest XML:    python main.py --config config_windows.yaml --mode ingest --path ..\mplus_topics\mplus_topics_2026-09-08.xml"
Write-Host "  4. Run eval:      python main.py --config config_windows.yaml --mode eval"
Write-Host "  5. Test query:    python main.py --config config_windows.yaml --mode query -q `"What are danger signs in a sick child?`""
Write-Host "  6. Start API:     python main.py --config config_windows.yaml --mode serve"
Write-Host ""
Write-Host "  API docs will be at: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host ""
