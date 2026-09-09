# 🩺 Avera Medical RAG

> **Offline Multimodal Medical AI Assistant for Resource-Constrained Edge Devices**  
> Designed for **NVIDIA Jetson Orin Nano (8GB)** and **Windows / Linux** edge environments.

---

## 🌟 Overview

**Avera Medical RAG** is an edge-native, fully offline medical decision support assistant. Built specifically for rural clinics and resource-limited healthcare facilities without reliable internet access, Avera provides grounded clinical intelligence, multilingual voice interaction, and multimodal image inspection directly on edge hardware.

---

## ✨ Key Features

- **🏥 Grounded Clinical Knowledge**: Ingests authoritative sources (MedlinePlus XML, WHO IMCI Chart Booklet) into a local vector knowledge base. Every statement cites its source (e.g. `[SOURCE 1]`).
- **🛡️ Strict Anti-Hallucination Safeguards**:
  - Semantic consistency checker detects and blocks ungrounded demographic assumptions (e.g. pediatric terms injected into adult queries).
  - Out-of-scope queries return standardized, multi-lingual safe clinical referrals.
- **🗣️ Indic Multilingual Voice Assistant**:
  - Speaks and understands 5 languages:
    - **English** (`Voice:en`)
    - **Tamil / தமிழ்** (`Voice:ta`)
    - **Hindi / हिन्दी** (`Voice:hi`)
    - **Telugu / తెలుగు** (`Voice:te`)
    - **Kannada / ಕನ್ನಡ** (`Voice:ka` / `kn`)
  - Real-time on-the-fly voice language switching.
  - Automatic Indic script detection ensuring natural neural voice synthesis.
- **⏹️ Interruptible Voice ("Stop While Speaking")**:
  - Non-blocking audio playback allows users to halt Avera's speech at any moment by pressing <kbd>Space</kbd>, <kbd>Enter</kbd>, or <kbd>Ctrl+C</kbd>, or speaking stop keywords (`stop`, `நில்`, `रुको`, `ఆగు`, `ನಿಲ್ಲು`).
- **⚡ Sequential Memory Management**:
  - Dynamically orchestrates VLM, Embedder, Reranker, and LLM to operate within an 8GB unified memory envelope.
- **🌐 Offline FastAPI Service**:
  - Production-ready single-worker REST API with rolling latency metrics and multipart image+text query endpoints.

---

## 📁 Repository Structure

```
Avera-RAG/
├── api/                    # FastAPI web server and schemas
│   ├── server.py
│   └── __init__.py
├── data/                   # Clinical reference datasets
│   └── imci_chart_booklet.jsonl
├── docs/                   # System design & architecture plans
│   └── jetson-medical-rag-implementation-plan.md
├── eval/                   # Benchmarks, regression suite & language tests
│   ├── benchmarks.py
│   ├── qa_pairs.jsonl
│   └── test_regression.py
├── ingestion/              # Document parsers, chunking, and loaders
│   ├── chunker.py
│   ├── loaders.py
│   └── pipeline.py
├── models/                 # LLM, VLM & memory lifecycle management
│   ├── llm.py
│   ├── model_manager.py
│   └── vlm.py
├── rag/                    # RAG pipeline, safety checks, prompts
│   ├── consistency_check.py
│   ├── pipeline.py
│   └── prompts.py
├── retrieval/              # Vector store (ChromaDB), BGE embeddings & reranker
│   ├── embedder.py
│   ├── reranker.py
│   └── vector_store.py
├── voice/                  # Offline ASR (faster-whisper) & TTS (Edge-TTS / Indic-TTS)
│   ├── asr.py
│   └── tts.py
├── config.yaml             # Production Jetson Orin Nano configuration
├── config_windows.yaml     # Windows local development configuration
├── main.py                 # Central CLI entry point
├── requirements.txt        # Jetson dependencies
└── requirements_windows.txt# Windows dependencies
```

---

## 🚀 Quick Start

### 1. Windows Setup
```powershell
# Clone the repository
git clone https://github.com/Devaprasath-22/Avera-RAG.git
cd Avera-RAG

# Run automated setup
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

### 2. Jetson Orin Nano Setup
```bash
git clone https://github.com/Devaprasath-22/Avera-RAG.git
cd Avera-RAG
bash setup_jetson.sh
```

---

## 💻 CLI Usage

All capabilities are accessible via `main.py`:

### 🎙️ Interactive Voice Mode
```powershell
python main.py --config config_windows.yaml --mode voice
```
When prompted, select your preferred voice:
```text
choose Avera Available lang: Voice:en, voice:ta, Voice:hi, Voice:te, Voice:ka

  [1] Voice:en — English
  [2] Voice:ta — Tamil (தமிழ்)
  [3] Voice:hi — Hindi (हिन्दी)
  [4] Voice:te — Telugu (తెలుగు)
  [5] Voice:ka — Kannada (ಕನ್ನಡ)
```
*Tip: You can interrupt Avera anytime while it is speaking by pressing Space, Enter, or Ctrl+C.*

### 💬 Single Query Mode
```powershell
# English
python main.py --config config_windows.yaml --mode query --question "What are the danger signs of dehydration?"

# Tamil
python main.py --config config_windows.yaml --mode query --question "எனக்கு வாந்தி உள்ளது" --language Voice:ta

# Hindi
python main.py --config config_windows.yaml --mode query --question "मुझे दस्त हो रहे हैं" --language Voice:hi
```

### 📚 Ingest Documents
```powershell
# Ingest clinical guidelines
python main.py --config config_windows.yaml --mode ingest --path data/imci_chart_booklet.jsonl
```

### 🌐 Start REST API
```powershell
python main.py --config config_windows.yaml --mode serve
```

### 🧪 Run Regression Tests
```powershell
python eval/test_regression.py
```

---

## ⚠️ Medical Disclaimer

*Avera is an artificial intelligence decision support assistant created to assist healthcare workers in rural or low-resource settings. It is NOT a substitute for professional clinical judgment, diagnosis, or treatment. Always verify critical recommendations with a licensed healthcare professional.*

---

## 📄 License

MIT License. Developed by **Devaprasath-22**.
