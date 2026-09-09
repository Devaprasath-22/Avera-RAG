# Jetson Orin Nano 8GB — Offline Medical Image + RAG + Voice Assistant
## Full Implementation Plan for Build Agent

**Target hardware:** NVIDIA Jetson Orin Nano 8GB (shared CPU/GPU memory — this is the #1 constraint that shapes every decision below)
**Goal:** Voice in (Whisper ASR) → Medical image analysis (VLM) + Document RAG → Grounded answer (LLM) → Voice out (TTS), 100% offline, low latency.

---

## 1. Model Recommendation & Comparison

### 1.1 Text LLM: **Qwen2.5-1.5B-Instruct** ✅ (not Base — see caveat)

| Dimension | 0.5B | **1.5B (recommended)** | 3B |
|---|---|---|---|
| Medical reasoning | Weak — loses thread across multi-hop clinical reasoning | Adequate for retrieval-grounded QA; struggles with novel inference not in context | Noticeably better unaided reasoning, but RAG reduces this advantage since answers should be grounded, not invented |
| RAG performance | Poor — frequently ignores retrieved context, hallucinates | Good — follows instruction-tuned RAG prompts, cites chunks reliably | Best raw performance, but marginal RAG gain over 1.5B doesn't justify 2x+ VRAM/latency |
| Context handling | Handles ~4-8K effectively before degrading | Handles 8-16K well (Qwen2.5 native ctx up to 32K) — enough for 5-8 retrieved chunks | Slightly more robust at long context, not decisive for your chunk counts |
| Hallucination tendency | High, especially with sparse/ambiguous context | Moderate — controllable with strict prompt grounding + low temperature | Lower, but RAG grounding matters far more than model size here |
| Inference speed (Jetson, INT4) | ~40-60 tok/s | **~20-30 tok/s** | ~8-14 tok/s |
| RAM/VRAM (INT4, KV cache incl.) | ~0.6-0.8 GB | **~1.3-1.6 GB** | ~2.5-3.2 GB |
| Suitability for 8GB Jetson | Fits easily but quality risk too high for medical use | **Best trade-off** — leaves headroom for VLM + embedder + Whisper + TTS running concurrently | Fits alone, but concurrent VLM+ASR+TTS pipeline will thrash memory/swap |

**Verdict:** 1.5B-Instruct is the sweet spot. Reserve 3B only if you drop the VLM and ASR/TTS to text-only, or move those stages off-device.

### 1.2 Coder variants — do not use
Qwen2.5-Coder models are pretrained/fine-tuned toward code syntax and structured programming tasks. This actively *hurts* natural clinical language generation and offers zero RAG-relevant advantage. Skip them entirely.

### 1.3 Vision (missing from your list — required)
Your candidate list is text-only. Add: **Qwen2-VL-2B-Instruct** (best accuracy/size ratio for medical image description; handles X-ray/derm/wound photos reasonably for general findings — **not** a diagnostic-grade radiology model, flag this to users) or **Moondream2** (1.8B, faster but less detailed) as a lighter fallback.

**Memory-saving architecture option:** Use Qwen2-VL-2B-Instruct for *both* image captioning/findings AND, once RAG context is retrieved, feed that context back into the same model for final synthesis — avoiding a second loaded LLM entirely. This is the recommended default for 8GB. If you need the text-only 1.5B model as a separate stronger synthesizer, only run the VLM and LLM sequentially (unload VLM weights before loading LLM), never concurrently resident.

---

## 2. Embedding Model
**BAAI/bge-small-en-v1.5** (33M params, 384-dim, ~130MB in FP16)
- Fast enough for ~5,000 pages (≈15-20K chunks) to embed in minutes on Jetson CPU/GPU
- Strong retrieval quality for general + biomedical English text relative to its size
- Alternative if you want domain bias: `NeuML/pubmedbert-base-embeddings` — better medical recall but ~4x larger/slower; only worth it if retrieval quality testing shows bge-small missing clinically relevant chunks.

## 3. Vector Database
**ChromaDB (persistent, embedded/SQLite-backed)**
- Zero external server, pure local file storage — ideal for offline Jetson
- Native metadata filtering (source doc, page, section) — useful for citations
- At ~15-20K chunks this is well within Chroma's comfortable range
- Alternative: **FAISS** (raw `IndexFlatIP` or `IndexHNSWFlat`) if you want maximum retrieval speed and are willing to hand-roll metadata storage separately (e.g., a SQLite side-table keyed by vector ID). Use FAISS only if Chroma's query latency becomes a bottleneck in testing.

## 4. Reranker (optional but recommended)
**cross-encoder/ms-marco-MiniLM-L-6-v2** (INT8) — retrieve top 20 with bge-small, rerank down to top 4-6 before feeding the LLM. This meaningfully improves grounding quality for a small latency cost (~50-100ms for 20 pairs on Jetson). Skip only if you profile and find latency budget too tight — retrieval quality will drop noticeably without it.

---

## 5. Document Processing Pipeline

```
Ingestion → OCR (if scanned) → Cleaning → Chunking → Metadata tagging
  → Embedding (batch) → Vector store upsert
```

| Stage | Tool |
|---|---|
| PDF ingestion | `PyMuPDF (fitz)` — fast, preserves layout/page numbers |
| DOCX ingestion | `python-docx` |
| TXT ingestion | direct read |
| OCR (scanned pages) | `Tesseract` (lightweight, offline) or `PaddleOCR` (better accuracy, more RAM — use only if Tesseract accuracy is insufficient on your docs) |
| Chunking | Recursive/semantic chunker (LangChain `RecursiveCharacterTextSplitter` or custom sentence-window) — see §6 |
| Metadata | source filename, page number, section heading (regex/heading-detection), chunk index |
| Embedding | bge-small-en-v1.5, batched (32-64 chunks/batch) |
| Vector storage | ChromaDB persistent collection |
| Retrieval | top-k=15-20 via cosine similarity, optional MMR for diversity |
| Reranking | cross-encoder MiniLM → top 4-6 final chunks |

Run ingestion as a **one-time offline batch job on the Jetson (or offloaded to a dev machine and copied over)** — this is not a runtime-latency-sensitive path, so use the more accurate OCR/chunking settings there.

## 6. Chunk Size & Overlap
**500-700 tokens per chunk, 60-100 token overlap (~12-15%).** Medical documents (guidelines, drug monographs, protocols) often have dense, context-dependent statements (dosages tied to preceding conditions) — overlap prevents splitting a dosage from its qualifying clause. Prefer splitting on section/heading boundaries first, then recursively on paragraph → sentence if a section exceeds the token budget.

---

## 7. Combining VLM Findings + RAG

1. VLM produces **structured findings** (force JSON output): `{"observations": [...], "region": "...", "confidence_notes": "..."}`
2. Build a **retrieval query** from the structured findings (not the raw image) — e.g., concatenate key observation terms.
3. Retrieve + rerank top medical-document chunks against that query.
4. Final prompt to the LLM (or same VLM in text mode) combines: system instruction ("only state what is supported by the provided context; do not diagnose"), VLM findings, retrieved chunks (with citations), and the user's question.
5. Require the model to **cite chunk IDs/sources inline** — this is your main lever against hallucination and gives you a faithfulness signal for free (see §13).

## 8. Recommended Architecture

```
[Mic] → faster-whisper (ASR) → text query
                                    │
[Image] → Qwen2-VL-2B (findings, JSON) ──┐
                                    │     │
                       query builder ←────┘
                                    │
                    bge-small-en-v1.5 (embed query)
                                    │
                        ChromaDB (top-20 retrieve)
                                    │
                  MiniLM cross-encoder (rerank → top 5)
                                    │
        Qwen2.5-1.5B-Instruct (or Qwen2-VL text mode)
        [VLM findings + retrieved chunks + question] → grounded answer
                                    │
                         Piper TTS / AI4Bharat Indic-TTS
                                    │
                                [Speaker]
```

Run VLM and LLM **sequentially, not concurrently resident**, to stay inside 8GB alongside Whisper+TTS (see budget table below).

## 9. Quantization
- **LLM (Qwen2.5-1.5B-Instruct):** INT4 (AWQ or GGUF Q4_K_M via llama.cpp with CUDA/Jetson build) — best latency/quality trade-off; INT8 if you see accuracy regressions in testing.
- **VLM (Qwen2-VL-2B):** INT4/INT8 via GGUF or TensorRT-LLM if available for Jetson; FP16 only if VRAM allows and INT4 vision degrades findings too much (test this — vision quantization is more failure-prone than text).
- **Embedding model:** FP16 is fine — small enough that INT8 gains little.
- **Reranker:** INT8.
- **Whisper:** INT8 via `faster-whisper` (CTranslate2 backend) — this is the standard, well-supported path on Jetson.

## 10. Estimated Memory & Latency (Jetson Orin Nano 8GB, shared memory)

| Component | Memory (quantized) | Latency |
|---|---|---|
| Qwen2.5-1.5B-Instruct INT4 | ~1.3-1.6 GB | ~20-30 tok/s |
| Qwen2-VL-2B-Instruct INT4 | ~1.8-2.2 GB | ~1-3s per image (findings) |
| bge-small-en-v1.5 FP16 | ~0.15 GB | <50ms/query embed |
| MiniLM reranker INT8 | ~0.1 GB | ~50-100ms for 20 pairs |
| faster-whisper small.en INT8 | ~0.3-0.5 GB | ~0.3-0.6x realtime |
| Piper TTS | ~0.1-0.2 GB | near-instant (fast vocoder) |
| **Peak concurrent (sequential loading)** | **~2.5-3.5 GB** | End-to-end voice-in → voice-out: **~3-6s** for a typical query with 1 image |

This leaves comfortable headroom under 8GB, including OS overhead — the design is intentionally conservative so you're not fighting swap/thermal throttling in the field.

## 11. Benchmarking Plan

| Metric | Method |
|---|---|
| Retrieval accuracy | Hand-build 50-100 Q&A pairs with known source passages; measure Recall@5/10 (does the gold chunk appear in top-k?) |
| RAG answer quality | Human rubric scoring (1-5) on relevance/completeness against a held-out Q&A set |
| Faithfulness/grounding | Check whether every claim in the answer maps to a cited chunk (manual or LLM-as-judge with a stronger offline/cloud model during dev only) |
| Hallucination rate | % of answers containing claims not traceable to retrieved context or VLM findings |
| VLM accuracy | Compare structured findings against a small labeled image set (ground-truth annotations you or a clinician provide) |
| End-to-end latency | Instrument each pipeline stage (ASR, VLM, retrieval, rerank, LLM, TTS) separately; log p50/p95 |

## 12. Evaluation Dataset Strategy (5,000 pages)
- Stratify your corpus by document type (guidelines, drug info, protocols) and sample proportionally when building test Q&A sets.
- Generate candidate Q&A pairs semi-automatically (use a larger model *offline during dev only*, e.g., via API, to draft questions from passages), then have a human (ideally with medical background) review/correct them — do not deploy on unreviewed synthetic eval data.
- Target 100-200 curated eval pairs minimum before calling retrieval/RAG quality "validated."

---

## 13. ASR + TTS Addition

**ASR: faster-whisper** (`small.en` or `base.en` for English; `small`/`medium` multilingual if Indian-language input needed), INT8 via CTranslate2 — well-tested on Jetson, real-time-capable at small/base sizes.

**TTS:** For fully offline operation, prefer **Piper TTS** (very lightweight, ONNX-based, near-instant on Jetson, good English voices) as the default. If Indian-language voice output is required, **Bhashini** is primarily a hosted/API service (not built for fully offline edge deployment) — for **offline** Indian-language TTS, use **AI4Bharat's open-source Indic-TTS/VITS models** instead, which are downloadable and run locally. Flag this trade-off explicitly: Bhashini is the right call only if occasional internet connectivity is acceptable; otherwise AI4Bharat's offline models are the correct substitute to stay within your "everything local" requirement.

---

## 14. Full Tech Stack

| Layer | Choice |
|---|---|
| ASR | faster-whisper (small.en/base.en, INT8) |
| VLM | Qwen2-VL-2B-Instruct (INT4) |
| LLM | Qwen2.5-1.5B-Instruct (INT4, GGUF via llama.cpp) |
| Embedding | BAAI/bge-small-en-v1.5 |
| Vector DB | ChromaDB (persistent) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (INT8) |
| OCR | Tesseract (fallback: PaddleOCR) |
| TTS | Piper TTS (offline default) / AI4Bharat Indic-TTS (offline Indian languages) |
| Orchestration | Python; simple FastAPI service or direct pipeline script — avoid heavy agent frameworks that add latency/memory overhead on-device |
| Runtime | llama.cpp (CUDA build for Jetson) or TensorRT-LLM if available for your JetPack version |

---

## 15. Build Order (for implementation agent)
1. Document pipeline first (ingest → chunk → embed → store) — validate retrieval quality on curated Q&A before touching models.
2. Stand up LLM (Qwen2.5-1.5B-Instruct, INT4) + retrieval + reranker as text-only RAG; benchmark faithfulness/hallucination.
3. Add VLM stage; test structured-findings JSON reliability separately before wiring into the RAG query builder.
4. Add ASR input, confirm end-to-end text pipeline latency budget holds.
5. Add TTS output last — it's the least failure-prone stage and easiest to swap.
6. Profile memory at each stage addition — catch VRAM regressions early rather than after full integration.
7. Run the benchmark suite (§11) after each major stage addition, not just at the end.
