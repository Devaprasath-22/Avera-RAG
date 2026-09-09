"""
FastAPI service for the Avera Medical RAG system.

Endpoints:
  POST /ingest        — ingest a document file or directory
  POST /query         — text-only RAG query
  POST /query/image   — multipart: image + question
  GET  /health        — system status (memory, loaded model, collection size)
  GET  /metrics       — rolling p50/p95 latency from last N queries

Single-worker design (workers=1) — models are NOT thread-safe.
The pipeline is kept as application state and shared across requests.

Start with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
"""

import logging
import os
import tempfile
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import psutil
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ingestion.pipeline import IngestionPipeline
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ── Config loading ────────────────────────────────────────────────────────────

def _load_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Application state ─────────────────────────────────────────────────────────

class AppState:
    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.ingest_pipeline: Optional[IngestionPipeline] = None
        # Rolling latency windows for /metrics
        self.latency_window: int = 100
        self.latency_records: Deque[Dict] = deque(maxlen=100)


_state = AppState()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Avera RAG API…")
    _state.config = _load_config()
    _state.latency_window = _state.config.get("logging", {}).get("latency_window", 100)
    _state.latency_records = deque(maxlen=_state.latency_window)

    _state.rag_pipeline = RAGPipeline(_state.config)
    _state.ingest_pipeline = IngestionPipeline(_state.config)
    logger.info("Avera RAG API ready.")
    yield
    # Shutdown
    logger.info("Shutting down Avera RAG API…")
    if _state.rag_pipeline:
        _state.rag_pipeline.manager.unload_all()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Avera Medical RAG API",
    description=(
        "Offline medical image analysis + document RAG + voice assistant. "
        "Runs 100% on-device (Jetson Orin Nano 8GB). "
        "This is an AI assistant — always verify clinical decisions with a qualified healthcare provider."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    keep_llm_loaded: bool = False


class IngestRequest(BaseModel):
    path: str           # file or directory path on the Jetson filesystem
    force: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: Dict[str, float]
    retrieval_count: int
    has_image: bool


class IngestResponse(BaseModel):
    status: str
    summary: Dict[str, Any]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_text(req: QueryRequest) -> QueryResponse:
    """
    Text-only RAG query.
    Returns a grounded answer with inline citations and source snippets.
    """
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    t0 = time.perf_counter()
    result = _state.rag_pipeline.query(
        question=req.question,
        keep_llm_loaded=req.keep_llm_loaded,
    )
    _state.latency_records.append({"total_ms": result.latency.get("total_ms", 0), "type": "text"})

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        latency_ms=result.latency,
        retrieval_count=result.retrieval_count,
        has_image=False,
    )


@app.post("/query/image", response_model=QueryResponse, tags=["RAG"])
async def query_with_image(
    question: str = Form(...),
    image: UploadFile = File(...),
    keep_llm_loaded: bool = Form(False),
) -> QueryResponse:
    """
    Multipart query: image + question.
    Runs VLM image analysis → enriched retrieval → grounded LLM answer.
    """
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    # Save uploaded image to a temp file
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = _state.rag_pipeline.query(
            question=question,
            image_path=tmp_path,
            keep_llm_loaded=keep_llm_loaded,
        )
        _state.latency_records.append(
            {"total_ms": result.latency.get("total_ms", 0), "type": "image"}
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        latency_ms=result.latency,
        retrieval_count=result.retrieval_count,
        has_image=True,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(req: IngestRequest) -> IngestResponse:
    """
    Ingest a document file or directory into the vector store.
    Idempotent — already-indexed documents are skipped unless force=True.
    """
    target = Path(req.path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    try:
        if target.is_dir():
            summary = _state.ingest_pipeline.ingest_directory(target, force=req.force)
        else:
            n = _state.ingest_pipeline.ingest_file(target, force=req.force)
            summary = {"chunks_added": n, "collection_size": _state.ingest_pipeline.store.count()}
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(status="ok", summary=summary)


@app.get("/health", tags=["System"])
async def health() -> Dict[str, Any]:
    """System status: memory, collection size, loaded model."""
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / 1024 ** 2
    virtual_mb = proc.memory_info().vms / 1024 ** 2

    gpu_info: Dict = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024 ** 2, 1),
                "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2, 1),
            }
    except ImportError:
        pass

    return {
        "status": "ok",
        "collection_size": _state.ingest_pipeline.store.count() if _state.ingest_pipeline else 0,
        "current_model": _state.rag_pipeline.manager._current.value if _state.rag_pipeline else "none",
        "memory": {
            "rss_mb": round(rss_mb, 1),
            "virtual_mb": round(virtual_mb, 1),
            **gpu_info,
        },
    }


@app.get("/metrics", tags=["System"])
async def metrics() -> Dict[str, Any]:
    """Rolling p50/p95 latency from the last N queries."""
    records = list(_state.latency_records)
    if not records:
        return {"message": "No queries recorded yet.", "window": _state.latency_window}

    latencies = sorted(r["total_ms"] for r in records)
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[min(int(n * 0.95), n - 1)]

    text_queries = sum(1 for r in records if r.get("type") == "text")
    image_queries = sum(1 for r in records if r.get("type") == "image")

    return {
        "query_count": n,
        "window_size": _state.latency_window,
        "latency_p50_ms": round(p50),
        "latency_p95_ms": round(p95),
        "latency_min_ms": round(latencies[0]),
        "latency_max_ms": round(latencies[-1]),
        "text_queries": text_queries,
        "image_queries": image_queries,
    }
