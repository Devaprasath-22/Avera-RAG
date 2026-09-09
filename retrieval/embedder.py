"""
Embedding model wrapper — BAAI/bge-small-en-v1.5.

Properties:
- 33M params, 384-dim, FP16
- Singleton: one model instance per process
- BGE query prefix applied automatically for retrieval
- CUDA device used when available (falls back to CPU)
"""

import logging
import time
from typing import List

logger = logging.getLogger(__name__)

# BGE models recommend this prefix for passage retrieval queries
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """
    Lazy-loaded singleton wrapper around sentence-transformers.

    The model is loaded once on first use and kept in memory.
    Call :meth:`embed` for batches, :meth:`embed_query` for single queries.
    """

    def __init__(self, config: dict) -> None:
        self.model_name: str = config["models"]["embedder"]
        self._model = None  # lazy init

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedder '{self.model_name}' on {device}…")
        t0 = time.perf_counter()

        self._model = SentenceTransformer(self.model_name, device=device)

        # Cast to FP16 on CUDA for ~2× memory saving (33M params → ~64 MB)
        if device == "cuda":
            self._model.half()

        elapsed = time.perf_counter() - t0
        logger.info(f"Embedder ready in {elapsed:.1f}s (device={device}, dtype=fp16)")

    def unload(self) -> None:
        """Release GPU/CPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Embedder unloaded.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed a list of passage texts (no query prefix).

        Returns
        -------
        List[List[float]]
            L2-normalised float32 embedding vectors.
        """
        if not texts:
            return []

        t0 = time.perf_counter()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # enables cosine via dot product
            convert_to_numpy=True,
        )
        elapsed = time.perf_counter() - t0
        logger.debug(f"Embedded {len(texts)} passages in {elapsed:.2f}s")
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single user query with the BGE retrieval prefix.

        Returns
        -------
        List[float]
            Single L2-normalised embedding vector.
        """
        prefixed = _BGE_QUERY_PREFIX + query
        results = self.embed([prefixed], batch_size=1)
        return results[0]
