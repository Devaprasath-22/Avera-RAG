"""
Cross-encoder reranker — MiniLM-L-6-v2 INT8.

Takes the top-K ANN results from ChromaDB and re-scores them using a
cross-encoder (query, passage) model, which is significantly more accurate
than bi-encoder cosine similarity for final passage selection.

Expected latency: ~50-100ms for 20 pairs on Jetson (INT8 CPU/GPU).
"""

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class Reranker:
    """
    Lazy-loaded MiniLM cross-encoder reranker.

    Usage:
        reranker = Reranker(config)
        top5 = reranker.rerank(query="...", results=chroma_results, top_n=5)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.model_name: str = config["models"]["reranker"]
        self.top_n: int = config["retrieval"]["rerank_top_n"]
        self._model = None  # lazy

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def _load(self) -> None:
        from sentence_transformers import CrossEncoder
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading reranker '{self.model_name}' on {device}…")
        t0 = time.perf_counter()

        try:
            self._model = CrossEncoder(
                self.model_name,
                device=device,
                local_files_only=True,
            )
        except Exception:
            self._model = CrossEncoder(
                self.model_name,
                device=device,
            )

        # Quantize to INT8 for ~2× speed on Jetson CPU
        if device == "cpu":
            try:
                import torch.quantization as tq
                self._model.model = tq.quantize_dynamic(
                    self._model.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                logger.info("Reranker quantized to INT8.")
            except Exception as e:
                logger.warning(f"INT8 quantization skipped: {e}")

        logger.info(f"Reranker ready in {time.perf_counter() - t0:.1f}s")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-score a list of retrieval results using cross-encoder scoring.

        Parameters
        ----------
        query : str
            The user query.
        results : List[dict]
            Output of VectorStore.query() — each dict has 'text', 'metadata', etc.
        top_n : int, optional
            How many to return. Defaults to config value.

        Returns
        -------
        List[dict]
            Results sorted descending by cross-encoder score, trimmed to top_n.
            Each result gains a 'rerank_score' key.
        """
        if not results:
            return []

        top_n = top_n or self.top_n
        t0 = time.perf_counter()

        pairs = [(query, r["text"]) for r in results]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for r, score in zip(results, scores):
            r["rerank_score"] = float(score)

        ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        top = ranked[:top_n]

        elapsed = time.perf_counter() - t0
        logger.debug(
            f"Reranked {len(results)} → {len(top)} in {elapsed*1000:.0f}ms "
            f"(top score: {top[0]['rerank_score']:.3f})"
        )
        return top
