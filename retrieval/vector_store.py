"""
ChromaDB persistent vector store wrapper.

Features:
- Persistent SQLite-backed storage (no external server)
- Cosine similarity (HNSW index)
- Metadata filtering support
- Doc-hash tracking for idempotent ingestion
- Stats endpoint for /health API
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Cosine distance threshold for retrieval filtering.
# ChromaDB cosine distance: 0 = identical vectors, 1 = orthogonal, 2 = opposite.
# Chunks with distance > this value are treated as irrelevant and discarded.
# Tune this value by running: python main.py --mode eval
#   Start conservative (0.8) and lower toward 0.5 as KB grows.
RETRIEVAL_DISTANCE_THRESHOLD: float = 0.65


class VectorStore:
    """
    Wraps ChromaDB for offline persistent vector storage.

    Each chunk is stored with its embedding, raw text, and metadata dict.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        persist_dir: str = config["retrieval"]["persist_dir"]
        collection_name: str = config["retrieval"]["collection_name"]

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB ready — collection '{collection_name}' "
            f"({self._collection.count()} chunks) at '{persist_dir}'"
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Upsert a batch of chunks.
        Metadata values are coerced to Chroma-compatible types (str/int/float/bool).
        """
        clean_metas = [
            {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in m.items()
            }
            for m in metadatas
        ]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=clean_metas,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        ANN search. Returns a list of result dicts ordered by relevance:
            {id, text, metadata, distance, score}
        where score = 1 - distance (cosine similarity, higher = more relevant).
        """
        n = min(top_k, self._collection.count() or 1)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results = []
        for i, (doc, meta, dist) in enumerate(
            zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0])
        ):
            if dist <= RETRIEVAL_DISTANCE_THRESHOLD:
                results.append(
                    {
                        "id": raw["ids"][0][i],
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                        "score": max(0.0, 1.0 - dist),  # clamp to [0,1]
                    }
                )

        if not results:
            logger.info(
                f"All {n} retrieved chunks exceeded distance threshold "
                f"({RETRIEVAL_DISTANCE_THRESHOLD}). Returning empty — will trigger fallback."
            )
        return results

    def list_doc_hashes(self) -> List[str]:
        """Return all unique `doc_hash` values stored (used for idempotency)."""
        if self._collection.count() == 0:
            return []
        try:
            result = self._client.get_collection(
                self._collection.name
            ).get(limit=50_000, include=["metadatas"])
            return list({m.get("doc_hash", "") for m in result["metadatas"]} - {""})
        except Exception as e:
            logger.warning(f"Could not list doc hashes: {e}")
            return []

    # ── Utility ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._collection.count()

    def stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.count(),
            "collection_name": self._collection.name,
        }

    def delete_collection(self) -> None:
        """Wipe and recreate the collection. Use with caution."""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Collection '{name}' deleted and recreated.")
