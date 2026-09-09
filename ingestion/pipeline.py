"""
Ingestion pipeline — orchestrates: load → chunk → embed → upsert.

Features:
- Idempotent: tracks ingested documents by content SHA-256 hash
- Batch embedding with configurable batch size
- Rich progress bar for directory ingestion
- Returns summary statistics
"""

import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ingestion.chunker import MedicalChunker
from ingestion.loaders import SUPPORTED_EXTENSIONS, get_loader
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)
console = Console()


def _file_hash(path: Path) -> str:
    """Return first 16 hex digits of SHA-256 of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Usage:
        pipeline = IngestionPipeline(config)
        pipeline.ingest_file("path/to/imci.jsonl")
        pipeline.ingest_directory("path/to/docs/")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self.chunker = MedicalChunker(
            chunk_size_tokens=config["ingestion"]["chunk_size"],
            overlap_tokens=config["ingestion"]["chunk_overlap"],
        )
        self.embedder = Embedder(config)
        self.store = VectorStore(config)

        # Pre-load known hashes to enable idempotency
        self._known_hashes: set[str] = set(self.store.list_doc_hashes())
        logger.info(
            f"IngestionPipeline ready — {len(self._known_hashes)} documents already indexed"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_file(self, path: str | Path, force: bool = False) -> int:
        """
        Ingest a single file into the vector store.

        Parameters
        ----------
        path : str or Path
            Path to the document.
        force : bool
            Re-ingest even if the file hash is already indexed.

        Returns
        -------
        int
            Number of chunks upserted (0 if skipped).
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        doc_hash = _file_hash(path)
        if doc_hash in self._known_hashes and not force:
            logger.info(f"Skipping (already indexed): {path.name} [{doc_hash}]")
            return 0

        logger.info(f"Ingesting: {path.name}")
        t_start = time.perf_counter()

        loader = get_loader(path)
        all_chunks = []

        for record in loader.load(path):
            text = record.get("text", "").strip()
            if not text:
                continue
            meta = {**record["metadata"], "doc_hash": doc_hash}
            chunks = self.chunker.chunk(text=text, metadata=meta)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning(f"No content extracted from: {path.name}")
            return 0

        # Batch embed → upsert
        batch_size = self.cfg["ingestion"]["batch_size"]
        total_upserted = 0

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embedder.embed(texts)
            ids = [
                hashlib.sha256(c.text.encode()).hexdigest()[:24]
                for c in batch
            ]
            metadatas = [c.metadata for c in batch]
            self.store.upsert(
                ids=ids,
                embeddings=embeddings,
                texts=texts,
                metadatas=metadatas,
            )
            total_upserted += len(batch)

        self._known_hashes.add(doc_hash)
        elapsed = time.perf_counter() - t_start

        logger.info(
            f"Ingested {path.name}: {total_upserted} chunks "
            f"({total_upserted / elapsed:.0f} chunks/s, {elapsed:.1f}s total)"
        )
        return total_upserted

    def ingest_directory(
        self,
        directory: str | Path,
        extensions: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest all supported documents in a directory (recursive).

        Returns a summary dict:
            {files_found, files_ingested, chunks_added, failed, collection_size}
        """
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        exts = extensions or SUPPORTED_EXTENSIONS
        files = sorted(f for f in directory.rglob("*") if f.suffix.lower() in exts)

        if not files:
            logger.warning(f"No supported files in {directory} (extensions: {exts})")
            return {"files_found": 0, "files_ingested": 0, "chunks_added": 0, "failed": []}

        total_chunks = 0
        ingested = 0
        failed: List[str] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description:<45}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting…", total=len(files))

            for f in files:
                desc = f.name[:42] + ("…" if len(f.name) > 42 else "")
                progress.update(task, description=desc)
                try:
                    n = self.ingest_file(f, force=force)
                    total_chunks += n
                    if n > 0:
                        ingested += 1
                except Exception as e:
                    logger.error(f"Failed to ingest {f.name}: {e}", exc_info=True)
                    failed.append(str(f))
                finally:
                    progress.advance(task)

        summary = {
            "files_found": len(files),
            "files_ingested": ingested,
            "chunks_added": total_chunks,
            "failed": failed,
            "collection_size": self.store.count(),
        }
        console.print(f"\n[bold green]✓ Ingestion complete[/bold green] — {summary}")
        return summary
