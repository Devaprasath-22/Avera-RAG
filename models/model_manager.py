"""
ModelManager — enforces sequential model loading on Jetson Orin Nano 8GB.

The Jetson uses unified CPU/GPU memory. Running VLM + LLM concurrently
would exhaust the 8 GB pool. This manager:
  - Tracks which model is currently resident
  - Forces unload of the current model before loading a new one
  - Logs RSS + GPU memory before/after each load for profiling

Usage:
    manager = ModelManager()
    manager.require("llm")   # unloads VLM if resident, loads LLM
    manager.require("vlm")   # unloads LLM if resident, loads VLM
    manager.unload_all()
"""

import logging
import time
from enum import Enum
from typing import Any, Dict, Optional

import psutil

logger = logging.getLogger(__name__)


class ModelName(str, Enum):
    LLM = "llm"
    VLM = "vlm"
    NONE = "none"


class ModelManager:
    """
    Singleton-style sequential model loader.

    The embedder and reranker are small enough to stay resident always.
    Only the LLM (~1.5 GB) and VLM (~2 GB) are swapped.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self._current: ModelName = ModelName.NONE
        self._llm = None
        self._vlm = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_llm(self):
        """Return the LLM backend, loading it (and unloading VLM) if necessary."""
        self.require(ModelName.LLM)
        return self._llm

    def get_vlm(self):
        """Return the VLM backend, loading it (and unloading LLM) if necessary."""
        self.require(ModelName.VLM)
        return self._vlm

    def require(self, model: ModelName) -> None:
        """Ensure `model` is loaded, unloading the other if needed."""
        if self._current == model:
            return  # already resident

        # Unload whatever is currently loaded
        if self._current != ModelName.NONE:
            logger.info(f"Unloading {self._current.value} to make room for {model.value}")
            self._unload_current()

        self._log_memory("before load")
        self._load(model)
        self._log_memory("after load")

    def unload_all(self) -> None:
        """Release all model memory."""
        if self._current != ModelName.NONE:
            self._unload_current()

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self, model: ModelName) -> None:
        t0 = time.perf_counter()
        if model == ModelName.LLM:
            from models.llm import LLMBackend
            self._llm = LLMBackend(self.cfg)
            self._llm.load()
        elif model == ModelName.VLM:
            from models.vlm import VLMBackend
            self._vlm = VLMBackend(self.cfg)
            self._vlm.load()
        self._current = model
        logger.info(f"Loaded {model.value} in {time.perf_counter()-t0:.1f}s")

    def _unload_current(self) -> None:
        if self._current == ModelName.LLM and self._llm is not None:
            self._llm.unload()
            self._llm = None
        elif self._current == ModelName.VLM and self._vlm is not None:
            self._vlm.unload()
            self._vlm = None
        self._current = ModelName.NONE
        self._clear_cuda_cache()

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    @staticmethod
    def _log_memory(label: str) -> None:
        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / 1024 ** 2
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mb = torch.cuda.memory_allocated() / 1024 ** 2
                logger.info(f"[Memory {label}] RSS={rss_mb:.0f}MB GPU={gpu_mb:.0f}MB")
                return
        except ImportError:
            pass
        logger.info(f"[Memory {label}] RSS={rss_mb:.0f}MB")
