"""
LLM backend — Qwen2.5-1.5B-Instruct via llama-cpp-python.

Quantization: Q4_K_M GGUF (~1.1 GB on disk, ~1.3-1.6 GB resident).
All GPU layers offloaded via n_gpu_layers=-1 (Jetson unified memory).
Expected throughput: ~20-30 tok/s on Jetson Orin Nano 8GB.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMBackend:
    """
    Wraps llama-cpp-python for Qwen2.5-1.5B-Instruct GGUF inference.

    The model is loaded lazily via :meth:`load` and can be explicitly
    unloaded with :meth:`unload` to free memory for the VLM stage.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        llm_cfg = config["models"]["llm"]
        self.model_path: str = str(Path(llm_cfg["path"]).expanduser().resolve())
        self.n_ctx: int = llm_cfg["n_ctx"]
        self.n_gpu_layers: int = llm_cfg["n_gpu_layers"]
        self.temperature: float = llm_cfg["temperature"]
        self.max_tokens: int = llm_cfg["max_tokens"]
        self.stop: List[str] = llm_cfg.get("stop", [])
        self._llm = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the GGUF model into memory (CUDA-offloaded)."""
        if self._llm is not None:
            return  # already loaded

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"GGUF model not found: {self.model_path}\n"
                f"Run setup_jetson.sh to download it."
            )

        if os.name == "nt":
            try:
                import torch
                torch_lib = Path(torch.__file__).parent / "lib"
                if torch_lib.exists():
                    os.add_dll_directory(str(torch_lib))
            except Exception:
                pass

        from llama_cpp import Llama

        logger.info(f"Loading LLM: {Path(self.model_path).name}")
        t0 = time.perf_counter()

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
            seed=-1,  # random seed each run
        )
        logger.info(f"LLM loaded in {time.perf_counter()-t0:.1f}s")

    def unload(self) -> None:
        """Release model from memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            logger.info("LLM unloaded.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a completion for the given prompt.

        Parameters
        ----------
        prompt : str
            Full formatted prompt (system + user turn already embedded).
        max_tokens : int, optional
            Override config max_tokens.
        temperature : float, optional
            Override config temperature.
        stop : List[str], optional
            Additional stop sequences.

        Returns
        -------
        str
            Generated text (stripped).
        """
        if self._llm is None:
            raise RuntimeError("LLM not loaded. Call load() first.")

        effective_stop = list(self.stop) + (stop or [])

        t0 = time.perf_counter()
        output = self._llm(
            prompt=prompt,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            stop=effective_stop or None,
            repeat_penalty=1.15,
            echo=False,
        )
        elapsed = time.perf_counter() - t0

        text = output["choices"][0]["text"].strip()
        usage = output.get("usage", {})
        completion_tokens = usage.get("completion_tokens", "?")

        logger.debug(
            f"LLM generated {completion_tokens} tokens in {elapsed:.1f}s "
            f"({int(completion_tokens) / elapsed if elapsed > 0 and isinstance(completion_tokens, int) else '?'} tok/s)"
        )
        return text

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Chat completion (Qwen instruct chat format).

        Parameters
        ----------
        messages : List[dict]
            List of {role: "system"|"user"|"assistant", content: str} dicts.

        Returns
        -------
        str
            Assistant response text.
        """
        if self._llm is None:
            raise RuntimeError("LLM not loaded. Call load() first.")

        t0 = time.perf_counter()
        output = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            stop=self.stop or None,
            repeat_penalty=1.22,
        )
        elapsed = time.perf_counter() - t0

        text = output["choices"][0]["message"]["content"].strip()
        usage = output.get("usage", {})
        logger.debug(
            f"LLM chat: {usage.get('prompt_tokens','?')}+{usage.get('completion_tokens','?')} tokens "
            f"in {elapsed:.1f}s"
        )
        return text

    def translate_to_clinical_query(self, question: str, language: str = "en") -> str:
        """
        Translate an Indian language patient question or symptom into concise English
        clinical keywords for vector search against WHO / MedlinePlus documents.
        """
        if language == "en" or not question.strip():
            return question

        if self._llm is None:
            self.load()

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a clinical query translator. Convert Indian language medical symptoms "
                    "and questions into clean English search queries for medical document retrieval.\n\n"
                    "IMPORTANT: Translate ONLY what the patient said. Do NOT add age groups "
                    "(child, infant, adult) unless the patient explicitly mentioned them.\n\n"
                    "Examples:\n"
                    "User: எனக்கு தலைவலி உள்ளது\nOutput: headache causes and treatment\n\n"
                    "User: காய்ச்சல் உள்ளது\nOutput: fever symptoms and assessment\n\n"
                    "User: எனக்கு வயிற்றுப்போக்கு\nOutput: diarrhea treatment and hydration\n\n"
                    "User: மூச்சு வாங்குகிறது\nOutput: difficulty breathing respiratory assessment\n\n"
                    "User: मुझे तेज बुखार है\nOutput: high fever causes and treatment\n\n"
                    "User: కడుపు నొప్పిగా ఉంది\nOutput: abdominal pain causes and treatment\n\n"
                    "User: నాకు దగ్గు వస్తోంది\nOutput: cough respiratory infection assessment"
                ),
            },
            {"role": "user", "content": question},
        ]
        try:
            output = self._llm.create_chat_completion(
                messages=prompt,
                max_tokens=32,
                temperature=0.0,
                repeat_penalty=1.1,
            )
            raw = output["choices"][0]["message"]["content"].strip()
            if raw.lower().startswith("output:"):
                raw = raw[7:].strip()
            return raw if raw else question
        except Exception as e:
            logger.warning(f"Clinical query translation failed: {e}")
            return question
