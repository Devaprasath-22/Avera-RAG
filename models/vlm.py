"""
VLM backend — Qwen2-VL-2B-Instruct via HuggingFace Transformers + BitsAndBytes INT4.

Memory: ~1.8-2.2 GB resident (INT4).
Latency: ~1-3s per image on Jetson Orin Nano 8GB.

Output contract: always returns a structured JSON dict:
    {
        "observations": ["finding 1", "finding 2", ...],
        "region": "anatomical region or image type",
        "confidence_notes": "any uncertainty or quality caveats",
        "raw": "full model output text (for debugging)"
    }

Caller must treat this as unverified AI output, NOT as a clinical diagnosis.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Instruction given to Qwen2-VL to produce structured JSON findings
_VLM_SYSTEM_PROMPT = (
    "You are a medical image analysis assistant. "
    "Describe only what is visually present in the image. "
    "Do NOT diagnose or prescribe. "
    "Always express uncertainty when findings are ambiguous. "
    "Respond ONLY with a valid JSON object matching exactly this schema:\n"
    '{"observations": [<list of strings>], "region": "<string>", '
    '"confidence_notes": "<string>"}'
)


class VLMBackend:
    """
    Qwen2-VL-2B-Instruct wrapper for medical image captioning.

    Load is deferred until :meth:`load` is called explicitly
    (via :class:`ModelManager`).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        vlm_cfg = config["models"]["vlm"]
        self.hf_id: str = vlm_cfg["hf_id"]
        self.load_in_4bit: bool = vlm_cfg.get("load_in_4bit", True)
        self.max_new_tokens: int = vlm_cfg.get("max_new_tokens", 256)
        self._model = None
        self._processor = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load Qwen2-VL model and processor."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        # ── Windows / CPU guard ──────────────────────────────────────────────
        # VLM requires CUDA for practical inference.
        # On CPU it would take 30-120s per image and is not usable.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "VLM requires a CUDA GPU and is not supported on CPU.\n"
                "On Windows without a GPU, set 'enabled: false' under models.vlm\n"
                "in config_windows.yaml. VLM will be available on Jetson."
            )

        logger.info(f"Loading VLM: {self.hf_id} (4bit={self.load_in_4bit})…")
        t0 = time.perf_counter()

        self._processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.hf_id,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # FP16 fallback (no BnB required)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.hf_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        self._model.eval()
        logger.info(f"VLM loaded in {time.perf_counter()-t0:.1f}s")

    def unload(self) -> None:
        """Release GPU/CPU memory."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("VLM unloaded.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def describe_image(
        self,
        image_path: str | Path,
        extra_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON findings for a medical image.

        Parameters
        ----------
        image_path : str or Path
            Local path to the image (JPEG, PNG, BMP supported).
        extra_instruction : str, optional
            Additional context to append to the prompt
            (e.g. "Focus on the chest region.").

        Returns
        -------
        dict
            {"observations": [...], "region": "...", "confidence_notes": "...", "raw": "..."}
            Falls back to empty observations + error note on parse failure.
        """
        if self._model is None:
            raise RuntimeError("VLM not loaded. Call load() first.")

        import torch
        from PIL import Image

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        user_text = _VLM_SYSTEM_PROMPT
        if extra_instruction:
            user_text += f"\n\nAdditional context: {extra_instruction}"

        # Build Qwen2-VL message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        t0 = time.perf_counter()
        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic
            )

        # Decode only generated tokens (skip prompt)
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        raw_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        elapsed = time.perf_counter() - t0
        logger.debug(f"VLM inference in {elapsed:.1f}s")

        return self._parse_findings(raw_text)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_findings(raw: str) -> Dict[str, Any]:
        """
        Parse the model's JSON output into a structured dict.
        Robust to minor JSON formatting issues from the model.
        """
        # Extract JSON block if surrounded by markdown fences
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            json_str = json_match.group(0) if json_match else None

        if json_str:
            try:
                parsed = json.loads(json_str)
                parsed.setdefault("observations", [])
                parsed.setdefault("region", "unspecified")
                parsed.setdefault("confidence_notes", "")
                parsed["raw"] = raw
                return parsed
            except json.JSONDecodeError:
                logger.warning(f"JSON parse failed for VLM output: {raw[:200]}")

        # Graceful fallback
        return {
            "observations": [raw] if raw else [],
            "region": "unspecified",
            "confidence_notes": "Could not parse structured output — raw text returned.",
            "raw": raw,
        }
