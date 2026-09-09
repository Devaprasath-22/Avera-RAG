"""
ASR backend — faster-whisper (CTranslate2 INT8).

Model: small.en (default) — ~244M params, ~0.3-0.5 GB resident.
Latency: 0.3-0.6× real-time on Jetson (fast enough for short utterances).

Features:
- Transcribes audio files (WAV/MP3/FLAC)
- Records from microphone with VAD-based silence detection
- Returns transcript + per-segment word-level confidence
"""

import io
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# VAD parameters for mic recording
_SILENCE_THRESHOLD_DB = -40    # dBFS below which audio is considered silence
_SILENCE_DURATION_S = 1.5      # seconds of silence to trigger end of utterance
_MAX_RECORDING_S = 30          # hard cap on single recording
_SAMPLE_RATE = 16_000          # Hz (Whisper native)

_MEDICAL_INITIAL_PROMPTS: Dict[str, str] = {
    "ta": "மருத்துவ அறிகுறிகள்: காய்ச்சல், கைச்சல், தலைவலி, இருமல், சளி, வயிற்றுப்போக்கு, வாந்தி, மூச்சுத்திணறல்.",
    "hi": "चिकित्सीय लक्षण: बुखार, सिरदर्द, खांसी, जुकाम, दस्त, उल्टी, सांस लेने में कठिनाई।",
    "te": "వైద్య లక్షణాలు: జ్వరం, తలనొప్పి, దగ్గు, జలుబు, విరేచనాలు, వాంతులు, శ్వాస ఆడకపోవడం.",
    "kn": "ವೈದ್ಯಕೀಯ ಲಕ್ಷಣಗಳು: ಜ್ವರ, ತಲೆನೋವು, ಕೆಮ್ಮು, ನೆಗಡಿ, ಭೇದಿ, ವಾಂತಿ, ಉಸಿರಾಟದ ತೊಂದರೆ.",
    "en": "Medical symptoms: fever, headache, cough, cold, diarrhea, vomiting, breathing difficulty.",
}


class ASRBackend:
    """
    Lazy-loaded faster-whisper ASR backend.

    Usage:
        asr = ASRBackend(config)
        text = asr.transcribe_file("audio.wav")
        text = asr.transcribe_mic()  # records until silence detected
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        asr_cfg = config["models"]["asr"]
        self.model_size: str = asr_cfg["model_size"]
        self.device: str = asr_cfg.get("device", "cuda")
        self.compute_type: str = asr_cfg.get("compute_type", "int8")
        self.language: Optional[str] = asr_cfg.get("language", "en")
        self.beam_size: int = asr_cfg.get("beam_size", 5)
        self._model = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info(
            f"Loading ASR: faster-whisper {self.model_size} "
            f"({self.compute_type} on {self.device})…"
        )
        t0 = time.perf_counter()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info(f"ASR loaded in {time.perf_counter()-t0:.1f}s")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("ASR unloaded.")

    # ── Transcription ─────────────────────────────────────────────────────────

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        Parameters
        ----------
        audio_path : str or Path
            Path to audio file (WAV, MP3, FLAC, OGG).
        language : str, optional
            Language code override (e.g. "ta", "hi", "te", "kn", "en").

        Returns
        -------
        dict
            {"text": str, "query_en": str, "language": str, "confidence": float, "duration_s": float}
        """
        if self._model is None:
            self.load()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        t0 = time.perf_counter()
        target_lang = language or self.language or None
        if target_lang:
            target_lang = str(target_lang).lower().replace("voice:", "").strip()
            if target_lang in ("ka", "kn"):
                target_lang = "kn"
        initial_prompt = _MEDICAL_INITIAL_PROMPTS.get(target_lang, "")

        segments, info = self._model.transcribe(
            str(audio_path),
            beam_size=self.beam_size,
            language=target_lang,
            initial_prompt=initial_prompt if initial_prompt else None,
            task="transcribe",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        texts = []
        confidences = []
        for seg in segments:
            texts.append(seg.text.strip())
            confidences.append(seg.avg_logprob)

        transcript = " ".join(texts).strip()
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        elapsed = time.perf_counter() - t0

        logger.info(
            f"Transcribed {audio_path.name} in {elapsed:.1f}s "
            f"(lang={info.language}, audio={info.duration:.1f}s, RTF={elapsed/info.duration:.2f}x)"
        )

        return {
            "text": transcript,
            "language": info.language,
            "confidence": round(avg_conf, 3),
            "duration_s": info.duration,
        }

    def transcribe_mic(
        self,
        language: Optional[str] = None,
        max_seconds: float = _MAX_RECORDING_S,
        silence_duration_s: float = _SILENCE_DURATION_S,
    ) -> Dict[str, Any]:
        """
        Record from the default microphone until silence is detected,
        then transcribe.

        Parameters
        ----------
        language : str, optional
            Language code override (e.g. "ta", "hi", "te", "kn", "en").
        max_seconds : float
            Maximum recording duration before forced stop.
        silence_duration_s : float
            Seconds of silence to detect end of utterance.

        Returns
        -------
        dict
            Same schema as :meth:`transcribe_file`.
        """
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
        except ImportError as e:
            raise ImportError(f"sounddevice/soundfile not installed: {e}")

        logger.info("Recording from microphone… (speak now)")
        frames = []
        silent_chunks = 0
        chunk_size = int(_SAMPLE_RATE * 0.1)  # 100ms chunks
        silence_chunks_needed = int(silence_duration_s / 0.1)
        max_chunks = int(max_seconds / 0.1)

        with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="float32") as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_size)
                frames.append(chunk.copy())

                # VAD: check RMS energy
                rms = np.sqrt(np.mean(chunk ** 2))
                db = 20 * np.log10(rms + 1e-10)
                if db < _SILENCE_THRESHOLD_DB:
                    silent_chunks += 1
                    if silent_chunks >= silence_chunks_needed and len(frames) > silence_chunks_needed:
                        logger.info("Silence detected — stopping recording.")
                        break
                else:
                    silent_chunks = 0

        audio = np.concatenate(frames, axis=0)
        logger.info(f"Recorded {len(audio) / _SAMPLE_RATE:.1f}s of audio.")

        # Save to temp WAV file and transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, _SAMPLE_RATE)
            return self.transcribe_file(tmp.name, language=language)
