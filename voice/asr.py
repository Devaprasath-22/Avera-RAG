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
_SILENCE_DURATION_S = 2.0      # seconds of silence to trigger end of utterance
_MAX_RECORDING_S = 30          # hard cap on single recording
_SAMPLE_RATE = 16_000          # Hz (Whisper native)

_MEDICAL_INITIAL_PROMPTS: Dict[str, str] = {
    "ta": (
        "மருத்துவ அறிகுறிகள்: காய்ச்சல், குளிர்க்காய்ச்சல், தலைவலி, தலைசுற்றல், மயக்கம், "
        "இருமல், சளி, மூச்சுத்திணறல், நெஞ்செரிச்சல், வயிற்றுப்போக்கு, வாந்தி, குமட்டல், "
        "வயிற்று வலி, மார்பு வலி, மூட்டு வலி, முதுகு வலி, தசை வலி, சோர்வு, பலவீனம், "
        "தோல் அரிப்பு, தடிப்பு, வீக்கம், காயம், ரத்தப்போக்கு, சர்க்கரை நோய், "
        "இரத்த அழுத்தம், காது வலி, கண் வலி, சிறுநீர் பிரச்சனை, மாதவிடாய் பிரச்சனை, "
        "தூக்கமின்மை, பதற்றம், மனஅழுத்தம்."
    ),
    "hi": (
        "चिकित्सीय लक्षण: बुखार, ठंड लगना, सिरदर्द, चक्कर आना, बेहोशी, खांसी, जुकाम, "
        "सांस लेने में कठिनाई, सीने में जलन, दस्त, उल्टी, मतली, पेट दर्द, सीने में दर्द, "
        "जोड़ों का दर्द, कमर दर्द, मांसपेशियों में दर्द, थकान, कमजोरी, त्वचा में खुजली, "
        "चकत्ते, सूजन, चोट, रक्तस्राव, मधुमेह, रक्तचाप, कान दर्द, आंख में दर्द, "
        "पेशाब में समस्या, मासिक धर्म की समस्या, नींद न आना, घबराहट, तनाव।"
    ),
    "te": (
        "వైద్య లక్షణాలు: జ్వరం, చలి జ్వరం, తలనొప్పి, తలతిరగడం, మూర్ఛ, దగ్గు, జలుబు, "
        "శ్వాస ఆడకపోవడం, గుండెల్లో మంట, విరేచనాలు, వాంతులు, వికారం, కడుపు నొప్పి, "
        "ఛాతీ నొప్పి, కీళ్ల నొప్పులు, నడుము నొప్పి, కండరాల నొప్పి, అలసట, బలహీనత, "
        "చర్మం దురద, దద్దుర్లు, వాపు, గాయం, రక్తస్రావం, మధుమేహం, రక్తపోటు, "
        "చెవి నొప్పి, కంటి నొప్పి, మూత్ర సమస్య, ఋతు సమస్య, నిద్రలేమి, ఆందోళన, ఒత్తిడి."
    ),
    "kn": (
        "ವೈದ್ಯಕೀಯ ಲಕ್ಷಣಗಳು: ಜ್ವರ, ಚಳಿ ಜ್ವರ, ತಲೆನೋವು, ತಲೆತಿರುಗುವಿಕೆ, ಮೂರ್ಛೆ, ಕೆಮ್ಮು, "
        "ನೆಗಡಿ, ಉಸಿರಾಟದ ತೊಂದರೆ, ಎದೆಯುರಿ, ಭೇದಿ, ವಾಂತಿ, ವಾಕರಿಕೆ, ಹೊಟ್ಟೆ ನೋವು, "
        "ಎದೆ ನೋವು, ಕೀಲು ನೋವು, ಬೆನ್ನು ನೋವು, ಸ್ನಾಯು ನೋವು, ಆಯಾಸ, ದೌರ್ಬಲ್ಯ, "
        "ಚರ್ಮದ ತುರಿಕೆ, ದದ್ದು, ಊತ, ಗಾಯ, ರಕ್ತಸ್ರಾವ, ಮಧುಮೇಹ, ರಕ್ತದೊತ್ತಡ, "
        "ಕಿವಿ ನೋವು, ಕಣ್ಣು ನೋವು, ಮೂತ್ರ ಸಮಸ್ಯೆ, ಋತುಚಕ್ರ ಸಮಸ್ಯೆ, ನಿದ್ರಾಹೀನತೆ, "
        "ಆತಂಕ, ಒತ್ತಡ."
    ),
    "en": (
        "Medical symptoms: fever, chills, headache, dizziness, vertigo, fainting, cough, "
        "cold, breathing difficulty, shortness of breath, heartburn, acid reflux, diarrhea, "
        "vomiting, nausea, abdominal pain, stomach pain, chest pain, joint pain, back pain, "
        "muscle pain, fatigue, weakness, skin itching, rash, swelling, injury, bleeding, "
        "diabetes, blood pressure, ear pain, eye pain, urinary problem, menstrual problem, "
        "insomnia, anxiety, stress."
    ),
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
        speech_chunks = 0
        has_spoken = False
        chunk_size = int(_SAMPLE_RATE * 0.1)  # 100ms chunks
        silence_chunks_needed = int(silence_duration_s / 0.1)
        initial_wait_chunks = int(5.0 / 0.1)  # wait up to 5s for user to begin
        max_chunks = int(max_seconds / 0.1)

        with sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype="float32") as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_size)
                frames.append(chunk.copy())

                # VAD: check RMS energy
                rms = np.sqrt(np.mean(chunk ** 2))
                db = 20 * np.log10(rms + 1e-10)
                if db < _SILENCE_THRESHOLD_DB:
                    if has_spoken:
                        silent_chunks += 1
                        if silent_chunks >= silence_chunks_needed:
                            logger.info("Silence detected — stopping recording.")
                            break
                    else:
                        if len(frames) >= initial_wait_chunks:
                            logger.info("No speech initiated — stopping recording.")
                            break
                else:
                    silent_chunks = 0
                    speech_chunks += 1
                    if speech_chunks >= 2:  # at least 200ms above threshold confirms speech
                        has_spoken = True

        audio = np.concatenate(frames, axis=0)
        logger.info(f"Recorded {len(audio) / _SAMPLE_RATE:.1f}s of audio.")

        # Save to temp WAV file and transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, _SAMPLE_RATE)
            return self.transcribe_file(tmp.name, language=language)
