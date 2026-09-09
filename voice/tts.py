"""
TTS backend — AI4Bharat Indic TTS (fully offline).

Uses AI4Bharat's VITS-based Indic TTS models via the Coqui TTS engine.
Models are downloaded from HuggingFace on first use and cached locally
— subsequent runs are 100% offline.

Supported language codes:
  Indo-Aryan  : hi (Hindi), bn (Bengali), mr (Marathi), gu (Gujarati),
                pa (Punjabi), or (Odia), as (Assamese), ur (Urdu),
                en (English)
  Dravidian   : ta (Tamil), te (Telugu), kn (Kannada), ml (Malayalam)

HuggingFace model IDs:
  Indo-Aryan  : ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4
  Dravidian   : ai4bharat/indic-tts-coqui-dravidian-gpu--t4

Memory: ~200-400 MB per loaded model group (CPU inference on Jetson).
Latency: ~0.5-2s per sentence (VITS, CPU).

Install:
    pip install TTS huggingface_hub
"""

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Language → model group mapping ───────────────────────────────────────────
_DRAVIDIAN_LANGS = {"ta", "te", "kn", "ml"}

_HF_MODEL_IDS = {
    "indo_aryan": "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4",
    "dravidian":  "ai4bharat/indic-tts-coqui-dravidian-gpu--t4",
}

_EDGE_VOICES = {
    "ta": "ta-IN-PallaviNeural",   # Tamil
    "hi": "hi-IN-SwaraNeural",     # Hindi
    "te": "te-IN-ShrutiNeural",    # Telugu
    "kn": "kn-IN-SapnaNeural",     # Kannada
    "en": "en-IN-NeerjaNeural",    # Indian English
}

_EDGE_VOICES_MALE = {
    "ta": "ta-IN-ValluvarNeural",
    "hi": "hi-IN-MadhurNeural",
    "te": "te-IN-MohanNeural",
    "kn": "kn-IN-GaganNeural",
    "en": "en-IN-PrabhatNeural",
}

_LANG_TO_SPEAKER: Dict[str, str] = {
    "hi": "Hindi Female",   "bn": "Bengali Female",
    "mr": "Marathi Female", "gu": "Gujarati Female",
    "pa": "Punjabi Female", "or": "Odia Female",
    "as": "Assamese Female","ur": "Urdu Female",
    "en": "English Female",
    "ta": "Tamil Female",   "te": "Telugu Female",
    "kn": "Kannada Female", "ml": "Malayalam Female",
}

_LANG_TO_SPEAKER_MALE: Dict[str, str] = {
    k: v.replace("Female", "Male") for k, v in _LANG_TO_SPEAKER.items()
}


def detect_script(text: str) -> Optional[str]:
    """Detect if text contains Indic scripts; returns 'ta', 'hi', 'te', 'kn', or None (English)."""
    for ch in text:
        cp = ord(ch)
        if 0x0B80 <= cp <= 0x0BFF:
            return "ta"
        if 0x0900 <= cp <= 0x097F:
            return "hi"
        if 0x0C00 <= cp <= 0x0C7F:
            return "te"
        if 0x0C80 <= cp <= 0x0CFF:
            return "kn"
    return None


def clean_for_speech(text: str) -> str:
    """Clean markdown, source citations, and formatting for natural TTS speaking."""
    import re
    # Remove citations like [SOURCE 1] or [SOURCE 1, p.4]
    t = re.sub(r"\[SOURCE\s*[^\]]*\]", "", text, flags=re.IGNORECASE)
    # Remove markdown headers and formatting
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"^[#\-\*•\s]+", "", t)
    t = re.sub(r"\n[#\-\*•\s]+", "\n", t)
    # Clean URLs
    t = re.sub(r"https?://\S+", "", t)
    return t.strip()


def is_key_pressed() -> bool:
    """Non-blocking check if a key was pressed (Windows or Linux)."""
    try:
        import msvcrt
        if msvcrt.kbhit():
            try:
                msvcrt.getch()
            except Exception:
                pass
            return True
    except ImportError:
        pass

    try:
        import sys, select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            try:
                sys.stdin.read(1)
            except Exception:
                pass
            return True
    except Exception:
        pass

    return False


class TTSBackend:
    """
    AI4Bharat Indic TTS backend — fully offline, multi-language.

    Models are lazy-loaded per language group (Indo-Aryan vs Dravidian)
    and cached in `model_cache_dir` after first download.

    Usage:
        tts = TTSBackend(config)
        tts.speak("आज का दिन अच्छा है")           # Hindi
        tts.speak("What are the danger signs?")   # English
        tts.speak("நாளை மீண்டும் வாருங்கள்")    # Tamil
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        tts_cfg = config.get("models", {}).get("tts", {})
        self.provider: str = tts_cfg.get("provider", "ai4bharat")
        self.language: str = tts_cfg.get("language", "hi")
        self.gender: str = tts_cfg.get("gender", "female")
        self.model_cache_dir: str = str(
            Path(tts_cfg.get("model_cache_dir", "~/avera_models/indic_tts")).expanduser()
        )
        self.sample_rate: int = tts_cfg.get("sample_rate", 22050)
        self._stop_requested: bool = False

        # Loaded models keyed by group name (lazy)
        self._models: Dict[str, Any] = {}

        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)

    # ── Language helpers ──────────────────────────────────────────────────────

    def _group(self, language: str) -> str:
        return "dravidian" if language in _DRAVIDIAN_LANGS else "indo_aryan"

    def _speaker(self, language: str) -> str:
        table = _LANG_TO_SPEAKER_MALE if self.gender.lower() == "male" else _LANG_TO_SPEAKER
        return table.get(language, table.get("hi", "Hindi Female"))

    # ── Model loading ─────────────────────────────────────────────────────────

    def _get_model(self, language: str):
        """Return the TTS model for the given language, loading/downloading if needed."""
        group = self._group(language)
        if group in self._models:
            return self._models[group]

        model_dir = Path(self.model_cache_dir) / group
        config_path = model_dir / "config.json"
        checkpoint_path = model_dir / "model_file.pth"

        # Download from HuggingFace if not cached
        if not checkpoint_path.exists():
            self._download_model(group, model_dir)

        # Load with Coqui TTS
        try:
            from TTS.utils.synthesizer import Synthesizer
        except ImportError:
            logger.warning("Coqui TTS not available. Falling back to native pyttsx3 voice engine.")
            self.provider = "pyttsx3"
            return None

        logger.info(f"Loading AI4Bharat Indic TTS ({group})…")
        t0 = time.perf_counter()

        synthesizer = Synthesizer(
            tts_checkpoint=str(checkpoint_path),
            tts_config_path=str(config_path),
            use_cuda=False,  # CPU inference — keeps GPU free for LLM/VLM
        )
        self._models[group] = synthesizer
        logger.info(f"Indic TTS ({group}) loaded in {time.perf_counter()-t0:.1f}s")
        return synthesizer

    def _download_model(self, group: str, model_dir: Path) -> None:
        """Download AI4Bharat VITS model from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed.\n"
                "Run: pip install huggingface_hub"
            )

        hf_id = _HF_MODEL_IDS[group]
        logger.info(
            f"Downloading AI4Bharat Indic TTS ({group}) from {hf_id}…\n"
            f"This is a one-time download (~200-400 MB). Cached at {model_dir}"
        )

        snapshot_download(
            repo_id=hf_id,
            local_dir=str(model_dir),
            ignore_patterns=["*.msgpack", "flax_model.*", "tf_model.*"],
        )
        logger.info(f"Model downloaded to {model_dir}")

    def unload(self, language: Optional[str] = None) -> None:
        """Free model memory. Pass language to unload just one group."""
        if language:
            group = self._group(language)
            if group in self._models:
                del self._models[group]
                logger.info(f"Indic TTS ({group}) unloaded.")
        else:
            self._models.clear()
            logger.info("All Indic TTS models unloaded.")

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
        language: Optional[str] = None,
    ) -> Path:
        """
        Synthesize text and write WAV/MP3 to output_path.

        Parameters
        ----------
        text : str
            Text to synthesize.
        output_path : str or Path
            Destination audio file.
        language : str, optional
            Language code override. Defaults to config language.

        Returns
        -------
        Path
            Resolved path to the written audio file.
        """
        output_path = Path(output_path)
        lang = language or self.language
        if lang:
            lang = str(lang).lower().replace("voice:", "").strip()
            if lang in ("ka", "kn"):
                lang = "kn"

        cleaned_text = clean_for_speech(text)
        if not cleaned_text:
            return output_path

        # Auto-detect script from text: if text is in Tamil/Hindi/Telugu/Kannada,
        # use that language's voice so Edge-TTS never throws NoAudioReceived
        detected = detect_script(cleaned_text)
        effective_lang = detected or lang or "en"

        if self.provider == "edge-tts":
            try:
                import asyncio
                import edge_tts
                voices = _EDGE_VOICES_MALE if self.gender.lower() == "male" else _EDGE_VOICES
                voice = voices.get(effective_lang, "en-IN-NeerjaNeural")

                async def _gen():
                    communicate = edge_tts.Communicate(cleaned_text, voice)
                    await communicate.save(str(output_path))

                asyncio.run(_gen())
                if output_path.exists() and output_path.stat().st_size > 0:
                    return output_path
            except Exception as e:
                logger.warning(f"edge-tts failed ({e}), falling back to pyttsx3")

            # Fallback to pyttsx3 when edge-tts fails
            try:
                import pyttsx3
                engine = pyttsx3.init()
                wav_path = output_path.with_suffix(".wav")
                engine.save_to_file(cleaned_text, str(wav_path))
                engine.runAndWait()
                return wav_path
            except Exception as e2:
                logger.warning(f"pyttsx3 fallback failed: {e2}")
                return output_path

        elif self.provider == "pyttsx3":
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.save_to_file(cleaned_text, str(output_path))
                engine.runAndWait()
                return output_path
            except Exception as e:
                logger.warning(f"pyttsx3 save_to_file failed: {e}")
                return output_path

        elif self.provider == "ai4bharat":
            try:
                speaker = self._speaker(effective_lang)
                t0 = time.perf_counter()
                model = self._get_model(effective_lang)

                if model is not None:
                    wav = model.tts(text=cleaned_text, speaker_name=speaker)
                    model.save_wav(wav=wav, path=str(output_path))
                    elapsed = (time.perf_counter() - t0) * 1000
                    logger.debug(
                        f"Indic TTS: '{cleaned_text[:50]}' → {output_path.name} "
                        f"in {elapsed:.0f}ms (lang={effective_lang}, speaker={speaker})"
                    )
                    return output_path
            except Exception as e:
                logger.warning(f"ai4bharat TTS failed ({e}), falling back to pyttsx3")

            # Fallback to pyttsx3 if AI4Bharat model fails
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.save_to_file(cleaned_text, str(output_path))
                engine.runAndWait()
                return output_path
            except Exception as e:
                logger.error(f"TTS fallback failed: {e}")
                return output_path

        return output_path

    def stop(self) -> None:
        """Immediately stop any currently playing speech."""
        self._stop_requested = True
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        import sys
        if sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    def speak(self, text: str, language: Optional[str] = None) -> bool:
        """
        Synthesize text and play immediately on the audio output.
        Returns True if interrupted/stopped early by user, False if completed normally.
        """
        if not text.strip() or self._stop_requested:
            return False

        lang = language or self.language
        suffix = ".mp3" if self.provider == "edge-tts" else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_path = Path(tmp.name)

        stopped = False
        try:
            resolved_path = self.synthesize_to_file(text, audio_path, language=lang)
            if self._stop_requested or is_key_pressed():
                self.stop()
                return True
            if resolved_path and resolved_path.exists() and resolved_path.stat().st_size > 0:
                stopped = self._play(resolved_path)
        except KeyboardInterrupt:
            self.stop()
            stopped = True
        except Exception as e:
            logger.warning(f"TTS speak failed: {e}")
        finally:
            try:
                if audio_path.exists():
                    audio_path.unlink()
            except OSError:
                pass
        return stopped

    def speak_streaming(
        self,
        text: str,
        chunk_size_words: int = 40,
        language: Optional[str] = None,
    ) -> bool:
        """
        Synthesize and play speech. Speaks short text in one natural flow,
        or splits by sentence for smooth prosody.
        Allows interrupting speech at any moment by pressing Space, Enter, or Ctrl+C.
        Returns True if stopped early, False if finished.
        """
        self._stop_requested = False
        try:
            cleaned = clean_for_speech(text)
            if not cleaned:
                return False

            if len(cleaned.split()) <= 60:
                return self.speak(cleaned, language=language)

            sentences = [s.strip() for s in cleaned.replace("\n", ". ").split(".") if s.strip()]
            for sentence in sentences:
                if self._stop_requested or is_key_pressed():
                    self.stop()
                    return True

                if len(sentence.split()) >= 3:
                    stopped = self.speak(sentence + ".", language=language)
                    if stopped:
                        return True

            return False
        except KeyboardInterrupt:
            self.stop()
            return True
        except Exception as e:
            logger.warning(f"TTS speak_streaming error: {e}")
            return False

    # ── Audio playback ────────────────────────────────────────────────────────

    def _play(self, audio_path: Path) -> bool:
        """
        Play an audio file on default output.
        Continuously checks for user keypress (Space/Enter) or Ctrl+C to halt immediately.
        Returns True if stopped early, False if finished normally.
        """
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return False

        try:
            import sounddevice as sd
            import soundfile as sf

            data, fs = sf.read(str(audio_path), dtype="float32")
            sd.play(data, fs)

            # Poll actively while audio is playing (non-blocking)
            while True:
                stream = sd.get_stream()
                if stream is None or not stream.active:
                    break
                if self._stop_requested or is_key_pressed():
                    sd.stop()
                    self._stop_requested = True
                    return True
                time.sleep(0.04)

            return False
        except KeyboardInterrupt:
            self.stop()
            return True
        except Exception as e:
            logger.debug(f"sounddevice failed ({e}), trying platform fallback")

        # Windows fallback: winsound
        import sys
        if sys.platform == "win32" and str(audio_path).endswith(".wav"):
            try:
                import winsound
                winsound.PlaySound(str(audio_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                while True:
                    if self._stop_requested or is_key_pressed():
                        winsound.PlaySound(None, winsound.SND_PURGE)
                        self._stop_requested = True
                        return True
                    time.sleep(0.05)
            except KeyboardInterrupt:
                winsound.PlaySound(None, winsound.SND_PURGE)
                self._stop_requested = True
                return True
            except Exception as e:
                logger.warning(f"winsound failed: {e}")

        # Linux fallback: aplay (Jetson / Ubuntu)
        try:
            proc = subprocess.Popen(["aplay", str(audio_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while proc.poll() is None:
                if self._stop_requested or is_key_pressed():
                    proc.terminate()
                    self._stop_requested = True
                    return True
                time.sleep(0.05)
        except KeyboardInterrupt:
            try:
                proc.terminate()
            except Exception:
                pass
            self._stop_requested = True
            return True
        except Exception:
            pass

        return False
