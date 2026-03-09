"""
Voice Input Module: OpenAI Whisper + sounddevice
Records audio from microphone and transcribes using Whisper.
"""

import io
import os
import logging
import tempfile
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # Whisper expects 16kHz
RECORD_SECONDS = 6   # Default recording duration


def record_audio(duration: int = RECORD_SECONDS) -> np.ndarray:
    """Record audio from default microphone."""
    try:
        import sounddevice as sd
        logger.info("Recording %d seconds of audio…", duration)
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio.flatten()
    except Exception as e:
        logger.error("Audio recording failed: %s", e)
        raise RuntimeError(f"Could not record audio: {e}")


def transcribe_audio(audio_array: np.ndarray, language: str = None) -> str:
    """Transcribe numpy audio array using Whisper."""
    try:
        import whisper
        from scipy.io import wavfile

        model = whisper.load_model("base")

        # Save to temp WAV so Whisper can load it
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            wavfile.write(tmp_path, SAMPLE_RATE, (audio_array * 32767).astype(np.int16))

        options = {"language": language} if language else {}
        result = model.transcribe(tmp_path, **options)
        os.unlink(tmp_path)

        return result.get("text", "").strip()

    except ImportError:
        raise RuntimeError("Whisper not installed. Run: pip install openai-whisper")
    except Exception as e:
        logger.error("Transcription error: %s", e)
        raise RuntimeError(f"Transcription failed: {e}")


def record_and_transcribe(duration: int = RECORD_SECONDS, language: str = None) -> str:
    """Full pipeline: record → transcribe → return text."""
    audio = record_audio(duration)
    return transcribe_audio(audio, language)
