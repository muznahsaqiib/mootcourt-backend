import os
import logging
import threading
from uuid import uuid4
import pyttsx3

logger = logging.getLogger(__name__)

AUDIO_DIR = "audio"
# pyttsx3 is not thread-safe; serialize TTS calls (e.g. judge + respondent in parallel).
_tts_lock = threading.Lock()

def text_to_speech(text: str) -> str:
    """
    Generate TTS audio file and return path to WAV file.
    pyttsx3 saves in a raw format; we use .wav extension for correct playback.
    """
    if not text or not str(text).strip():
        raise ValueError("TTS requires non-empty text")

    os.makedirs(AUDIO_DIR, exist_ok=True)
    filename = f"reply_{uuid4().hex}.wav"
    output_path = os.path.join(AUDIO_DIR, filename)

    with _tts_lock:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.save_to_file(text.strip(), output_path)
            engine.runAndWait()
        except Exception as e:
            logger.exception("TTS failed")
            raise RuntimeError(f"TTS failed: {e}") from e

    if not os.path.exists(output_path):
        raise RuntimeError("TTS failed to create file")

    return output_path
