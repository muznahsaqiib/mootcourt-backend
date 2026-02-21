import os
import logging
from uuid import uuid4
import pyttsx3

logger = logging.getLogger(__name__)

AUDIO_DIR = "audio"

def text_to_speech(text: str) -> str:
    """
    Generate TTS audio file and return relative path.
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)

    filename = f"reply_{uuid4().hex}.mp3"
    output_path = os.path.join(AUDIO_DIR, filename)

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.save_to_file(text, output_path)
    engine.runAndWait()

    if not os.path.exists(output_path):
        raise RuntimeError("TTS failed to create file")

    return output_path  # return relative path
