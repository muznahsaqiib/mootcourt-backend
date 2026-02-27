# app/rag/moot_rag/audio/tts.py
import asyncio
import logging
import io
import edge_tts

logger = logging.getLogger(__name__)

# Distinct voices per courtroom role — makes VR experience more immersive
VOICES = {
    "default":    "en-US-GuyNeural",
    "judge":      "en-US-ChristopherNeural",  # authoritative
    "respondent": "en-GB-RyanNeural",          # distinct accent
    "petitioner": "en-US-EricNeural",          # neutral
}

async def _tts_async(text: str, voice: str = "default") -> bytes:
    """Core async TTS — streams audio chunks and returns raw bytes."""
    voice_name = VOICES.get(voice, VOICES["default"])
    communicate = edge_tts.Communicate(text.strip(), voice_name)

    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])

    audio_bytes = buf.getvalue()
    if not audio_bytes:
        raise RuntimeError(f"edge-tts produced empty audio for voice={voice_name}")
    return audio_bytes


def text_to_speech(text: str, voice: str = "default") -> bytes:
    """
    Sync wrapper — returns raw audio bytes directly (no disk I/O).
    Safe to call via asyncio.to_thread() from FastAPI routes.

    Args:
        text:  Text to synthesize.
        voice: One of 'default', 'judge', 'respondent', 'petitioner'.

    Returns:
        Raw MP3 audio bytes.

    Raises:
        ValueError:   If text is empty.
        RuntimeError: If TTS produces no output.
    """
    if not text or not text.strip():
        raise ValueError("TTS requires non-empty text")
    return asyncio.run(_tts_async(text, voice))


def tts_to_bytes(tts_output) -> bytes:
    """
    Compatibility shim — TTS now returns bytes directly, but this handles
    both bytes (new) and legacy file paths (old pyttsx3 behaviour).
    """
    if isinstance(tts_output, (bytes, bytearray, memoryview)):
        return bytes(tts_output)
    if isinstance(tts_output, str):
        with open(tts_output, "rb") as f:
            return f.read()
    raise RuntimeError(f"Unexpected TTS output type: {type(tts_output).__name__}")