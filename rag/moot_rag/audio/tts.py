# audio/tts.py
import os
import logging

logger = logging.getLogger(__name__)

def text_to_speech(text: str, output: str = "reply.mp3") -> str:
    """Synthesize `text` to `output` and return absolute path.

    Ensures output directory exists and raises/logs any errors.
    """
    try:
        # ensure directory exists
        out_path = os.path.abspath(output)
        out_dir = os.path.dirname(out_path) or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        try:
            import pyttsx3
        except ModuleNotFoundError as e:
            logger.error("pyttsx3 is not installed. Install with 'pip install pyttsx3' to enable TTS.")
            raise

        engine = pyttsx3.init()
        engine.setProperty("rate", 150)

        # Save to file and block until finished
        engine.save_to_file(text, out_path)
        engine.runAndWait()

        # Verify file was created
        if not os.path.exists(out_path):
            logger.error(f"TTS completed but output file not found: {out_path}")
            raise RuntimeError("TTS failed to produce output file")

        logger.info(f"TTS saved to: {out_path}")
        return out_path
    except Exception as e:
        logger.exception("TTS failed")
        raise
