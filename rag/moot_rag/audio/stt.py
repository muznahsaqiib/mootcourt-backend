import io
import logging
import whisper
import soundfile as sf
import numpy as np
import librosa

logger = logging.getLogger(__name__)
model = whisper.load_model("base")  # load once globally


def speech_to_text(wav_stream: io.BytesIO) -> str:
    """
    Transcribe WAV audio to text using Whisper.
    Expects mono or stereo 16-bit WAV; resamples to 16 kHz if needed.
    """
    wav_stream.seek(0)

    try:
        audio, sr = sf.read(wav_stream)
    except Exception as e:
        logger.exception("STT: failed to read WAV")
        raise ValueError(f"Invalid or unsupported audio: {e}") from e

    if audio.size == 0:
        return ""

    # Whisper expects mono: (n_samples,) not (n_samples, n_channels)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    try:
        result = model.transcribe(audio, language=None, fp16=False)
    except Exception as e:
        logger.exception("STT: Whisper transcribe failed")
        raise RuntimeError(f"Transcription failed: {e}") from e

    text = (result.get("text") or "").strip()
    return text
