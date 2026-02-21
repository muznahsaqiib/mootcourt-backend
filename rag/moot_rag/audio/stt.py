import io
import whisper
import soundfile as sf
import numpy as np

model = whisper.load_model("base")  # load once globally

def speech_to_text(wav_stream: io.BytesIO) -> str:
    wav_stream.seek(0)

    # Read WAV from memory → numpy array
    audio, sr = sf.read(wav_stream)

    # Ensure 16kHz
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Convert to float32 (important for Whisper stability)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    result = model.transcribe(audio)
    return result["text"].strip()
