# audio/stt.py
import whisper

def speech_to_text(audio_path):
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # faster than "small"
    result = model.transcribe(audio_path)
    return result["text"]
