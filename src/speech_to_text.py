import whisper

# Load Whisper model (choose a model size: tiny, base, small, medium, large)
MODEL_SIZE = "large"  # You can use "small" or "medium" for better accuracy
MODEL_PATH = "whisper-model"  # Optional: Path to save the model

def tamil_speech_to_text(audio_path):
    """Convert Tamil speech to text using OpenAI's Whisper model."""
    # Load the Whisper model
    model = whisper.load_model(MODEL_SIZE, download_root=MODEL_PATH)

    # Transcribe the audio file
    result = model.transcribe(audio_path, language="ta")  # "ta" for Tamil

    return result["text"]