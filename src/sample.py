from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
import torch

# Load the model and feature extractor
model = Wav2Vec2ForSequenceClassification.from_pretrained("harshit345/xlsr-wav2vec-speech-emotion-recognition")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("harshit345/xlsr-wav2vec-speech-emotion-recognition")

def detect_emotion(audio_path):
    """Predict emotion from speech."""
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Preprocess the audio
    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted emotion
    predicted_class = torch.argmax(logits, dim=-1).item()
    print("Predicted Emotion:", predicted_class)

# Example usage
detect_emotion("P:\\EmoAI\\emoai\\src\\angry.wav")