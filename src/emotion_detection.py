from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
import torch

# Load the model and feature extractor
model = Wav2Vec2ForSequenceClassification.from_pretrained("harshit345/xlsr-wav2vec-speech-emotion-recognition")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("harshit345/xlsr-wav2vec-speech-emotion-recognition")

# Map numerical labels to emotion names
emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
}

def detect_emotion(audio_path):
    """Predict emotion from speech."""
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample to 16,000 Hz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

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
    predicted_emotion = emotion_map.get(predicted_class, "unknown")  # Default to "unknown" if label is not in the map
    print("Predicted Emotion:", predicted_emotion)
    return predicted_emotion
