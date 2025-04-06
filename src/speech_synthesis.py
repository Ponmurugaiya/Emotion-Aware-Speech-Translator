import os
import torch
import soundfile as sf
import librosa
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.inference import EncoderClassifier
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SpeechT5 model, processor, and vocoder
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", ignore_mismatched_sizes=True).to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

# Load a pre-trained speaker embedding model from SpeechBrain
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", run_opts={"device": device})

def extract_speaker_embedding(reference_audio_path):
    """Extracts speaker embedding from reference audio."""
    if not os.path.exists(reference_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")

    # Load the reference audio file
    waveform, sample_rate = sf.read(reference_audio_path)
    
    # Ensure the waveform is in the correct format (1D array)
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    
    # Resample to 16kHz if necessary (SpeechBrain model expects 16kHz)
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
    
    # Convert waveform to tensor and move to the correct device
    waveform = torch.tensor(waveform).unsqueeze(0).to(device)
    
    # Extract speaker embedding
    with torch.no_grad():
        speaker_embedding = classifier.encode_batch(waveform)
    
    # Reshape the speaker embedding to [batch_size, embedding_dim]
    speaker_embedding = speaker_embedding.squeeze(1)  # Shape: [1, 512]
    
    return speaker_embedding

def generate_speech(text, emotion, output_path):
    """Generates speech using SpeechT5 with a reference speaker."""
    print(f"Generating speech for text: {text}")

    reference_audio_path = utils.get_reference_audio_path(emotion)

    # Extract speaker embedding from the reference audio
    speaker_embedding = extract_speaker_embedding(reference_audio_path)

    # Process input text
    inputs = processor(text=text, return_tensors="pt").to(device)

    # Generate speech
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

    # Save to file
    sf.write(output_path, speech.cpu().numpy(), samplerate=16000)
    print(f"✅ Speech saved as {output_path}")
