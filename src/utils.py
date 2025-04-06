import os

def get_reference_audio_path(emotion, gender="male"):
    """Get reference audio path based on detected emotion."""
    base_path = "P:\\EmoAI\\emoai\\datasets\\audio"
    
    # Construct the path to the emotion and gender folder
    emotion_path = os.path.join(base_path, emotion, gender)
    
    # Check if the emotion folder exists
    if not os.path.exists(emotion_path):
        print(f"❌ No reference audio found for emotion: {emotion}. Using default emotion 'happy'.")
        # Fallback to "happy" if the emotion folder doesn't exist
        emotion_path = os.path.join(base_path, "happy", gender)
    
    # Check if the fallback folder exists
    if not os.path.exists(emotion_path):
        print(f"❌ No reference audio found for fallback emotion 'happy'. No audio available.")
        return None
    
    # List files in the folder
    files = os.listdir(emotion_path)
    if files:
        # Return the path to the first file in the folder
        return os.path.join(emotion_path, files[0])
    else:
        print(f"❌ No audio files found in folder: {emotion_path}")
        return None