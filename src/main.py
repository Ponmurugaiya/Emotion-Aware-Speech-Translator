from translate import translate_text
from speech_to_text import tamil_speech_to_text
from emotion_detection import detect_emotion
from speech_synthesis import generate_speech

def main(audio_file):
    print("🎤 Tamil Speech Recognition...")
    tamil_text = tamil_speech_to_text(audio_file)
    print(f"✅ Tamil Text: {tamil_text}")

    print("🔄 Translating Tamil to English...")
    english_text = translate_text(tamil_text)
    print(f"✅ English Text: {english_text}")

    print("😃 Detecting Emotion...")
    emotion = detect_emotion(audio_file)
    print(f"✅ Detected Emotion: {emotion}")

    print("🗣 Generating Emotion-Aware Speech...")
    generate_speech(english_text, emotion, "final_output.wav")
    print("✅ Speech Generation Complete: final_output.wav")

if __name__ == "__main__":
    main("P:\\EmoAI\\emoai\\google_voice.wav")  # Replace with your input file
