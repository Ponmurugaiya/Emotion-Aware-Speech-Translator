from flask import Flask, request, render_template, send_file
import os
from translate import translate_text
from speech_to_text import tamil_speech_to_text
from emotion_detection import detect_emotion
from speech_synthesis import generate_speech
import utils

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        if "audio_file" not in request.files:
            return "No file uploaded", 400
        audio_file = request.files["audio_file"]
        if audio_file.filename == "":
            return "No file selected", 400

        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(file_path)

        # Process the audio file
        print("🎤 Tamil Speech Recognition...")
        tamil_text = tamil_speech_to_text(file_path)
        print(f"✅ Tamil Text: {tamil_text}")

        print("🔄 Translating Tamil to English...")
        english_text = translate_text(tamil_text)
        print(f"✅ English Text: {english_text}")

        print("😃 Detecting Emotion...")
        emotion = detect_emotion(file_path)
        print(f"✅ Detected Emotion: {emotion}")

        print("🗣 Generating Emotion-Aware Speech...")
        output_file = os.path.join(OUTPUT_FOLDER, "final_output.wav")
        generate_speech(english_text, emotion, output_file)
        print("✅ Speech Generation Complete: final_output.wav")

        # Return the result to the user
        return render_template(
            "result.html",
            tamil_text=tamil_text,
            english_text=english_text,
            emotion=emotion,
            audio_file=output_file,
        )

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)