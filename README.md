# EmoAI – Emotion-Aware Speech Translation System

**EmoAI** is a Python-based application that converts **Tamil speech** to **English speech**, preserving the **emotional tone** of the original audio. The system performs **speech recognition**, **translation**, **emotion detection**, and **emotion-aware speech synthesis** using state-of-the-art deep learning models. It offers both a **CLI interface** and a **web interface** built with Flask.

---

## Features

* **Tamil Speech-to-Text**: Converts Tamil audio into text using OpenAI Whisper.
* **Translation**: Translates Tamil text to English using Google Translate API.
* **Emotion Detection**: Detects emotion in the original speech (happy, sad, angry, fear, disgust) using Wav2Vec2.
* **Emotion-Aware Speech Synthesis**: Generates English speech while preserving the speaker's emotion using SpeechT5 and SpeechBrain.
* **Web Interface**: Upload audio files, view transcribed text, translated text, detected emotion, and download synthesized speech.
* **CLI Interface**: Run the full pipeline on local audio files directly from the command line.

---

## Project Structure

```
EmoAI/
├── src/
│   ├── app.py                # Flask web application
│   ├── main.py               # CLI script to run full pipeline
│   ├── sample.py             # Sample test for emotion detection
│   ├── speech_to_text.py     # Tamil speech-to-text module (Whisper)
│   ├── translate.py          # Tamil-to-English translation module
│   ├── emotion_detection.py  # Emotion detection module (Wav2Vec2)
│   ├── speech_synthesis.py   # Emotion-aware speech synthesis (SpeechT5 + SpeechBrain)
│   └── utils.py              # Helper functions
├── uploads/                  # Folder to store uploaded audio files (Flask)
├── output/                   # Folder to save generated speech outputs
├── requirements.txt          # Python dependencies

```

---

## Installation

1. **Clone the repository:**

```bash
git clone <repository_url>
cd EmoAI
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Optional**: Ensure CUDA is available for GPU acceleration (recommended for SpeechT5).

---

## Usage

### CLI

Run the full pipeline on a local audio file:

```bash
python src/main.py
```

Modify `main.py` to use your own Tamil audio file:

```python
main("path/to/your/audio.wav")
```

### Web Interface

1. Run the Flask app:

```bash
python src/app.py
```

2. Open your browser at `http://127.0.0.1:5000/`.
3. Upload a Tamil audio file and view:

   * Transcribed Tamil text
   * Translated English text
   * Detected emotion
   * Downloadable emotion-aware English speech

---

## Models & Dependencies

* **Speech Recognition:** OpenAI Whisper (`large` model recommended)
* **Translation:** Google Translate API (`googletrans==4.0.0-rc1`)
* **Emotion Detection:** `harshit345/xlsr-wav2vec-speech-emotion-recognition`
* **Speech Synthesis:** Microsoft SpeechT5 + SpeechBrain speaker embeddings
* **Audio Processing:** `librosa`, `soundfile`, `pydub`

---

## Reference Audio Dataset

Place emotion-specific reference audios in:

```
datasets/audio/<emotion>/<gender>/
```

For example, in this project:

```
datasets/audio/angry/female/angry_female_02.wav
datasets/audio/angry/male/angry_male_01.wav
```

The system automatically selects a reference audio based on the detected emotion. If the exact emotion/gender audio is unavailable, it will **fall back to the "happy" emotion**.

---

## License

This project is licensed under the **MIT License** – see `LICENSE` file for details.

---

## Acknowledgements

* OpenAI Whisper – Speech recognition
* Hugging Face Transformers – Wav2Vec2 and SpeechT5 models
* SpeechBrain – Speaker embedding extraction
* Google Translate – Text translation

