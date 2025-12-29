import numpy as np
import sys
import os
import json
from vosk import Model, KaldiRecognizer

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import RATE

# Keywords for Vosk to recognize
KEY_PHRASES = {
    "hey driver make it hotter": "🔥 Command Detected: Make it Hotter 🔥",
    "hey driver make it colder": "🥶 Command Detected: Make it Colder 🥶",
    "hey driver stop here": "❌ Command Detected: Stop Here ❌",
    "hey driver open the trunk": "🚗 Command Detected: Open The Trunk 🚗"
}

# Load Vosk model for speech recognition
VOSK_PATH = "models/vosk-model-small-en-us-0.15"
vosk_model = Model(VOSK_PATH)
recognizer = KaldiRecognizer(vosk_model, RATE)

def process_text(text):
    if text in ("", " "):
        print("👂 No Speech Detected")
        return

    if text in KEY_PHRASES:
        print(KEY_PHRASES[text])
    else:
        print("Non command speech detected: ", text)

def detect_keywords(audio_queue_keywords):
    """Thread to detect spoken hot/cold commands using Vosk."""
    while True:
        if not audio_queue_keywords.empty():
            audio_data = audio_queue_keywords.get()

            result = None 
            # Vosk expects 16 bit integers
            int16_audio = (audio_data * 32767).astype(np.int16)
            # Sent to vosk
            if recognizer.AcceptWaveform(int16_audio.tobytes()):
                result = json.loads(recognizer.Result())
            # else case is when Vosk wasnt able to determine a complete sentence 
            else:
                # Log partial recognition if it's not a full match
                partial_result = json.loads(recognizer.PartialResult())
                result = partial_result  # Set result to partial

            # Only proceed if we have a valid result
            if result:
                text = result.get("text", "").lower()
                process_text(text)