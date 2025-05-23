import numpy as np
import sys
import os
import json
from vosk import Model, KaldiRecognizer

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import RATE

# Keywords for Vosk to recognize
HOT_PHRASES = ["Hey Uber make it hotter", "hey ober make it hotter"]
COLD_PHRASES = ["Hey Uber make it colder", "hey ober make it colder"]

# Load Vosk model for speech recognition
VOSK_PATH = "models/vosk-model-small-en-us-0.15"
vosk_model = Model(VOSK_PATH)
recognizer = KaldiRecognizer(vosk_model, RATE)

def detect_keywords(audio_queue_keywords):
    """Thread to detect spoken hot/cold commands using Vosk."""
    while True:
        if not audio_queue_keywords.empty():
            audio_data = audio_queue_keywords.get()

            # Process the audio for speech recognition
            result = None  # Initialize result variable
            int16_audio = (audio_data * 32767).astype(np.int16)
            if recognizer.AcceptWaveform(int16_audio.tobytes()):
                result = json.loads(recognizer.Result())
            else:
                # Log partial recognition if it's not a full match
                partial_result = json.loads(recognizer.PartialResult())
                result = partial_result  # Set result to partial

            # Only proceed if we have a valid result
            if result:
                text = result.get("text", "")
                print(f"Processing recognized text: {text}")

                # Check if any hot or cold phrase is detected
                if any(phrase in text.lower() for phrase in HOT_PHRASES):
                    print("🔥 Command Detected: Make it Hotter 🔥")
                elif any(phrase in text.lower() for phrase in COLD_PHRASES):
                    print("🥶 Command Detected: Make it Colder 🥶")