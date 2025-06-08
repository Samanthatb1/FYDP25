import numpy as np
import sys
import os
import json
from vosk import Model, KaldiRecognizer
from rapidfuzz import fuzz

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import RATE

# Keywords for Vosk to recognize
WAKEUP_PHRASES = ["hey driver", "hey alert rider", "hey alert writer", "a driver", "a alert rider", "a alert writer", "low rider", "low writer"]

KEYWORDS = {
    "hotter": ["make it warmer", "turn the heat up", "warmer", "hotter", "increase the temperature", "turn on heat"],
    "colder": ["make it cooler", "turn the heat down", "cooler", "colder", "turn on the ac", "lower temperature", "turn down the heat"],
    "stop": ["stop here", "you can stop", "pull over", "this is fine"],
    "thanks": ["thank you", "thanks", "appreciate it"],
    "trunk": ["open the trunk", "pop the trunk", "can you open the back"]
}

# Load Vosk model for speech recognition
VOSK_PATH = "models/vosk-model-small-en-us-0.15"
vosk_model = Model(VOSK_PATH)
recognizer = KaldiRecognizer(vosk_model, RATE)

def detect_keywords(audio_queue_keywords):
    """Thread to detect spoken hot/cold commands using Vosk."""
    while True:
        if not audio_queue_keywords.empty():
            audio_data = audio_queue_keywords.get()
            int16_audio = (audio_data * 32767).astype(np.int16)

            # Process the audio for speech recognition
            result = None  # Initialize result variable
            if recognizer.AcceptWaveform(int16_audio.tobytes()):
                result = json.loads(recognizer.Result())
            else:
                # Log partial recognition if it's not a full match
                partial_result = json.loads(recognizer.PartialResult())
                result = partial_result  # Set result to partial            

            # Only proceed if we have a valid result
            if result:
                text = result.get("text", "").lower()
                print(f"Processing recognized text: {text}")

                # check if wakeup phrase is in the text
                if any(wake in text for wake in WAKEUP_PHRASES):
                    command = get_command(text)

                    if command:
                        print(f"COMMAND DETECTED: {command}")
                            
                    else:
                        print("WAKEUP PHRASE DETECTED BUT NO KNOWN COMMAND MATCHED")

# use fuzz to do fuzzy matching for phrases 
def get_command(text, threshold=80):
    for command, phrases in KEYWORDS.items():
        for phrase in phrases:
            similarity = fuzz.partial_ratio(text, phrase)
            if similarity > threshold:
                return command
    return None