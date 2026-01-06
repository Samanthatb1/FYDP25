import numpy as np
import sys
import os
import json
import queue
from vosk import Model, KaldiRecognizer
from rapidfuzz import fuzz

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import RATE

# Keywords for Vosk to recognize
WAKEUP_PHRASES = [
    "hey driver"
]

KEYWORDS = {
    "hotter": [
        "make it warmer",
        "turn the heat up",
        "warmer",
        "hotter",
        "increase the temperature",
        "turn on heat",
    ],
    "colder": [
        "make it cooler",
        "turn the heat down",
        "cooler",
        "colder",
        "turn on the ac",
        "lower temperature",
        "turn down the heat",
    ],
    "stop": [
        "stop here",
        "you can stop",
        "pull over",
        "this is fine",
    ],
    "thanks": [
        "thank you",
        "thanks",
        "appreciate it",
    ],
    "trunk": [
        "open the trunk",
        "pop the trunk",
        "can you open the back",
    ],
}

# Load Vosk model for speech recognition
VOSK_PATH = "models/vosk-model-small-en-us-0.15"
vosk_model = Model(VOSK_PATH)
recognizer = KaldiRecognizer(vosk_model, RATE)

def detect_keywords(audio_queue_keywords, command_queue=None):
    """Thread to detect spoken commands using Vosk with fuzzy matching."""
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

                # Check if wakeup phrase is in the text
                if any(wake in text for wake in WAKEUP_PHRASES):
                    commands = get_commands(text)

                    if commands:
                        for command in commands:
                            print(f"COMMAND DETECTED: {command}")
                            if command_queue is not None:
                                try:
                                    command_queue.put_nowait(command)
                                except queue.Full:
                                    print(f"Command queue full; dropping '{command}'.")
                    else:
                        print("WAKEUP PHRASE DETECTED BUT NO KNOWN COMMAND MATCHED")


# Use fuzz to do fuzzy matching for phrases and return all commands in spoken order
def get_commands(text, threshold=80):
    candidates = []
    for command, phrases in KEYWORDS.items():
        best_pos = None

        for phrase in phrases:
            # Prefer explicit substring matches to retain position
            if phrase in text:
                pos = text.find(phrase)
                best_pos = pos if best_pos is None else min(best_pos, pos)
                continue

            # Fallback to fuzzy match; approximate position using first word
            similarity = fuzz.partial_ratio(text, phrase)
            if similarity > threshold:
                anchor = phrase.split()[0] if phrase.split() else phrase
                pos = text.find(anchor) if anchor in text else len(text)
                best_pos = pos if best_pos is None else min(best_pos, pos)

        if best_pos is not None:
            candidates.append((best_pos, command))

    # Sort by position to preserve spoken order; drop duplicates while keeping order
    candidates.sort(key=lambda item: item[0])
    ordered_commands = []
    seen = set()
    for _, command in candidates:
        if command not in seen:
            ordered_commands.append(command)
            seen.add(command)

    return ordered_commands
