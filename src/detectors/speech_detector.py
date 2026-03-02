import numpy as np
import sys
import os
import json
import time
import queue
from vosk import Model, KaldiRecognizer
from rapidfuzz import fuzz

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import RATE

# Keywords for Vosk to recognize
WAKEUP_PHRASES = [
    "excuse me driver",
    "excuse me the driver",
    "the excuse me driver",
    "hey driver",
    "he driver",
    "a driver",
    "ivor",
    "use me driver",
]

KEYWORDS = {
    "hotter": [
        "turn up the heat",       
    ],
    "colder": [
        "make it cooler",    
    ],
    "stop": [
        "pull over",
        "pullover",          # Vosk merges the two words ~40% of the time
    ],
    "thanks": [
        "thank you",        
    ],
    "trunk": [
        "open the trunk",    
    ]#,
    # "wakeup_no_cmd": []
}

# Load Vosk model for speech recognition
VOSK_PATH = "models/vosk-model-small-en-us-0.15"  # small model for Pi compatibility
vosk_model = Model(VOSK_PATH)
recognizer = KaldiRecognizer(vosk_model, RATE)

def detect_keywords(audio_queue_keywords, command_queue=None, heartbeat=None):
    """Thread to detect spoken commands using Vosk with fuzzy matching."""
    global recognizer

    while True:
        try:
            audio_data = audio_queue_keywords.get()

            if heartbeat is not None:
                heartbeat[0] = time.monotonic()
            int16_audio = (audio_data * 32767).astype(np.int16)

            if recognizer.AcceptWaveform(int16_audio.tobytes()):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
                print(f"Processing recognized text: {text}")

                if any(wake in text for wake in WAKEUP_PHRASES):
                    commands = get_commands(text)

                    if commands:
                        for command in commands:
                            print(f"COMMAND DETECTED: {command} 🗣️🗣️🗣️🗣️🗣️🗣️")
                            if command_queue is not None:
                                try:
                                    command_queue.put_nowait((1, time.monotonic(), command))
                                except queue.Full:
                                    print(f"Command queue full; dropping '{command}'.")
                    else:
                        print("WAKEUP PHRASE DETECTED BUT NO KNOWN COMMAND MATCHED")
            else:
                pass  # partial results are not actionable; don't print every chunk

        except Exception as e:
            print(f"Error in keyword detector: {e} — reinitialising recognizer")
            recognizer = KaldiRecognizer(vosk_model, RATE)
            time.sleep(0.5)


# Use fuzz to do fuzzy matching for phrases and return all commands in spoken order
def get_commands(text, threshold=70):
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
