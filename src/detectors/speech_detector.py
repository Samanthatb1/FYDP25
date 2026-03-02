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
    "ivor",
    "use me driver",
    "the driver",
    "a driver"
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

def detect_keywords(audio_queue_keywords, command_queue=None):
    """Thread to detect spoken commands using Vosk with fuzzy matching."""
    # In noisy environments (e.g. a moving car), Vosk's VAD never detects silence
    # so AcceptWaveform never returns True on its own. Force-flush the recognizer
    # every FLUSH_EVERY chunks so accumulated speech is always processed.
    # At CHUNK=9600 and RATE=16000, each chunk is 0.6s → flush every ~3s.
    FLUSH_EVERY = 5
    chunks_since_flush = 0

    # When the wakeup phrase is heard but no command follows in the same flush
    # window (e.g. the user pauses between "excuse me driver" and "thank you"),
    # hold the wakeup text and prepend it to the next flush before matching.
    pending_wakeup_text = None

    while True:
        audio_data = audio_queue_keywords.get()
        int16_audio = (audio_data * 32767).astype(np.int16)
        chunks_since_flush += 1

        got_final = recognizer.AcceptWaveform(int16_audio.tobytes())

        if got_final:
            result = json.loads(recognizer.Result())
            chunks_since_flush = 0
        elif chunks_since_flush >= FLUSH_EVERY:
            # Force-flush: grab whatever Vosk has accumulated so far
            result = json.loads(recognizer.FinalResult())
            chunks_since_flush = 0
        else:
            # Partial result - don't process, just log for debugging
            partial_result = json.loads(recognizer.PartialResult())
            partial_text = partial_result.get("partial", "").lower()
            if partial_text:
                print(f"Partial recognition: {partial_text}")
            continue

        text = result.get("text", "").lower()

        # If we're waiting for a follow-up command after a wakeup phrase,
        # combine the held text with the new text and check for commands.
        if pending_wakeup_text is not None:
            combined = (pending_wakeup_text + " " + text).strip()
            pending_wakeup_text = None
            print(f"Processing combined text: {combined}")
            commands = get_commands(combined)
            if commands:
                for command in commands:
                    print(f"COMMAND DETECTED: {command} 🗣️🗣️🗣️🗣️🗣️🗣️")
                    if command_queue is not None:
                        try:
                            command_queue.put_nowait((1, time.monotonic(), command))
                        except queue.Full:
                            print(f"Command queue full; dropping '{command}'.")
            else:
                print("NO COMMAND FOUND IN FOLLOW-UP")
            continue

        if not text:
            continue

        print(f"Processing recognized text: {text}")

        # Check if wakeup phrase is in the text
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
                # Wakeup heard but no command yet — wait one more flush window
                print("WAKEUP PHRASE DETECTED, WAITING FOR COMMAND...")
                pending_wakeup_text = text


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
