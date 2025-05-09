import sounddevice as sd
import numpy as np
import json
import queue
import threading
from vosk import Model, KaldiRecognizer

# Audio parameters
RATE = 16000  # 16kHz sample rate
CHUNK = 64000  # 4 seconds of audio per chunk

# Keywords for Vosk to recognize
HOT_PHRASES = ["make it hotter", "please make it hotter"]

# Queue for audio data
audio_queue = queue.Queue(maxsize=10)

# Load Vosk model for speech recognition
VOSK_PATH = "vosk-model-small-en-us-0.15"  # Make sure this path is correct
vosk_model = Model(VOSK_PATH)
recognizer = KaldiRecognizer(vosk_model, RATE)

def audio_callback(indata, frames, time_info, status):
    """Callback function for audio input stream."""
    if status:
        print("Audio status:", status)

    # Convert to 16-bit mono PCM data
    audio_data = (indata[:, 0] * 32767).astype(np.int16)  # Convert float32 to int16
    audio_queue.put(audio_data)
    print(f"Audio data length: {len(audio_data)}")  # Log audio data length

def detect_keywords():
    """Thread to detect spoken hot/cold commands using Vosk."""
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            
            print(f"Processing audio data of length {len(audio_data)}")  # Check length of audio being processed
            
            # Process the audio for speech recognition
            result = None  # Initialize result variable
            if recognizer.AcceptWaveform(audio_data.tobytes()):
                result = json.loads(recognizer.Result())
                print(f"Recognized text (final): {result.get('text', '')}")  # Added log for every recognized text
            else:
                # Log partial recognition if it's not a full match
                partial_result = json.loads(recognizer.PartialResult())
                print(f"Partial recognized text: {partial_result.get('text', '')}")
                result = partial_result  # Set result to partial

            # Only proceed if we have a valid result
            if result:
                text = result.get("text", "")
                print(f"Processing recognized text: {text}")

                # Check if the phrase "make it hotter" is detected
                if any(phrase in text.lower() for phrase in HOT_PHRASES):
                    print("🔥 Command Detected: Make it Hotter 🔥")

def start_threads():
    """Start detection thread for keywords."""
    threading.Thread(target=detect_keywords, daemon=True).start()

def main():
    """Main function to set up the audio stream and start threads."""
    print("Starting the hot phrase detection system.")
    start_threads()

    with sd.InputStream(
        channels=1,
        samplerate=RATE,
        blocksize=CHUNK,
        dtype='float32',
        callback=audio_callback
    ):
        print("Listening for hot phrases. Press Ctrl+C to stop.")
        while True:
            pass  # Keep the program running, the detection happens in the background

if __name__ == "__main__":
    main()
