import numpy as np
import sounddevice as sd
import time
import queue
import threading

import os

from detectors.siren_detector import detect_siren
from detectors.speech_detector import detect_keywords

from constants import RATE, CHUNK

# Queue for audio data
audio_queue_siren = queue.Queue(maxsize=10)  # Queue for siren detection
audio_queue_keywords = queue.Queue(maxsize=10)  # Queue for keyword detection

def audio_callback(indata, frames, time_info, status):
    """Callback function for audio input stream."""
    if status:
        print("Audio status:", status)

    # Flatten and convert to tensor
    audio_data = indata[:, 0].astype(np.float32)  # Ensure it's mono

    if len(audio_data) < 64:
        print("⚠️ Audio buffer too short")
        return False
    if np.all(audio_data == 0):
        print("⚠️ Empty audio data")
        return False
    
    # Put audio data into both queues
    try:
        audio_queue_siren.put_nowait(audio_data)
        audio_queue_keywords.put_nowait(audio_data)
    except queue.Full:
        print("[Audio Callback] ⚠️ Queue full, skipping audio frame")

def start_threads():
    """Start detection threads for siren and keywords."""
    threading.Thread(target=detect_siren, args=(audio_queue_siren,), daemon=True).start()
    threading.Thread(target=detect_keywords, args=(audio_queue_keywords,), daemon=True).start()

def main():
    """Main function to set up the audio stream and start threads."""
    print("Starting the detection system.")
    start_threads()

    with sd.InputStream(
        channels=1,
        samplerate=RATE,
        blocksize=CHUNK,
        dtype='float32',
        callback=audio_callback
    ):
        print("Listening for sirens and keywords. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)

if __name__ == "__main__":
    main()
