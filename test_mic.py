import sounddevice as sd
import numpy as np
import wave

DURATION = 10  # seconds
SAMPLE_RATE = 44100
FILENAME = "mic_test.wav"

def record_to_wav():
    device_info = sd.query_devices(kind='input')
    print("devices: ", device_info)

    print("Recording from mic...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print(f"Recording complete. Saving to {FILENAME}...")

    # Save as WAV file
    with wave.open(FILENAME, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(recording.tobytes())

    print(f"Saved '{FILENAME}'. You can transfer it to another machine and play it back.")

if __name__ == "__main__":
    try:
        record_to_wav()
    except Exception as e:
        print("Error recording:", e)
