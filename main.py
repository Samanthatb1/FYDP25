import numpy as np
from tflite_runtime.interpreter import Interpreter
import sounddevice as sd
import time
import csv
# import scipy.signal
import json
import queue
import threading
from vosk import Model, KaldiRecognizer

# Audio parameters
RATE = 16000  # 16kHz sample rate, required by YAMNet
CHUNK = 32000  # 2 second of audio

# Band-pass filter parameters
LOW_CUT = 1000  # Hz (low frequency cutoff for siren-like sounds)
HIGH_CUT = 5000  # Hz (high frequency cutoff for siren-like sounds)

# Keywords for Vosk to recognize
HOT_PHRASES = ["hey Uber make it hotter", "hey ober make it hotter"]
COLD_PHRASES = ["hey Uber make it colder", "hey ober make it colder"]

# Queue for audio data
audio_queue_siren = queue.Queue(maxsize=10)  # Queue for siren detection
audio_queue_keywords = queue.Queue(maxsize=10)  # Queue for keyword detection

# Load YAMNet model
print("Loading YAMNet TFLite model...")
interpreter = Interpreter(model_path="yamnet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("YAMNet TFLite model loaded.")


# Load class names for YAMNet
class_names = []
with open('yamnet_class_map.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row['display_name'])


# Load Vosk model for speech recognition
VOSK_PATH = "vosk-model-small-en-us-0.15"
vosk_model = Model(VOSK_PATH)
recognizer = KaldiRecognizer(vosk_model, RATE)

# def bandpass_filter(audio_data, low_cut, high_cut, sample_rate):
#     """Apply a band-pass filter to the audio data."""
#     nyquist = 0.5 * sample_rate
#     low = low_cut / nyquist
#     high = high_cut / nyquist
#     b, a = scipy.signal.butter(4, [low, high], btype='band')
#     return scipy.signal.filtfilt(b, a, audio_data)

# def has_siren_frequencies(audio_data, low_cut, high_cut, sample_rate):
#     """Check if the audio contains siren-like frequencies."""
#     filtered_audio = bandpass_filter(audio_data, low_cut, high_cut, sample_rate)
    
#     # Compute the Power (sum of squares) in the filtered signal
#     energy = np.sum(filtered_audio ** 2)
#     print(f"Power in 1–5 kHz: {energy:.2f}")

#     return energy > 0.01  # Adjust threshold based on testing


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)

    # Convert to mono float32
    audio_data = indata[:, 0].astype(np.float32)

    # Resample from 44100 to 16000 using linear interpolation
    # TODO do i even need this because my macs sample rate is 48000.0 and it worked fine
    # input_length = len(audio_data)
    # output_length = int(input_length * 16000 / 44100)
    # resampled_indices = np.linspace(0, input_length - 1, output_length)
    # resampled_data = np.interp(resampled_indices, np.arange(input_length), audio_data).astype(np.float32)

    # # Debug: print once every few seconds
    # print(f"Resampled audio length: {len(resampled_data)}")

    # Add to queues
    if not audio_queue_siren.full():
        audio_queue_siren.put(audio_data)
    if not audio_queue_keywords.full():
        audio_queue_keywords.put(audio_data)


def detect_siren():
    """Thread to detect sirens using YAMNet."""
    while True:
        time.sleep(0.01)
        print("listening for siren")
        if not audio_queue_siren.empty():
            audio_data = audio_queue_siren.get()
            
            # Apply the band-pass filter check for siren-like frequencies
            # TODO scipy not compatible with arm7
            # if not has_siren_frequencies(audio_data, LOW_CUT, HIGH_CUT, RATE):
            #     print("NO siren range frequencies")
            #     continue  # Skip if no siren-like frequencies

            print("siren range frequencies")

            # Preprocess and reshape input
            audio_input = audio_data.astype(np.float32).flatten()
            interpreter.set_tensor(input_details[0]['index'], audio_input)

            # Run inference
            interpreter.invoke()
            scores = interpreter.get_tensor(output_details[0]['index'])[0]

            # Get top 5 classes
            top_classes = np.argsort(scores)[-5:][::-1]


            print("\nTop 5 predicted classes:")
            for i in top_classes:
                print(f'{class_names[i]}: {scores[i]:.3f}')

            # Check for siren-related classes
            siren_classes = ['Siren', 'Civil defense siren', 'Police car (siren)',
                             'Ambulance (siren)', 'Fire engine, fire truck (siren)']
            if any(class_names[i] in siren_classes for i in top_classes):
                print("🚨 ALERT: Potential siren detected!")

def detect_keywords():
    """Thread to detect spoken hot/cold commands using Vosk."""
    print("queue size: ", audio_queue_keywords.qsize())

    while True:
        time.sleep(0.01)
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
                    print("❄️ Command Detected: Make it Colder ❄️")

def start_threads():
    """Start detection threads for siren and keywords."""
    threading.Thread(target=detect_siren, daemon=True).start()
    threading.Thread(target=detect_keywords, daemon=True).start()

def main():
    print("Starting the detection system.")
    start_threads()

    device_info = sd.query_devices(kind='input')
    print("devices: ", device_info)

    try:
        with sd.InputStream(
            device=10,  # Make sure this is your mic's index
            channels=1,
            samplerate=44100.0,
            blocksize=int(44100 * 2),
            dtype='float32',
            callback=audio_callback
        ):
            print("Input stream started successfully.")
            while True:
                time.sleep(0.1)
    except Exception as e:
        print(f"Failed to start input stream: {e}")



if __name__ == "__main__":
    main()
