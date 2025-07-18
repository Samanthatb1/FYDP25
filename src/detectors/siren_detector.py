import numpy as np
import os
import csv
import sys
from queue import Queue
import tflite_runtime.interpreter as tflite

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bandpass_filter import has_siren_frequencies
from constants import RATE

# Load TFLite model
print("Loading YAMNet TFLite model...")
interpreter = tflite.Interpreter(model_path="models/yamnet/yamnet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("YAMNet TFLite model loaded successfully.")

# Load class names
class_names = []
with open('models/yamnet_class_map.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row['display_name'])

def detect_siren(audio_queue_siren):
    """Thread to detect sirens using YAMNet."""
    while True:
        if not audio_queue_siren.empty():
            audio_data = audio_queue_siren.get()
            
            # Apply the band-pass filter check for siren-like frequencies
            if not has_siren_frequencies(audio_data, RATE):
                print("NO siren range frequencies")
                continue  # Skip if no siren-like frequencies

            print("siren range frequencies")

            # Preprocess for YAMNet
            audio_tensor = audio_data.astype(np.float32).flatten()
            interpreter.set_tensor(input_details[0]['index'], audio_tensor)

            # Run inference
            interpreter.invoke()
            scores = interpreter.get_tensor(output_details[0]['index'])[0]

            # Get top 5 classes
            top_classes = np.argsort(scores)[-5:][::-1]
            
            print("\nTop 5 predicted classes:")
            for i in top_classes:
                print(f'{class_names[i]}: {scores[0][i]:.3f}')

            # Check for siren-related classes
            siren_classes = ['Siren', 'Civil defense siren', 'Police car (siren)',
                            'Ambulance (siren)', 'Fire engine, fire truck (siren)']
            if any(class_names[i] in siren_classes for i in top_classes):
                print("🚨 ALERT: Potential siren detected!")