import numpy as np
import sys
import os
import csv
import tensorflow as tf
import tensorflow_hub as hub

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bandpass_filter import has_siren_frequencies 
from constants import RATE

# Load YAMNet model
print("Loading YAMNet model...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print("YAMNet model loaded successfully.")

# Load class names for YAMNet
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

            # Run the YAMNet model
            audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
            scores, _, _ = yamnet_model(audio_tensor)

            # Get top 5 classes
            top_classes = tf.argsort(scores, axis=-1, direction='DESCENDING')[0][:5]

            print("\nTop 5 predicted classes:")
            for i in top_classes:
                print(f'{class_names[i]}: {scores[0][i].numpy():.3f}')

            # Check for siren-related classes
            siren_classes = ['Siren', 'Civil defense siren', 'Police car (siren)',
                            'Ambulance (siren)', 'Fire engine, fire truck (siren)']
            if any(class_names[i] in siren_classes for i in top_classes):
                print("🚨 ALERT: Potential siren detected!")