import numpy as np
import sys
import os
import csv
import queue
import time
import tensorflow as tf
import tensorflow_hub as hub

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bandpass_filter import has_siren_frequencies 
from constants import RATE

# Load YAMNet model
print("Loading YAMNet model...")
local_model_path = "models/yamnet/9616fd04ec2360621642ef9455b84f4b668e219e"
yamnet_model = hub.load(local_model_path)
# yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print("YAMNet model loaded successfully.")

# Load class names for YAMNet
class_names = []
with open('models/yamnet_class_map.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row['display_name'])


def detect_siren(audio_queue_siren, command_queue=None):
    """Thread to detect sirens using YAMNet."""
    SIREN_ALERT_COOLDOWN = 3.0
    last_alert_time = 0.0

    siren_classes = frozenset([
        'Siren', 'Civil defense siren', 'Police car (siren)',
        'Ambulance (siren)', 'Fire engine, fire truck (siren)',
        'Alarm', 'Buzzer', 'Emergency vehicle',
        'Vehicle horn, car horn, honking',
    ])
    SIREN_SCORE_THRESHOLD = 0.15

    while True:
        try:
            audio_data = audio_queue_siren.get()

            # YAMNet is slow on a Pi — skip to the freshest chunk so we don't
            # process audio that is many seconds old.
            while not audio_queue_siren.empty():
                try:
                    audio_data = audio_queue_siren.get_nowait()
                except queue.Empty:
                    break

            if not has_siren_frequencies(audio_data, RATE):
                print("👂 No siren range frequencies")
                continue

            print("👂 Siren range frequencies")

            audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
            scores, _, _ = yamnet_model(audio_tensor)

            if scores.shape[0] == 0:
                continue

            top_classes = tf.argsort(scores, axis=-1, direction='DESCENDING')[0][:5]

            if any(class_names[i] in siren_classes
                   and scores[0][i].numpy() > SIREN_SCORE_THRESHOLD
                   for i in top_classes):
                print("🚨 ALERT: Siren Detected! 🚨")
                if command_queue is not None:
                    now = time.monotonic()
                    if now - last_alert_time < SIREN_ALERT_COOLDOWN:
                        continue
                    last_alert_time = now
                    try:
                        command_queue.put_nowait((0, now, "siren detected"))
                    except queue.Full:
                        print("Command queue full; dropping 'siren detected'.")

        except Exception as e:
            print(f"Error in siren detector: {e}")
            time.sleep(0.5)