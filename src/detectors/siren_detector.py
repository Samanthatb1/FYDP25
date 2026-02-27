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
    # Throttle how often we enqueue siren alerts to avoid flooding the UI queue.
    # This is necessary because if a siren is played for 5 seconds, that means originally
    # we'd add 25 siren commands to the queue, so it essentially hogs the queue
    SIREN_ALERT_COOLDOWN = 3.0  # seconds between queued siren alerts
    last_alert_time = 0.0

    while True:
        audio_data = audio_queue_siren.get()

        # Apply the singal analysis filter check for siren-like frequencies
        if not has_siren_frequencies(audio_data, RATE):
            print("👂 No siren range frequencies")
            continue  # Skip if no siren-like frequencies

        print("👂 Siren range frequencies")

        # Run the YAMNet model
        audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
        scores, _, _ = yamnet_model(audio_tensor)

        # Get top 5 classes
        top_classes = tf.argsort(scores, axis=-1, direction='DESCENDING')[0][:5]

        # print("Model Classified: ")
        # for i in top_classes:
        #     print(f'{class_names[i]}: {scores[0][i].numpy():.3f}')

        # Check for siren-related classes
        siren_classes = set(['Siren', 'Civil defense siren', 'Police car (siren)',
                        'Ambulance (siren)', 'Fire engine, fire truck (siren)',
                        'Alarm', 'Buzzer', 'Emergency vehicle',
                        'Vehicle horn, car horn, honking'])

        # Require a minimum confidence score so YAMNet must be fairly certain,
        # not just vaguely placing a siren class somewhere in the top 5.
        SIREN_SCORE_THRESHOLD = 0.15

        if any(class_names[i] in siren_classes and scores[0][i].numpy() > SIREN_SCORE_THRESHOLD
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
                        print("Command queue full; dropping 'siren detected' notification.")