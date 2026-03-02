import gc
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


def detect_siren(audio_queue_siren, command_queue=None, heartbeat=None):
    """Thread to detect sirens using YAMNet."""
    SIREN_ALERT_COOLDOWN = 3.0
    YAMNET_MIN_INTERVAL = 3.0  # seconds — cap inference rate to protect Pi CPU
    GC_INTERVAL = 60.0         # force a GC pass every 60s to reclaim TF memory
    last_alert_time = 0.0
    last_yamnet_time = 0.0
    last_gc_time = time.monotonic()

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

            if heartbeat is not None:
                heartbeat[0] = time.monotonic()

            # Skip to the freshest chunk so we don't process stale audio.
            while not audio_queue_siren.empty():
                try:
                    audio_data = audio_queue_siren.get_nowait()
                except queue.Empty:
                    break

            # Periodic GC to reclaim TF intermediate tensors before memory fills up.
            now = time.monotonic()
            if now - last_gc_time > GC_INTERVAL:
                gc.collect()
                last_gc_time = now

            if not has_siren_frequencies(audio_data, RATE):
                continue

            if now - last_yamnet_time < YAMNET_MIN_INTERVAL:
                continue

            print("👂 Siren range frequencies — running YAMNet")

            audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
            scores, _, _ = yamnet_model(audio_tensor)

            # Record time AFTER inference so the interval is always measured from
            # when we finished, not when we started. On a slow/throttled Pi,
            # inference can take longer than YAMNET_MIN_INTERVAL, which would
            # cause back-to-back runs if we stamped before the call.
            last_yamnet_time = time.monotonic()

            detected = scores.shape[0] > 0 and any(
                class_names[i] in siren_classes
                and scores[0][i].numpy() > SIREN_SCORE_THRESHOLD
                for i in tf.argsort(scores, axis=-1, direction='DESCENDING')[0][:5]
            )

            # Explicitly release TF tensors so they don't accumulate in memory.
            del audio_tensor, scores
            gc.collect()

            # Brief yield so the audio callback thread gets CPU time.
            time.sleep(0.05)

            if detected:
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