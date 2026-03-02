import gc
import csv
import os
import sys
import time
from queue import Empty

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bandpass_filter import has_siren_frequencies
from constants import RATE


def detect_siren(audio_queue, result_queue):
    """
    Entry point for the siren-detection process.

    TensorFlow and YAMNet are imported and loaded here — not at module
    level — so the main process never loads TF into its address space.
    Running in a separate process also eliminates GIL contention with the
    audio callback and the Vosk speech-recognition thread.
    """
    import tensorflow as tf
    import tensorflow_hub as hub

    print("Loading YAMNet model...")
    yamnet_model = hub.load("models/yamnet/9616fd04ec2360621642ef9455b84f4b668e219e")
    print("YAMNet model loaded successfully.")

    class_names = []
    with open('models/yamnet_class_map.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row['display_name'])

    SIREN_ALERT_COOLDOWN = 3.0
    YAMNET_MIN_INTERVAL = 3.0
    GC_INTERVAL = 60.0
    SIREN_SCORE_THRESHOLD = 0.15

    siren_classes = frozenset([
        'Siren', 'Civil defense siren', 'Police car (siren)',
        'Ambulance (siren)', 'Fire engine, fire truck (siren)',
        'Alarm', 'Buzzer', 'Emergency vehicle',
        'Vehicle horn, car horn, honking',
    ])

    last_alert_time = 0.0
    last_yamnet_time = 0.0
    last_gc_time = time.monotonic()

    while True:
        try:
            audio_data = audio_queue.get()

            # Skip to the freshest available chunk.
            while not audio_queue.empty():
                try:
                    audio_data = audio_queue.get_nowait()
                except Empty:
                    break

            now = time.monotonic()
            if now - last_gc_time > GC_INTERVAL:
                gc.collect()
                last_gc_time = now

            if not has_siren_frequencies(audio_data, RATE):
                continue

            if now - last_yamnet_time < YAMNET_MIN_INTERVAL:
                continue

            audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
            scores, _, _ = yamnet_model(audio_tensor)
            last_yamnet_time = time.monotonic()

            detected = scores.shape[0] > 0 and any(
                class_names[i] in siren_classes
                and scores[0][i].numpy() > SIREN_SCORE_THRESHOLD
                for i in tf.argsort(scores, axis=-1, direction='DESCENDING')[0][:5]
            )

            del audio_tensor, scores
            gc.collect()

            if detected:
                now = time.monotonic()
                if now - last_alert_time >= SIREN_ALERT_COOLDOWN:
                    last_alert_time = now
                    print("🚨 ALERT: Siren Detected! 🚨")
                    try:
                        result_queue.put_nowait("siren detected")
                    except Exception:
                        pass

        except Exception as e:
            print(f"Error in siren detector: {e}")
            time.sleep(0.5)
