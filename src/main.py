import numpy as np
import sounddevice as sd
import time
import queue
import threading
import multiprocessing
import signal
import sys
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk

from detectors.speech_detector import detect_keywords
from constants import RATE, CHUNK

AUDIO_GAIN = 3.0

audio_queue_keywords = queue.Queue(maxsize=10)

_last_audio_callback = time.monotonic()
_speech_heartbeat = [time.monotonic()]

# Assigned in main() before the audio stream starts.
_mp_audio_queue = None


def audio_callback(indata, frames, time_info, status):
    """Callback function for audio input stream."""
    global _last_audio_callback
    _last_audio_callback = time.monotonic()

    if status:
        print("Audio status:", status)

    raw = indata[:, 0].astype(np.float32)

    # Siren detector (separate process) gets raw audio.
    try:
        _mp_audio_queue.put_nowait(raw)
    except Exception:
        pass

    # Speech detector (thread) gets gained audio.
    gained = np.clip(raw * AUDIO_GAIN, -1.0, 1.0)
    try:
        audio_queue_keywords.put_nowait(gained)
    except queue.Full:
        pass


def start_audio_stream():
    """Start the audio stream, restarting automatically on failure."""
    while True:
        try:
            with sd.InputStream(
                channels=1,
                samplerate=RATE,
                blocksize=CHUNK,
                dtype="float32",
                callback=audio_callback,
            ):
                print("Audio stream started — listening for sirens and keywords.")
                while True:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Audio stream error: {e}. Restarting in 2 seconds...")
            time.sleep(2)


def run_command_display(command_queue, image_dir=None):
    """
    Run a Tkinter UI that shows images for commands from the queue.

    The image file must match the command name with a .png extension.
    When no command is waiting, blank.png is shown.
    Images are automatically resized to fit the screen.
    """
    root = tk.Tk()
    root.title("Command Display")
    root.attributes('-fullscreen', True)
    root.config(cursor='none')

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    display_size = (screen_width, screen_height)

    image_dir = Path(image_dir) if image_dir else Path(__file__).resolve().parent.parent
    cache = {}
    image_aliases = {"siren detected": "siren"}

    blank_path = image_dir / "blank.png"
    pil_blank = Image.open(str(blank_path))
    pil_blank = pil_blank.resize(display_size, Image.Resampling.LANCZOS)
    blank_image = ImageTk.PhotoImage(pil_blank)
    cache["blank"] = blank_image

    label = tk.Label(root)
    label.pack()
    label.config(image=blank_image)
    label.image = blank_image

    def load_image_for(command_name):
        key = image_aliases.get(command_name, command_name)
        if key in cache:
            return cache[key]
        image_path = image_dir / f"{key}.png"
        if image_path.exists():
            pil_img = Image.open(str(image_path))
            pil_img = pil_img.resize(display_size, Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(pil_img)
        else:
            print(f"No image found for '{command_name}' at {image_path}, using blank.")
            img = blank_image
        cache[key] = img
        return img

    last_warning_print = [0.0]

    def update_display():
        try:
            try:
                _, _, command_name = command_queue.get_nowait()
            except queue.Empty:
                now = time.monotonic()
                audio_stale = now - _last_audio_callback
                speech_stale = now - _speech_heartbeat[0]

                if now - last_warning_print[0] > 10:
                    if audio_stale > 5:
                        print(f"WARNING: no audio for {audio_stale:.0f}s")
                    if speech_stale > 10:
                        print(f"WARNING: speech detector silent for {speech_stale:.0f}s")
                    if audio_stale > 5 or speech_stale > 10:
                        last_warning_print[0] = now

                if label.image != blank_image:
                    label.config(image=blank_image)
                    label.image = blank_image

                root.after(200, update_display)
                return

            img = load_image_for(command_name)
            label.config(image=img)
            label.image = img
            display_duration_ms = 4000 if command_name == "siren detected" else 10000
            root.after(display_duration_ms, update_display)

        except Exception as e:
            print(f"Error in update_display: {e}")
            root.after(500, update_display)

    root.after(0, update_display)
    root.mainloop()


def handle_exit(sig, frame):
    print("Shutting down cleanly...")
    sys.exit(0)


def main():
    global _mp_audio_queue

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    print("Starting the detection system.")

    command_queue = queue.PriorityQueue(maxsize=20)

    # Siren detection runs in a separate PROCESS so TensorFlow's memory
    # and CPU usage are fully isolated from the main process.
    _mp_audio_queue = multiprocessing.Queue(maxsize=10)
    siren_result_queue = multiprocessing.Queue(maxsize=20)

    from detectors.siren_detector import detect_siren
    siren_proc = multiprocessing.Process(
        target=detect_siren,
        args=(_mp_audio_queue, siren_result_queue),
        daemon=True,
    )
    siren_proc.start()

    # Bridge thread forwards siren detections into the shared command queue.
    def bridge_siren_results():
        while True:
            try:
                result = siren_result_queue.get()
                try:
                    command_queue.put_nowait((0, time.monotonic(), result))
                except queue.Full:
                    pass
            except Exception:
                pass

    threading.Thread(target=bridge_siren_results, daemon=True).start()

    # Speech detection stays as a thread (Vosk is lightweight).
    threading.Thread(
        target=detect_keywords,
        args=(audio_queue_keywords, command_queue, _speech_heartbeat),
        daemon=True,
    ).start()

    threading.Thread(target=start_audio_stream, daemon=True).start()
    print("Audio stream running. Starting command display window.")

    run_command_display(command_queue, "src/images")


if __name__ == "__main__":
    main()
