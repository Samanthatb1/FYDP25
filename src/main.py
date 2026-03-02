import numpy as np
import sounddevice as sd
import time
import queue
import threading
import signal
import sys
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk

from detectors.siren_detector import detect_siren
from detectors.speech_detector import detect_keywords

from constants import RATE, CHUNK

# Software gain for microphones without hardware gain control
# Increase this if the mic is too quiet (try 4.0, 5.0, etc.)
AUDIO_GAIN = 3.0

# Queue for audio data
audio_queue_siren = queue.Queue(maxsize=10)  # Queue for siren detection
audio_queue_keywords = queue.Queue(maxsize=10)  # Queue for keyword detection

_last_audio_callback = time.monotonic()

def audio_callback(indata, frames, time_info, status):
    """Callback function for audio input stream."""
    global _last_audio_callback
    _last_audio_callback = time.monotonic()

    if status:
        print("Audio status:", status)

    audio_data = indata[:, 0].astype(np.float32)

    audio_data = audio_data * AUDIO_GAIN
    audio_data = np.clip(audio_data, -1.0, 1.0)

    try:
        audio_queue_siren.put_nowait(audio_data)
    except queue.Full:
        pass
    try:
        audio_queue_keywords.put_nowait(audio_data)
    except queue.Full:
        pass


def start_detection_threads(command_queue):
    """Start detection threads for siren and keywords."""
    threading.Thread(
        target=detect_siren, args=(audio_queue_siren, command_queue), daemon=True
    ).start()
    threading.Thread(
        target=detect_keywords, args=(audio_queue_keywords, command_queue), daemon=True
    ).start()


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
    When no command is waiting, blank.png (or a generated placeholder) is shown.
    Images are automatically resized to fit the screen.
    
    Args:
        command_queue: Queue containing commands to display
        image_dir: Directory containing image files
    """

    root = tk.Tk()
    root.title("Command Display")
    root.attributes('-fullscreen', True)
    root.config(cursor='none')

    # Automatically detect screen size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    display_size = (screen_width, screen_height)
    print(f"Detected screen size: {screen_width}x{screen_height}")

    image_dir = Path(image_dir) if image_dir else Path(__file__).resolve().parent.parent
    cache = {}
    image_aliases = {"siren detected": "siren"}

    blank_path = image_dir / "blank.png"
    # Load and resize blank image
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
            # Load and resize image using PIL
            pil_img = Image.open(str(image_path))
            pil_img = pil_img.resize(display_size, Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(pil_img)
        else:
            print(f"No image found for '{command_name}' at {image_path}, using blank.")
            img = blank_image

        cache[key] = img
        return img

    def update_display():
        try:
            try:
                _, _, command_name = command_queue.get_nowait()
            except queue.Empty:
                if label.image != blank_image:
                    label.config(image=blank_image)
                    label.image = blank_image

                stale = time.monotonic() - _last_audio_callback
                if stale > 5:
                    print(f"WARNING: no audio callback for {stale:.0f}s — mic may be stalled")

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
    """Main entry point to set up audio, detection, and command display."""
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    print("Starting the detection system.")

    command_queue = queue.PriorityQueue(maxsize=20)
    start_detection_threads(command_queue)

    threading.Thread(target=start_audio_stream, daemon=True).start()
    print("Audio stream running. Starting command display window.")

    run_command_display(command_queue, "src/images")


if __name__ == "__main__":
    main()
