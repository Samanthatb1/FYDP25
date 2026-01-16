import numpy as np
import sounddevice as sd
import time
import queue
import threading
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk

from detectors.siren_detector import detect_siren
from detectors.speech_detector import detect_keywords

from constants import RATE, CHUNK

# Queue for audio data
audio_queue_siren = queue.Queue(maxsize=10)  # Queue for siren detection
audio_queue_keywords = queue.Queue(maxsize=10)  # Queue for keyword detection

def audio_callback(indata, frames, time_info, status):
    """Callback function for audio input stream."""
    if status:
        print("Audio status:", status)

    # Flatten and convert for tensor
    # audio data stores CHUNK number of samples that store the amplitude 
    audio_data = indata[:, 0].astype(np.float32)  # Ensure it's mono

    # Put audio data into both queues
    audio_queue_siren.put(audio_data)
    audio_queue_keywords.put(audio_data)


def start_detection_threads(command_queue):
    """Start detection threads for siren and keywords."""
    threading.Thread(
        target=detect_siren, args=(audio_queue_siren, command_queue), daemon=True
    ).start()
    threading.Thread(
        target=detect_keywords, args=(audio_queue_keywords, command_queue), daemon=True
    ).start()


def start_audio_stream():
    """Start the audio stream on a background thread."""
    with sd.InputStream(
        channels=1,
        samplerate=RATE,
        blocksize=CHUNK,
        dtype="float32",
        callback=audio_callback,
    ):
        print("Listening for sirens and keywords. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)

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
    
    # Make fullscreen without title bar/window decorations
    root.attributes('-fullscreen', True)
    
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
        if not command_queue.empty():
            _, _, command_name = command_queue.get()
            img = load_image_for(command_name)
            label.config(image=img)
            label.image = img
            # Show siren alert briefly; other commands stay visible longer.
            display_duration_ms = 3000 if command_name == "siren detected" else 10000
            root.after(display_duration_ms, update_display)
        else:
            if label.image != blank_image:
                label.config(image=blank_image)
                label.image = blank_image
            # Poll a little faster when idle so new commands appear quickly.
            root.after(200, update_display)

    root.after(0, update_display)
    root.mainloop()


def main():
    """Main entry point to set up audio, detection, and command display."""
    print("Starting the detection system.")

    command_queue = queue.PriorityQueue(maxsize=20)
    start_detection_threads(command_queue)

    threading.Thread(target=start_audio_stream, daemon=True).start()
    print("Audio stream running. Starting command display window.")

    run_command_display(command_queue, "src/images")


if __name__ == "__main__":
    main()
