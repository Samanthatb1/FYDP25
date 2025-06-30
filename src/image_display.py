import tkinter as tk
import threading
import time
import os
from PIL import Image, ImageTk

ICON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'icons'))


def show_command_image(command, duration=3):
    image_path = os.path.join(ICON_DIR, f'{command}.png')
    print(f"looking for: {image_path}")
    if not os.path.exists(image_path):
        print(f"No image found for command: {command}")
        return

    def display():
        root = tk.Tk()
        root.attributes('-fullscreen', True)
        root.configure(background='white')
        root.attributes('-topmost', True)

        # Hide title bar, borders, etc.
        root.overrideredirect(True)

        # Load and display the image
        image = Image.open(image_path)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        image = image.resize((screen_width, screen_height))
        photo = ImageTk.PhotoImage(image)

        label = tk.Label(root, image=photo)
        label.pack()

        # Close the window after `duration` seconds
        def close_after_delay():
            time.sleep(duration)
            root.destroy()

        threading.Thread(target=close_after_delay, daemon=True).start()
        root.mainloop()

    threading.Thread(target=display, daemon=True).start()
