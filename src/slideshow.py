import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path

IMAGE_DISPLAY_SECONDS = 5
BLANK_DISPLAY_SECONDS = 3

IMAGE_DIR = Path(__file__).resolve().parent / "images"
BLANK_PATH = IMAGE_DIR / "blank.png"


def run_slideshow():
    root = tk.Tk()
    root.title("Image Slideshow")
    root.attributes("-fullscreen", True)
    root.config(cursor="none")

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    display_size = (screen_width, screen_height)

    def load(path):
        img = Image.open(str(path)).resize(display_size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() == ".png" and p.name != "blank.png"
    )

    blank_photo = load(BLANK_PATH)

    label = tk.Label(root)
    label.pack()
    label.config(image=blank_photo)
    label.image = blank_photo

    # Build a flat sequence: [img1, blank, img2, blank, ...]
    sequence = []
    for img_path in images:
        sequence.append((img_path, IMAGE_DISPLAY_SECONDS * 1000))
        sequence.append((BLANK_PATH, BLANK_DISPLAY_SECONDS * 1000))

    cache = {"blank.png": blank_photo}
    index = [0]

    def show_next():
        if not sequence:
            return

        path, duration_ms = sequence[index[0]]
        index[0] = (index[0] + 1) % len(sequence)

        key = path.name
        if key not in cache:
            cache[key] = load(path)

        photo = cache[key]
        label.config(image=photo)
        label.image = photo

        root.after(duration_ms, show_next)

    root.after(0, show_next)
    root.mainloop()


if __name__ == "__main__":
    run_slideshow()
