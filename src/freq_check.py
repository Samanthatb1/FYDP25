"""
Records a short audio sample and reports the peak frequency and its magnitude.
"""

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # Hz
DURATION = 3         # seconds to record


def main():
    print(f"Recording for {DURATION} seconds... speak now!")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording done.")

    samples = audio[:, 0]

    fft_vals = np.fft.rfft(samples)
    magnitudes = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / SAMPLE_RATE)

    # Ignore frequencies below this to exclude 60/120 Hz electrical hum
    LOW_CUTOFF_HZ = 200

    valid_mask = freqs >= LOW_CUTOFF_HZ
    valid_magnitudes = magnitudes[valid_mask]
    valid_freqs = freqs[valid_mask]

    # Frequency with the highest magnitude (dominant frequency)
    peak_idx = np.argmax(valid_magnitudes)
    peak_freq = valid_freqs[peak_idx]
    peak_mag = valid_magnitudes[peak_idx]

    # Magnitude at the highest frequency bin (Nyquist limit, within valid range)
    max_freq = valid_freqs[-1]
    max_freq_mag = valid_magnitudes[-1]

    print(f"\n--- Dominant frequency (highest magnitude) ---")
    print(f"Frequency  : {peak_freq:.2f} Hz")
    print(f"Magnitude  : {peak_mag:.4f}")

    print(f"\n--- Magnitude at highest frequency bin ---")
    print(f"Frequency  : {max_freq:.2f} Hz")
    print(f"Magnitude  : {max_freq_mag:.4f}")


if __name__ == "__main__":
    main()
