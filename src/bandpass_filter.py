import numpy as np
import scipy.signal

# Band-pass filter parameters
LOW_CUT = 1000  # Hz (low frequency cutoff for siren-like sounds)
HIGH_CUT = 5000  # Hz (high frequency cutoff for siren-like sounds)

def bandpass_filter(audio_data, low_cut, high_cut, sample_rate):
    """Apply a band-pass filter to the audio data."""
    nyquist = 0.5 * sample_rate
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    return scipy.signal.filtfilt(b, a, audio_data)

def has_siren_frequencies(audio_data, sample_rate):
    """Check if the audio contains siren-like frequencies."""
    filtered_audio = bandpass_filter(audio_data, LOW_CUT, HIGH_CUT, sample_rate)
    
    # Compute the Power (sum of squares) in the filtered signal
    energy = np.sum(filtered_audio ** 2)
    print(f"Power in 1–5 kHz: {energy:.2f}")

    return energy > 0.01  # Adjust threshold based on testing