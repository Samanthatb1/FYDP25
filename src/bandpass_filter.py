import numpy as np
from scipy.signal import butter, sosfilt      # For designing and applying the bandpass filter
from scipy.fft import rfft, rfftfreq          # For performing FFT and getting corresponding frequencies

LOW_CUT = 1000  # Hz (lower edge of typical siren range)
HIGH_CUT = 5000  # Hz (upper edge of typical siren range)

# Pre-compute filter coefficients once at import time — calling butter() on every
# audio chunk (1.6×/second) was wasting significant CPU for no reason.
_DEFAULT_SOS = butter(N=4, Wn=[LOW_CUT, HIGH_CUT], btype='bandpass', fs=16000, output='sos')

def bandpass_filter(audio_data, sampling_rate, lowcut=LOW_CUT, highcut=HIGH_CUT, order=4):
    if lowcut == LOW_CUT and highcut == HIGH_CUT and sampling_rate == 16000:
        sos = _DEFAULT_SOS
    else:
        sos = butter(N=order, Wn=[lowcut, highcut], btype='bandpass', fs=sampling_rate, output='sos')
    return sosfilt(sos, audio_data)

# fft = fast fourier transform
# basically allows us to get the frequencies and corresponding magnitudes for all 
    # the frequencies in the sample
# our data analysis shows that strong magnitudes of large frequencies (1000-5000)
    # are a good pre-check for siren audios
# references: https://realpython.com/python-scipy-fft/
def peak_fft_magnitude_in_range(filtered_data, sampling_rate, lowcut=LOW_CUT, highcut=HIGH_CUT):
    # filtered_data is in the time domain (x axis time, y axis amplitude)

    # Compute magnitudes of each frequency (convert to frequency domain) -> array of magnitudes
    fft_scores = np.abs(rfft(filtered_data)) # rfft = real value fast fourier transform

    # Compute corresponding frequency bins
    # ~ freqs[0] = 0 Hz
    freqs = rfftfreq(len(filtered_data), 1 / sampling_rate)

    # Ignore frequencies outside of 1000-5000Hz with this mask
    mask = (freqs >= lowcut) & (freqs <= highcut)
    if not np.any(mask):
        print("No frequencies found in target range.")
        return 0.0

    # Apply the mask
    masked_scores = fft_scores[mask]
    masked_freqs = freqs[mask]

    peak_index = np.argmax(masked_scores)
    peak_magnitude = masked_scores[peak_index]

    return peak_magnitude


def has_siren_frequencies(audio_data, sampling_rate, 
                          energy_threshold=0.000120, # "How much sound energy, on average, is sitting in the 1000–5000 Hz range during this chunk?"
                          fft_magnitude_threshold=130):
    audio_data = audio_data - np.mean(audio_data) # center around 0

    filtered = bandpass_filter(audio_data, sampling_rate)

    # compute the energy of the filtered signal on the RAW (un-normalized) audio
    # so that absolute loudness is preserved — quiet sounds like laughter won't
    # be artificially amplified past this gate
    energy = np.sum(filtered ** 2) / len(filtered)

    if energy < energy_threshold:
        return False

    # normalize only AFTER the energy gate, so the FFT shape analysis is
    # scale-independent while the loudness check above still reflects reality
    if np.max(np.abs(filtered)) > 0:
        filtered = filtered / np.max(np.abs(filtered))

    # get the peak magnitude for frequency in target range (1000-5000Hz)
    peak_magnitude = peak_fft_magnitude_in_range(filtered, sampling_rate)
    return peak_magnitude > fft_magnitude_threshold
