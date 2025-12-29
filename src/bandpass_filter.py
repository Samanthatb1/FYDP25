import numpy as np
from scipy.signal import butter, sosfilt      # For designing and applying the bandpass filter
from scipy.fft import rfft, rfftfreq          # For performing FFT and getting corresponding frequencies

LOW_CUT = 1000  # Hz (lower edge of typical siren range)
HIGH_CUT = 5000  # Hz (upper edge of typical siren range)

def bandpass_filter(audio_data, sampling_rate, lowcut=LOW_CUT, highcut=HIGH_CUT, order=4):
    # Defines the bandpass filter
    sos = butter(N=order, Wn=[lowcut, highcut], btype='bandpass', fs=sampling_rate, output='sos')
    # Actually applies the filter to the data
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

    # Find the index of the peak magnitude
    peak_index = np.argmax(masked_scores)
    peak_magnitude = masked_scores[peak_index]
    peak_frequency = masked_freqs[peak_index]

    # Print the frequency with the highest magnitude
    print(f"Strongest frequency: {peak_frequency:.2f} Hz (magnitude: {peak_magnitude:.2f})")

    return peak_magnitude


def has_siren_frequencies(audio_data, sampling_rate, 
                          energy_threshold=0.001,
                          fft_magnitude_threshold=130):
    # Normalize audio to range [-1, 1]
    # eg. 1 is the highest energy possible
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    audio_data = audio_data - np.mean(audio_data) # center around 0

    filtered = bandpass_filter(audio_data, sampling_rate)

    # compute the energy of the filtered signal
    # the average power of the signal within the siren band (1000–5000 Hz)
    energy = np.sum(filtered ** 2) / len(filtered)

    # we found this threshold from data analysis
    if energy < energy_threshold:
        return False
    
    print("energy: ", energy , "J")

    # get the peak magnitude for frequency in target range (1000-5000Hz)
    peak_magnitude = peak_fft_magnitude_in_range(filtered, sampling_rate)
    return peak_magnitude > fft_magnitude_threshold
