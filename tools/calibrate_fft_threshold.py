"""
Calibration script for the FFT peak magnitude threshold.

Runs the exact same pipeline as production (bandpass → energy gate →
normalise → FFT peak magnitude) and logs the peak magnitude per label so
you can pick a threshold that separates siren audio from false positives.

The energy gate step is intentionally kept in: if a chunk wouldn't reach
the FFT check in production (energy too low), it is skipped here too.

Usage
-----
  # Analyse a .wav file:
  python tools/calibrate_fft_threshold.py --file path/to/siren.wav --label police_80

  # Record live from the microphone for 5 seconds:
  python tools/calibrate_fft_threshold.py --live --duration 5 --label talking

  # Print a summary of all logged values and suggest a threshold:
  python tools/calibrate_fft_threshold.py --summary

Results are appended to tools/fft_log.csv.
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from scipy.signal import butter, sosfilt
from scipy.fft import rfft, rfftfreq

# ── Audio parameters (must match constants.py / production pipeline) ──────────
RATE              = 16000
CHUNK             = 9600
AUDIO_GAIN        = 3.0
LOW_CUT           = 1000
HIGH_CUT          = 5000
ENERGY_THRESHOLD  = 0.000120   # current production value — chunks below this
                                # never reach the FFT check, so we skip them too

LOG_PATH        = Path(__file__).parent / "fft_log.csv"
LOG_FIELDNAMES  = ["label", "chunk_index", "peak_magnitude", "passed_energy_gate",
                   "energy", "timestamp"]

SIREN_LABELS = {"police_80", "police_20", "ambulance_80", "ambulance_20",
                "fire_truck_80", "fire_truck_20"}

# ── Signal processing (mirrors bandpass_filter.py exactly) ────────────────────

def _bandpass_filter(audio_data):
    sos = butter(N=4, Wn=[LOW_CUT, HIGH_CUT], btype="bandpass", fs=RATE, output="sos")
    return sosfilt(sos, audio_data)


def _peak_fft_magnitude(normalised_chunk):
    """Peak FFT magnitude in the siren band for a normalised chunk."""
    fft_scores = np.abs(rfft(normalised_chunk))
    freqs = rfftfreq(len(normalised_chunk), 1 / RATE)
    mask = (freqs >= LOW_CUT) & (freqs <= HIGH_CUT)
    if not np.any(mask):
        return 0.0
    return float(np.max(fft_scores[mask]))


def process_chunk(chunk):
    """
    Mirror the full production pipeline for one chunk.
    Returns (energy, passed_energy_gate, peak_magnitude).
    peak_magnitude is None if the chunk failed the energy gate.
    """
    chunk = chunk - np.mean(chunk)          # DC offset removal
    filtered = _bandpass_filter(chunk)
    energy = float(np.sum(filtered ** 2) / len(filtered))

    if energy < ENERGY_THRESHOLD:
        return energy, False, None

    # Normalise (same as production)
    max_val = np.max(np.abs(filtered))
    if max_val > 0:
        filtered = filtered / max_val

    peak = _peak_fft_magnitude(filtered)
    return energy, True, peak


# ── Audio source helpers ───────────────────────────────────────────────────────

def chunks_from_file(wav_path: str):
    rate, data = wavfile.read(wav_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2_147_483_648.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != RATE:
        print(f"  Warning: file sample rate is {rate} Hz; expected {RATE} Hz.")
    data = np.clip(data * AUDIO_GAIN, -1.0, 1.0)
    return [data[s : s + CHUNK] for s in range(0, len(data) - CHUNK + 1, CHUNK)]


def chunks_from_mic(duration_s: float):
    total = int(RATE * duration_s)
    print(f"  Recording {duration_s:.0f}s from microphone — make noise now …")
    rec = sd.rec(total, samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    data = np.clip(rec[:, 0] * AUDIO_GAIN, -1.0, 1.0)
    print("  Recording complete.")
    return [data[s : s + CHUNK] for s in range(0, len(data) - CHUNK + 1, CHUNK)]


# ── Logging ───────────────────────────────────────────────────────────────────

def append_rows(rows):
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def print_stats(label, magnitudes, skipped):
    total = len(magnitudes) + skipped
    if not magnitudes:
        print(f"\n  Label : {label}  — all {total} chunk(s) failed energy gate, no FFT data.")
        return
    m = np.array(magnitudes)
    print(f"\n  Label : {label}  ({len(m)} chunks passed energy gate, {skipped} skipped)")
    print(f"  Min   : {m.min():.1f}")
    print(f"  p10   : {np.percentile(m, 10):.1f}")
    print(f"  p25   : {np.percentile(m, 25):.1f}")
    print(f"  Median: {np.median(m):.1f}")
    print(f"  p75   : {np.percentile(m, 75):.1f}")
    print(f"  p90   : {np.percentile(m, 90):.1f}")
    print(f"  Max   : {m.max():.1f}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary():
    if not LOG_PATH.exists():
        print("No fft_log.csv found. Run some recordings first.")
        return

    label_mags: dict  = defaultdict(list)
    label_skip: dict  = defaultdict(int)

    with open(LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            label = row["label"]
            if row["passed_energy_gate"] == "True" and row["peak_magnitude"]:
                label_mags[label].append(float(row["peak_magnitude"]))
            else:
                label_skip[label] += 1

    print("\n══════════════════════════════════════════")
    print("  FFT peak magnitude calibration summary")
    print(f"  Log file : {LOG_PATH}")
    print(f"  Current production threshold : 130")
    print("══════════════════════════════════════════")

    for label in sorted(label_mags.keys() | label_skip.keys()):
        print_stats(label, label_mags[label], label_skip[label])

    # Suggest a threshold if siren and non-siren data both exist
    siren_mags = []
    for l in SIREN_LABELS:
        siren_mags.extend(label_mags.get(l, []))

    noise_labels = set(label_mags.keys()) - SIREN_LABELS
    noise_mags = []
    for l in noise_labels:
        noise_mags.extend(label_mags[l])

    if siren_mags and noise_mags:
        siren_min = np.percentile(siren_mags, 10)
        noise_max = np.percentile(noise_mags, 90)
        midpoint  = (siren_min + noise_max) / 2

        print("\n──────────────────────────────────────────")
        print(f"  Siren  p10 (lowest siren magnitude) : {siren_min:.1f}")
        print(f"  Noise  p90 (highest noise magnitude): {noise_max:.1f}")
        if siren_min > noise_max:
            print(f"\n  Clean separation! Suggested threshold: {midpoint:.1f}")
        else:
            print(f"\n  Overlap detected — classes not cleanly separable by FFT magnitude alone.")
            print(f"  Suggested threshold (midpoint): {midpoint:.1f}")
        print(f"  Current threshold (130) is {'above' if 130 > noise_max else 'below or within'} the noise p90.")
        print("──────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate the FFT peak magnitude threshold for siren detection."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", metavar="WAV", help="Path to a .wav file to analyse")
    source.add_argument("--live", action="store_true", help="Record from the microphone")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Recording duration in seconds (--live only, default: 5)")
    parser.add_argument("--label", default="unlabelled",
                        help='Label, e.g. "police_80", "laughter", "talking"')
    parser.add_argument("--summary", action="store_true",
                        help="Print a summary of all logged values and exit")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if not args.file and not args.live:
        parser.print_help()
        return

    print(f"\n  Label            : {args.label}")
    print(f"  Energy threshold : {ENERGY_THRESHOLD} (chunks below this are skipped)")
    print(f"  Chunk size       : {CHUNK} samples ({CHUNK/RATE:.2f}s each)\n")

    chunks = chunks_from_file(args.file) if args.file else chunks_from_mic(args.duration)
    if not chunks:
        print("  No complete chunks obtained — audio too short?")
        return

    rows = []
    magnitudes = []
    skipped = 0
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    for i, chunk in enumerate(chunks):
        energy, passed, peak = process_chunk(chunk)
        row = {
            "label": args.label,
            "chunk_index": i,
            "energy": f"{energy:.8f}",
            "passed_energy_gate": str(passed),
            "peak_magnitude": f"{peak:.2f}" if peak is not None else "",
            "timestamp": timestamp,
        }
        rows.append(row)

        if passed and peak is not None:
            magnitudes.append(peak)
            print(f"  chunk {i:>3}:  energy = {energy:.6f}  →  peak FFT mag = {peak:.1f}")
        else:
            skipped += 1
            print(f"  chunk {i:>3}:  energy = {energy:.6f}  →  below energy gate (skipped)")

    append_rows(rows)
    print_stats(args.label, magnitudes, skipped)
    print(f"\n  Results appended to {LOG_PATH}")
    print("  Run with --summary after collecting siren + noise samples for a threshold suggestion.\n")


if __name__ == "__main__":
    main()
