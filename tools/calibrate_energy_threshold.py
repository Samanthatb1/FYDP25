"""
Calibration script for the bandpass energy threshold.

Logs raw (pre-normalization) energy values from the siren band (1000-5000 Hz)
for audio samples you provide, so you can pick a threshold that separates
siren audio from false positives like laughter or speech.

Usage
-----
  # Record live from the microphone for 5 seconds:
  python tools/calibrate_energy_threshold.py --live --duration 5 --label siren

  # Analyse an existing .wav file:
  python tools/calibrate_energy_threshold.py --file path/to/laugh.wav --label laughter

  # Print a summary comparing all labels logged so far:
  python tools/calibrate_energy_threshold.py --summary

Results are appended to tools/energy_log.csv so you can build up a dataset
across multiple recordings and labels.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from scipy.signal import butter, sosfilt

# ── Audio parameters (must match constants.py) ────────────────────────────────
RATE = 16000       # Hz
CHUNK = 9600       # samples per chunk (~0.6 s), same as production
AUDIO_GAIN = 3.0   # same software gain applied in main.py
LOW_CUT = 1000     # Hz
HIGH_CUT = 5000    # Hz

LOG_PATH = Path(__file__).parent / "energy_log.csv"
LOG_FIELDNAMES = ["label", "chunk_index", "energy", "timestamp"]


# ── Signal processing (mirrors bandpass_filter.py) ────────────────────────────

def bandpass_filter(audio_data):
    sos = butter(N=4, Wn=[LOW_CUT, HIGH_CUT], btype="bandpass", fs=RATE, output="sos")
    return sosfilt(sos, audio_data)


def raw_energy(audio_chunk):
    """Energy of the bandpass-filtered signal WITHOUT any prior normalization."""
    chunk = audio_chunk - np.mean(audio_chunk)          # remove DC offset
    filtered = bandpass_filter(chunk)
    return float(np.sum(filtered ** 2) / len(filtered))


# ── Audio source helpers ───────────────────────────────────────────────────────

def chunks_from_file(wav_path):
    """Yield CHUNK-sized float32 mono chunks from a .wav file."""
    rate, data = wavfile.read(wav_path)

    # Convert to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    # Mix to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample naively if sample rate differs (simple repeat/drop — good enough
    # for threshold calibration; use librosa for production resampling)
    if rate != RATE:
        print(f"  ⚠  File sample rate is {rate} Hz, expected {RATE} Hz. "
              f"Results may be slightly off.")

    # Apply the same software gain used in production
    data = np.clip(data * AUDIO_GAIN, -1.0, 1.0)

    for start in range(0, len(data) - CHUNK + 1, CHUNK):
        yield data[start : start + CHUNK]


def chunks_from_mic(duration_s):
    """Record `duration_s` seconds from the default mic and yield chunks."""
    total_samples = int(RATE * duration_s)
    print(f"  🎙  Recording {duration_s}s from microphone … (speak/play audio now)")
    recording = sd.rec(total_samples, samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    data = recording[:, 0]
    data = np.clip(data * AUDIO_GAIN, -1.0, 1.0)
    print("  Recording complete.")
    for start in range(0, len(data) - CHUNK + 1, CHUNK):
        yield data[start : start + CHUNK]


# ── Logging ───────────────────────────────────────────────────────────────────

def append_rows(rows):
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def print_stats(label, energies):
    if not energies:
        print(f"  (no data for '{label}')")
        return
    e = np.array(energies)
    print(f"\n  Label : {label}  ({len(e)} chunks)")
    print(f"  Min   : {e.min():.6f}")
    print(f"  p10   : {np.percentile(e, 10):.6f}")
    print(f"  p25   : {np.percentile(e, 25):.6f}")
    print(f"  Median: {np.median(e):.6f}")
    print(f"  p75   : {np.percentile(e, 75):.6f}")
    print(f"  p90   : {np.percentile(e, 90):.6f}")
    print(f"  Max   : {e.max():.6f}")


# ── Summary across all logged labels ──────────────────────────────────────────

def print_summary():
    if not LOG_PATH.exists():
        print("No energy_log.csv found. Run some recordings first.")
        return

    from collections import defaultdict
    label_energies = defaultdict(list)

    with open(LOG_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_energies[row["label"]].append(float(row["energy"]))

    print("\n══════════════════════════════════════════")
    print("  Energy calibration summary")
    print(f"  Log file: {LOG_PATH}")
    print("══════════════════════════════════════════")

    for label, energies in sorted(label_energies.items()):
        print_stats(label, energies)

    # Suggest a threshold if both siren and non-siren labels exist
    siren_labels = {l for l in label_energies if "siren" in l.lower()}
    noise_labels  = set(label_energies.keys()) - siren_labels

    if siren_labels and noise_labels:
        siren_min  = min(np.percentile(label_energies[l], 10) for l in siren_labels)
        noise_max  = max(np.percentile(label_energies[l], 90) for l in noise_labels)
        midpoint   = (siren_min + noise_max) / 2

        print("\n──────────────────────────────────────────")
        print(f"  Siren  p10 (lowest siren energy) : {siren_min:.6f}")
        print(f"  Noise  p90 (highest noise energy): {noise_max:.6f}")
        if siren_min > noise_max:
            print(f"\n  ✅  Clean separation! Suggested threshold: {midpoint:.6f}")
        else:
            print(f"\n  ⚠  Overlap detected — classes are not cleanly separable by energy alone.")
            print(f"     Suggested threshold (midpoint): {midpoint:.6f}")
            print(f"     Consider combining with spectral flatness or YAMNet score gating.")
        print("──────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate the bandpass energy threshold for siren detection."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", metavar="WAV", help="Path to a .wav file to analyse")
    source.add_argument("--live", action="store_true", help="Record from the microphone")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Recording duration in seconds (--live only, default: 5)")
    parser.add_argument("--label", default="unlabelled",
                        help='Label for this recording, e.g. "siren", "laughter", "speech"')
    parser.add_argument("--summary", action="store_true",
                        help="Print a summary of all logged values and exit")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if not args.file and not args.live:
        parser.print_help()
        return

    print(f"\nLabel: '{args.label}'")
    print(f"Chunk size: {CHUNK} samples ({CHUNK/RATE:.2f}s each)\n")

    if args.file:
        chunk_iter = chunks_from_file(args.file)
    else:
        chunk_iter = chunks_from_mic(args.duration)

    rows = []
    energies = []
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    for i, chunk in enumerate(chunk_iter):
        e = raw_energy(chunk)
        energies.append(e)
        rows.append({"label": args.label, "chunk_index": i,
                     "energy": f"{e:.8f}", "timestamp": timestamp})
        print(f"  chunk {i:>3}:  energy = {e:.6f}")

    append_rows(rows)
    print_stats(args.label, energies)
    print(f"\n  Results appended to {LOG_PATH}")
    print("  Run with --summary after collecting siren + noise samples for a threshold suggestion.\n")


if __name__ == "__main__":
    main()
