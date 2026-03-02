"""
Siren energy study — recreates the spreadsheet table from the FYDP report.

Records or reads audio for each condition (controls + siren types at different
volumes) across N trials and produces a results table with per-trial average
bandpass energy and an overall average per condition.

Study conditions
----------------
Controls  : silence, laughter, talking
Sirens    : police, ambulance, fire_truck  ×  80 % / 20 % speaker volume

Table structure (mirrors original spreadsheet)
-----------------------------------------------
  Category        | Volume | Trial 1 | Trial 2 | … | Trial N | Average Energy
  ----------------+--------+---------+---------+---+---------+---------------
  Silence         | N/A    |  …
  Laughter        | N/A    |  …
  Talking         | N/A    |  …
  Police Siren    | 80 %   |  …
  Police Siren    | 20 %   |  …
  Ambulance Siren | 80 %   |  …
  Ambulance Siren | 20 %   |  …
  Fire Truck Siren| 80 %   |  …
  Fire Truck Siren| 20 %   |  …

Usage
-----
  # Record a single trial live (5 s):
  python tools/siren_energy_study.py --live --label police_80 --trial 1

  # Analyse an existing .wav file:
  python tools/siren_energy_study.py --file path/to/clip.wav --label laughter --trial 3

  # Print the full results table:
  python tools/siren_energy_study.py --table

  # Export table to CSV:
  python tools/siren_energy_study.py --table --export results/siren_study.csv

Results are accumulated in tools/study_results.csv so you can collect data
across multiple sessions.

Label conventions
-----------------
  Controls : silence | laughter | talking
  Sirens   : police_80 | police_20
             ambulance_80 | ambulance_20
             fire_truck_80 | fire_truck_20
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from scipy.signal import butter, sosfilt

# ── Audio parameters (must match constants.py / production pipeline) ──────────
RATE       = 16000   # Hz
CHUNK      = 9600    # samples per chunk (~0.6 s)
AUDIO_GAIN = 3.0     # software gain applied in main.py
LOW_CUT    = 1000    # Hz  (siren bandpass low edge)
HIGH_CUT   = 5000    # Hz  (siren bandpass high edge)

RESULTS_PATH = Path(__file__).parent / "study_results.csv"
RESULTS_FIELDNAMES = ["label", "trial", "avg_energy", "num_chunks", "timestamp"]

# ── Display metadata for each label ───────────────────────────────────────────
LABEL_META = {
    # label            : (display name,              volume display)
    "silence"          : ("Silence (Control)",        "N/A"),
    "laughter"         : ("Laughter (Control)",       "N/A"),
    "talking"          : ("Talking (Control)",        "N/A"),
    "police_80"        : ("Police Siren",             "80%"),
    "police_20"        : ("Police Siren",             "20%"),
    "ambulance_80"     : ("Ambulance Emergency Siren","80%"),
    "ambulance_20"     : ("Ambulance Emergency Siren","20%"),
    "fire_truck_80"    : ("Fire Truck Siren",         "80%"),
    "fire_truck_20"    : ("Fire Truck Siren",         "20%"),
}

# Canonical display order for the table
TABLE_ORDER = [
    "silence",
    "laughter",
    "talking",
    "police_80",
    "police_20",
    "ambulance_80",
    "ambulance_20",
    "fire_truck_80",
    "fire_truck_20",
]

SIREN_LABELS = {"police_80", "police_20", "ambulance_80", "ambulance_20",
                "fire_truck_80", "fire_truck_20"}

# ── Signal processing (mirrors bandpass_filter.py) ────────────────────────────

def _bandpass_filter(audio_data):
    sos = butter(N=4, Wn=[LOW_CUT, HIGH_CUT], btype="bandpass", fs=RATE, output="sos")
    return sosfilt(sos, audio_data)


def _raw_energy(chunk):
    """Mean-squared bandpass energy for one chunk, without any prior normalisation."""
    chunk = chunk - np.mean(chunk)        # remove DC offset
    filtered = _bandpass_filter(chunk)
    return float(np.sum(filtered ** 2) / len(filtered))


def trial_avg_energy(chunks):
    """Return the average energy across all chunks for a single trial."""
    energies = [_raw_energy(c) for c in chunks]
    return float(np.mean(energies)), len(energies)


# ── Audio source helpers ───────────────────────────────────────────────────────

def chunks_from_file(wav_path: str):
    """Yield CHUNK-sized float32 mono chunks from a .wav file."""
    rate, data = wavfile.read(wav_path)

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2_147_483_648.0
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo → mono

    if rate != RATE:
        print(f"  Warning: file sample rate is {rate} Hz; expected {RATE} Hz. "
              "Results may be slightly off.")

    data = np.clip(data * AUDIO_GAIN, -1.0, 1.0)

    chunks = []
    for start in range(0, len(data) - CHUNK + 1, CHUNK):
        chunks.append(data[start : start + CHUNK])
    return chunks


def chunks_from_mic(duration_s: float):
    """Record duration_s seconds from the default microphone and return chunks."""
    total_samples = int(RATE * duration_s)
    print(f"  Recording {duration_s:.0f}s from microphone — make noise now …")
    recording = sd.rec(total_samples, samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    data = recording[:, 0]
    data = np.clip(data * AUDIO_GAIN, -1.0, 1.0)
    print("  Recording complete.")
    chunks = []
    for start in range(0, len(data) - CHUNK + 1, CHUNK):
        chunks.append(data[start : start + CHUNK])
    return chunks


# ── Persistence ───────────────────────────────────────────────────────────────

def save_trial(label: str, trial: int, avg_energy: float, num_chunks: int):
    write_header = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "label":      label,
            "trial":      trial,
            "avg_energy": f"{avg_energy:.8f}",
            "num_chunks": num_chunks,
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        })


def load_results() -> dict:
    """Return {label: {trial: avg_energy}} from study_results.csv."""
    data: dict = defaultdict(dict)
    if not RESULTS_PATH.exists():
        return data
    with open(RESULTS_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            trial = int(row["trial"])
            energy = float(row["avg_energy"])
            # Keep the most recent entry if a trial was re-recorded
            data[label][trial] = energy
    return data


# ── Table display ─────────────────────────────────────────────────────────────

def _all_trials(results: dict) -> list[int]:
    """Return a sorted list of all trial numbers seen across any label."""
    trials = set()
    for trial_map in results.values():
        trials.update(trial_map.keys())
    return sorted(trials) if trials else list(range(1, 6))


def print_table(results: dict):
    trials = _all_trials(results)
    num_t  = len(trials)

    # Column widths
    col_name   = 26
    col_vol    =  6
    col_trial  = 12
    col_avg    = 14

    header = (
        f"{'Category':<{col_name}} {'Vol':>{col_vol}}"
        + "".join(f"  {'Trial '+str(t):>{col_trial}}" for t in trials)
        + f"  {'Avg Energy':>{col_avg}}"
    )
    sep = "─" * len(header)

    print()
    print("═" * len(header))
    print("  Siren Energy Study — Results")
    print(f"  Data file: {RESULTS_PATH}")
    print("═" * len(header))
    print(header)
    print(sep)

    siren_avgs = []

    for label in TABLE_ORDER:
        meta = LABEL_META.get(label, (label, "N/A"))
        display_name, volume = meta
        trial_map = results.get(label, {})

        trial_cols = ""
        trial_vals = []
        for t in trials:
            val = trial_map.get(t)
            if val is not None:
                trial_cols += f"  {val:>{col_trial}.6f}"
                trial_vals.append(val)
            else:
                trial_cols += f"  {'—':>{col_trial}}"

        if trial_vals:
            avg = np.mean(trial_vals)
            avg_str = f"{avg:.6f}"
            if label in SIREN_LABELS:
                siren_avgs.append(avg)
        else:
            avg_str = "—"

        print(f"{display_name:<{col_name}} {volume:>{col_vol}}{trial_cols}  {avg_str:>{col_avg}}")

    print(sep)

    if siren_avgs:
        overall = np.mean(siren_avgs)
        print(f"\n  Average energy of siren conditions: {overall:.8f}")

    print()


def export_table(results: dict, export_path: str):
    """Write the study table to a CSV file."""
    trials = _all_trials(results)
    out = Path(export_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="") as f:
        fieldnames = (["Category", "Volume"]
                      + [f"Trial {t}" for t in trials]
                      + ["Average Energy"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        siren_avgs = []
        for label in TABLE_ORDER:
            meta = LABEL_META.get(label, (label, "N/A"))
            display_name, volume = meta
            trial_map = results.get(label, {})

            row = {"Category": display_name, "Volume": volume}
            trial_vals = []
            for t in trials:
                val = trial_map.get(t)
                row[f"Trial {t}"] = f"{val:.6f}" if val is not None else ""
                if val is not None:
                    trial_vals.append(val)

            if trial_vals:
                avg = float(np.mean(trial_vals))
                row["Average Energy"] = f"{avg:.6f}"
                if label in SIREN_LABELS:
                    siren_avgs.append(avg)
            else:
                row["Average Energy"] = ""

            writer.writerow(row)

        if siren_avgs:
            writer.writerow({})
            writer.writerow({
                "Category": "Average Energy of Siren",
                "Average Energy": f"{np.mean(siren_avgs):.8f}",
            })

    print(f"  Table exported to {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run or inspect the siren energy study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Label conventions:
  Controls : silence | laughter | talking
  Sirens   : police_80 | police_20
             ambulance_80 | ambulance_20
             fire_truck_80 | fire_truck_20
        """,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", metavar="WAV",
                        help="Path to a .wav file to analyse for one trial")
    source.add_argument("--live", action="store_true",
                        help="Record live from the microphone for one trial")
    source.add_argument("--table", action="store_true",
                        help="Print the results table and exit")

    parser.add_argument("--label", default="unlabelled",
                        help="Condition label (see label conventions above)")
    parser.add_argument("--trial", type=int, default=1,
                        help="Trial number (default: 1)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Live recording duration in seconds (default: 5)")
    parser.add_argument("--export", metavar="CSV",
                        help="Export table to this CSV path (use with --table)")
    parser.add_argument("--list-labels", action="store_true",
                        help="List all known label names and exit")

    args = parser.parse_args()

    if args.list_labels:
        print("\nKnown labels and their display names:")
        for lbl, (name, vol) in LABEL_META.items():
            print(f"  {lbl:<20} → {name}  (volume: {vol})")
        print()
        return

    if args.table:
        results = load_results()
        if not results:
            print("No results yet. Run some trials first.")
            return
        print_table(results)
        if args.export:
            export_table(results, args.export)
        return

    if not args.file and not args.live:
        parser.print_help()
        return

    # Validate label
    if args.label not in LABEL_META:
        known = ", ".join(LABEL_META.keys())
        print(f"  Warning: '{args.label}' is not a recognised label.")
        print(f"  Known labels: {known}")
        print("  Proceeding anyway — data will still be saved.\n")

    print(f"\n  Label : {args.label}")
    print(f"  Trial : {args.trial}")
    print(f"  Chunk : {CHUNK} samples ({CHUNK/RATE:.2f} s each)\n")

    if args.file:
        chunks = chunks_from_file(args.file)
    else:
        chunks = chunks_from_mic(args.duration)

    if not chunks:
        print("  No complete chunks obtained — audio too short?")
        sys.exit(1)

    avg_energy, num_chunks = trial_avg_energy(chunks)

    print(f"\n  Chunks processed : {num_chunks}")
    print(f"  Average energy   : {avg_energy:.8f}")

    save_trial(args.label, args.trial, avg_energy, num_chunks)
    print(f"  Saved to         : {RESULTS_PATH}")
    print("\n  Run --table after collecting all trials to see the full results table.\n")


if __name__ == "__main__":
    main()
