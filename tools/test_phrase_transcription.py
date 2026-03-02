"""
Phrase transcription accuracy tester.

Say each phrase aloud when prompted. The script records it, runs it through
the same Vosk model used in production, and shows you exactly what was
transcribed. Repeat each phrase multiple times to get a success rate.

Usage
-----
  python tools/test_phrase_transcription.py
"""

import json
import sys
import os
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from constants import RATE

# ── Config ────────────────────────────────────────────────────────────────────

VOSK_PATH = "models/vosk-model-small-en-us-0.15"
RECORD_SECONDS = 3      # how long to record each attempt
ATTEMPTS_PER_PHRASE = 5 # how many times to say each phrase
AUDIO_GAIN = 3.0        # must match main.py

PHRASES_TO_TEST = [
    "turn up the heat",
    "make it cooler",
    "pull over",
    "thank you",
    "open the trunk",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def record(duration_s):
    samples = int(RATE * duration_s)
    audio = sd.rec(samples, samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    data = audio[:, 0]
    return np.clip(data * AUDIO_GAIN, -1.0, 1.0)


def transcribe(audio, recognizer):
    int16 = (audio * 32767).astype(np.int16)
    recognizer.AcceptWaveform(int16.tobytes())
    result = json.loads(recognizer.FinalResult())
    return result.get("text", "").strip().lower()


def phrase_matched(transcription, target_phrase):
    """True if the target phrase appears verbatim in the transcription."""
    return target_phrase.lower() in transcription


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nLoading Vosk model from {VOSK_PATH} ...")
    model = Model(VOSK_PATH)
    print("Model loaded.\n")

    all_results = {}

    for phrase in PHRASES_TO_TEST:
        print(f"{'─' * 50}")
        print(f"  Phrase: \"{phrase}\"")
        print(f"  You will be asked to say it {ATTEMPTS_PER_PHRASE} times.")
        input("  Press Enter when ready...\n")

        successes = 0
        transcriptions = []

        for attempt in range(1, ATTEMPTS_PER_PHRASE + 1):
            # Fresh recognizer each attempt so previous audio doesn't bleed in
            recognizer = KaldiRecognizer(model, RATE)

            print(f"  [{attempt}/{ATTEMPTS_PER_PHRASE}] Say: \"{phrase}\"  (recording {RECORD_SECONDS}s...)")
            audio = record(RECORD_SECONDS)
            transcription = transcribe(audio, recognizer)

            matched = phrase_matched(transcription, phrase)
            status = "✅" if matched else "❌"
            print(f"           Got: \"{transcription}\"  {status}\n")

            transcriptions.append(transcription)
            if matched:
                successes += 1

        rate = successes / ATTEMPTS_PER_PHRASE * 100
        print(f"  Result: {successes}/{ATTEMPTS_PER_PHRASE} exact matches  ({rate:.0f}% success rate)\n")
        all_results[phrase] = {
            "successes": successes,
            "attempts": ATTEMPTS_PER_PHRASE,
            "rate": rate,
            "transcriptions": transcriptions,
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 50}")
    print("  SUMMARY")
    print(f"{'═' * 50}")
    print(f"  {'Phrase':<25} {'Success rate'}")
    print(f"  {'──────':<25} {'────────────'}")
    for phrase, r in all_results.items():
        bar = "█" * r["successes"] + "░" * (r["attempts"] - r["successes"])
        print(f"  {phrase:<25} {r['rate']:>3.0f}%  {bar}")
    print()

    # Flag any phrases that consistently came out differently —
    # those alternatives might be worth adding as fuzzy-match variants
    print("  Most common transcriptions per phrase:")
    for phrase, r in all_results.items():
        from collections import Counter
        counts = Counter(r["transcriptions"])
        top = counts.most_common(3)
        print(f"\n  \"{phrase}\"")
        for transcription, count in top:
            tag = " ← exact" if phrase_matched(transcription, phrase) else ""
            print(f"    {count}x  \"{transcription}\"{tag}")
    print()


if __name__ == "__main__":
    main()
