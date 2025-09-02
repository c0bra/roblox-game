#!/usr/bin/env python3
"""
beats_dump.py — Inspect beat & downbeat times with madmom

Usage examples:
  python beats_dump.py input.wav
  python beats_dump.py "Heaven's Edge (Vocals).mp3" --meter 4 --fps 100
  python beats_dump.py input.mp3 --beats-only
  python beats_dump.py input.mp3 --meter 3 --click-wav click.wav

Outputs:
  - Prints a quick summary to stdout.
  - Writes CSV: <input>.beats.csv  (time_s, is_downbeat)
  - Optional WAV click track via --click-wav (simple ticks on beats).
"""

import argparse
import csv
import os
import sys
import numpy as np

from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor  # type: ignore
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor  # type: ignore

def dump_csv(path, beat_times, is_downbeat):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "is_downbeat"])
        for t, d in zip(beat_times, is_downbeat):
            w.writerow([f"{t:.6f}", int(bool(d))])
    return path

def synth_downbeats(beat_times, meter=4):
    is_db = np.zeros_like(beat_times, dtype=bool)
    is_db[::meter] = True
    return is_db

def make_click_wav(path, beat_times, sr=44100):
    """Generate a simple click track WAV with short impulses at each beat.
       Uses only stdlib wave; no external deps.
    """
    import wave, struct, math

    dur = max(beat_times[-1] + 1.0, 1.0)  # seconds
    nframes = int(dur * sr)
    data = [0.0] * nframes

    # Click: 2 ms sine burst at 2 kHz
    click_len = int(0.002 * sr)
    freq = 2000.0
    for t in beat_times:
        idx = int(t * sr)
        for i in range(click_len):
            j = idx + i
            if 0 <= j < nframes:
                # simple sine, decay
                val = 0.9 * math.sin(2*math.pi*freq*(i/sr)) * (1 - i/click_len)
                data[j] += val

    # Normalize/clamp
    mx = max(1e-9, max(abs(x) for x in data))
    scale = 0.95 / mx
    data_i16 = [int(max(-1.0, min(1.0, x*scale)) * 32767) for x in data]

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        frames = b''.join(struct.pack('<h', x) for x in data_i16)
        wf.writeframes(frames)
    return path

def main():
    ap = argparse.ArgumentParser(description="Dump beat/downbeat times with madmom")
    ap.add_argument("input", help="Audio file (wav/mp3/flac/m4a)")
    ap.add_argument("--meter", choices=["auto", "3", "4"], default="auto",
                    help="Assumed meter for downbeat tracking; try 4 for pop/rock")
    ap.add_argument("--fps", type=int, default=100, help="DBN processing FPS (timing resolution)")
    ap.add_argument("--beats-only", action="store_true", help="Use plain beat tracker (no downbeats)")
    ap.add_argument("--click-wav", help="Optional path to write a WAV click track aligned to beats")
    args = ap.parse_args()

    inp = args.input
    if not os.path.isfile(inp):
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(2)

    base, _ = os.path.splitext(inp)
    csv_path = base + ".beats.csv"

    beat_times = None
    is_downbeat = None

    if not args.beats_only:
        # Try downbeat tracking with chosen meter(s)
        meters_to_try = []
        if args.meter in ("3", "4"):
            meters_to_try = [int(args.meter)]
        else:
            meters_to_try = [4, 3]
        success = False
        for m in meters_to_try:
            try:
                rnn = RNNDownBeatProcessor()
                act = rnn(inp)
                dbn = DBNDownBeatTrackingProcessor(beats_per_bar=m, fps=args.fps)
                beats = dbn(act)
                beat_times = beats[:, 0].astype(float)
                is_downbeat = (beats[:, 1].astype(int) == 1)
                success = len(beat_times) >= 2
                if success:
                    print(f"[madmom] Downbeat tracker OK with meter={m} (beats={len(beat_times)})")
                    break
            except Exception as e:
                # fall through to next meter or beat-only
                pass
        if not success:
            print("[madmom] Downbeat tracking failed; falling back to beat-only.")
            args.beats_only = True

    if args.beats_only:
        rnn_b = RNNBeatProcessor()
        act_b = rnn_b(inp)
        dbn_b = DBNBeatTrackingProcessor(fps=args.fps)
        beat_times = np.asarray(dbn_b(act_b)).astype(float)
        # synthesize downbeats each 4 beats by default, or chosen meter
        meter = int(args.meter) if args.meter in ("3", "4") else 4
        is_downbeat = synth_downbeats(beat_times, meter=meter)
        print(f"[madmom] Beat tracker OK (beats={len(beat_times)}), synthesized downbeats every {meter} beats.")

    # Save CSV
    dump_csv(csv_path, beat_times, is_downbeat)
    print(f"Wrote CSV: {csv_path}")

    # Optional click track
    if args.click_wav:
        out_wav = args.click_wav
        make_click_wav(out_wav, beat_times)
        print(f"Wrote click WAV: {out_wav}")

    # Quick preview to stdout (first 12 beats)
    print("First few beats:")
    for t, d in list(zip(beat_times, is_downbeat))[:12]:
        print(f"  {t:8.3f}s  {'DB' if d else '  '}")

if __name__ == "__main__":
    main()
