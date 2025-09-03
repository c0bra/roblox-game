#!/usr/bin/env python3
"""
pyin_bass_shift_to_midi.py
---------------------------------
Pitch-shift a bass stem UP (normalize F0), run pYIN (Vamp) to get notes,
then shift those notes back DOWN by the same semitones and write MIDI.

Why: pYIN struggles on very-low fundamentals. Shifting +12 or +24 semitones
makes the fundamental easier to detect; we preserve durations, and then
transpose MIDI back down.

Requirements:
  - sonic-annotator (CLI) with the pYIN Vamp plugin installed
  - Python: librosa, soundfile, pretty_midi

Usage:
  python pyin_bass_shift_to_midi.py input.wav output.mid --semitones 24
"""

import argparse, os, sys, math, tempfile, subprocess
from typing import List, Tuple

import librosa
import soundfile as sf
import pretty_midi as pm


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(max(hz, 1e-9) / 440.0)


def midi_to_int_safe(x: float) -> int:
    return int(round(max(0, min(127, x))))


def run_sonic_annotator_pyin(wav_path: str) -> List[Tuple[float, float, float]]:
    """Run sonic-annotator pYIN plugin, return rows as (time, duration, frequencyHz)."""
    cmd = [
        "sonic-annotator",
        "-d", "vamp:pyin:pyin:notes",
        "-w", "csv",
        "--csv-stdout",
        wav_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        print(e.output, file=sys.stderr)
        raise SystemExit("sonic-annotator/pYIN failed; is the plugin installed?")

    rows: List[Tuple[float, float, float]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            t = float(parts[0])
            d = float(parts[1])
            hz = float(parts[2])
            rows.append((t, d, hz))
        except ValueError:
            continue
    return rows


def main():
    ap = argparse.ArgumentParser(description="Shift bass up, run pYIN, transpose MIDI back down")
    ap.add_argument("input", help="Bass stem (wav/flac/mp3)")
    ap.add_argument("output", help="Output MIDI path")
    ap.add_argument("--semitones", type=int, default=24, help="Pitch shift UP before analysis (12 or 24 typical)")
    ap.add_argument("--program", type=int, default=32, help="MIDI program for Electric Bass (default=32)")
    ap.add_argument("--velocity", type=int, default=100, help="Fixed MIDI velocity")
    ap.add_argument("--print-ffmpeg", action="store_true", help="Print ffmpeg commands for pitch-shift and exit")
    args = ap.parse_args()

    if args.print_ffmpeg:
        print("# +12 semitones (keep duration):")
        print('ffmpeg -i in.wav -af "asetrate=sr*2,aresample=sr,atempo=0.5" out_+12.wav')
        print("# +24 semitones (keep duration):")
        print('ffmpeg -i in.wav -af "asetrate=sr*4,aresample=sr,atempo=0.5,atempo=0.5" out_+24.wav')
        sys.exit(0)

    # Load and pitch-shift UP
    y, sr = librosa.load(args.input, mono=True, sr=None)
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=args.semitones)

    with tempfile.TemporaryDirectory() as td:
        tmp_wav = os.path.join(td, "shifted.wav")
        sf.write(tmp_wav, y_shift, sr)

        # Run pYIN on shifted audio
        rows = run_sonic_annotator_pyin(tmp_wav)

    # Build MIDI, transposing notes DOWN
    midi = pm.PrettyMIDI(initial_tempo=120.0)
    inst = pm.Instrument(program=args.program, name="Bass (pYIN shifted back)")
    transpose = -args.semitones

    for (t, d, hz) in rows:
        if d <= 0 or hz <= 0:
            continue
        midi_pitch = hz_to_midi(hz) + transpose
        pitch_i = midi_to_int_safe(midi_pitch)
        start = max(0.0, t)
        end = max(start + 0.02, t + d)
        inst.notes.append(pm.Note(
            velocity=max(1, min(127, args.velocity)),
            pitch=pitch_i,
            start=float(start),
            end=float(end),
        ))

    midi.instruments.append(inst)
    midi.write(args.output)
    print(f"Wrote {args.output} with {len(inst.notes)} notes "
          f"(analyzed at +{args.semitones} st, transposed back).")


if __name__ == "__main__":
    main()