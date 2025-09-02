#!/usr/bin/env python3
"""
pyin_csv_to_midi_quant.py — Convert pYIN 'notes' CSV to MIDI and quantize to a beat grid.

Inputs:
  1) pYIN CSV (no header): time_seconds, duration_seconds, frequency_hz
  2) Beat-times file (one float seconds per line)
  3) Output MIDI path

Usage:
  python pyin_csv_to_midi_quant.py pyin_notes.csv beat_times.txt output.mid
"""
import sys, csv, math
import pretty_midi as pm

def hz_to_midi(hz: float) -> int:
    if hz <= 0:
        return 60
    return int(round(69 + 12 * math.log2(hz / 440.0)))

def load_beats(path):
    beats = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                beats.append(float(s))
            except ValueError:
                # ignore non-numeric lines
                pass
    if not beats:
        raise SystemExit(f"No beats found in {path}")
    beats.sort()
    return beats

def nearest(beats, t):
    # binary search for nearest beat to time t
    import bisect
    i = bisect.bisect_left(beats, t)
    if i == 0:
        return beats[0]
    if i == len(beats):
        return beats[-1]
    before = beats[i-1]
    after = beats[i]
    return before if (t - before) <= (after - t) else after

def main():
    if len(sys.argv) != 4:
        print("Usage: python pyin_csv_to_midi_quant.py pyin_notes.csv beat_times.txt output.mid")
        sys.exit(1)

    csv_in, beats_in, midi_out = sys.argv[1], sys.argv[2], sys.argv[3]

    beats = load_beats(beats_in)

    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0, name="pYIN Quantized Notes")

    with open(csv_in) as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 3:
                continue
            try:
                t = float(row[0].strip())
                d = float(row[1].strip())
                hz = float(row[2].strip())
            except ValueError:
                continue

            start_raw = max(0.0, t)
            end_raw = max(start_raw + 0.01, t + d)
            pitch = hz_to_midi(hz)

            # Quantize start/end to nearest beats
            qs = nearest(beats, start_raw)
            qe = nearest(beats, end_raw)
            # Ensure end is after start (at least 20 ms)
            if qe <= qs:
                # move end to the next beat if possible; else add small epsilon
                import bisect
                idx = bisect.bisect_right(beats, qs)
                if idx < len(beats):
                    qe = beats[idx]
                else:
                    qe = qs + 0.02

            inst.notes.append(pm.Note(velocity=100, pitch=pitch, start=float(qs), end=float(qe)))

    midi.instruments.append(inst)
    midi.write(midi_out)
    print(f"Wrote {midi_out}")

if __name__ == "__main__":
    main()
