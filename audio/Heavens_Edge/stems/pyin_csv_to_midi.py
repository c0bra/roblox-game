#!/usr/bin/env python3
"""
pyin_csv_to_midi.py — Convert pYIN 'notes' CSV to MIDI.

Expected CSV columns (no header):
  time_seconds, duration_seconds, frequency_hz

Example row:
  7.836734694,0.348299319,292.796

Usage:
  python pyin_csv_to_midi.py input.csv output.mid
"""
import sys, csv, math
import pretty_midi as pm

def hz_to_midi(hz: float) -> int:
    if hz <= 0:  # fallback to middle C if weird
        return 60
    return int(round(69 + 12 * math.log2(hz / 440.0)))

def main():
    if len(sys.argv) < 3:
        print("Usage: python pyin_csv_to_midi.py input.csv output.mid")
        sys.exit(1)

    csv_in, midi_out = sys.argv[1], sys.argv[2]

    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0, name="pYIN Notes")  # Acoustic Grand Piano

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
                # skip malformed rows
                continue
            start = max(0.0, t)
            end = max(start + 0.01, t + d)
            pitch = hz_to_midi(hz)
            inst.notes.append(pm.Note(velocity=100, pitch=pitch, start=start, end=end))

    midi.instruments.append(inst)
    midi.write(midi_out)
    print(f"Wrote {midi_out} with {len(inst.notes)} notes.")

if __name__ == "__main__":
    main()
