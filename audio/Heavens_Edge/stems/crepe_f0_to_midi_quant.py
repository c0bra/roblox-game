#!/usr/bin/env python3
"""
crepe_f0_to_midi_quant.py — Convert CREPE f0 CSV to MIDI and quantize to a beat grid.

Inputs:
  1) CREPE f0 CSV (with header): time,frequency,confidence
  2) Beat-times file (one float seconds per line)
  3) Output MIDI path

Usage:
  python crepe_f0_to_midi_quant.py crepe_f0.csv beat_times.txt output.mid
"""
import sys, csv, math
import pretty_midi as pm
import numpy as np

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

def segment_notes(times, frequencies, confidences, min_conf=0.3, min_duration=0.05):
    """
    Segment continuous voiced regions into notes.
    """
    if len(times) == 0:
        return []
    
    notes = []
    start_idx = None
    frame_duration = 0.01  # Default 10ms frame spacing
    
    # Calculate frame duration from time differences
    if len(times) > 1:
        frame_duration = times[1] - times[0]
    
    for i in range(len(times)):
        # Start of a voiced segment
        if start_idx is None and confidences[i] >= min_conf:
            start_idx = i
        # End of a voiced segment
        elif start_idx is not None and confidences[i] < min_conf:
            end_idx = i
            # Check minimum duration
            duration = times[end_idx-1] - times[start_idx]
            if duration >= min_duration:
                # Calculate average frequency for the segment
                avg_freq = np.mean(frequencies[start_idx:end_idx])
                pitch = hz_to_midi(avg_freq)
                notes.append((times[start_idx], times[end_idx-1] + frame_duration, pitch))
            start_idx = None
    
    # Handle case where file ends with a voiced segment
    if start_idx is not None:
        duration = times[-1] - times[start_idx]
        if duration >= min_duration:
            avg_freq = np.mean(frequencies[start_idx:])
            pitch = hz_to_midi(avg_freq)
            notes.append((times[start_idx], times[-1] + frame_duration, pitch))
    
    return notes

def main():
    if len(sys.argv) != 4:
        print("Usage: python crepe_f0_to_midi_quant.py crepe_f0.csv beat_times.txt output.mid")
        sys.exit(1)

    csv_in, beats_in, midi_out = sys.argv[1], sys.argv[2], sys.argv[3]

    beats = load_beats(beats_in)

    # Load CREPE f0 data
    times = []
    frequencies = []
    confidences = []
    
    with open(csv_in) as f:
        r = csv.reader(f)
        header = next(r)  # Skip header
        for row in r:
            if not row or len(row) < 3:
                continue
            try:
                t = float(row[0].strip())
                f_hz = float(row[1].strip())
                conf = float(row[2].strip())
                times.append(t)
                frequencies.append(f_hz)
                confidences.append(conf)
            except ValueError:
                continue

    # Convert to notes
    raw_notes = segment_notes(times, frequencies, confidences, min_conf=0.2, min_duration=0.03)
    
    if not raw_notes:
        print("Warning: No notes found in the input data")
        # Create empty MIDI file
        midi = pm.PrettyMIDI()
        inst = pm.Instrument(program=0, name="CREPE Quantized Notes")
        midi.instruments.append(inst)
        midi.write(midi_out)
        print(f"Wrote {midi_out}")
        return

    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=6, name="CREPE Quantized Notes")  # Program 33 = Electric Bass (finger)

    for start_time, end_time, pitch in raw_notes:
        start_raw = max(0.0, start_time)
        end_raw = max(start_raw + 0.01, end_time)
        
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
        
        # Only add note if it has positive duration
        if qe > qs:
            inst.notes.append(pm.Note(velocity=100, pitch=pitch, start=float(qs), end=float(qe)))

    midi.instruments.append(inst)
    midi.write(midi_out)
    print(f"Wrote {len(inst.notes)} notes to {midi_out}")

if __name__ == "__main__":
    main()