#!/usr/bin/env python3
# pyin_to_lanes.py — turn pYIN notes CSV into 3-lane assignments

import csv, math, numpy as np
from lane_assign_melody import Note, assign_melody_to_lanes, MELODY_PRESETS

def hz_to_midi(hz: float) -> int:
    if hz <= 0: return 60
    return int(round(69 + 12 * math.log2(hz / 440.0)))

def load_pyin_notes_csv(path: str):
    notes = []
    with open(path) as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 3: 
                continue
            t = float(row[0]); d = float(row[1]); hz = float(row[2])
            if d <= 0 or hz <= 0: 
                continue
            pitch = hz_to_midi(hz)
            notes.append(Note(t=t, d=d, pitch=pitch, strength=1.0))
    return notes

def load_beats_txt(path: str):
    beats = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s:
                beats.append(float(s))
    return np.asarray(sorted(beats), float)

# --- usage example ---
pyin_csv   = "input_(Vocals)_htdemucs_vamp_pyin_pyin_notes.csv"      # time,duration,frequencyHz (no header)
beat_times = "drum_beats.csv"      # one time (s) per line from your temp grid
difficulty = "Easy"              # "Easy" | "Medium" | "Hard" | "TwoLane"

notes = load_pyin_notes_csv(pyin_csv)
beats = load_beats_txt(beat_times)

settings = MELODY_PRESETS[difficulty]
assignments = assign_melody_to_lanes(notes, beats, settings, subdiv=4)
# assignments: list of (time_s, lane_idx, midi_pitch, duration_s)

# Dump a simple CSV for your game:
with open("vocal_lanes.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_s", "lane", "pitch", "dur_s"])
    for t, lane, pitch, dur in assignments:
        w.writerow([f"{t:.6f}", lane, pitch, f"{dur:.6f}"])

print(f"Wrote vocal_lanes.csv with {len(assignments)} notes mapped to lanes.")
