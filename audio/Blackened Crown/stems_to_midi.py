#!/usr/bin/env python3
import argparse, os, sys, math
import numpy as np
import librosa
import pretty_midi

GM = {"kick":36, "snare":38, "hats":42}

def detect_onsets(path, sr=None, backtrack=False, pre_max=20, post_max=20,
                  pre_avg=100, post_avg=100, delta=0.2, wait=10, hop_length=512):
    y, sr = librosa.load(path, sr=sr, mono=True)
    # onset strength (spectral flux)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr,
                                        backtrack=backtrack, units='time',
                                        pre_max=pre_max, post_max=post_max,
                                        pre_avg=pre_avg, post_avg=post_avg,
                                        delta=delta, wait=wait, hop_length=hop_length)
    return np.asarray(onsets, dtype=float)

def quantize_times(times, bpm, division):
    """Snap times (sec) to 1/division notes at BPM."""
    if division <= 0: return times
    sec_per_beat = 60.0 / bpm
    grid = sec_per_beat / division
    return np.round(times / grid) * grid

def merge_close(times, min_sep_s):
    if len(times)==0: return times
    times = np.sort(times)
    keep = [times[0]]
    for t in times[1:]:
        if t - keep[-1] >= min_sep_s:
            keep.append(t)
    return np.array(keep)

def add_hits(pm, times, midi_note, velocity, hold_ms, bpm, ppq):
    sec_per_beat = 60.0 / bpm
    ticks_per_sec = ppq / sec_per_beat
    dur_ticks = max(1, int((hold_ms/1000.0)*ticks_per_sec))
    inst = pm.instruments[0]
    for t in times:
        start = max(0.0, float(t))
        end   = start + (dur_ticks / ticks_per_sec)
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=midi_note, start=start, end=end))

def main():
    ap = argparse.ArgumentParser(description="Convert drum stems to a GM-drum MIDI")
    ap.add_argument("--kick",  help="kick stem wav",  default=None)
    ap.add_argument("--snare", help="snare stem wav", default=None)
    ap.add_argument("--hats",  help="hats stem wav",  default=None)
    ap.add_argument("-o", "--out", required=True, help="output .mid")
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--ppq", type=int, default=480)
    ap.add_argument("--hold-ms", type=int, default=60, help="note length")
    ap.add_argument("--vel", type=int, default=100, help="MIDI velocity 1-127")
    ap.add_argument("--min-sep-ms", type=float, default=50.0, help="merge hits closer than this")
    ap.add_argument("--quant-div", type=int, default=0, help="0=off, else snap to 1/div notes")
    ap.add_argument("--delta", type=float, default=0.2, help="onset detection sensitivity (lower = more hits)")
    args = ap.parse_args()

    pm = pretty_midi.PrettyMIDI(resolution=args.ppq)
    drum = pretty_midi.Instrument(program=0, is_drum=True)  # channel 10 in MIDI players
    pm.instruments.append(drum)

    stems = [
        (args.kick,  GM["kick"]),
        (args.snare, GM["snare"]),
        (args.hats,  GM["hats"]),
    ]

    for path, note in stems:
        if not path: continue
        if not os.path.exists(path):
            print(f"warn: missing {path}, skipping", file=sys.stderr); continue
        times = detect_onsets(path, delta=args.delta)
        times = merge_close(times, args.min_sep_ms/1000.0)
        if args.quant_div>0:
            times = quantize_times(times, args.bpm, args.quant_div)
            times = merge_close(times, args.min_sep_ms/1000.0)
        add_hits(pm, times, note, args.vel, args.hold_ms, args.bpm, args.ppq)

    pm.write(args.out)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
