#!/usr/bin/env python3
"""
onsets_to_midi.py — Detect onsets with madmom and write a MIDI file.

Usage (basic):
  python onsets_to_midi.py "vocals.wav" --output vocals_onsets.mid

Usage (tuned for vocals):
  python onsets_to_midi.py "vocals.wav" --model rnn --fps 200 --threshold 0.15 \
      --smooth 0.05 --combine 0.03 --pre-avg 0.10 --post-avg 0.10 \
      --pitch 60 --velocity 96 --dur-ms 120 --output vocals_onsets.mid

Quantize to beat grid (optional):
  python onsets_to_midi.py "vocals.wav" --quantize --meter 4 --subdiv 6 \
      --max-snap-ms 120 --pitch 62 --output vocals_onsets_quant.mid
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple, List

# --- NumPy 1.24+ compatibility shim (aliases removed upstream) ---
import numpy as _np
for _name, _alias in {
    "float": float,
    "int": int,
    "complex": complex,
    "bool": bool,
    "object": object,
}.items():
    if not hasattr(_np, _name):
        try:
            setattr(_np, _name, _alias)
        except Exception:
            pass
# --- End NumPy shim ---

# --- collections ABC shim for Python 3.10+ (madmom compatibility) ---
import collections as _collections
try:
    import collections.abc as _cabc
    for _name in ("MutableMapping","MutableSet","MutableSequence","Mapping","Set","Sequence"):
        if not hasattr(_collections, _name) and hasattr(_cabc, _name):
            setattr(_collections, _name, getattr(_cabc, _name))
except Exception:
    pass
# --- End shim ---

# madmom imports
from madmom.features.onsets import (  # type: ignore
    RNNOnsetProcessor,
    CNNOnsetProcessor,
    OnsetPeakPickingProcessor
)
from madmom.features.downbeats import (  # type: ignore
    RNNDownBeatProcessor,
    DBNDownBeatTrackingProcessor
)
from madmom.features.beats import (  # type: ignore
    RNNBeatProcessor as _RNNBeatProc,
    DBNBeatTrackingProcessor as _DBNBeatProc
)

import pretty_midi  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Detect onsets with madmom and export as MIDI notes")
    ap.add_argument("input", help="Audio file (wav/mp3/flac/m4a)")
    ap.add_argument("--output", "-o", help="Output MIDI path (.mid). Default: alongside input")

    # Onset model & params
    ap.add_argument("--model", choices=["rnn", "cnn"], default="rnn", help="Onset model (rnn is robust default)")
    ap.add_argument("--fps", type=int, default=200, help="Onset activation sampling rate (Hz)")
    ap.add_argument("--threshold", type=float, default=0.15, help="Peak-picking threshold (lower = more onsets)")
    ap.add_argument("--smooth", type=float, default=0.05, help="Smoothing (seconds)")
    ap.add_argument("--combine", type=float, default=0.03, help="Combine window to merge double-triggers (seconds)")
    ap.add_argument("--pre-max", type=float, default=0.02, help="Local max window before (seconds)")
    ap.add_argument("--post-max", type=float, default=0.02, help="Local max window after (seconds)")
    ap.add_argument("--pre-avg", type=float, default=0.10, help="Adaptive threshold window before (seconds)")
    ap.add_argument("--post-avg", type=float, default=0.10, help="Adaptive threshold window after (seconds)")

    # MIDI properties
    ap.add_argument("--pitch", type=int, default=60, help="MIDI pitch for onset notes (default C4=60)")
    ap.add_argument("--velocity", type=int, default=96, help="MIDI velocity (1..127)")
    ap.add_argument("--dur-ms", type=float, default=120.0, help="Duration per onset note (milliseconds)")
    ap.add_argument("--initial-tempo", type=float, default=120.0, help="Constant tempo metadata in MIDI (BPM)")

    # Optional quantization to beat grid
    ap.add_argument("--quantize", action="store_true", help="Snap onset times to beat grid from madmom")
    ap.add_argument("--meter", choices=["auto", "3", "4"], default="auto", help="Meter for downbeat tracker")
    ap.add_argument("--subdiv", type=int, default=4, help="Beat subdivisions for grid (e.g., 4=16ths, 6=triplet 8ths)")
    ap.add_argument("--max-snap-ms", type=float, default=120.0, help="Max distance to snap onsets (ms)")

    return ap.parse_args()


def track_beats_and_downbeats(audio_path: str, fps: int = 100, meter: str = "auto") -> Tuple[_np.ndarray, _np.ndarray]:
    """Robust beat & downbeat with fallbacks; returns (beat_times, is_downbeat)"""
    def _try_dbn(beats_per_bar):
        rnn = RNNDownBeatProcessor()
        act = rnn(audio_path)
        dbn = DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar, fps=fps)
        return dbn(act)

    # Try downbeat tracking with specific meters first
    meters = [[4], [3]] if meter == "auto" else [[int(meter)]]
    for m in meters:
        try:
            beats = _np.asarray(_try_dbn(m))
            bt = beats[:, 0].astype(float)
            is_db = (beats[:, 1].astype(int) == 1)
            if len(bt) >= 2:
                return bt, is_db
        except Exception:
            pass

    # Fallback: plain beats, synthesize downbeats
    rnn_b = _RNNBeatProc()
    act_b = rnn_b(audio_path)
    dbn_b = _DBNBeatProc(fps=fps)
    bt = _np.asarray(dbn_b(act_b)).astype(float)
    m = int(meter) if meter in ("3", "4") else 4
    is_db = _np.zeros_like(bt, dtype=bool)
    is_db[::m] = True
    return bt, is_db


def build_quant_grid(beat_times: _np.ndarray, subdiv: int) -> _np.ndarray:
    grid: List[float] = []
    for i in range(len(beat_times) - 1):
        start = beat_times[i]
        end = beat_times[i + 1]
        for k in range(subdiv):
            grid.append(start + (end - start) * (k / subdiv))
    grid.append(float(beat_times[-1]))
    return _np.asarray(grid, dtype=float)


def nearest_grid_time(grid: _np.ndarray, t: float) -> Tuple[float, int, float]:
    idx = _np.searchsorted(grid, t)
    cand = []
    if 0 <= idx < len(grid): cand.append(idx)
    if idx - 1 >= 0: cand.append(idx - 1)
    if idx + 1 < len(grid): cand.append(idx + 1)
    if not cand:
        return float(grid[0]), 0, abs(t - grid[0])
    best = min(cand, key=lambda j: abs(grid[j] - t))
    return float(grid[best]), int(best), abs(grid[best] - t)


def seconds_to_midi(onsets: _np.ndarray, output_path: str, pitch: int, velocity: int,
                    dur_ms: float, initial_tempo_bpm: float) -> None:
    """Create a single-track MIDI where each onset becomes a short note at fixed pitch."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo_bpm)
    inst = pretty_midi.Instrument(program=0, is_drum=False, name="Onsets")
    dur = max(0.01, dur_ms / 1000.0)
    vel = int(max(1, min(127, velocity)))
    for t in onsets:
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=float(t), end=float(t + dur)))
    pm.instruments.append(inst)
    pm.write(output_path)


def main() -> None:
    args = parse_args()

    inp = args.input
    if not os.path.isfile(inp):
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(2)
    outp = args.output or os.path.splitext(inp)[0] + ".onsets.mid"

    # 1) Onset activations
    if args.model == "cnn":
        proc = CNNOnsetProcessor(fps=args.fps)
    else:
        proc = RNNOnsetProcessor(fps=args.fps)
    act = proc(inp)

    # 2) Peak picking
    picker = OnsetPeakPickingProcessor(
        fps=args.fps,
        threshold=args.threshold,
        smooth=args.smooth,
        combine=args.combine,
        pre_max=args.pre_max,
        post_max=args.post_max,
        pre_avg=args.pre_avg,
        post_avg=args.post_avg,
    )
    onsets = picker(act)  # seconds (np.ndarray)

    # 3) Optional quantization to beat grid
    if args.quantize and len(onsets) > 0:
        beat_times, is_db = track_beats_and_downbeats(inp, fps=100, meter=args.meter)
        grid = build_quant_grid(beat_times, subdiv=args.subdiv)
        max_snap_s = args.max_snap_ms / 1000.0
        snapped = []
        for t in onsets:
            q_t, _, d = nearest_grid_time(grid, t)
            snapped.append(q_t if d <= max_snap_s else t)
        onsets = _np.asarray(snapped, dtype=float)

    # 4) Emit MIDI
    seconds_to_midi(onsets, outp, args.pitch, args.velocity, args.dur_ms, args.initial_tempo)

    print(f"Detected {len(onsets)} onsets. Wrote MIDI: {outp}")


if __name__ == "__main__":
    main()
