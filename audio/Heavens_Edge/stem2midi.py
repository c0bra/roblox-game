#!/usr/bin/env python3
"""
stem2midi.py — Single-stem -> quantized MIDI

Pipeline:
  1) Beat & downbeat tracking with madmom (builds a timing grid).
  2) Transcribe notes with Spotify Basic Pitch.
  3) Clean up notes (min conf/velocity, short-note removal, micro-gap merge).
  4) Quantize onsets/offsets to the beat grid.
  5) Export MIDI (pretty_midi).

Install (example):
  pip install basic-pitch madmom pretty_midi mido numpy soundfile librosa

Notes:
  - Basic Pitch returns a PrettyMIDI object; we read notes from it for robustness.
  - We quantize to a grid derived from madmom's downbeat/beat tracker.
  - MIDI is exported with a *constant* tempo (initial_tempo). The note times
    are absolute seconds, so playback aligns with the original audio even if
    your DAW grid shows a different tempo map. (Variable-tempo export can be
    added later if needed.)

References:
  - Basic Pitch usage: predict() -> (model_output, midi_data, note_events)
    https://github.com/spotify/basic-pitch
  - madmom downbeat tracking: RNNDownBeatProcessor + DBNDownBeatTrackingProcessor
    https://madmom.readthedocs.io/en/v0.16/modules/features/downbeats.html
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Import transcription & MIDI libs
from basic_pitch.inference import predict  # type: ignore
import pretty_midi  # type: ignore


# --- Compatibility shim for Python 3.10+ ---
# madmom (0.16.x) imports ABCs from `collections` which moved to `collections.abc`.
# This shim backfills attributes on `collections` before importing madmom.
import collections as _collections
try:
    import collections.abc as _cabc  # py>=3.3
    for _name in ("MutableMapping","MutableSet","MutableSequence","Mapping","Set","Sequence"):
        if not hasattr(_collections, _name) and hasattr(_cabc, _name):
            setattr(_collections, _name, getattr(_cabc, _name))
except Exception:
    pass
# --- End shim ---

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

# Beat/downbeat tracking

from madmom.features.beats import RNNBeatProcessor as _RNNBeatProc, DBNBeatTrackingProcessor as _DBNBeatProc  # type: ignore
from madmom.features.downbeats import (  # type: ignore
    RNNDownBeatProcessor,
    DBNDownBeatTrackingProcessor,
)


@dataclass
class Note:
    onset: float      # seconds
    offset: float     # seconds
    pitch: int        # MIDI note number
    strength: float   # 0..1 (from Basic Pitch velocity/127 as proxy)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-stem -> quantized MIDI with madmom + Basic Pitch")
    p.add_argument("--input", "-i", required=True, help="Path to mono/stem audio (wav/flac/mp3/m4a)")
    p.add_argument("--output", "-o", help="Output MIDI path (.mid). Default: alongside input")
    # Grid / quantization
    p.add_argument("--subdiv", type=int, default=4, help="Beat subdivisions (e.g., 4=16ths, 3=triplets)")
    p.add_argument("--meter", choices=["auto", "3", "4"], default="auto", help="Assumed meter for downbeat tracking; try 4 for pop/rock if auto fails")
    p.add_argument("--max-snap-ms", type=float, default=80.0, help="Max distance to snap onsets/offsets (ms)")
    # Cleanup
    p.add_argument("--min-note-ms", type=float, default=90.0, help="Drop notes shorter than this duration (ms)")
    p.add_argument("--merge-gap-ms", type=float, default=60.0, help="Merge same-pitch notes separated by <= this gap (ms)")
    p.add_argument("--min-conf", type=float, default=0.6, help="Drop notes with (velocity/127) < min-conf")
    # MIDI
    p.add_argument("--ppq", type=int, default=960, help="Ticks per quarter note (for file writing)")
    p.add_argument("--velocity-mode", choices=["amp", "fixed"], default="amp", help="Use amplitude or fixed velocity")
    p.add_argument("--fixed-vel", type=int, default=96, help="Fixed velocity if velocity-mode=fixed (1..127)")
    p.add_argument("--initial-tempo", type=float, default=120.0, help="Constant tempo to store in MIDI meta (BPM)")
    # Debug
    p.add_argument("--save-debug", action="store_true", help="Save CSVs of raw/clean/quantized notes and beat grid")
    p.add_argument("--print-backend", action="store_true",
               help="Just print which Basic Pitch backend will be used and exit")
    return p.parse_args()


def secs_to_ms(x: float) -> float:
    return x * 1000.0


def ms_to_secs(x: float) -> float:
    return x / 1000.0



def track_beats_and_downbeats(audio_path: str, fps: int = 100, meter: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust beat & downbeat tracker with fallbacks.
    - Try DBNDownBeatTrackingProcessor with chosen meter(s).
    - If it errors (ragged results etc.), fall back to plain beat tracking and
      synthesize downbeats by modulo the assumed meter.
    """
    def _try_dbn_downbeat(beats_per_bar):
        rnn = RNNDownBeatProcessor()
        act = rnn(audio_path)
        dbn = DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar, fps=fps)
        return dbn(act)

    # 1) Preferred path: downbeat tracker
    try_meters = []
    if meter in ("3", "4"):
        try_meters = [[int(meter)]]
    else:
        try_meters = [[4], [3]]  # common meters

    for m in try_meters:
        try:
            beats = _np.asarray(_try_dbn_downbeat(m))
            # beats: shape (N, 2): time (s), beat_number_in_bar (1..meter)
            beat_times = beats[:, 0].astype(float)
            is_downbeat = (beats[:, 1].astype(int) == 1)
            if len(beat_times) >= 2:
                return beat_times, is_downbeat
        except Exception as e:
            # continue to next meter or fallback
            pass

    # 2) Fallback: plain beat tracking, synthesize downbeats
    try:
        rnn_b = _RNNBeatProc()
        act_b = rnn_b(audio_path)
        dbn_b = _DBNBeatProc(fps=fps)
        beat_times = _np.asarray(dbn_b(act_b)).astype(float)
        if meter in ("3", "4"):
            m = int(meter)
        else:
            m = 4  # default
        is_downbeat = _np.zeros_like(beat_times, dtype=bool)
        is_downbeat[::m] = True  # every m-th beat as downbeat
        return beat_times, is_downbeat
    except Exception:
        raise SystemExit("Beat tracking failed (madmom). Try a different file or install ffmpeg.")


def build_quant_grid(beat_times: np.ndarray, subdiv: int) -> np.ndarray:
    """
    Build grid times at given subdivision between consecutive beats.

    For each [b_i, b_{i+1}], create subdiv points inclusive of b_i, exclusive of b_{i+1}.
    Returns array of grid times in seconds.
    """
    grid_times: List[float] = []
    for i in range(len(beat_times) - 1):
        start = beat_times[i]
        end = beat_times[i + 1]
        # linear spacing: include start, exclude end to avoid duplicates
        for k in range(subdiv):
            t = start + (end - start) * (k / subdiv)
            grid_times.append(float(t))
    # Append the final beat time as a grid point (handy for snapping offsets near the end)
    grid_times.append(float(beat_times[-1]))
    return np.asarray(grid_times, dtype=float)


def nearest_grid_time(grid: np.ndarray, t: float) -> Tuple[float, int, float]:
    """
    Returns (nearest_time, index, abs_distance_seconds)
    """
    idx = np.searchsorted(grid, t)
    candidates = []
    if 0 <= idx < len(grid):
        candidates.append(idx)
    if idx - 1 >= 0:
        candidates.append(idx - 1)
    if idx + 1 < len(grid):
        candidates.append(idx + 1)
    if not candidates:
        return float(grid[0]), 0, abs(t - grid[0])
    best_idx = min(candidates, key=lambda j: abs(grid[j] - t))
    return float(grid[best_idx]), int(best_idx), abs(grid[best_idx] - t)


def transpose_pretty_midi_to_notes(pm: pretty_midi.PrettyMIDI) -> List[Note]:
    """
    Extract all notes from PrettyMIDI object into our Note list.
    """
    notes: List[Note] = []
    for inst in pm.instruments:
        for n in inst.notes:
            # Strength proxy from velocity
            strength = max(0.0, min(1.0, n.velocity / 127.0))
            notes.append(Note(onset=float(n.start), offset=float(n.end), pitch=int(n.pitch), strength=strength))
    # Sort by onset then pitch
    notes.sort(key=lambda x: (x.onset, x.pitch))
    return notes


def run_basic_pitch(audio_path: str) -> List[Note]:
    """
    Run Basic Pitch and return a list of Note objects.
    """
    _, midi_data, note_events = predict(audio_path)  # note_events not used here
    if isinstance(midi_data, pretty_midi.PrettyMIDI):
        return transpose_pretty_midi_to_notes(midi_data)
    # Fallback: if midi_data isn't a PrettyMIDI instance, try to decode from note_events
    notes: List[Note] = []
    if isinstance(note_events, (list, tuple)):
        for ev in note_events:
            # be defensive about field names
            onset = float(ev.get('start_time', ev.get('onset_time', ev.get('onset', 0.0))))
            offset = float(ev.get('end_time', ev.get('offset_time', ev.get('offset', onset + 0.1))))
            pitch = int(ev.get('note_number', ev.get('pitch', 60)))
            velocity = int(ev.get('velocity', 96))
            strength = max(0.0, min(1.0, velocity / 127.0))
            notes.append(Note(onset=onset, offset=offset, pitch=pitch, strength=strength))
    notes.sort(key=lambda x: (x.onset, x.pitch))
    return notes


def cleanup_notes(notes: List[Note], min_note_ms: float, merge_gap_ms: float, min_conf: float) -> List[Note]:
    """
    - drop notes shorter than min_note_ms
    - merge same-pitch notes separated by <= merge_gap_ms
    - drop notes with strength < min_conf
    """
    if not notes:
        return []

    min_note_s = ms_to_secs(min_note_ms)
    merge_gap_s = ms_to_secs(merge_gap_ms)

    # Filter by strength and duration
    filtered = [n for n in notes if n.strength >= min_conf and (n.offset - n.onset) >= min_note_s]
    if not filtered:
        return []

    # Merge same-pitch neighbors separated by small gaps
    merged: List[Note] = []
    # Group by pitch for merging contiguity
    by_pitch: dict[int, List[Note]] = {}
    for n in filtered:
        by_pitch.setdefault(n.pitch, []).append(n)
    for pitch, ns in by_pitch.items():
        ns.sort(key=lambda x: x.onset)
        current = ns[0]
        for nxt in ns[1:]:
            gap = nxt.onset - current.offset
            if gap <= merge_gap_s and gap >= -1e-4:
                # merge: extend current
                current.offset = max(current.offset, nxt.offset)
                current.strength = max(current.strength, nxt.strength)
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
    # Re-sort merged across all pitches
    merged.sort(key=lambda x: (x.onset, x.pitch))
    # Enforce min duration again (after merging)
    merged = [n for n in merged if (n.offset - n.onset) >= min_note_s]
    return merged


def quantize_notes(notes: List[Note], grid: np.ndarray, max_snap_ms: float) -> List[Note]:
    """
    Snap onsets to nearest grid (within threshold), and offsets to the next
    grid line at or after the onset (also within threshold). Enforce >= 1 grid step.
    """
    if not notes or len(grid) < 2:
        return notes
    max_snap_s = ms_to_secs(max_snap_ms)

    quantized: List[Note] = []
    for n in notes:
        q_on, q_idx, d_on = nearest_grid_time(grid, n.onset)
        onset = q_on if d_on <= max_snap_s else n.onset

        # For offset: choose the grid point not earlier than onset
        # If the nearest is before, advance to at least next grid index
        _, idx_guess, _ = nearest_grid_time(grid, max(onset, n.offset))
        # ensure at least 1 step after onset grid index
        min_idx = max(q_idx + 1, np.searchsorted(grid, onset, side="right"))
        idx = max(min_idx, idx_guess)
        if idx >= len(grid):
            # Clamp to last grid + small epsilon beyond
            offset = max(onset + 1e-3, grid[-1] + 1e-3)
        else:
            cand = grid[idx]
            d_off = abs(cand - n.offset)
            offset = cand if d_off <= max_snap_s else max(onset + 1e-3, n.offset)

        # Guarantee nonzero duration
        if offset <= onset:
            offset = onset + (grid[1] - grid[0]) * 0.5

        quantized.append(Note(onset=onset, offset=offset, pitch=n.pitch, strength=n.strength))

    return quantized


def write_debug_csvs(
    out_base: str,
    raw: List[Note],
    clean: List[Note],
    quant: List[Note],
    beat_times: np.ndarray,
    subdiv: int
) -> None:
    def dump_notes(path: str, xs: List[Note]) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["onset_s", "offset_s", "pitch", "strength_0_1", "dur_ms"])
            for n in xs:
                w.writerow([f"{n.onset:.6f}", f"{n.offset:.6f}", n.pitch, f"{n.strength:.3f}", f"{secs_to_ms(n.offset-n.onset):.1f}"])

    dump_notes(out_base + ".raw_notes.csv", raw)
    dump_notes(out_base + ".clean_notes.csv", clean)
    dump_notes(out_base + ".quantized_notes.csv", quant)

    # Beat grid CSV
    grid = build_quant_grid(beat_times, subdiv)
    with open(out_base + ".beat_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grid_index", "time_s"])
        for i, t in enumerate(grid):
            w.writerow([i, f"{t:.6f}"])


def export_midi(
    notes: List[Note],
    output_path: str,
    initial_tempo_bpm: float = 120.0,
    program: int = 0,
    is_drum: bool = False,
) -> None:
    """
    Export notes (absolute seconds) to a MIDI file using pretty_midi.
    A constant initial tempo is stored in metadata for DAWs which care,
    but timing is encoded in seconds so the playback aligns.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo_bpm)
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name="Stem2MIDI")
    for n in notes:
        vel = int(round(max(1, min(127, n.strength * 127))))  # strength proxy
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=n.pitch, start=n.onset, end=n.offset))
    pm.instruments.append(inst)
    pm.write(output_path)


def main() -> None:
    args = parse_args()

    if args.print_backend:
        backend = "Unknown"
        try:
            import onnxruntime  # noqa
            backend = "ONNX"
        except ImportError:
            try:
                import tflite_runtime  # noqa
                backend = "TFLite"
            except ImportError:
                try:
                    import tensorflow  # noqa
                    backend = "TensorFlow"
                except ImportError:
                    pass
        print(f"Basic Pitch backend: {backend}")
        return

    in_path = args.input
    if not os.path.isfile(in_path):
        raise SystemExit(f"Input not found: {in_path}")
    out_path = args.output
    if not out_path:
        base, _ = os.path.splitext(in_path)
        out_path = base + ".quant.mid"

    # 1) Beat & downbeat tracking -> build timing grid
    beat_times, is_downbeat = track_beats_and_downbeats(in_path, fps=100, meter=args.meter)
    if len(beat_times) < 2:
        raise SystemExit("Could not detect enough beats to build a grid.")
    grid = build_quant_grid(beat_times, subdiv=args.subdiv)

    # 2) Transcribe with Basic Pitch
    raw_notes = run_basic_pitch(in_path)

    # 3) Cleanup
    clean_notes = cleanup_notes(
        raw_notes,
        min_note_ms=args.min_note_ms,
        merge_gap_ms=args.merge_gap_ms,
        min_conf=args.min_conf,
    )

    # 4) Quantize
    quant_notes = quantize_notes(clean_notes, grid, max_snap_ms=args.max_snap_ms)

    # Optional velocity mode
    if args.velocity_mode == "fixed":
        for n in quant_notes:
            n.strength = args.fixed_vel / 127.0

    # 5) Export
    export_midi(quant_notes, out_path, initial_tempo_bpm=args.initial_tempo)

    # Debug artifacts
    if args.save_debug:
        base, _ = os.path.splitext(out_path)
        write_debug_csvs(base, raw_notes, clean_notes, quant_notes, beat_times, args.subdiv)

    print(f"Wrote MIDI: {out_path}")
    if args.save_debug:
        print(f"Wrote debug CSVs next to: {out_path}")

# --- Basic Pitch backend check ---
def _bp_backend():
    try:
        import onnxruntime  # noqa
        has_onnx = True
    except Exception:
        has_onnx = False
    try:
        import tflite_runtime  # noqa
        has_tflite = True
    except Exception:
        has_tflite = False
    try:
        import tensorflow  # noqa
        has_tf = True
    except Exception:
        has_tf = False

    # Crude mirror of basic-pitch's preference order if multiple exist:
    if has_onnx:
        return "ONNX"
    if has_tflite:
        return "TFLite"
    if has_tf:
        return "TensorFlow"
    return "Unknown"

print(f"[basic-pitch] Selected backend: {_bp_backend()}")
# --- end check ---

if __name__ == "__main__":
    main()
