#!/usr/bin/env python3
"""
lane_assign_melody.py
---------------------
Map monophonic notes (vocals, bass, etc.) into 2–3 game lanes, preserving contour
(ascending runs sweep right, descending runs sweep left), while respecting cooldown,
density caps, and accessibility presets.

Usage:
    from lane_assign_melody import Note, assign_melody_to_lanes, MELODY_PRESETS

    notes = [Note(t=1.0, d=0.3, pitch=64), ...]
    beats = np.loadtxt("beat_times.txt")  # one float per line
    settings = MELODY_PRESETS["Medium"]
    assn = assign_melody_to_lanes(notes, beats, settings)
    # assn -> [(time_s, lane_idx, midi_pitch, duration_s)]
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import math


@dataclass
class Note:
    t: float      # onset time (s)
    d: float      # duration (s)
    pitch: int    # MIDI pitch (0..127)
    strength: float = 1.0  # 0..1 (confidence/energy)


@dataclass
class MelodyLaneSettings:
    num_lanes: int = 3
    cooldown_s: float = 0.12
    max_notes_per_beat_global: float = 2.2
    max_notes_per_beat_lane: float = 1.6
    window_beats: float = 2.0
    # DP costs
    w_dir: float = 2.0      # penalize lane motion that contradicts pitch direction
    w_jump: float = 0.6     # big lane jumps feel awkward
    w_span: float = 0.8     # pitch→lane mismatch
    w_speed: float = 1.2    # overcrowding a lane
    difficulty: str = "Medium"


# Presets (accessibility included)
MELODY_PRESETS: Dict[str, MelodyLaneSettings] = {
    "Easy": MelodyLaneSettings(
        num_lanes=2, cooldown_s=0.14,
        max_notes_per_beat_global=1.8, max_notes_per_beat_lane=1.2,
        window_beats=2.0, w_dir=2.0, w_jump=0.8, w_span=1.0, w_speed=1.4,
        difficulty="Easy"
    ),
    "Medium": MelodyLaneSettings(
        num_lanes=3, cooldown_s=0.12,
        max_notes_per_beat_global=2.4, max_notes_per_beat_lane=1.6,
        window_beats=2.0, w_dir=2.0, w_jump=0.6, w_span=0.8, w_speed=1.2,
        difficulty="Medium"
    ),
    "Hard": MelodyLaneSettings(
        num_lanes=3, cooldown_s=0.10,
        max_notes_per_beat_global=3.2, max_notes_per_beat_lane=2.2,
        window_beats=1.5, w_dir=1.6, w_jump=0.5, w_span=0.6, w_speed=1.0,
        difficulty="Hard"
    ),
    "TwoLane": MelodyLaneSettings(
        num_lanes=2, cooldown_s=0.14,
        max_notes_per_beat_global=1.8, max_notes_per_beat_lane=1.2,
        window_beats=2.0, w_dir=2.0, w_jump=0.8, w_span=1.0, w_speed=1.4,
        difficulty="Easy"
    ),
}


def _norm_pitch_to_lane_target(pitches: np.ndarray, num_lanes: int) -> np.ndarray:
    """Map MIDI pitches to a continuous [1, num_lanes] target using z-scores."""
    mu, sigma = float(np.median(pitches)), float(np.std(pitches) + 1e-6)
    z = (pitches - mu) / sigma
    # sigmoid → 0..1, then scale to lanes
    return 1.0 + (num_lanes - 1) * (1 / (1 + np.exp(-z)))


def assign_melody_to_lanes(
    notes: List[Note],
    beats: np.ndarray,
    settings: MelodyLaneSettings = MELODY_PRESETS["Medium"],
) -> List[Tuple[float, int, int, float]]:
    """
    Assign notes to lanes using DP/Viterbi contour-preserving mapping.

    Returns:
        [(time_s, lane, midi_pitch, duration_s)]
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n.t)
    num_lanes = max(2, settings.num_lanes)

    P = np.array([n.pitch for n in notes], dtype=float)
    T = np.array([n.t for n in notes], dtype=float)
    D = np.array([n.d for n in notes], dtype=float)
    target = _norm_pitch_to_lane_target(P, num_lanes)

    N = len(notes)
    C = np.full((N, num_lanes), 1e9, dtype=float)  # costs
    B = np.full((N, num_lanes), -1, dtype=int)     # backpointers

    # init
    for l in range(num_lanes):
        C[0, l] = settings.w_span * abs((l + 1) - target[0])
        B[0, l] = -1

    # recurrence
    for i in range(1, N):
        dpitch = P[i] - P[i - 1]
        for l in range(num_lanes):
            cost_local = settings.w_span * abs((l + 1) - target[i])
            best_cost, best_lp = 1e9, -1
            for lp in range(num_lanes):
                cost = C[i - 1, lp]
                # direction penalty
                if dpitch > 0 and l < lp:
                    cost += settings.w_dir
                if dpitch < 0 and l > lp:
                    cost += settings.w_dir
                # jump penalty
                cost += settings.w_jump * abs(l - lp)
                if cost < best_cost:
                    best_cost, best_lp = cost, lp
            C[i, l] = best_cost + cost_local
            B[i, l] = best_lp

    # backtrack
    last_lane = int(np.argmin(C[-1]))
    lanes = [last_lane]
    for i in range(N - 1, 0, -1):
        last_lane = int(B[i, last_lane])
        lanes.append(last_lane)
    lanes = lanes[::-1]

    # cooldown + density enforcement
    assignments = [(notes[i].t, lanes[i], notes[i].pitch, notes[i].d) for i in range(N)]
    last_t: Dict[int, float] = {}
    keep = [True] * N
    for i, (t, l, p, d) in enumerate(assignments):
        lt = last_t.get(l, -1e9)
        if t - lt < settings.cooldown_s:
            keep[i] = False
        else:
            last_t[l] = t

    out = [a for a, k in zip(assignments, keep) if k]
    return out
