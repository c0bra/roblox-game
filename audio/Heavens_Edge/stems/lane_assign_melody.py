#!/usr/bin/env python3
"""
lane_assign_melody.py  (contour-preserving lane mapper, 2–3 lanes)

Inputs:  list[Note(t,d,pitch,strength)], beat times (np.ndarray)
Output:  list[(time_s, lane_idx, midi_pitch, duration_s)]  # lane_idx is 0-based

Key features:
- Pitch→lane target from robust percentiles (P20–P80)
- Direction-aware inertia: ascending runs sweep right, descending sweep left
- Anti-collapse: discourage long streaks on the same lane
- Cooldown and simple density control via presets (incl. 2-lane accessibility)
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import math

# ---------------- Models & Presets ----------------

@dataclass
class Note:
    t: float
    d: float
    pitch: int
    strength: float = 1.0

@dataclass
class MelodyLaneSettings:
    num_lanes: int = 3
    cooldown_s: float = 0.12
    max_notes_per_beat_global: float = 2.4
    max_notes_per_beat_lane: float = 1.6
    window_beats: float = 2.0
    # DP costs
    w_dir: float = 2.0      # penalize lane motion that contradicts direction
    w_jump: float = 0.6     # penalize large lane jumps
    w_span: float = 0.8     # distance from pitch target lane
    w_inertia: float = 0.8  # extra cost to "not move" when direction is clear
    # Post-DP heuristics
    streak_lookback: int = 6
    streak_penalty: float = 0.5
    dir_semitones: float = 1.0  # min Δpitch to consider a direction (avoid jitter)
    difficulty: str = "Medium"

MELODY_PRESETS: Dict[str, MelodyLaneSettings] = {
    "Easy": MelodyLaneSettings(
        num_lanes=2, cooldown_s=0.14,
        max_notes_per_beat_global=1.8, max_notes_per_beat_lane=1.2,
        window_beats=2.0, w_dir=2.2, w_jump=0.9, w_span=1.0, w_inertia=1.0,
        streak_lookback=5, streak_penalty=0.6, dir_semitones=1.0, difficulty="Easy"
    ),
    "Medium": MelodyLaneSettings(
        num_lanes=3, cooldown_s=0.12,
        max_notes_per_beat_global=2.4, max_notes_per_beat_lane=1.6,
        window_beats=2.0, w_dir=2.0, w_jump=0.6, w_span=0.8, w_inertia=0.8,
        streak_lookback=6, streak_penalty=0.5, dir_semitones=1.0, difficulty="Medium"
    ),
    "Hard": MelodyLaneSettings(
        num_lanes=3, cooldown_s=0.10,
        max_notes_per_beat_global=3.2, max_notes_per_beat_lane=2.2,
        window_beats=1.5, w_dir=1.6, w_jump=0.5, w_span=0.6, w_inertia=0.6,
        streak_lookback=7, streak_penalty=0.4, dir_semitones=0.8, difficulty="Hard"
    ),
    # Accessibility: two-lane / one-hand
    "TwoLane": MelodyLaneSettings(
        num_lanes=2, cooldown_s=0.14,
        max_notes_per_beat_global=1.8, max_notes_per_beat_lane=1.2,
        window_beats=2.0, w_dir=2.2, w_jump=0.9, w_span=1.0, w_inertia=1.0,
        streak_lookback=5, streak_penalty=0.6, dir_semitones=1.0, difficulty="Easy"
    ),
}

# ---------------- Helpers ----------------

def _target_from_percentiles(pitches: np.ndarray, num_lanes: int) -> np.ndarray:
    """Map MIDI pitches → continuous lane target in [0, num_lanes-1] using robust scaling."""
    if len(pitches) < 3:
        # trivial map: put everything mid-lane
        return np.full_like(pitches, (num_lanes-1)/2, dtype=float)
    p20 = float(np.percentile(pitches, 20))
    p80 = float(np.percentile(pitches, 80))
    if p80 <= p20:
        p20, p80 = float(np.min(pitches)), float(np.max(pitches) + 1e-6)
    x = (pitches - p20) / (p80 - p20)
    x = np.clip(x, 0.0, 1.0)
    return x * (num_lanes - 1)  # 0..L-1

def build_subbeat_grid(beats: np.ndarray, subdiv: int) -> np.ndarray:
    """Interpolate a dense grid between beats with `subdiv` slots per beat."""
    if subdiv <= 1 or len(beats) < 2:
        return beats.copy()
    g = []
    for i in range(len(beats) - 1):
        b0, b1 = beats[i], beats[i+1]
        step = (b1 - b0) / subdiv
        for k in range(subdiv):
            g.append(b0 + k * step)
    g.append(float(beats[-1]))
    return np.asarray(g, float)

def _nearest(arr: np.ndarray, x: float) -> float:
    i = np.searchsorted(arr, x)
    cand = []
    if i < len(arr): cand.append(arr[i])
    if i > 0:         cand.append(arr[i-1])
    return min(cand, key=lambda v: abs(v - x)) if cand else x

def _next_at_or_after(arr: np.ndarray, x: float) -> float:
    i = np.searchsorted(arr, x)
    if i >= len(arr):  # extend with last step size (fallback)
        return arr[-1] + (arr[-1] - arr[-2] if len(arr) > 1 else 0.25)
    return arr[i]

def quantize_notes_to_grid(
    notes: List[Note],
    grid: np.ndarray,
    snap_ms: float = 60.0,
    min_dur_s: float = 0.08,
    mode: str = "start_to_nearest_end_to_next",
) -> List[Note]:
    """Quantize monophonic notes to a sub-beat grid and enforce min duration/overlap rules."""
    if not notes: return []
    tol = snap_ms / 1000.0
    out: List[Note] = []
    last_end = -1e9
    for n in sorted(notes, key=lambda x: x.t):
        s_raw, e_raw = n.t, n.t + max(n.d, 0.0)
        if mode == "both_to_nearest":
            s_q = _nearest(grid, s_raw); e_q = _nearest(grid, e_raw)
            if e_q <= s_q: e_q = s_q + min_dur_s
        else:  # "start_to_nearest_end_to_next"
            s_q = _nearest(grid, s_raw)
            if abs(s_q - s_raw) > tol:
                s_q = s_raw  # too far: leave start (or skip if you prefer)
            e_q = _next_at_or_after(grid, max(e_raw, s_q + 1e-6))
            if e_q - s_q < min_dur_s:
                nxt = _next_at_or_after(grid, e_q + 1e-6)
                e_q = max(s_q + min_dur_s, nxt)
        # prevent overlap (monophonic)
        if s_q < last_end:
            if last_end - s_q <= tol:
                s_q = last_end
            else:
                continue  # drop colliding note
        if e_q - s_q < min_dur_s:
            e_q = s_q + min_dur_s
        out.append(Note(t=float(s_q), d=float(e_q - s_q), pitch=int(n.pitch), strength=n.strength))
        last_end = e_q
    return out

# ---------------- Main ----------------

def assign_melody_to_lanes(
    notes: List[Note],
    beats: np.ndarray,
    settings: MelodyLaneSettings = MELODY_PRESETS["Medium"],
    *,
    subdiv: int = 4,                 # 4=16ths, 3=triplet 8ths, 2=8ths
    snap_ms: float = 60.0,
    quant_mode: str = "start_to_nearest_end_to_next",
) -> List[Tuple[float, int, int, float]]:
    """Quantize to sub-beat grid from `beats` then do contour-preserving DP lane mapping."""
    if not notes:
        return []

    # 1) Build grid from beats and quantize notes
    grid = build_subbeat_grid(beats, subdiv=subdiv)
    q_notes = quantize_notes_to_grid(notes, grid, snap_ms=snap_ms, min_dur_s=0.08, mode=quant_mode)
    if not q_notes:
        return []

    # 2) DP/Viterbi lane mapping (same logic as before)
    q_notes = sorted(q_notes, key=lambda n: n.t)
    num_lanes = max(2, settings.num_lanes)

    P = np.asarray([n.pitch for n in q_notes], float)
    T = np.asarray([n.t for n in q_notes], float)
    D = np.asarray([n.d for n in q_notes], float)

    target = _target_from_percentiles(P, num_lanes)  # 0..L-1

    N = len(q_notes)
    C = np.full((N, num_lanes), 1e9, float)
    B = np.full((N, num_lanes), -1, int)
    S = np.zeros((N, num_lanes), int)

    for l in range(num_lanes):
        C[0, l] = settings.w_span * abs(l - target[0])
        B[0, l] = -1
        S[0, l] = 1

    for i in range(1, N):
        dp = P[i] - P[i-1]
        dir_clear = abs(dp) >= settings.dir_semitones
        for l in range(num_lanes):
            cost_local = settings.w_span * abs(l - target[i])
            best_cost, best_lp, best_streak = 1e9, -1, 1
            for lp in range(num_lanes):
                cost = C[i-1, lp]
                if dir_clear:
                    if dp > 0 and l < lp: cost += settings.w_dir
                    if dp < 0 and l > lp: cost += settings.w_dir
                    if (dp > 0 and l == lp) or (dp < 0 and l == lp):
                        cost += settings.w_inertia
                cost += settings.w_jump * abs(l - lp)
                streak = (S[i-1, lp] + 1) if l == lp else 1
                if streak > settings.streak_lookback:
                    cost += settings.streak_penalty * (streak - settings.streak_lookback)
                if cost < best_cost:
                    best_cost, best_lp, best_streak = cost, lp, streak
            C[i, l] = best_cost + cost_local
            B[i, l] = best_lp
            S[i, l] = best_streak

    last_lane = int(np.argmin(C[-1]))
    lanes = [last_lane]
    for i in range(N-1, 0, -1):
        last_lane = int(B[i, last_lane])
        lanes.append(last_lane)
    lanes = lanes[::-1]

    # 3) Cooldown pass (greedy)
    last_t: Dict[int, float] = {}
    keep = [True]*N
    for i, (t, l) in enumerate(zip(T, lanes)):
        lt = last_t.get(l, -1e9)
        if t - lt < settings.cooldown_s:
            moved = False
            for nl in [l-1, l+1]:
                if 0 <= nl < num_lanes:
                    lt2 = last_t.get(nl, -1e9)
                    if t - lt2 >= settings.cooldown_s:
                        lanes[i] = nl
                        last_t[nl] = t
                        moved = True
                        break
            if not moved:
                keep[i] = False
        else:
            last_t[l] = t

    out = []
    for i, k in enumerate(keep):
        if not k: continue
        n = q_notes[i]
        out.append((n.t, lanes[i], n.pitch, n.d))
    return out