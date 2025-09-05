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
from bisect import bisect_left, bisect_right

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
    """Return a grid made from beats with `subdiv` slots per beat (1 = beats only)."""
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

def _nearest_and_index(arr: np.ndarray, x: float) -> tuple[float, int]:
    i = bisect_left(arr, x)
    if i == 0:
        return float(arr[0]), 0
    if i >= len(arr):
        j = len(arr) - 1
        return float(arr[j]), j
    before, after = arr[i-1], arr[i]
    if (x - before) <= (after - x):
        return float(before), i-1
    else:
        return float(after), i

def quantize_notes_to_grid_strict(
    notes: list["Note"],
    grid: np.ndarray,
    min_dur_s: float = 0.02,  # matches the epsilon behavior in your MIDI script
) -> list["Note"]:
    """
    Snap start and end to NEAREST grid line (like your MIDI script). If end <= start,
    push end to the NEXT grid if possible, else extend by min_dur_s.
    Also enforces monophonic non-overlap by nudging start to the last end.
    """
    if not notes:
        return []
    out: list[Note] = []
    last_end = -1e9

    for n in sorted(notes, key=lambda x: x.t):
        s_raw, e_raw = n.t, n.t + max(n.d, 0.0)

        s_q, s_idx = _nearest_and_index(grid, s_raw)
        e_q, e_idx = _nearest_and_index(grid, e_raw)

        if e_q <= s_q:
            # move end to the next grid if possible; else add small epsilon
            nxt_idx = max(s_idx + 1, bisect_right(grid, s_q))
            if nxt_idx < len(grid):
                e_q = float(grid[nxt_idx])
            else:
                e_q = s_q + min_dur_s

        # prevent overlap (monophonic): if quantized start < previous end, move start up to last_end
        if s_q < last_end:
            s_q = last_end
            if e_q <= s_q:
                # extend minimally
                e_q = s_q + min_dur_s

        out.append(Note(t=float(s_q), d=float(e_q - s_q), pitch=int(n.pitch), strength=n.strength))
        last_end = e_q
    return out

def condense_simultaneous_per_lane(
    assn: list[tuple[float, int, int, float]],
    *,
    tol: float = 1e-4,          # times within tol are “the same”
    pick: str = "longest",      # "longest" | "earliest_end"
) -> list[tuple[float, int, int, float]]:
    """
    Merge multiple notes that start at (nearly) the same time on the same lane.
    Strategy:
      - Group by (lane, rounded start)
      - Keep the note with the longest duration (or earliest end)
      - End time = max end among the group (so we preserve sustain)
    """
    if not assn:
        return assn

    # sort and group by (lane, quantized start)
    assn = sorted(assn, key=lambda x: (x[1], x[0], x[3]))
    groups: dict[tuple[int, float], list[tuple[float,int,int,float]]] = {}
    for t, lane, pitch, dur in assn:
        # quantize start for grouping
        key = (lane, round(t / tol) * tol if tol > 0 else t)
        groups.setdefault(key, []).append((t, lane, pitch, dur))

    merged: list[tuple[float,int,int,float]] = []
    for (lane, _), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(items) == 1:
            merged.append(items[0])
            continue
        # pick representative
        if pick == "earliest_end":
            rep = min(items, key=lambda x: x[0] + x[3])
        else:  # "longest"
            rep = max(items, key=lambda x: x[3])
        # stretch end to cover the latest among items
        start = rep[0]
        end = max(t + d for (t, _, _, d) in items)
        merged.append((start, rep[1], rep[2], max(end - start, 0.02)))

    # keep global time order
    merged.sort(key=lambda x: x[0])
    return merged

# ---------------- Main ----------------

def assign_melody_to_lanes(
    notes: List[Note],
    beats: np.ndarray,
    settings: MelodyLaneSettings = MELODY_PRESETS["Medium"],
    *,
    subdiv: int = 1,   # 1 = beat grid (exactly like your MIDI script). Use 2/3/4 for sub-beats.
) -> List[Tuple[float, int, int, float]]:
    """
    Quantize notes to a (sub-)beat grid using strict 'nearest' snapping for start/end,
    then run contour-preserving DP lane mapping and a cooldown pass.
    """
    if not notes:
        return []

    # 1) Build grid and STRICT-quantize (replicates your MIDI script behavior)
    grid = build_subbeat_grid(beats, subdiv=subdiv)
    q_notes = quantize_notes_to_grid_strict(notes, grid, min_dur_s=0.02)
    if not q_notes:
        return []

    # 2) DP/Viterbi lane mapping (unchanged logic)
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
    out = condense_simultaneous_per_lane(out, tol=1e-4, pick="longest")

    return out