#!/usr/bin/env python3
"""
Kick / Snare / Hats adaptive separator (grid-aware, clustering, soft-masks)
-----------------------------------------------------------------------------
Given a drum(-ish) stem and a beat grid, this script:
  1) Detects percussive onsets (aubio HFC if available; else librosa fallback),
     and snaps them to the nearest grid time (within a tolerance).
  2) Extracts short-time features per hit (band energies, centroid, flatness,
     zero-crossing rate, attack/decay) and clusters hits into 3 classes via
     k-means (Kick/Snare/Hats) with automatic heuristic labeling.
  3) Builds per-class time–frequency templates and constructs soft masks to
     render three separated WAVs: kick.wav, snare.wav, hats.wav.
  4) Exports a CSV of the labeled events and a MIDI with three tracks.

This is a pragmatic prototype optimized for *adaptivity* rather than SOTA.
It should give noticeably cleaner K/S/H than fixed EQ splits when given a
reasonable drum stem and a decent beat grid.

USAGE
-----
python kick_snare_hat_separator.py \
    --audio drums.wav \
    --beats beat_times.txt \
    --outdir out_dir \
    --sr 44100 --hop 256 --win 1024

DEPENDENCIES
------------
- numpy, scipy, soundfile, librosa, scikit-learn, pretty_midi
- aubio (optional; used if installed for HFC onsets)

NOTES
-----
- Grid snapping: only onsets within --snap-ms of a grid time are kept (or moved).
- Hats-first trick: we create a high-band template from the most "hat-like"
  cluster and subtract it early when building other class masks.

"""
from __future__ import annotations
import argparse
import os
import sys
import csv
import math
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import librosa
import librosa.display  # noqa: F401 (useful during iteration)
from scipy.signal import medfilt
from sklearn.cluster import KMeans
import pretty_midi as pm

# -------------------------
# Utilities
# -------------------------

def load_beats(path: str) -> np.ndarray:
    times = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                times.append(float(s))
            except ValueError:
                pass
    if not times:
        raise SystemExit(f"No beat times found in {path}")
    return np.asarray(sorted(times), dtype=float)


def nearest(arr: np.ndarray, x: float) -> Tuple[float, int, float]:
    i = np.searchsorted(arr, x)
    cand = []
    if 0 <= i < len(arr):
        cand.append(i)
    if i - 1 >= 0:
        cand.append(i - 1)
    if i + 1 < len(arr):
        cand.append(i + 1)
    if not cand:
        return float(arr[0]), 0, abs(x - arr[0])
    j = min(cand, key=lambda k: abs(arr[k] - x))
    return float(arr[j]), int(j), abs(arr[j] - x)

def build_subbeat_grid(beats: np.ndarray, subdiv: int) -> np.ndarray:
    """Interpolate a dense grid between beats using `subdiv` steps per beat."""
    if subdiv <= 1:
        return beats.copy()
    grid: list[float] = []
    for i in range(len(beats) - 1):
        b0, b1 = beats[i], beats[i+1]
        dt = (b1 - b0) / subdiv
        for k in range(subdiv):
            grid.append(b0 + k * dt)
    grid.append(float(beats[-1]))
    return np.asarray(grid, dtype=float)


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(max(hz, 1e-9) / 440.0)


@dataclass
class Hit:
    t: float  # onset time (s)
    i: int    # frame index
    conf: float  # onset strength (arbitrary units)


# -------------------------
# Onset detection (aubio HFC preferred)
# -------------------------

def detect_onsets(audio: np.ndarray, sr: int, hop: int, use_aubio: bool = True,
                  backtrack: bool = False) -> List[Hit]:
    hits: List[Hit] = []
    if use_aubio:
        try:
            import aubio  # type: ignore
            o = aubio.onset("hfc", hop, hop, sr)
            o.set_silence(-40)  # dB
            o.set_threshold(0.3)
            o.set_minioi_ms(30)

            # aubio expects float32 frames
            x = audio.astype(np.float32)
            pos = 0
            frame_idx = 0
            while pos + hop <= len(x):
                frame = x[pos:pos+hop]
                if o(frame):
                    t = float(o.get_last_s())
                    hits.append(Hit(t=t, i=frame_idx, conf=float(o.get_onset_detection_function())))
                pos += hop
                frame_idx += 1
            if hits:
                return hits
        except Exception:
            warnings.warn("aubio unavailable or failed; falling back to librosa onsets.")

    # librosa fallback: percussive onset strength
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop, aggregate=np.median)
    on_ix = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop,
                                       backtrack=backtrack, units='frames', pre_max=5, post_max=5,
                                       pre_avg=30, post_avg=30, delta=0.1, wait=5)
    times = librosa.frames_to_time(on_ix, sr=sr, hop_length=hop)
    for idx, t in zip(on_ix, times):
        conf = float(onset_env[min(idx, len(onset_env)-1)])
        hits.append(Hit(t=t, i=int(idx), conf=conf))
    return hits


# -------------------------
# Feature extraction per hit
# -------------------------

def stft_multi(y: np.ndarray, sr: int, n_fft: int, hop: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window='hann', center=True)
    mag = np.abs(S)
    phase = np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return mag, phase, freqs


def band_indices(freqs: np.ndarray, bands: List[Tuple[float, float]]) -> List[np.ndarray]:
    idxs = []
    for lo, hi in bands:
        idxs.append(np.where((freqs >= lo) & (freqs < hi))[0])
    return idxs


def extract_features_per_hit(mag: np.ndarray, freqs: np.ndarray, hits: List[Hit],
                             hop: int, sr: int, win_frames: int = 6) -> np.ndarray:
    """Extract per-hit features using a small window around each hit frame.
    Features: [E_low, E_mid, E_high, centroid, flatness, zcr_est, attack_slope, decay_tau]
    """
    bands = [(0, 140), (140, 4000), (4000, freqs[-1] + 1)]
    bidx = band_indices(freqs, bands)

    feats = []
    zcr_win = int(0.025 * sr)  # 25 ms approx for zcr est from magnitude changes

    for h in hits:
        i0 = max(0, h.i - 1)
        i1 = min(mag.shape[1] - 1, h.i + win_frames)
        M = mag[:, i0:i1+1]
        # Band energies
        e = []
        for bi in bidx:
            e.append(float(np.sum(M[bi, :]**2) + 1e-8))
        E_total = sum(e) + 1e-12
        e_norm = [x / E_total for x in e]

        # Spectral centroid & flatness at the onset frame
        col = mag[:, h.i]
        centroid = float(np.sum(freqs * col) / (np.sum(col) + 1e-9))
        geo_mean = float(np.exp(np.mean(np.log(col + 1e-12))))
        arith_mean = float(np.mean(col) + 1e-12)
        flatness = geo_mean / arith_mean

        # Rough ZCR estimate from frame-to-frame magnitude sign-changes
        # (proxy; actual ZCR would use time-domain, but stem may be huge)
        col_prev = mag[:, max(0, h.i-1)]
        zcr_est = float(np.mean((np.sign(col - col_prev) != 0)))

        # Attack slope (energy rise) and decay tau
        env = np.sum(M**2, axis=0)
        env = env / (np.max(env) + 1e-12)
        # slope between first two frames
        attack_slope = float(env[1] - env[0]) if env.size >= 2 else 0.0
        # decay tau: frames until env falls below 1/e
        tau = 0.0
        for k in range(1, env.size):
            if env[k] <= 1/math.e:
                tau = k
                break
        feats.append([*e_norm, centroid, flatness, zcr_est, attack_slope, tau])

    return np.asarray(feats, dtype=float)


# -------------------------
# Clustering & labeling
# -------------------------

def cluster_hits(X: np.ndarray, k: int = 3, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    lab = km.fit_predict(X)
    return lab


def auto_label_clusters(X: np.ndarray, labels: np.ndarray) -> Dict[int, str]:
    """Heuristic mapping cluster_id -> {kick,snare,hats}.
    Uses band energy distribution + centroid + flatness + tau.
    """
    mapping: Dict[int, str] = {}
    stats = {}
    for c in np.unique(labels):
        Xi = X[labels == c]
        mean = Xi.mean(axis=0)
        E_low, E_mid, E_high, centroid, flatness, zcr_est, attack, tau = mean
        stats[c] = dict(E_low=E_low, E_mid=E_mid, E_high=E_high,
                        centroid=centroid, flatness=flatness, tau=tau)
    # Rank by E_high -> hats, by E_low -> kick, else snare
    by_high = sorted(stats.items(), key=lambda kv: kv[1]['E_high'], reverse=True)
    by_low = sorted(stats.items(), key=lambda kv: kv[1]['E_low'], reverse=True)

    hats_c = by_high[0][0]
    kick_c = by_low[0][0] if by_low[0][0] != hats_c else by_low[1][0]
    # remaining is snare
    remaining = [c for c in stats.keys() if c not in (hats_c, kick_c)]
    snare_c = remaining[0] if remaining else hats_c

    mapping[kick_c] = 'kick'
    mapping[snare_c] = 'snare'
    mapping[hats_c] = 'hats'
    return mapping


# -------------------------
# Template building & masking
# -------------------------

def build_class_templates(mag: np.ndarray, freqs: np.ndarray, hits: List[Hit], labels: np.ndarray,
                          mapping: Dict[int, str], win_frames: int = 6) -> Dict[str, np.ndarray]:
    """Average per-class magnitude spectra around each hit -> 1D template per class."""
    tpl: Dict[str, List[np.ndarray]] = {'kick': [], 'snare': [], 'hats': []}
    for h, lab in zip(hits, labels):
        cls = mapping.get(lab, None)
        if cls is None:
            continue
        i0 = max(0, h.i)
        i1 = min(mag.shape[1]-1, h.i + win_frames)
        spec = np.mean(mag[:, i0:i1+1], axis=1)
        tpl[cls].append(spec)
    tpls = {}
    for cls, lst in tpl.items():
        if lst:
            m = np.mean(np.stack(lst, axis=1), axis=1)
            tpls[cls] = m / (np.max(m) + 1e-12)
        else:
            tpls[cls] = np.ones(mag.shape[0], dtype=float) * (1.0 / mag.shape[0])
    return tpls


def soft_masks_from_templates(mag: np.ndarray, tpls: Dict[str, np.ndarray],
                              hits: List[Hit], labels: np.ndarray, mapping: Dict[int, str],
                              time_sigma_frames: int = 3) -> Dict[str, np.ndarray]:
    """Construct soft masks per class by summing Gaussian time windows * spectral templates."""
    F, T = mag.shape
    masks = {cls: np.zeros((F, T), dtype=float) for cls in ('kick', 'snare', 'hats')}

    # Precompute Gaussian time kernels centered on each hit
    t_grid = np.arange(T)
    for h, lab in zip(hits, labels):
        cls = mapping.get(lab, None)
        if cls is None:
            continue
        w = np.exp(-0.5 * ((t_grid - h.i) / max(1e-6, time_sigma_frames))**2)
        # scale by hit confidence
        w *= max(0.1, h.conf)
        tpl = tpls[cls][:, None]
        masks[cls] += tpl * w[None, :]

    # Normalize so sum of masks <= 1 at each bin
    eps = 1e-8
    total = masks['kick'] + masks['snare'] + masks['hats'] + eps
    for cls in masks:
        masks[cls] = masks[cls] / total
    return masks

# --- NEW: event-aware, frequency-selective refinement to reduce kick-in-snare bleed ---

def _freq_prior(freqs: np.ndarray, kind: str) -> np.ndarray:
    """Simple smooth priors in frequency for each class (0..1)."""
    f = freqs
    if kind == 'kick_low':
        # Low-shelf around 150 Hz
        return 1.0 / (1.0 + np.exp((f - 150.0) / 30.0))
    if kind == 'snare_mid':
        # Broad mid band 200 Hz .. 4 kHz
        return 1.0 / (1.0 + np.exp((200.0 - f) / 50.0)) * 1.0 / (1.0 + np.exp((f - 4000.0) / 400.0))
    if kind == 'hat_high':
        # High-pass above ~5 kHz
        return 1.0 / (1.0 + np.exp((5000.0 - f) / 400.0))
    return np.ones_like(f)


def refine_masks_eventwise(masks: Dict[str, np.ndarray], hits: List[Hit], labels: np.ndarray,
                           mapping: Dict[int, str], freqs: np.ndarray,
                           time_sigma_frames: int = 3,
                           kick_duck_in_snare: float = 0.6,
                           boost_kick_low: float = 0.3) -> Dict[str, np.ndarray]:
    """
    Reduce kick bleed in snare by ducking snare mask at low freqs around kick hits.
    Optionally boost kick mask in low band around its own events.
    """
    F, T = masks['snare'].shape
    t_grid = np.arange(T)
    low_prior = _freq_prior(freqs, 'kick_low')[:, None]  # (F,1)

    sn = masks['snare']
    kk = masks['kick']

    for h, lab in zip(hits, labels):
        cls = mapping.get(lab)
        if cls != 'kick':
            continue
        w = np.exp(-0.5 * ((t_grid - h.i) / max(1, time_sigma_frames))**2)
        w = w[None, :]  # (1,T)
        # Duck snare low band near kick hits
        sn *= (1.0 - kick_duck_in_snare * (low_prior @ w))
        # Slightly boost kick low band near its own hits (helps mask competition)
        kk *= (1.0 + boost_kick_low * (low_prior @ w))

    # Re-normalize softly to keep masks within [0,1] and sum≈1
    eps = 1e-8
    for cls in ('kick','snare','hats'):
        masks[cls] = np.clip(masks[cls], 0.0, None)
    total = masks['kick'] + masks['snare'] + masks['hats'] + eps
    for cls in ('kick','snare','hats'):
        masks[cls] = masks[cls] / total
    return masks

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Adaptive K/S/H separator with grid-aware clustering")
    ap.add_argument('--audio', required=True, help='Input drum stem (wav/flac/mp3)')
    ap.add_argument('--beats', required=True, help='Text file: one beat time (s) per line')
    ap.add_argument('--outdir', required=True, help='Output directory')
    ap.add_argument('--sr', type=int, default=44100, help='Resample rate')
    ap.add_argument('--hop', type=int, default=256, help='STFT hop length')
    ap.add_argument('--win', type=int, default=1024, help='STFT window (n_fft)')
    ap.add_argument('--snap-ms', type=float, default=60.0, help='Max snap distance to grid (ms)')
ap.add_argument('--subdiv', type=int, default=1, help='Subdivisions per beat for quantization (1=beats, 2=8ths, 4=16ths, 6=triplet 8ths)')
    ap.add_argument('--backtrack', action='store_true', help='Backtrack onsets to nearest peaks (librosa)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load audio
    y, sr = librosa.load(args.audio, mono=True, sr=args.sr)

    # Compute STFT (complex) once for masking
    S = librosa.stft(y, n_fft=args.win, hop_length=args.hop, window='hann', center=True)
    mag = np.abs(S)
    phase = np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=args.win)

    # Onsets
    hits = detect_onsets(y, sr=sr, hop=args.hop, use_aubio=True, backtrack=args.backtrack)

    # Snap to grid (with optional sub-beat quantization)
beats = load_beats(args.beats)
quant_grid = build_subbeat_grid(beats, args.subdiv)
snap_tol = args.snap_ms / 1000.0
snapped: List[Hit] = []
used_frames = set()
for h in hits:
    q_t, _, d = nearest(quant_grid, h.t)
    if d <= snap_tol:
        f = int(round(q_t * sr / args.hop))
        if f in used_frames:
            continue
        used_frames.add(f)
        snapped.append(Hit(t=q_t, i=f, conf=h.conf))
# If too sparse, also include unsnapped onsets
if len(snapped) < max(4, len(hits)//4):
    snapped = hits

if not snapped:
    raise SystemExit("No onsets detected after snapping; try increasing --snap-ms or disabling grid snap.")

    # Features & clustering
    X = extract_features_per_hit(mag, freqs, snapped, hop=args.hop, sr=sr, win_frames=6)
    labels = cluster_hits(X, k=3, seed=args.seed)
    mapping = auto_label_clusters(X, labels)

    # Templates and soft masks
    tpls = build_class_templates(mag, freqs, snapped, labels, mapping, win_frames=6)
    masks = soft_masks_from_templates(mag, tpls, snapped, labels, mapping, time_sigma_frames=3)

    # Apply masks (use mixture phase)
    sep_specs = {cls: masks[cls] * S for cls in masks}
    outs = {cls: librosa.istft(sep_specs[cls], hop_length=args.hop, window='hann', length=len(y)) for cls in sep_specs}

    # Write WAVs
    for cls, sig in outs.items():
        sf.write(os.path.join(args.outdir, f"{cls}.wav"), sig, sr)

    # Export events CSV
    csv_path = os.path.join(args.outdir, "events.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "cluster", "label", "conf"])
        for h, lab in zip(snapped, labels):
            w.writerow([f"{h.t:.6f}", int(lab), mapping.get(lab, 'unk'), f"{h.conf:.4f}"])

    # Export MIDI with 3 tracks
    midi = pm.PrettyMIDI(initial_tempo=120.0)
    prog = dict(kick=115, snare=117, hats=116)  # use percussion channel? we'll just pick distinct programs
    for cls in ('kick', 'snare', 'hats'):
        inst = pm.Instrument(program=0, is_drum=True, name=cls)
        for h, lab in zip(snapped, labels):
            if mapping.get(lab) != cls:
                continue
            t = max(0.0, h.t)
            # short ticks; downstream can map to drum lanes
            inst.notes.append(pm.Note(velocity=100, pitch=36 if cls=='kick' else (38 if cls=='snare' else 42),
                                      start=t, end=t+0.05))
        midi.instruments.append(inst)
    midi_path = os.path.join(args.outdir, "events.mid")
    midi.write(midi_path)

    # Summary
    counts = {cls: 0 for cls in ('kick','snare','hats')}
    for lab in labels:
        counts[mapping.get(lab, 'unk')] = counts.get(mapping.get(lab,'unk'),0)+1
    print("Done.")
    print("WAVs:")
    for cls in ('kick','snare','hats'):
        print("  ", os.path.join(args.outdir, f"{cls}.wav"))
    print("Events CSV:", csv_path)
    print("MIDI:", midi_path)
    print("Counts:", counts)


if __name__ == "__main__":
    main()

# ==========================
# lane_assign_drums.py (module)
# ==========================

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import numpy as np

# ---- Shared event schema ----
@dataclass
class DrumEvent:
    t: float                 # onset time (seconds)
    track: str               # 'kick' | 'snare' | 'hats' (or 'mix')
    strength: float = 1.0    # onset strength 0..1

@dataclass
class DrumLaneSettings:
    num_lanes: int = 3                 # accessibility: 2 for one-hand/limited
    cooldown_s: float = 0.12           # min spacing on a lane (Easy ~0.14, Hard ~0.09)
    max_notes_per_beat_global: float = 3.0
    max_notes_per_beat_lane: float = 2.0
    window_beats: float = 2.0          # sliding window length for density caps
    max_same_lane_streak: int = 6      # anti-boredom; alternate if exceeded
    interleave_when_single_source: bool = True
    # difficulty scaling helpers
    difficulty: str = "Medium"         # "Easy" | "Medium" | "Hard"

# quick presets
DRUM_PRESETS: Dict[str, DrumLaneSettings] = {
    "Easy":   DrumLaneSettings(num_lanes=3, cooldown_s=0.14, max_notes_per_beat_global=2.2, max_notes_per_beat_lane=1.5, window_beats=2.0, max_same_lane_streak=5, difficulty="Easy"),
    "Medium": DrumLaneSettings(num_lanes=3, cooldown_s=0.12, max_notes_per_beat_global=3.0, max_notes_per_beat_lane=2.0, window_beats=2.0, max_same_lane_streak=6, difficulty="Medium"),
    "Hard":   DrumLaneSettings(num_lanes=3, cooldown_s=0.10, max_notes_per_beat_global=4.0, max_notes_per_beat_lane=3.0, window_beats=1.5, max_same_lane_streak=7, difficulty="Hard"),
    # Accessibility: one-hand / two-lane
    "TwoLane":DrumLaneSettings(num_lanes=2, cooldown_s=0.14, max_notes_per_beat_global=2.2, max_notes_per_beat_lane=1.5, window_beats=2.0, max_same_lane_streak=5, difficulty="Easy"),
}


def _apply_cooldown(assignments: List[Tuple[float,int]], cooldown_s: float) -> List[Tuple[float,int]]:
    """Ensure min spacing per lane; drop or shift late arrivals."""
    last_t = {}
    out: List[Tuple[float,int]] = []
    for t, lane in assignments:
        lt = last_t.get(lane, -1e9)
        if t - lt >= cooldown_s:
            out.append((t, lane))
            last_t[lane] = t
        else:
            # try neighbor lane as a quick fallback
            for nl in [lane-1, lane+1]:
                if nl < 0 or nl >= max(lane+1, max(last_t.keys(), default=-1)+1):
                    continue
                lt2 = last_t.get(nl, -1e9)
                if t - lt2 >= cooldown_s:
                    out.append((t, nl))
                    last_t[nl] = t
                    break
            # else drop
    return out


def _cap_density(assignments: List[Tuple[float,int]], beats: np.ndarray, settings: DrumLaneSettings) -> List[Tuple[float,int]]:
    """Slide over windows measured in beats; prune overflow by weakest/most redundant hits."""
    if len(assignments) == 0:
        return assignments
    # make a tempo map (beat index per time)
    def beat_at_time(t):
        i = np.searchsorted(beats, t)
        if i == 0: return 0.0
        if i >= len(beats): return float(len(beats)-1)
        t0, t1 = beats[i-1], beats[i]
        return (i-1) + (t - t0)/max(1e-6, (t1 - t0))

    # sort by time
    assignments = sorted(assignments, key=lambda x: x[0])
    keep = np.ones(len(assignments), dtype=bool)
    # evaluate windows centered at each event
    for idx, (t, lane) in enumerate(assignments):
        b = beat_at_time(t)
        b0, b1 = b - settings.window_beats/2, b + settings.window_beats/2
        # collect events in window
        window_idx = [j for j,(tj, lj) in enumerate(assignments) if b0 <= beat_at_time(tj) <= b1]
        total = sum(1 for j in window_idx if keep[j])
        # global cap
        max_total = math.ceil(settings.max_notes_per_beat_global * settings.window_beats)
        if total > max_total:
            # drop later/weak duplicates in the window (simple heuristic)
            drop_count = total - max_total
            for j in reversed(window_idx):
                if drop_count <= 0: break
                if keep[j]:
                    keep[j] = False
                    drop_count -= 1
        # lane cap
        per_lane_counts: Dict[int,int] = {}
        for j in window_idx:
            if keep[j]:
                per_lane_counts[assignments[j][1]] = per_lane_counts.get(assignments[j][1],0)+1
        for ln, cnt in per_lane_counts.items():
            max_lane = math.ceil(settings.max_notes_per_beat_lane * settings.window_beats)
            if cnt > max_lane:
                need = cnt - max_lane
                # drop from this lane in window (latest first)
                for j in reversed([k for k in window_idx if assignments[k][1]==ln and keep[k]]):
                    if need<=0: break
                    keep[j]=False
                    need -= 1
    return [a for a,k in zip(assignments, keep) if k]


def assign_drums_to_lanes(
    events: List[DrumEvent],
    beats: np.ndarray,
    settings: DrumLaneSettings = DRUM_PRESETS["Medium"],
) -> List[Tuple[float,int,str]]:
    """
    Map drum events to lanes with interleaving and accessibility constraints.
    Returns [(time_s, lane_idx, track_label)].
    """
    num_lanes = max(2, settings.num_lanes)
    # 1) direct map when all present
    lane_for_track = {"kick":0, "snare":1 if num_lanes>1 else 0, "hats":2 if num_lanes>2 else (1 if num_lanes>1 else 0)}

    # seed naive assignments
    assn: List[Tuple[float,int,str]] = []
    for ev in sorted(events, key=lambda e: e.t):
        lane = lane_for_track.get(ev.track, 0)
        assn.append((ev.t, lane, ev.track))

    # 2) interleave when only one source dominates in a local window
    if settings.interleave_when_single_source and len(assn) >= 3:
        W = settings.window_beats
        # simple run-length scan on same track
        run_start = 0
        while run_start < len(assn):
            run_track = assn[run_start][2]
            run_end = run_start+1
            while run_end < len(assn) and assn[run_end][2] == run_track:
                run_end += 1
            run_len = run_end - run_start
            if run_len >= settings.max_same_lane_streak:
                # alternate lanes across the run
                for r, idx in enumerate(range(run_start, run_end)):
                    t, base_lane, tr = assn[idx]
                    assn[idx] = (t, (base_lane + (r % num_lanes)) % num_lanes, tr)
            run_start = run_end

    # 3) per-lane cooldown & density caps
    tl = [(t, lane) for (t,lane,_) in assn]
    tl = _apply_cooldown(tl, settings.cooldown_s)
    tl = _cap_density(tl, beats, settings)

    # stitch track labels back approximately by time
    tl = sorted(tl, key=lambda x: x[0])
    out: List[Tuple[float,int,str]] = []
    j = 0
    for t,lane in tl:
        # best effort: match to nearest original event time
        k = min(range(len(assn)), key=lambda i: abs(assn[i][0]-t))
        out.append((t, lane, assn[k][2]))
    return out


# ===========================
# lane_assign_melody.py (module)
# ===========================

from dataclasses import dataclass
from typing import List, Tuple, Optional
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

MELODY_PRESETS: Dict[str, MelodyLaneSettings] = {
    "Easy":   MelodyLaneSettings(num_lanes=2, cooldown_s=0.14, max_notes_per_beat_global=1.8, max_notes_per_beat_lane=1.2, window_beats=2.0, w_dir=2.0, w_jump=0.8, w_span=1.0, w_speed=1.4, difficulty="Easy"),
    "Medium": MelodyLaneSettings(num_lanes=3, cooldown_s=0.12, max_notes_per_beat_global=2.4, max_notes_per_beat_lane=1.6, window_beats=2.0, w_dir=2.0, w_jump=0.6, w_span=0.8, w_speed=1.2, difficulty="Medium"),
    "Hard":   MelodyLaneSettings(num_lanes=3, cooldown_s=0.10, max_notes_per_beat_global=3.2, max_notes_per_beat_lane=2.2, window_beats=1.5, w_dir=1.6, w_jump=0.5, w_span=0.6, w_speed=1.0, difficulty="Hard"),
    # Accessibility one-hand 2-lane
    "TwoLane":MelodyLaneSettings(num_lanes=2, cooldown_s=0.14, max_notes_per_beat_global=1.8, max_notes_per_beat_lane=1.2, window_beats=2.0, w_dir=2.0, w_jump=0.8, w_span=1.0, w_speed=1.4, difficulty="Easy"),
}


def _norm_pitch_to_lane_target(pitches: np.ndarray, num_lanes: int) -> np.ndarray:
    # z-score within a rolling window approximated by global stats for simplicity
    mu, sigma = float(np.median(pitches)), float(np.std(pitches) + 1e-6)
    z = (pitches - mu) / sigma
    # map to [1, num_lanes]
    return 1.0 + (num_lanes - 1) * (1/(1+np.exp(-z)))  # sigmoid


def assign_melody_to_lanes(
    notes: List[Note],
    beats: np.ndarray,
    settings: MelodyLaneSettings = MELODY_PRESETS["Medium"],
) -> List[Tuple[float,int,int,float]]:
    """
    DP/Viterbi lane mapping that preserves contour and accessibility limits.
    Returns [(time_s, lane, pitch, duration_s)].
    """
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: n.t)
    num_lanes = max(2, settings.num_lanes)

    # cooldown helper state (approximate): last time used per lane
    def cooldown_penalty(t, lane, last_t):
        lt = last_t.get(lane, -1e9)
        gap = t - lt
        if gap >= settings.cooldown_s:
            return 0.0
        # penalize inversely with gap
        return settings.w_speed * (1.0 - min(1.0, gap / settings.cooldown_s))

    P = np.array([n.pitch for n in notes], dtype=float)
    T = np.array([n.t for n in notes], dtype=float)
    D = np.array([n.d for n in notes], dtype=float)
    target = _norm_pitch_to_lane_target(P, num_lanes)

    # DP tables
    N = len(notes)
    C = np.full((N, num_lanes), 1e9, dtype=float)
    B = np.full((N, num_lanes), -1, dtype=int)

    # init (no previous lane)
    for l in range(num_lanes):
        # span cost to target
        C[0, l] = settings.w_span * abs((l+1) - target[0])
        B[0, l] = -1

    # recurrence
    for i in range(1, N):
        dpitch = P[i] - P[i-1]
        for l in range(num_lanes):
            # local cost independent of prev lane
            cost_local = settings.w_span * abs((l+1) - target[i])
            best = (1e9, -1)
            for lp in range(num_lanes):
                cost = C[i-1, lp]
                # direction penalty
                if dpitch > 0 and l < lp:  # went left while pitch went up
                    cost += settings.w_dir
                if dpitch < 0 and l > lp:  # went right while pitch went down
                    cost += settings.w_dir
                # jump penalty
                cost += settings.w_jump * abs(l - lp)
                if cost < best[0]:
                    best = (cost, lp)
            C[i, l] = best[0] + cost_local
            B[i, l] = best[1]

    # backtrack best path
    last_lane = int(np.argmin(C[-1]))
    lanes = [last_lane]
    for i in range(N-1, 0, -1):
        last_lane = int(B[i, last_lane])
        lanes.append(last_lane)
    lanes = lanes[::-1]

    # enforce cooldown and density caps greedily
    assignments = [(notes[i].t, lanes[i]) for i in range(N)]

    # cooldown
    last_t: Dict[int,float] = {}
    keep = [True]*N
    for i,(t,l) in enumerate(assignments):
        lt = last_t.get(l, -1e9)
        if t - lt < settings.cooldown_s:
            # try neighbors
            moved = False
            for nl in [l-1, l+1]:
                if 0 <= nl < num_lanes:
                    lt2 = last_t.get(nl, -1e9)
                    if t - lt2 >= settings.cooldown_s:
                        assignments[i] = (t, nl)
                        lanes[i] = nl
                        last_t[nl] = t
                        moved = True
                        break
            if not moved:
                keep[i] = False
        else:
            last_t[l] = t

    assignments = [a for a,k in zip(assignments, keep) if k]
    kept_notes = [n for n,k in zip(notes, keep) if k]

    # simple density cap by sliding beat window
    def beat_at_time(t):
        i = np.searchsorted(beats, t)
        if i == 0: return 0.0
        if i >= len(beats): return float(len(beats)-1)
        t0,t1 = beats[i-1], beats[i]
        return (i-1) + (t - t0)/max(1e-6, (t1 - t0))

    arr = list(zip(range(len(assignments)), assignments))
    arr.sort(key=lambda x: x[1][0])
    keep2 = [True]*len(arr)
    for idx,(i,(t,l)) in enumerate(arr):
        b = beat_at_time(t)
        b0, b1 = b - settings.window_beats/2, b + settings.window_beats/2
        win = [j for j,(ii,(tt,ll)) in enumerate(arr) if b0 <= beat_at_time(tt) <= b1 and keep2[j]]
        total = len(win)
        if total > math.ceil(settings.max_notes_per_beat_global*settings.window_beats):
            # drop interior low-strength notes first
            win_sorted = sorted(win, key=lambda j: kept_notes[arr[j][0]].strength)
            for j in win_sorted[: total - math.ceil(settings.max_notes_per_beat_global*settings.window_beats) ]:
                keep2[j] = False
        # per-lane cap
        per = {}
        for j in win:
            ln = arr[j][1][1]
            per[ln] = per.get(ln,0)+1
        for ln,cnt in per.items():
            lim = math.ceil(settings.max_notes_per_beat_lane*settings.window_beats)
            if cnt > lim:
                over = [j for j in win if arr[j][1][1]==ln]
                # drop weakest on that lane in window
                over_sorted = sorted(over, key=lambda j: kept_notes[arr[j][0]].strength)
                for j in over_sorted[: cnt - lim]:
                    keep2[j] = False

    final_idx = [arr[j][0] for j,k in enumerate(keep2) if k]
    out: List[Tuple[float,int,int,float]] = []
    for i in sorted(final_idx):
        t,l = assignments[i]
        n = kept_notes[i]
        out.append((t, l, n.pitch, n.d))
    return out

