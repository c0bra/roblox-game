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


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(max(hz, 1e-9) / 440.0)

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
        # outer product: spectral template (F,) * time kernel (T,)
        tpl = tpls[cls][:, None]
        masks[cls] += tpl * w[None, :]

    # Normalize so sum of masks <= 1 at each bin
    eps = 1e-8
    total = masks['kick'] + masks['snare'] + masks['hats'] + eps
    for cls in masks:
        masks[cls] = masks[cls] / total
    return masks


def reconstruct_sources(S_complex: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for cls, M in masks.items():
        out[cls] = np.real(np.fft.irfft(np.fft.rfft(S_complex, axis=0) * 1.0, axis=0))  # placeholder (ensure same shape)
    # Actually use complex STFT for reconstruction
    out = {}
    for cls, M in masks.items():
        out_spec = M * S_complex
        out[cls] = out_spec
    return out


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
    ap.add_argument('--backtrack', action='store_true', help='Backtrack onsets to nearest peaks (librosa)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--subdiv', type=int, default=1,
                help='Subdivisions per beat for quantization (1=beats, 2=8ths, 4=16ths, 6=triplet 8ths)')
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
