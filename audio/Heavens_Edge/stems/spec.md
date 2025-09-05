# Lane Assignment Pipeline Specification

## Overview
This project converts raw audio stems (vocals, bass, drums) into **3-lane (or 2-lane accessibility) note charts** for a rhythm game.  
It involves:

1. **Audio preprocessing**  
   - Stem separation (kick/snare/hats, vocals, bass).
   - Onset/pitch extraction (pYIN, Aubio, madmom).
   - Beat/tempo tracking.

2. **Quantization**  
   - Align notes to beat or sub-beat grid (nearest or next).
   - Ensure minimum duration and monophonic overlap rules.

3. **Lane assignment**  
   - Drums: deterministic mapping (kick/snare/hats) with interleaving when collapsed.  
   - Melody (vocals/bass): dynamic programming (DP/Viterbi) to preserve melodic contour.  
   - Apply cooldowns, density limits, and difficulty scaling.

4. **Post-processing**  
   - Condense simultaneous notes per lane.  
   - Prune excessive density based on difficulty.  
   - Export CSV/MIDI for use in game engine.

---

## Functional Requirements

### Input
- **Audio stems**: `.wav` or `.mp3`.  
- **Beat times**: text file with one float (seconds) per line.  
- **pYIN CSV**: `time, duration, frequency` (Hz).  
- **Aubio onsets**: optional `time` file for percussion.  

### Output
- **Note assignments**: CSV of `(time_s, lane, pitch, dur_s)`.  
- **Optional MIDI**: with instrument programs per track.  
- **Logs/metrics**: quantization accuracy, dropped notes count.

---

## Modules

### 1. Audio → Notes
- **Vocals/Bass**:  
  - Run `pYIN` (Vamp plugin) → `time, duration, frequency`.  
  - Convert frequency → MIDI pitch.  
  - Strength default = 1.0 (confidence may come from pYIN).  

- **Drums**:  
  - Separate with Spleeter/Demucs (kick/snare/hats).  
  - Run Aubio/madmom onset detection (HFC for percussion).  
  - Tag each onset with track = kick/snare/hats.  

### 2. Quantization
- Build sub-beat grid from beat times (`subdiv = 1..4`).  
- Snap note starts/ends:
  - **Start** → nearest beat.  
  - **End** → nearest beat (or next ≥ start).  
- Enforce:
  - `min_dur_s = 0.02–0.08` depending on instrument.  
  - Monophonic non-overlap (shift start or drop).  

### 3. Lane Assignment

#### Drums
- Direct map: `kick→0, snare→1, hats→2` (or `0/1` if 2 lanes).  
- Interleave when only 1–2 tracks present:
  - Sliding window, rotate lanes to avoid monotony.  
- Apply rules:
  - **Cooldown**: ≥90–120 ms between same-lane hits.  
  - **Max streak**: after 5–6 hits, force lane alternation.  
  - **Density cap**: per-beat limit per lane & global.

#### Melody
- Use DP/Viterbi with costs:
  - `w_dir`: penalty if lane motion contradicts pitch direction.  
  - `w_jump`: penalty for large lane jumps.  
  - `w_span`: distance from target lane (pitch-mapped percentiles).  
  - `w_inertia`: penalty for staying put when contour is clear.  
  - `streak_penalty`: discourage long runs on one lane.  
- Accessibility: presets for 2 lanes (`TwoLane`).  
- Post-pass: greedy cooldown enforcement (reassign to neighbor or drop).

### 4. Post-Processing
- Condense simultaneous notes per lane (choose longest duration).  
- Prune overflow notes based on difficulty:  
  1. Drop duplicates < `min_ioi` apart.  
  2. Drop weakest (low strength/confidence).  
  3. Always keep downbeats and run anchors.  

### 5. Export
- **CSV** (preferred for Roblox pipeline):  
```
time_s,lane,pitch,dur_s
7.836735,1,62,0.348299
```
- **MIDI** (for debugging in DAW).  

---

## Difficulty Presets

### Drums
- Easy: cooldown 140 ms, max_notes_per_beat_lane=1.5  
- Medium: cooldown 120 ms, max_notes_per_beat_lane=2.0  
- Hard: cooldown 100 ms, max_notes_per_beat_lane=3.0  
- TwoLane: same as Easy but 2 lanes.

### Melody
- Easy: 2 lanes, cooldown 140 ms, stricter density caps.  
- Medium: 3 lanes, cooldown 120 ms.  
- Hard: 3 lanes, cooldown 100 ms, lighter penalties.
- TwoLane: 2 lanes, same as Easy.

---

## Automation Requirements (for AI Tool)

1. **Environment Setup**
 - Python ≥3.10  
 - `librosa`, `aubio`, `madmom`, `pretty_midi`, `numpy`, `scipy`.  
 - Vamp host (Sonic Annotator) for `pYIN`.  
 - Ensure consistent NumPy compatibility (patch `np.float`, `np.bool` if needed).

2. **Pipeline Orchestration**
 - Step 1: Generate stems with Demucs.  
 - Step 2: Run `pYIN` → vocal/bass CSV.  
 - Step 3: Run Aubio → drum onsets.  
 - Step 4: Quantize → grid (from tempo).  
 - Step 5: Lane assignment (use presets).  
 - Step 6: Condense/prune.  
 - Step 7: Export CSV/MIDI.  

3. **Configurable Parameters**
 - `subdiv` (default=4).  
 - `snap_ms` tolerance.  
 - Difficulty level.  
 - Number of lanes (2/3).  
 - Export format(s).  

4. **Validation**
 - No overlapping notes in same lane.  
 - Note density within difficulty caps.  
 - Logs: dropped notes count, average IOI per lane.  

---

## Deliverables
- `lane_assign_drums.py` — drum assignment module.  
- `lane_assign_melody.py` — melody assignment module.  
- `quantize.py` — note quantization helpers.  
- `pipeline.py` — orchestrator script to process stems → lanes.  
- `spec.md` — this document.  
- Example data + tests:
- Input: short pYIN CSV + beat grid.  
- Output: CSV/MIDI with valid lane assignments.  
