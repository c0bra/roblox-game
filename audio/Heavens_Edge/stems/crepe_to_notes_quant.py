# save as crepe_to_notes_quant.py
import argparse, numpy as np, librosa, csv
import pretty_midi as pm
import crepe

def hz_to_midi(hz): 
    return 69 + 12*np.log2(np.maximum(hz,1e-6)/440.0)

def load_grid_csv(path):
    # CSV with a column of beat times in seconds (header or not)
    times=[]
    with open(path) as f:
        r=csv.reader(f)
        for row in r:
            try:
                t=float(row[0])
                times.append(t)
            except: pass
    return np.asarray(times, float)

def load_grid_from_midi(midipath):
    m=pm.PrettyMIDI(midipath)
    # Use tempo changes to compute beat times on a dense grid (quarter notes)
    beats=m.get_beats()  # seconds
    return np.asarray(beats, float)

def nearest(grid, t):
    i=np.searchsorted(grid, t)
    cand=[max(0,i-1), min(len(grid)-1,i)]
    j=min(cand, key=lambda k: abs(grid[k]-t))
    return grid[j], j, abs(grid[j]-t)

def segment_notes(times, f0_hz, conf, 
                  min_conf=0.3, 
                  min_len_s=0.08, 
                  pitch_jump_cents=80, 
                  smooth_win=7):
    # mask unvoiced
    voiced = conf >= min_conf
    # smooth f0 in semitones
    midi = hz_to_midi(f0_hz)
    midi_sm = librosa.decompose.nn_filter(midi[np.newaxis,:], aggregate=np.median, metric='cosine', width=smooth_win)[0]
    # segment by voiced gaps or large pitch jumps
    segs=[]
    start=None
    for i in range(1,len(times)):
        new_seg=False
        if not voiced[i-1] and voiced[i]:
            start=i
        if voiced[i-1] and not voiced[i]:
            new_seg=True
        if voiced[i] and voiced[i-1]:
            if abs(midi_sm[i]-midi_sm[i-1])> pitch_jump_cents/100.0:
                new_seg=True
        if new_seg and start is not None:
            end=i
            if times[end]-times[start] >= min_len_s:
                pitch=np.median(midi_sm[start:end])
                segs.append((times[start], times[end], int(round(pitch))))
            start=None
    # tail
    if start is not None and (times[-1]-times[start])>=min_len_s:
        pitch=int(round(np.median(midi_sm[start:])))
        segs.append((times[start], times[-1], pitch))
    return segs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--grid-csv", help="CSV with beat times (s)")
    ap.add_argument("--grid-midi", help="MIDI to derive beat grid from")
    ap.add_argument("--snap-ms", type=float, default=120)
    ap.add_argument("--out", default="vocal_notes.mid")
    ap.add_argument("--fixed-vel", type=int, default=96)
    args=ap.parse_args()

    y, sr = librosa.load(args.audio, mono=True, sr=None)
    # CREPE: returns times (s), freqs (Hz), confidence [0-1]
    time, freq, conf, _ = crepe.predict(y, sr, viterbi=True, step_size=10)  # 10 ms frames

    segs = segment_notes(time, freq, conf,
                         min_conf=0.25,       # looser for vocals
                         min_len_s=0.06,      # keep short syllables
                         pitch_jump_cents=60, # split when pitch shifts
                         smooth_win=9)

    # Optional quantization to your temp grid
    if args.grid_csv or args.grid_midi:
        grid = load_grid_csv(args.grid_csv) if args.grid_csv else load_grid_from_midi(args.grid_midi)
        maxd = args.snap_ms/1000.0
        q_segs=[]
        for (s,e,p) in segs:
            qs,_,ds = nearest(grid, s)
            qe,_,de = nearest(grid, e)
            s2 = qs if ds<=maxd else s
            e2 = max(s2 + 0.02, qe if de<=maxd else e)
            q_segs.append((s2,e2,p))
        segs=q_segs

    # Emit MIDI
    m=pm.PrettyMIDI(initial_tempo=120)
    inst=pm.Instrument(program=0, name="VocalNotes")
    vel=max(1,min(127,args.fixed_vel))
    for s,e,p in segs:
        inst.notes.append(pm.Note(velocity=vel, pitch=int(p), start=float(s), end=float(e)))
    m.instruments.append(inst)
    m.write(args.out)
    print(f"Wrote {len(segs)} notes -> {args.out}")

if __name__=="__main__":
    main()