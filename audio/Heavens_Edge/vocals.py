import pretty_midi

BPM = 120
GRID_DIV = 4   # 1/16th notes
MIN_DUR = 0.12 # seconds

pm = pretty_midi.PrettyMIDI("vocals.mid")
out = pretty_midi.PrettyMIDI(initial_tempo=BPM)
inst = pretty_midi.Instrument(program=0)

sec_per_beat = 60.0 / BPM
grid = sec_per_beat / GRID_DIV

for note in pm.instruments[0].notes:
    dur = note.end - note.start
    if dur < MIN_DUR:  # ignore tiny blips
        continue
    # snap to grid
    s = round(note.start / grid) * grid
    e = max(s+grid, round(note.end / grid) * grid)
    # bin pitch (optional: 3 lanes)
    if note.pitch < 65: pitch = 60
    elif note.pitch < 72: pitch = 64
    else: pitch = 67
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=s, end=e))

out.instruments.append(inst)
out.write("vocals_simple.mid")
