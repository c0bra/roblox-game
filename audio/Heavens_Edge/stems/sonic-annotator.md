sonic-annotator -d vamp:beatroot-vamp:beatroot:beats \
  --force -w csv --csv-omit-filename --csv-one-file drum_beats.csv 1_heavens_edge_\(Drums\).mp3


sonic-annotator -d vamp:beatroot-vamp:beatroot:beats \
  --force -w midi 1_heavens_edge_\(Drums\).mp3

sonic-annotator -d vamp:pyin:pyin:notes \
  --force -w midi "input_(Vocals)_htdemucs.wav"

sonic-annotator -d vamp:pyin:pyin:notes \
  --force -w csv "input_(Vocals)_htdemucs.wav"

sonic-annotator -d vamp:vamp-aubio:aubioonset:onsets \
  --force -w midi "input_(Bass)_htdemucs.wav"

sonic-annotator -d vamp:vamp-aubio:aubioonset:onsets \
  --force -w csv "input_(Bass)_htdemucs.wav"
g
python crepe_to_notes_quant.py "input_(Vocals)_htdemucs.wav" --grid-csv drum_beats.csv --out vocal_notes.mid

python pyin_csv_to_midi_quant.py "input_(Vocals)_htdemucs_vamp_pyin_pyin_notes.csv" drum_beats.csv pyin_notes.mid


## Testing

python pyin_bass_shift_to_midi.py "input_(Bass)_htdemucs.wav" bass_notes.mid --semitones 24



## Other commands

sonic-annotator -d vamp:beatroot-vamp:beatroot:beats \
  -w csv --csv-omit-filename --csv-one-file blackened_crown_drum_beats.csv \
  --force "1_Blackened Crown(1)_(Drums).mp3"


python kick_snare_hat_separator.py \
  --audio "blackend_crown_drums.wav" \
  --beats "blackened_crown_drum_beats.csv" \
  --outdir ./drums --subdiv 4

