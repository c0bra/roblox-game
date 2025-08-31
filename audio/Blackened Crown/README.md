ffmpeg -y -i 1_Blackened\ Crown\(1\)_\(Vocals\).mp3 -ac 1 -ar 44100 \
  -af "highpass=f=80,lowpass=f=5000,afftdn=nf=-25, dynaudnorm" \
  clean.wav

ffmpeg -y -i 1_Blackened\ Crown\(1\)_\(Drums\).mp3 -af "dynaudnorm" drums_norm.wav

./aubio_clean2midi.sh notes.txt cleaned.mid 128 480 4 80 48 84
./aubio_clean2midi.sh notes.txt cleaned.mid 128 480 8 15 0 127 0
./aubio_clean2midi.sh snared_notes.txt snare.mid 128 480 8 15 0 127 0
./aubio_clean2midi.sh drum_notes.txt drum.mid 128 480 8 15 0 127 0

# Drums

ffmpeg -y -i 1_Blackened\ Crown\(1\)_\(Drums\).mp3 -af "highpass=f=150,lowpass=f=3000,volume=2" snare.wav
ffmpeg -y -i 1_Blackened\ Crown\(1\)_\(Drums\).mp3 -af "lowpass=f=150,volume=2" kick.wav
ffmpeg -y -i 1_Blackened\ Crown\(1\)_\(Drums\).mp3 -af "highpass=f=5000,volume=3" hats.wav


./stems_to_midi.py --kick kick.wav --snare snare.wav --hats hats.wav \
  -o drums.mid --bpm 128 --ppq 480 --hold-ms 60 --vel 105 \
  --min-sep-ms 45 --quant-div 4 --delta 0.18
