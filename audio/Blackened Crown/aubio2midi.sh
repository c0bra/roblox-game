#!/usr/bin/env bash
# Usage: ./aubio2midi.sh notes.txt out.mid [BPM]
# notes.txt lines: PITCH START_SEC END_SEC  (floats)
# Requires: midicsv (csvmidi)

IN="$1"
OUT="$2"
BPM="${3:-120}"            # default tempo
PPQ=480                    # pulses per quarter note (MIDI ticks per beat)

if [ -z "$IN" ] || [ -z "$OUT" ]; then
  echo "Usage: $0 notes.txt out.mid [BPM]" >&2
  exit 1
fi

CSV="$(mktemp)"
awk -v PPQ="$PPQ" -v BPM="$BPM" -v VEL=100 '
function tticks(sec) { return int(sec * PPQ * BPM / 60 + 0.5) }
BEGIN {
  # MIDI header track
  printf "0, 0, Header, 1, 2, %d\n", PPQ
  print  "1, 0, Start_track"
  printf "1, 0, Tempo, %d\n", int(60000000 / BPM)
  print  "1, 0, Program_c, 0, 1"   # set a basic instrument (Acoustic Grand)
}
# Expect: pitch start end
NF>=3 {
  pitch = int($1 + 0)
  t_on  = tticks($2)
  t_off = tticks($3)
  if (t_off <= t_on) t_off = t_on + 1
  printf "1, %d, Note_on_c, 0, %d, %d\n", t_on,  pitch, VEL
  printf "1, %d, Note_off_c, 0, %d, 0\n",  t_off, pitch
  if (t_off > last) last = t_off
}
END {
  printf "1, %d, End_track\n", (last + 1)
  print  "0, 0, End_of_file"
}
' "$IN" > "$CSV"

csvmidi "$CSV" "$OUT"
rm -f "$CSV"
echo "Wrote $OUT"

