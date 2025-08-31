#!/usr/bin/env bash
# Usage: ./drum_onsets_to_midi.sh kick_onsets.csv snare_onsets.csv hats_onsets.csv out.mid [BPM] [PPQ] [HOLD_MS]
KCSV="$1"; SCSV="$2"; HCSV="$3"; OUT="$4"
BPM="${5:-120}"; PPQ="${6:-480}"; HOLD="${7:-50}"  # 50ms note length

if [ -z "$OUT" ]; then
  echo "Usage: $0 kick.csv snare.csv hats.csv out.mid [BPM] [PPQ] [HOLD_MS]" >&2; exit 1
fi
command -v csvmidi >/dev/null || { echo "csvmidi (midicsv) not found"; exit 1; }

tmp=$(mktemp)

awk -v BPM="$BPM" -v PPQ="$PPQ" -v HOLD="$HOLD" '
function sec_to_ticks(sec){ return int(sec * PPQ * BPM / 60.0 + 0.5) }
function dur_ticks(){ return int((HOLD/1000.0) * PPQ * BPM / 60.0 + 1) }
function header(){
  print  "0, 0, Header, 1, 2, " PPQ
  print  "1, 0, Start_track"
  printf "1, 0, Tempo, %d\n", int(60000000 / BPM)
  print  "1, 0, Program_c, 9, 1"  # Channel 10 (index 9) is standard drums; Program ignored for drums
}
function emit_note(ticks, pitch){
  vel=100; off=ticks + dur_ticks()
  printf "1, %d, Note_on_c, 9, %d, %d\n", ticks, pitch, vel
  printf "1, %d, Note_off_c, 9, %d, 0\n", off, pitch
  if (off>last) last=off
}
BEGIN{ header() }
FNR==1 { next } # skip CSV header rows Sonic Annotator writes
FILENAME==kick && NF>=1 { t=sec_to_ticks($1); emit_note(t,36); next }
FILENAME==snare && NF>=1{ t=sec_to_ticks($1); emit_note(t,38); next }
FILENAME==hats && NF>=1 { t=sec_to_ticks($1); emit_note(t,42); next }
END{ printf "1, %d, End_track\n0, 0, End_of_file\n", (last?last:0)+1 }
' kick="$KCSV" snare="$SCSV" hats="$HCSV" "$KCSV" "$SCSV" "$HCSV" > "$tmp"

csvmidi "$tmp" "$OUT" >/dev/null 2>&1 && echo "Wrote $OUT (BPM=$BPM)"
rm -f "$tmp"
