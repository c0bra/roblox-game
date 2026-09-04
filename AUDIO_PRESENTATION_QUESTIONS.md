# Bands Battle Audio Presentation Specification Questions

- **Status:** Interview complete; 12 of 12 questions resolved and reconciled
- **Started:** 2026-08-31
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#77-audio-presentation)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Content dependency:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)
- **Rhythm dependency:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)
- **Encounter dependency:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Abilities dependency:** [`ABILITIES_AND_COOPERATIVE_ACTIONS.md`](ABILITIES_AND_COOPERATIVE_ACTIONS.md)
- **UI/settings dependency:** [`UI_UX.md`](UI_UX.md)
- **Working record:** [`AUDIO_PRESENTATION_WORKING.md`](AUDIO_PRESENTATION_WORKING.md)
- **Canonical result:** [`AUDIO_PRESENTATION.md`](AUDIO_PRESENTATION.md)

## 1. Interview method

This interview uses four checkpoints of three questions. It begins with stable
song playback and personal performance response, then resolves shared semantic
cues, device/accessibility mixing, and the complete authoring/runtime contract.

Answers are persisted after each checkpoint. When all twelve questions are
resolved, they are reconciled into canonical `AUDIO_PRESENTATION.md`. Exact
gain/EQ/compression values, asset formats, Roblox implementation, and final sound
design remain downstream unless a boundary is necessary for fairness or clarity.

## 2. Fixed inherited decisions

- The approved full song is the encounter clock. Difficulty never changes song
  speed, duration, pitch, structure, or musical identity.
- Every player hears the complete song at a stable backing level. The selected
  playable role gains additional local performance response; ordinary errors
  never silence the song.
- Drums, vocals, guitar, and bass are initial examples, not a closed roster.
  Piano, synthesizer, percussion, strings, and other authentic song-specific
  roles may use one or more controllable layers or a human-approved equivalent.
- A playable role requires independently controllable audio or an approved
  equivalent aligned with its chart. Sparse, atmospheric, or absent parts are
  neither fabricated nor forced into the roster.
- Perfect receives the strongest crisp/clear response, Great a confident normal
  accent, Good a softer response, and Miss a brief duck/filtered stumble without
  silence. Movement returns to neutral backing without a Miss sound.
- Downing makes the local selected role muffled/distant while the complete song
  continues; successful recovery returns it through an on-beat swell.
- Performance response is primarily local. One player's weak play does not
  damage teammates' song mixes. Teammates receive meaningful shared combat and
  group-event audio instead of every local judgment.
- Shared crowd, arena, and combat response follows identified aggregate band
  events. Band Calls and Crescendos widen/strengthen the ensemble. Acolytes add
  fixed support presentation but never fake instrument performance or scores.
- Priority is critical boss/timing cues, local selected role/judgment response,
  core song, then other combat effects, crowd, and ambience. Critical cues may
  duck nonessential effects but never remove the audible song pulse.
- Critical cues use rhythm, register, timbre/envelope, source identity, and
  multimodal reinforcement rather than volume alone. They remain useful on
  phone speakers, in mono, and without deep bass or stereo separation.
- Repeated note hits strengthen the performed role rather than layering an
  unrelated noisy effect on every input.
- Automatic gameplay cues cannot be muted with player pings. Ping muting never
  hides attacks, targeting, movement, downing, revival, group actions, or phase
  changes.
- Solo pause freezes song and encounter exactly. Resume uses a separate visible/
  audible beat count and continues the approved song from the frozen instant.
- UI/UX defines setting values/profile scope and renders captions. Audio applies
  master, song, local role, timing/boss cue, voice, combat, crowd, ambience,
  dynamic-range, and related outputs without changing gameplay semantics.
- Caption/subtitle metadata identifies speaker or sound source. Every critical
  audio fact has visual or optional haptic reinforcement; accessibility settings
  never alter difficulty, rewards, public identity, or maximum contribution.
- The existing browser prototype's role-specific `backing` plus `stem` assets are
  useful evidence, not a closed runtime schema or the canonical source of truth.

## 3. Question plan

### Checkpoint A - Song playback, controllable layers, and local response

#### AP-01 - Runtime song/layer model and neutral full-mix reference

- **Status:** Resolved 2026-09-01.
- **Decision needed:** How does each client reproduce the approved complete song
  while retaining responsive control over the selected playable role?
- **Must resolve:** Master/reference mix, backing bed, one-or-more local layers,
  arbitrary roles, role/audio map, neutral reconstruction, missing/dirty stems,
  equivalent controllable treatment, duplicate roles, phase alignment, loudness,
  and nonplayable/source-only layers.

#### AP-02 - Judgment, hold, suspension, downing, and recovery response

- **Status:** Resolved 2026-09-01.
- **Decision needed:** How does authentic local performance alter the selected
  role without becoming noisy, misleading, or musically destructive?
- **Must resolve:** Perfect/Great/Good/Miss envelopes, early/late, repeated notes,
  rapid overlap, holds and early release, zero input, movement/other suspension,
  downed/recovery/return, response bounds, role-specific character, local-only
  privacy, no rollback, and no fabricated onset.

#### AP-03 - Start, clock alignment, pause, drift, rejoin, and playback failure

- **Status:** Resolved 2026-09-01.
- **Decision needed:** How does the audible mix remain aligned to the one musical
  clock through loading, countdown, pause, synchronization change, and rejoin?
- **Must resolve:** Preload/readiness, shared future start, phase-locked layers,
  time source, local latency, pause/resume count, output-device change, drift/
  correction, unsafe confidence, disconnect/rejoin fade, end/stop, critical
  asset failure, and no audible tempo/pitch manipulation.

### Checkpoint B - Critical cues, shared events, and world response

#### AP-04 - Critical cue taxonomy, priority, masking, and ducking

- **Status:** Resolved 2026-09-01.
- **Decision needed:** Which audible cues are protected and how do they remain
  distinct when music, combat, players, and several threats overlap?
- **Must resolve:** Timing, Telegraph/Commit/Impact/Recovery, targeting, position/
  movement, survival/recovery, group, terminal/system cues, priority tiers,
  rhythmic/register/timbre identity, ducking, pulse protection, repeat/rate
  behavior, simultaneous impacts, distance, and missing-cue handling.

#### AP-05 - Combat, group actions, acolytes, and aggregate band audio

- **Status:** Resolved 2026-09-01.
- **Decision needed:** How do personal and shared combat events sound powerful
  without leaking private performance or multiplying noise with roster size?
- **Must resolve:** Attack/Defend/Special, Ward/down/revival, Resolve/Momentum,
  Signature, Band Call, Crescendo tiers, consumables, risk/avoidance, acolyte
  functions/suppression, human versus fixed contribution, aggregation/roster
  scaling, duplicate roles, source/target, commitment, cancellation, and caps.

#### AP-06 - Hub, onboarding, Results, dialogue, pings, crowd, and ambience

- **Status:** Resolved 2026-09-01.
- **Decision needed:** How does the broader experience use sound for orientation,
  learning, feedback, and atmosphere without competing with core music or
  becoming required communication?
- **Must resolve:** Hub landmarks/restoration/shards, menu transitions, practice/
  calibration/count-in, Results/rewards/unlocks, dialogue/voice, subtitles,
  preset pings/muting, crowd/ambience responsiveness, scene transitions, optional
  versus critical content, focus/background behavior, and age-appropriate tone.

### Checkpoint C - Buses, devices, accessibility, captions, and haptics

#### AP-07 - Bus catalog, player settings, loudness, and dynamic range

- **Status:** Resolved 2026-09-01.
- **Decision needed:** Which mix buses and presets provide useful control without
  letting players accidentally remove an essential cue?
- **Must resolve:** Master/song/local role/timing-boss/voices/combat/crowd/
  ambience buses, nesting, defaults/ranges/mute, cue floors/alternatives,
  compression/quiet/night profiles, loudness/headroom/limiting, ducking sidechain,
  preview/apply timing, per-device scope, and reset.

#### AP-08 - Phone, headphones, mono, spatialization, and output profiles

- **Status:** Resolved 2026-09-01.
- **Decision needed:** How does the same semantic mix remain clear across weak
  phone speakers, headphones, desktop, stereo/mono, and device changes?
- **Must resolve:** midrange/transient strategy, bass/stereo independence, mono
  fold-down, spatial versus nonspatial sources, source direction, distance/
  occlusion, dialogue, headroom, device/output detection, Bluetooth/latency
  change, profile suggestions, fallback, and representative-device targets.

#### AP-09 - Caption/source metadata, accessible alternatives, and haptic requests

- **Status:** Resolved 2026-09-01.
- **Decision needed:** Which metadata lets UI present every important audible
  fact accessibly and lets devices reinforce it without creating another rhythm
  channel?
- **Must resolve:** subtitle versus sound caption, speaker/source/direction,
  criticality, timing/duration, repetition/coalescing, musical description,
  localization, text/background settings, non-speech alternatives, haptic event/
  strength/duration/rate, reduced-haptics, platform absence, privacy, and no
  sound-only or haptic-only meaning.

### Checkpoint D - Cue definitions, authoring, performance, and completeness

#### AP-10 - Audio definition, event lifecycle, concurrency, and idempotency

- **Status:** Resolved 2026-09-01.
- **Decision needed:** Which fields and runtime rules make every song response
  and semantic cue deterministic rather than ad hoc client sound effects?
- **Must resolve:** Stable identity/revision, source event, class/priority,
  asset/variant selection, musical/exact boundary, local/shared scope, bus,
  spatial/source/target, envelope, ducking, concurrency/polyphony, cooldown/
  coalescing, commit/cancel/stop, fallback, caption/haptic keys, deduplication,
  and private attribution.

#### AP-11 - Asset/package validation, streaming, budgets, and degradation

- **Status:** Resolved 2026-09-01.
- **Decision needed:** How are audio assets proven aligned, complete, performant,
  and safely degradable before an encounter can start or publish?
- **Must resolve:** Formats/transcodes, duration/sample alignment, loop/seek,
  loudness/peak/noise/phase, layer reconstruction, cue coverage, preload/stream,
  memory/voice/decoder/network budgets, cache, low-device profiles, decorative
  degradation, missing assets, platform substitution/equivalence, and review.

#### AP-12 - Semantic outputs, test matrix, and Content Authoring reconciliation

- **Status:** Resolved 2026-09-01.
- **Decision needed:** Which facts, catalogs, validators, evidence, and handoffs
  make Audio Presentation implementation-ready without inventing sound behavior?
- **Must resolve:** Mix/layer/cue/caption/haptic state outputs, consumer requests,
  settings/persistence handoff, privacy/analytics, role/difficulty/roster/event/
  device/accessibility matrix, objective and human listening gates, failure tests,
  browser/prototype evidence, Content Authoring register, deferred values, and
  final completion audit.

## 4. Completion criteria

`AUDIO_PRESENTATION.md` is complete only when:

- AP-01 through AP-12 are resolved;
- every playable role reconstructs the approved song and supports bounded local
  response without assuming a conventional four-instrument roster;
- local judgments, holds, suspensions, downing, and return are musical,
  deterministic, private, and never silence or retime the song;
- critical semantic cues remain distinct, prioritized, mono/phone compatible,
  multimodal, and protected from optional mix/mute behavior;
- shared combat/group/world audio communicates aggregate meaning without public
  ranking, fake performance, roster-scaled noise, or player-miss leakage;
- buses, dynamic range, devices, captions, alternatives, and haptics honor UI/
  accessibility settings without changing gameplay;
- every cue definition and runtime state has deterministic identity, lifecycle,
  concurrency, fallback, and deduplication; and
- assets, streaming/budgets, validation, outputs, tests, and Content Authoring
  requirements leave no implementation-agent design choice.

## 5. Change log

- **2026-08-31:** Created the concise 12-question Audio Presentation interview
  from the approved GDD, Systems Map, canonical gameplay/UI specifications,
  Content Authoring baseline, and existing browser backing/stem evidence.
- **2026-09-01:** Approved AP-01 through AP-03, completing neutral role-layer
  reconstruction, local performance response, and clock/playback lifecycle
  checkpoint A. Progress is 3 of 12 questions.
- **2026-09-01:** Approved AP-04 through AP-06, completing protected critical
  cues, aggregate combat/group sound, and hub/onboarding/Results/world audio
  checkpoint B. Progress is 6 of 12 questions.
- **2026-09-01:** Approved AP-07 through AP-09, completing bus/dynamic-range,
  device/mono/spatial, and caption/haptic accessibility checkpoint C. Progress
  is 9 of 12 questions.
- **2026-09-01:** Approved AP-10 through AP-12, completing cue lifecycle,
  asset/budget/degradation, and semantic-output/verification checkpoint D. All
  twelve questions were reconciled into canonical `AUDIO_PRESENTATION.md`.
