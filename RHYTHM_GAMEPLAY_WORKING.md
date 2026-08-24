# Bands Battle Rhythm Gameplay Working Record

- **Status:** Complete decision record; 12 of 12 questions reconciled
- **Started:** 2026-08-21
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#41-rhythm-gameplay)
- **Interview plan:** [`RHYTHM_GAMEPLAY_QUESTIONS.md`](RHYTHM_GAMEPLAY_QUESTIONS.md)
- **Authoring dependency:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)
- **Canonical result:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)

## 1. Role of this record

This document persists owner decisions while the Rhythm Gameplay interview is in
progress. It is not canonical until reconciled into `RHYTHM_GAMEPLAY.md`.

## 2. Inherited boundary

Rhythm owns the player-facing musical clock, chart playback, legal rhythm-action
interpretation, input-to-note matching, judgments, hold progress, scoring groups,
pre-combat normalized contribution, chart suspension/re-entry, application of
calibration and Hold Assist, and solo pause/resume timing.

It does not own chart authoring, physical device bindings, saved settings,
combat conversion, boss state, runtime audio mixing, results presentation, or
analytics storage.

## 3. Approved inputs

The system consumes an exact approved content revision containing musical clock,
role/difficulty chart, canonical-event lineage, phrase/passage identities,
Activity Map/candidates, and encounter timing. It also receives encounter state,
selected role/difficulty, semantic physical actions, calibration/Hold Assist
settings, and accessibility requirements from their owning systems.

## 4. Decision record

### Checkpoint A — Input matching and holds

#### RG-01 — Tap matching and stray inputs

- **Status:** Approved.
- A physical press first considers unresolved notes on the same logical lane.
  It claims the closest note whose timing window contains the adjusted press
  time; an exact distance tie goes to the earlier note.
- One press claims at most one note and one note can resolve only once.
- An unresolved note becomes a Late Miss when its late edge expires.
- A same-lane press before the early edge can resolve the upcoming note
  immediately as an Early Miss when it is close enough that the intended note
  is unambiguous. The tunable association range may not reach through a nearer
  note or allow an arbitrary tap to erase a distant future note.
- A wrong-lane press or a press with no plausibly associated note is a stray. It
  neither creates damage nor buffers, claims, or improves a future note. It may
  be retained privately for diagnostics and playtest analysis.
- Hardware-generated contact bounce is filtered and does not count as another
  physical attempt.

#### RG-02 — Rapid and overlapping actions

- **Status:** Approved.
- Repeat requires a release and a new press. Legal repeat speed is constrained
  by the authored difficulty envelope; Rhythm does not add an unrelated
  runtime rate limit.
- Alternating lanes rapidly is legal.
- One active hold may coexist with taps on either other lane. This combination
  is sequential lane activity, not a forbidden chord.
- First release excludes multiple simultaneously active holds and excludes a
  new note on the lane of an active hold. Content validation must reject those
  chart patterns.

#### RG-03 — Hold progress and Hold Assist

- **Status:** Approved.
- A hold's initial press receives the ordinary timing grade. Its continuing
  contribution is proportional to the authored duration actually maintained.
- A short, globally defined and playtest-tuned release grace period ignores
  accidental contact loss. If release persists beyond grace, future hold
  contribution stops; this does not create a second Miss and the ended hold
  cannot be re-grabbed.
- Release at the authored endpoint is not a separate timing judgment.
- Movement, downing, or another participation suspension ends future hold
  contribution without fabricating a Miss.
- Hold Assist leaves the initial timing judgment unchanged and automatically
  maintains an accepted hold through its endpoint unless participation is
  suspended. Assisted holds have the same possible contribution, rewards, and
  public presentation as manually maintained holds.

### Checkpoint B — Scoring groups and semantic output

#### RG-04 — Note and phrase aggregation

- **Status:** Approved.
- Starting grade factors are Perfect 100%, Great 80%, Good 50%, and Miss 0%.
  They are balance data and may change through playtesting without changing the
  four-grade semantic contract.
- A Tap earns its event weight multiplied by its grade factor.
- A Hold divides its available weight into an onset portion and a duration
  portion. The initial press grades only the onset. Maintained contribution is
  earned over authored duration, preventing either part from making the other
  irrelevant. The exact onset/duration weighting is playtest-tunable.
- A phrase remains open until every event belonging to it has resolved,
  including the relevant portion of a hold crossing its boundary. It then emits
  one immediate phrase summary.
- Adjacent phrases remain distinct scoring groups when chained into a passage;
  chaining creates no forced downtime.
- Immediate feedback consumes per-event grade, timing direction, and hold
  state. Detailed early/late distributions and improvement guidance are
  aggregate result facts and do not alter combat conversion.

#### RG-05 — Passage normalization

- **Status:** Approved.
- Every passage has one fixed maximum pre-combat contribution budget. For every
  role with playable material, approved event weights sum to that same maximum
  on Easy, Normal, and Hard. Raw note count and density do not alter the budget.
- A sparse but genuinely playable role may distribute the budget over fewer
  authentic events; it is not padded with fabricated notes.
- Earned contribution is the sum of resolved event contribution, capped at the
  passage budget. The same content revision and performance facts always
  produce the same result.
- Movement, downing, and other valid participation suspensions create no Miss,
  but their skipped material earns zero contribution. The system does not
  renormalize the remaining notes upward, so suspending cannot preserve full
  output.
- Rhythm keeps three different facts distinct:
  - **Execution quality:** accuracy over material actually judged while active.
  - **Participation coverage:** the share of the passage budget during which
    the player was eligible and participating.
  - **Earned contribution:** normalized value earned against the full passage
    budget, including zero output for legitimately skipped material.
- A role with no authentic playable material in a passage is inactive there;
  it does not receive a fake full-budget opportunity or a performance penalty.

#### RG-06 — Intent and consumer handoff

- **Status:** Approved.
- Every resolved event has stable content identity plus encounter-instance and
  player identity so consumers can order and deduplicate semantic results.
- The immediate event result includes phrase/passage identity, role,
  difficulty, adjusted timing delta and early/late direction, grade, available
  and earned weight, maintained fraction where applicable, participation state,
  and effective combat intent.
- Phrase closure emits an identified aggregate containing available and earned
  normalized weight, execution quality, participation coverage, and separate
  Attack/Defend/Special portions when attribution changed within the phrase.
- A queued intent becomes effective on the next playable beat or note. Events
  before that boundary retain the prior intent and are never reinterpreted.
- A hold's onset remains attributed to the intent active when it was pressed.
  Maintained contribution on a crossing hold is split exactly at the new
  intent's effective musical boundary.
- Rhythm publishes semantic performance facts to Combat, UI, Audio, Results,
  and later Analytics. It never calculates damage, Ward, healing, ability
  charge, or any other combat effect.

### Checkpoint C — Suspension, re-entry, and pause

#### RG-07 — Movement and authored suspension

- **Status:** Approved.
- Rhythm suspends participation when an accepted dash begins and when an
  encounter-owned authored inactive state takes effect. The owning system
  decides whether movement or encounter state is valid; Rhythm applies the
  supplied boundary.
- Every unresolved visible note inside a suspension clears without a Miss and
  earns no contribution. An active hold ends at the boundary, retaining earned
  onset and duration value but earning nothing afterward and receiving no extra
  judgment.
- The staff visually withdraws or marks the interruption so skipped material
  cannot look like player failure. Previously resolved events remain immutable.
- After landing and the movement system's settling period, or when authored
  inactivity ends, Rhythm chooses the first eligible note whose preview meets
  the minimum readable lead time. Re-entry does not have to wait for a phrase
  boundary.
- Notes before that fair re-entry point are skipped without Misses, contribution,
  or later backfill. If no remaining note in the current phrase can be previewed
  fairly, re-entry naturally advances to the next eligible material.

#### RG-08 — Downing, recovery, disconnect, and rejoin

- **Status:** Approved.
- Downing immediately suspends the ordinary chart and ends any active hold at
  the state boundary. Unresolved and incoming ordinary notes clear without
  Misses. Resolved contribution and performance history remain valid.
- A cooperative revival or solo recovery request exists only when triggered by
  live state. The Activity Map selector chooses the earliest candidate valid for
  the current song position, role, difficulty, roster, phase, and conflicts; it
  is not a fixed revival timestamp authored in advance.
- A solo recovery or universal-beat challenge is an explicitly identified
  temporary scoring stream. Its results go to the recovery consumer and never
  masquerade as ordinary passage contribution or accuracy.
- Teammates contributing to revival continue to perform their authentic chart;
  Rhythm labels the applicable normalized result for the revival consumer while
  Combat or Cooperative Actions owns its conversion and group accumulation.
- After successful recovery, the player's ordinary chart remains suspended
  through the owning protection/settling interval, then returns at the first
  eligible note with adequate preview.
- Disconnect absence creates neither Misses nor contribution. A permitted
  reconnect or inactive-player resume first confirms content/clock readiness,
  then returns at a safe note with adequate preview. Nothing during absence is
  replayed, backfilled, or newly awarded, and prior accepted results remain.

#### RG-09 — Solo pause and resume

- **Status:** Approved.
- Solo pause freezes immediately: master song time, encounter time, notes and
  judgment windows, maintained holds, telegraphs, impacts, timers, and other
  clock-driven encounter behavior all retain their exact relative state.
- The pause surface hides upcoming chart material and boss attacks and provides
  no timeline scrub or other way to inspect the frozen future.
- Release or contact changes while paused do not judge notes. Before resumption,
  a visible and audible beat countdown runs without advancing encounter time or
  accepting Tap judgments.
- The countdown is phase-aligned to the frozen musical position. At its end,
  song and encounter resume from the exact frozen instant rather than skipping
  to another boundary.
- A held input must be physically re-established by countdown end. Otherwise
  its normal release grace begins when gameplay resumes; pause itself never
  grants or removes hold contribution.
- Cooperative encounters never freeze the shared song. Their pause/menu surface
  must state that limitation and treat the local player through ordinary
  multiplayer inactivity rules if participation stops.

### Checkpoint D — Calibration, synchronization, and resilience

#### RG-10 — Calibration application and clock authority

- **Status:** Approved.
- The saved calibration profile is selected by device/control profile. It
  supplies a bounded input-timing correction and visual-alignment correction
  relative to the player's perceived audio.
- Rhythm applies the input correction before matching a physical press to a
  chart event. It retains both the raw timestamp and adjusted timing delta for
  diagnostics without exposing raw device latency as player failure.
- Visual presentation may shift by the approved correction, but the master song
  clock, audio playback position, chart event time, encounter event time,
  scoring budget, and judgment-window width remain unchanged.
- Calibration made during an encounter is saved for the next encounter. It does
  not reinterpret already resolved events or change alignment partway through a
  song.
- Allowed offsets are bounded balance/configuration data. An extreme requested
  value produces a private recalibration warning rather than an advantage,
  larger timing window, or shifted boss timeline.

#### RG-11 — Multiplayer synchronization from the player's perspective

- **Status:** Approved.
- Each player hears and performs against a synchronized local representation of
  the shared encounter clock. Input feedback is judged locally and displayed
  immediately so ordinary round-trip latency is not part of the rhythm action.
- Stable content-event, encounter-instance, player, and result identities let
  the server validate, order, and deduplicate the semantic performance used for
  shared combat consequences.
- Ordinary transport delay cannot change or visibly roll back an already shown
  grade. Duplicate or late copies of an accepted result do not apply its shared
  effect twice.
- Every player has an independent chart and judgment stream. Another player's
  latency, correction, suspension, or weak performance cannot shift local note
  timing, audio, or grades.
- Small clock drift uses gradual presentation convergence outside active strike
  windows. It may not change authored tempo, move through an unresolved timing
  window, or reinterpret resolved events.

#### RG-12 — Desync recovery and output completeness

- **Status:** Approved.
- When clock, content-revision, or event-order confidence falls below the safe
  threshold, Rhythm suspends scoring immediately. Unresolved notes clear with no
  Miss and no contribution. In multiplayer, the shared song and encounter
  continue for everyone else.
- The affected player receives an obvious, nonpunitive synchronization state
  instead of ordinary judgment feedback. Once clock/content confidence returns,
  the chart resumes at the first eligible note with minimum preview lead.
- Confirmed earlier results remain immutable. The uncertain interval is recorded
  as synchronization suspension rather than poor execution. If recovery cannot
  complete within the operational limit, Rhythm marks the affected interval or
  session timing-invalid and hands control to Multiplayer; it does not invent
  results.
- Rhythm's semantic output catalog covers clock readiness, event resolution,
  hold progress/end, phrase and passage lifecycle/results, intent attribution,
  participation/coverage state, suspension cause, fair re-entry, pause/resume,
  calibration profile application, drift state, synchronization loss/recovery,
  and timing invalidation.
- UI, Audio, Results, and Analytics receive the same underlying semantic facts
  at the aggregation level they need. Shape/label, sound, optional haptics,
  Hold Assist, and other accessible presentation may change how facts are
  conveyed or an input is maintained, never the resulting grade or value.

## 5. Content Authoring reconciliation register

- First-release chart validation must reject overlapping holds and notes placed
  on the lane of an active hold.
- Difficulty-density validation must account for physical release/repress
  feasibility in repeat patterns.
- Each role/difficulty chart must provide deterministic event weights that sum
  to the common passage budget wherever that role has authentic playable
  material.
- Content must retain exact musical boundaries and crossing-hold lineage so
  runtime can close phrases and split sustained contribution deterministically.
- Activity Map/runtime chart data must expose enough eligible-note lookahead to
  enforce a consistent minimum preview lead after any live suspension.
- Required recovery validation must prove an eligible candidate exists within
  its allowed delay for every supported role, difficulty, and roster; revival
  remains dynamically selected rather than placed at a fixed timestamp.
- Runtime content must expose stable event/phrase/passage identities and an
  immutable content revision so multiplayer validation and deduplication cannot
  confuse results across charts or revisions.

## 6. Open handoffs

- `COMBAT.md` will consume normalized performance and intent attribution.
- `BOSS_ENCOUNTERS.md` will supply encounter-active/suspended state and consume
  musical-boundary facts.
- `UI_UX.md` will present preview, judgments, early/late direction, phrase
  summaries, calibration, pause/resume, and desync feedback.
- `AUDIO_PRESENTATION.md` will consume clock and judgment events without owning
  their semantics.
- `MULTIPLAYER.md` and later technical architecture will define transport and
  authority while preserving the player-facing synchronization rules decided
  here.

## 7. Change log

- **2026-08-21:** Created the working record. Progress is 0 of 12 questions.
- **2026-08-21:** Approved RG-01 through RG-03. Progress is 3 of 12 questions.
  Added Early Miss resolution for an identifiable premature attempt.
- **2026-08-21:** Approved RG-04 through RG-06. Progress is 6 of 12 questions.
  Established grade factors, equal passage budgets, separate execution and
  coverage facts, and intent-segmented semantic output.
- **2026-08-21:** Approved RG-07 through RG-09. Progress is 9 of 12 questions.
  Established no-miss suspension/re-entry, dynamic recovery selection,
  reconnect handling, and exact-time solo pause/resume.
- **2026-08-21:** Approved RG-10 through RG-12 and reconciled all twelve
  decisions into canonical `RHYTHM_GAMEPLAY.md`.
