# Bands Battle Rhythm Gameplay Specification Questions

- **Status:** Interview complete; 12 of 12 questions resolved and reconciled
- **Started:** 2026-08-21
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#41-rhythm-gameplay)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Authoring contract:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)
- **Working record:** [`RHYTHM_GAMEPLAY_WORKING.md`](RHYTHM_GAMEPLAY_WORKING.md)
- **Canonical result:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)

## 1. Interview method

This interview uses four checkpoints of three questions. It does not re-ask
settled GDD decisions. Defaults are proposed so the owner can approve a group or
change only the behavior that matters.

Answers are persisted after each checkpoint. When all twelve questions are
resolved, they are reconciled into canonical `RHYTHM_GAMEPLAY.md`.

## 2. Fixed inherited decisions

- The full song is the encounter clock and never changes speed by difficulty.
- Runtime consumes the approved musical clock, charts, phrases/passages,
  Activity Maps, and candidates from `CONTENT_AUTHORING.md`.
- Rhythm uses three stable inputs and supports Tap, Hold, Repeat, Alternate, and
  Rest. First release excludes chords, swipes, flicks, and drag gestures.
- Mobile uses three large fixed pads; keyboard uses `Z`/`X`/`C`; gamepad uses
  left/bottom/right face buttons. Bindings and physical input interpretation are
  supplied by Input/Settings rather than owned here.
- Perfect/Great/Good/Miss and the Easy/Normal/Hard starting windows are settled
  by GD-07 and GD-08; final numbers remain playtest tuning.
- Every difficulty has equal maximum pre-difficulty contribution per passage.
- The same chart feeds Attack, Defend, and Special. Rhythm reports performance;
  Combat owns conversion into effects.
- Moving, downing, disconnect absence, and authored inactivity do not create
  artificial misses.
- Hold Assist judges the initial press but removes the need to maintain contact.
- Equipment never changes judgments, calibration, chart contents, note density,
  phrase availability, or song speed.

## 3. Question plan

### Checkpoint A — Input matching and holds

#### RG-01 — Tap matching and stray inputs [Resolved]

- **Decision needed:** How does a physical press claim a chart note, and what
  happens when no eligible note exists on that lane?
- **Must resolve:** Nearest-note selection, tie-breaking, one-input/one-note
  consumption, late expiration, wrong-lane presses, stray taps, and contact
  bounce.
- **Owner decision:** A press claims the closest unresolved same-lane note in
  its timing window, with an exact tie going to the earlier note. One press can
  claim at most one note and every note resolves once. An unclaimed note becomes
  a Late Miss when its window expires. A same-lane press just before the early
  edge, but close enough to identify the upcoming note unambiguously, resolves
  that note immediately as an Early Miss; the association limit must not let an
  arbitrary tap erase a distant future note. Wrong-lane presses and presses
  with no plausibly associated note are stray inputs: they do not buffer or
  help a future note and may be retained as private diagnostics. Hardware
  contact bounce is ignored.

#### RG-02 — Rapid and overlapping actions [Resolved]

- **Decision needed:** Which rapid or simultaneous input patterns are legal
  under the no-chord grammar?
- **Must resolve:** Repeat release/repress, alternation, minimum separation,
  one active hold plus other-lane taps, overlapping holds, and notes on a held
  lane.
- **Owner decision:** Repeat notes require release and repress. The authored
  difficulty envelope, rather than a separate runtime throttle, limits legal
  repeat speed. Alternation is legal. One active hold may coexist with taps on
  the other two lanes, which is not treated as a chord. First release excludes
  multiple overlapping holds and new notes on a lane whose hold is active.

#### RG-03 — Hold progress and Hold Assist [Resolved]

- **Decision needed:** How is maintained contribution measured, interrupted, and
  assisted?
- **Must resolve:** Initial grade, progress checkpoints, early release, re-grab,
  endpoint behavior, suspension/cancellation, and reward-neutral Hold Assist.
- **Owner decision:** The initial press receives the timing grade and maintained
  contribution follows the fraction of authored hold duration completed. A
  short, playtest-tuned grace period absorbs accidental contact loss. Releasing
  beyond it ends future contribution without adding another Miss, and the hold
  cannot be re-grabbed. Endpoint release timing is not judged. Movement,
  downing, or another participation suspension ends future hold contribution
  without a Miss. Hold Assist uses the same initial judgment and then maintains
  the hold automatically until its endpoint or a participation suspension; it
  receives identical contribution, rewards, and public treatment.

### Checkpoint B — Scoring groups and semantic output

#### RG-04 — Note and phrase aggregation [Resolved]

- **Decision needed:** How do note grades and hold progress become a phrase-level
  performance result?
- **Must resolve:** Grade weights, missed/partial holds, phrase completion,
  chained passages, and data needed for immediate feedback versus results.
- **Owner decision:** Starting grade factors are Perfect 100%, Great 80%, Good
  50%, and Miss 0%, subject to playtest tuning as balance data. Tap contribution
  applies the grade factor to that event's normalized weight. A hold separates
  its graded initial press from contribution earned over maintained duration so
  the onset grade does not replace sustain performance. A phrase closes after
  its final resolvable event and produces an immediate summary; adjacent
  phrases remain distinct scoring groups even when they chain into one passage.
  Detailed early/late distributions are retained for results and improvement
  guidance rather than combat conversion.

#### RG-05 — Passage normalization [Resolved]

- **Decision needed:** How is performance normalized so role, density,
  difficulty, and sparse material do not change maximum available contribution?
- **Must resolve:** Denominator, authored weights, partial availability,
  suspension exclusion, caps, and deterministic output.
- **Owner decision:** Each playable role and difficulty receives the same fixed
  maximum pre-combat contribution budget for a passage. Approved event weights
  distribute that budget across the chart, so density and raw note count do not
  change its maximum. Earned contribution is capped by the passage budget and
  is deterministic for the same chart revision and performance facts. Material
  skipped because of movement, downing, or another valid suspension creates no
  Miss but still yields no contribution and is not renormalized away. Rhythm
  reports execution quality separately from participation coverage and earned
  passage contribution so suspended time cannot falsify accuracy or become an
  output exploit.

#### RG-06 — Intent and consumer handoff [Resolved]

- **Decision needed:** What exact semantic result does Rhythm send to Combat and
  other consumers, and where does intent switching split attribution?
- **Must resolve:** Per-note versus scoring-group events, queued intent boundary,
  early/late and grade facts, normalized contribution, event identity, and
  prohibition on Rhythm calculating damage or Ward.
- **Owner decision:** Rhythm emits an immediate, uniquely identified semantic
  result for each resolved chart event to feedback consumers and an identified
  aggregate when a phrase closes. A phrase result may contain separate Attack,
  Defend, and Special contribution portions. A queued intent takes effect on the
  next playable beat or note; earlier performance retains its old intent. For a
  hold crossing that boundary, its initial press remains with its original
  intent and maintained contribution is split at the effective boundary.
  Rhythm reports grades, timing direction/delta, maintained fraction, available
  and earned normalized weight, execution quality, coverage, and attribution;
  Combat alone converts those facts into damage, Ward, charge, or other effects.

### Checkpoint C — Suspension, re-entry, and pause

#### RG-07 — Movement and authored suspension [Resolved]

- **Decision needed:** When does chart participation suspend and resume around
  movement, boss transitions, and other authored inactive states?
- **Must resolve:** Commitment boundary, visible-note treatment, crossing holds,
  no-miss intervals, re-entry preview, and next eligible phrase/note.
- **Owner decision:** Rhythm participation suspends when an accepted movement
  begins or an authored inactive state takes effect. Unresolved visible notes
  clear without Misses or contribution and an active hold ends without another
  judgment. After landing and the movement system's settling period, or after
  authored inactivity ends, Rhythm resumes at the first eligible note with
  sufficient preview lead; it need not wait for a phrase boundary. Material
  before that fair re-entry point remains skipped without backfill.

#### RG-08 — Downing, recovery, disconnect, and rejoin [Resolved]

- **Decision needed:** How does Rhythm respond to survival/network state changes
  without falsifying performance history?
- **Must resolve:** Immediate stop, unresolved notes, prior contribution,
  recovery challenge handoff, settling protection, safe-boundary rejoin, and
  absence evidence.
- **Owner decision:** Downing immediately suspends the player's ordinary chart,
  clears unresolved notes without Misses, and leaves prior resolved history
  intact. Revival or solo recovery is requested only after the live need exists;
  the Activity Map selects the earliest fair candidate rather than relying on a
  pre-mapped timestamp. Challenge performance remains distinct from ordinary
  chart contribution. Recovery, reconnect, and permitted inactive-player return
  resume after their owning protection/settling rules at the first eligible note
  with adequate preview. No absent interval is backfilled, judged, or awarded.

#### RG-09 — Solo pause and resume [Resolved]

- **Decision needed:** What freezes in solo, how resumption is counted in, and
  where pausing is prohibited?
- **Must resolve:** Audio/clock freeze, open judgments/holds, countdown, safe
  musical boundary, exploit prevention, and difference from multiplayer pause.
- **Owner decision:** Solo pause takes effect immediately and freezes the song,
  encounter clock, open judgment windows, hold progress, and all clock-driven
  encounter behavior. The pause surface hides upcoming notes and attacks and
  offers no timeline inspection. Resume gives a phase-aligned visible and
  audible beat countdown, then continues at the exact frozen musical time. A
  held input must be re-established by countdown end or ordinary release grace
  begins at resume. Countdown taps do not judge chart notes. Cooperative
  encounters cannot pause their shared song.

### Checkpoint D — Calibration, synchronization, and resilience

#### RG-10 — Calibration application and clock authority [Resolved]

- **Decision needed:** How does a saved device/control offset affect judgments
  while every gameplay system still shares one encounter clock?
- **Must resolve:** Input timestamp adjustment, visual offset, audio offset,
  offset changes mid-song, bounds, and what remains unshifted.
- **Owner decision:** A saved device/control profile adjusts input timestamps
  and visual cue alignment against the player's perceived audio. It never moves
  the approved song, encounter clock, chart event times, boss events, or audio
  itself. Raw and adjusted timing facts remain distinguishable. Profile changes
  made during an encounter apply to the next encounter. Offsets are bounded;
  extreme values prompt recalibration rather than expanding judgment windows.

#### RG-11 — Multiplayer synchronization from the player's perspective [Resolved]

- **Decision needed:** What does each player experience when local playback and
  server encounter time differ?
- **Must resolve:** Local immediate judgment, authoritative event identity,
  drift correction, teammate independence, late/duplicate messages, and no
  visible rollback of an accepted input.
- **Owner decision:** Each player receives immediate local feedback against a
  synchronized local representation of the shared song clock. Stable event and
  encounter identities let the server validate shared consequences and ignore
  duplicates. Ordinary network delay does not change or visibly roll back the
  displayed grade, and one player's lag never changes another player's chart.
  Small drift converges gradually outside critical strike moments without
  changing authored tempo or crossing active judgment windows.

#### RG-12 — Desync recovery and output completeness [Resolved]

- **Decision needed:** What happens when timing confidence is lost, and which
  events must Rhythm expose for UI, Audio, Results, and Analytics?
- **Must resolve:** Soft correction versus suspension, resync boundary, player
  feedback, invalid-session evidence, semantic event catalog, accessibility
  invariants, and completion audit.
- **Owner decision:** When timing confidence becomes unreliable, Rhythm
  suspends scoring immediately and clears unresolved notes without Misses or
  contribution while the shared multiplayer song continues. A clear syncing
  state replaces ordinary chart feedback. After clock/content confidence is
  restored, play resumes at the first eligible note with fair preview. Prior
  confirmed results remain. Rhythm exposes identified note, hold, phrase,
  passage, intent-attribution, participation, pause, calibration, synchronization,
  suspension, and resume facts to its consumers. Accessibility presentation may
  change cues and physical effort but never these semantic results.

## 4. Completion criteria

`RHYTHM_GAMEPLAY.md` is complete only when:

- RG-01 through RG-12 are resolved;
- input matching is deterministic for every legal chart pattern;
- stray, suspended, downed, disconnected, and inactive states cannot fabricate
  misses or contribution;
- difficulty and role normalization preserve equal maximum passage output;
- Hold Assist preserves gameplay value without changing initial judgment;
- consumers receive semantic performance facts without Rhythm owning combat;
- solo pause and multiplayer synchronization have explicit player-facing rules;
  and
- every new authored-data need is registered for Content Authoring
  reconciliation.

## 5. Change log

- **2026-08-21:** Created the concise 12-question plan from the approved GDD,
  systems map, and Content Authoring contract.
- **2026-08-21:** Resolved RG-01 through RG-03. Clarified that an identifiable
  premature same-lane attempt can resolve the upcoming note as an Early Miss.
- **2026-08-21:** Resolved RG-04 through RG-06, establishing tunable grade
  factors, equal passage budgets, separate execution/coverage measures, and
  beat-boundary intent attribution.
- **2026-08-21:** Resolved RG-07 through RG-09, establishing fair no-miss
  suspension and re-entry, dynamically selected recovery timing, absence
  handling, and exact-time solo pause/resume.
- **2026-08-21:** Resolved RG-10 through RG-12 and reconciled all twelve answers
  into canonical `RHYTHM_GAMEPLAY.md`.
