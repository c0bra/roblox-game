# Bands Battle Rhythm Gameplay

- **Status:** Approved
- **Approved:** 2026-08-21
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#41-rhythm-gameplay)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Authoring dependency:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)
- **Decision source:** [`RHYTHM_GAMEPLAY_WORKING.md`](RHYTHM_GAMEPLAY_WORKING.md)
- **Interview plan:** [`RHYTHM_GAMEPLAY_QUESTIONS.md`](RHYTHM_GAMEPLAY_QUESTIONS.md)

## 1. Role and authority

This document defines how an approved role/difficulty chart and semantic player
actions become deterministic musical judgments and normalized pre-combat
performance. It keeps all players aligned to the full-song encounter clock while
preserving immediate local response, fair suspension, calibration, solo pause,
and multiplayer recovery behavior.

Rhythm owns chart playback, action interpretation, input-to-note matching,
Perfect/Great/Good/Miss, hold progress, scoring groups, passage normalization,
intent attribution as a performance fact, chart suspension and fair re-entry,
calibration application, Hold Assist behavior, solo pause timing, and semantic
performance output.

Rhythm does not own:

- creation or approval of charts and Activity Maps;
- physical device bindings or persistence of player settings;
- Attack/Defend/Special selection validity or combat-effect conversion;
- boss, movement, Ward, downed, ability, roster, reward, or analytics state;
- audio mixing or results-screen presentation; or
- client/server implementation architecture and transport.

Owning systems supply those states or consume Rhythm's semantic facts. A system
boundary here is a design contract, not a required Roblox module layout.

## 2. Governing invariants

1. **One approved musical clock:** the full song is the encounter timeline and
   never changes speed by difficulty.
2. **Authentic chart material:** runtime plays the exact approved content
   revision; it never invents ordinary instrument notes.
3. **Immediate, deterministic judgment:** the same chart revision, adjusted
   action times, and participation state produce the same results.
4. **One action, at most one note:** neither retries, duplicate messages, nor
   contact bounce can multiply a result.
5. **No artificial misses:** movement, downing, disconnect absence, authored
   inactivity, pause, and synchronization suspension never masquerade as weak
   execution.
6. **Suspension still costs opportunity:** skipped material earns no output and
   is not renormalized away.
7. **Equal passage ceiling:** every playable role and difficulty has the same
   maximum pre-combat passage contribution.
8. **Rhythm reports; Combat converts:** grade and normalized contribution never
   directly calculate Resolve pressure, Ward, charge, healing, or damage here.
9. **Accessibility is value-neutral:** accessible input maintenance and cue
   presentation never change the underlying grade, maximum, reward treatment,
   or public result.
10. **Resolved history is immutable:** state changes and synchronization recovery
    do not reinterpret an accepted event.

## 3. Runtime inputs

Rhythm consumes an immutable approved content revision containing:

- song, encounter, schema, and content-revision identity;
- the human-approved exact-time and musical-position clock;
- selected role/difficulty chart with stable event identities and lane/timing;
- Tap/Hold/Repeat/Alternate/Rest semantics and hold duration;
- canonical-event lineage and approved normalized event weights;
- phrase and passage identities and boundaries;
- Activity Map facts and recovery/re-entry candidates; and
- encounter timing and authored active/inactive intervals.

Runtime-owned inputs include the encounter instance, player, selected role and
difficulty, semantic physical actions, queued/effective combat intent,
movement/encounter/survival/connection participation state, saved calibration
and Hold Assist settings, and accessibility presentation requirements.

Rhythm rejects a mismatched or incompatible content revision explicitly rather
than attempting to combine identities or timing from different packages.

## 4. Clock and playable lifecycle

The approved song clock relates exact elapsed time to measure, beat, and
subdivision. Chart events, phrase boundaries, intent boundaries, boss events,
pauses, and synchronization evidence refer to that clock. Presentation may
predict or offset local rendering, but those adaptations do not create a second
gameplay timeline.

A player's ordinary chart has these semantic participation states:

- **Preparing:** content and clock are loading or synchronization confidence is
  not yet sufficient.
- **Previewing:** the next eligible material is visible with enough lead time,
  but its judgment window has not opened.
- **Participating:** actions may claim notes and earn contribution.
- **Suspended:** ordinary chart events produce neither judgments nor output for
  a known cause such as movement, downing, authored inactivity, disconnect, AFK,
  recovery, input-device unavailability, or synchronization loss.
- **Paused:** solo-only exact-time freeze of the song and encounter.
- **Complete:** no further ordinary material exists for this player.

Transitions record their exact musical boundary and cause. Merely reaching the
end of a phrase does not suspend a chained passage.

## 5. Action grammar and matching

First release uses three stable logical lanes and supports:

- **Tap:** one press for one indicated event;
- **Hold:** a judged starting press followed by maintained contact;
- **Repeat:** release and repress the same lane for successive events;
- **Alternate:** successive events across different lanes; and
- **Rest:** intentional absence of input.

First release excludes chords, swipes, flicks, drag gestures, simultaneous
overlapping holds, and a new event on a lane whose hold is active. One active
hold may coexist with taps on the other two lanes; this is not a chord.

Legal repeat speed comes from the authored difficulty complexity envelope. The
runtime does not add a separate arbitrary repeat throttle. Contact-bounce
filtering removes hardware-generated duplicates without suppressing intentional
release/repress actions.

### Press-to-note resolution

For a semantic lane press:

1. Apply the saved bounded timing correction to obtain adjusted action time.
2. Consider unresolved events on the same logical lane.
3. If one or more timing windows contain the action, claim the closest event.
4. If two are exactly equidistant, claim the earlier event.
5. Resolve at most one event and prevent that event from resolving again.

An unclaimed event resolves as a Late Miss when its late edge expires. A press
just before the early edge may resolve the upcoming same-lane event immediately
as an Early Miss when it lies inside the tunable association range and is the
unambiguous intended target. That range may not cross a nearer unresolved event
or let an arbitrary tap erase a distant future note.

A wrong-lane press or a press with no plausibly associated event is a stray. It
does not buffer, improve, claim, or damage a future note. Private diagnostics may
retain it for playtesting and contact-quality analysis.

## 6. Judgments and immediate feedback

The first playtest windows are:

| Difficulty | Perfect | Great | Good | Miss |
|---|---:|---:|---:|---:|
| Easy | ±90 ms | ±150 ms | ±230 ms | Beyond ±230 ms |
| Normal | ±60 ms | ±110 ms | ±170 ms | Beyond ±170 ms |
| Hard | ±45 ms | ±85 ms | ±135 ms | Beyond ±135 ms |

These are tuning data, not immutable release constants. Equipment, builds, gear,
encounter pressure, and rewards never modify them.

Starting contribution factors are Perfect 100%, Great 80%, Good 50%, and Miss
0%. These factors are also playtest-tunable while preserving the four-grade
semantic contract and monotonic ordering.

On resolution, the local player receives immediate pad/staff response,
instrument response, and a compact grade. Great and Good show early/late
direction; Perfect needs no direction. Early Miss and Late Miss may remain
distinguishable in private/result evidence while sharing nonshaming player-facing
Miss language. Critical feedback combines shape/label with color and may add
restrained optional haptics.

## 7. Holds and Hold Assist

A hold divides its normalized available weight into:

- an **onset portion**, earned by the initial press grade; and
- a **duration portion**, earned according to the authored duration maintained.

The exact onset/duration allocation is tuning data. A short global release grace
period ignores accidental contact loss. Release beyond grace ends future
duration contribution without another Miss. An ended hold cannot be re-grabbed,
and release at the authored endpoint is not separately judged.

Movement, downing, authored suspension, or another participation boundary ends
future hold contribution at that boundary while preserving contribution already
earned.

Hold Assist uses the same initial press and grade, then maintains an accepted
hold automatically until its endpoint or a participation suspension. It has the
same maximum contribution, rewards, analytics semantics, and public treatment as
manual maintenance. Private diagnostics may distinguish the setting only for
accessibility quality testing, never performance comparison.

## 8. Phrases, passages, and normalization

A phrase is the immediate readability, scoring, and combat group. Adjacent
phrases may chain without downtime into a passage. A phrase closes only after
all of its resolvable event portions are settled, including the applicable
portion of a boundary-crossing hold, then emits one summary.

Every passage defines one fixed maximum pre-combat contribution budget. For each
role with authentic playable material, approved event weights sum to that same
budget on Easy, Normal, and Hard. Sparse material may distribute the budget over
fewer authentic events; it is never padded with fabricated actions. A role with
no authentic playable material is inactive for that passage and receives
neither a fake opportunity nor a performance penalty.

For a Tap, earned weight is event weight multiplied by grade factor. Holds add
their graded onset and maintained-duration portions. Total earned contribution
is deterministic and capped at the passage budget.

Rhythm keeps three measures separate:

- **Execution quality:** earned judgment quality over material actually judged
  while participating.
- **Participation coverage:** share of the passage opportunity during which the
  player was eligible and participating.
- **Earned contribution:** normalized value earned against the full passage
  budget, including zero output for legitimately skipped material.

Suspension creates no Miss and does not lower execution quality by pretending
the player attempted those notes. It also does not renormalize remaining notes
upward, so movement, absence, or synchronization loss cannot retain full output.

Detailed early/late distributions belong to Results and private improvement
guidance. Combat consumes normalized contribution, not raw note count or raw
timing deltas.

## 9. Intent attribution and consumer boundary

Attack, Defend, and Special use the same chart. Rhythm receives intent state and
labels earned performance; it neither decides whether Special is available nor
converts an intent portion into an effect.

A queued intent becomes effective on the next playable beat or note. Earlier
resolved material retains its prior intent. A hold's onset remains assigned to
the intent effective when pressed; duration contribution crossing the effective
boundary is divided exactly at that musical boundary.

A phrase result may therefore contain multiple intent portions, each with its
own available and earned normalized weight. Switching never regrades or creates
notes and never reinterprets resolved events.

## 10. Semantic result contract

Every result carries stable content revision, chart event or scoring-group,
encounter instance, player, role, and difficulty identity as applicable. This
supports ordering and deduplication without prescribing network transport.

### Event-level facts

An event resolution exposes:

- event, phrase, and passage identity;
- adjusted timing delta and early/late direction;
- Perfect/Great/Good/Miss and Early/Late Miss evidence where applicable;
- available and earned normalized weight;
- hold onset, maintained fraction, end cause, and assisted/manual equivalence;
- effective intent and any intent split; and
- participation state and suspension cause at resolution.

### Group-level facts

Phrase and passage output exposes:

- lifecycle boundary and completion state;
- available, judged, skipped, and earned normalized weight;
- execution quality and participation coverage;
- Attack/Defend/Special contribution portions;
- early/late distribution for eligible Results/Analytics consumers; and
- invalid, suspended, or incomplete evidence without fabricating a score.

UI and Audio consume immediate semantic facts. Combat consumes identified
normalized portions. Results consumes aggregates and timing distributions.
Analytics observes the same contract later without becoming gameplay authority.

## 11. Movement and authored suspension

An accepted dash suspends chart participation when movement begins. An
encounter-owned inactive state suspends at its supplied boundary. Unresolved
visible notes clear without Misses or contribution, the staff communicates that
play has intentionally withdrawn, and an active hold stops earning at the
boundary.

After landing plus the movement system's settling period, or after authored
inactivity ends, Rhythm selects the first eligible note whose preview meets the
minimum readable lead. Re-entry need not wait for a phrase boundary. Earlier
material remains skipped with no backfill. If nothing remaining in the current
phrase can be previewed fairly, re-entry advances to the next eligible material.

Rhythm reports incomplete phrase/coverage facts. Combat and Rewards decide any
separate consequences such as losing an unbanked positional Risk Bonus.

## 12. Downing, recovery, absence, and return

Downing immediately suspends the ordinary chart, ends an active hold, and clears
unresolved/incoming ordinary notes without Misses. Resolved history remains.

Cooperative revival and solo emergency recovery are triggered by live state.
The Activity Map selector chooses the earliest candidate valid for the current
song position, role, difficulty, roster, phase, and conflicts; revival is not a
fixed event timestamp authored in advance.

A solo recovery or universal-beat challenge is an explicitly identified
temporary scoring stream. Its output goes only to the recovery consumer and does
not appear as ordinary passage accuracy or contribution. A teammate performing
for cooperative revival continues using authentic ordinary chart material;
Rhythm labels the applicable normalized result while Combat or Cooperative
Actions owns its conversion and accumulation.

After successful recovery, ordinary participation stays suspended through the
owning protection/settling interval, then returns at the first fairly previewed
eligible note.

Disconnect and permitted AFK return follow the same no-miss/no-backfill rule.
Re-entry first confirms exact content revision and clock confidence, then uses a
safe note with adequate preview. Absence earns no contribution, prior accepted
history remains, and a player who left while downed returns downed.

If a cooperative player has no usable input profile after active-device loss,
UI/UX requests an identified input-unavailable suspension. It creates no Misses
or contribution, does not pause shared song time, records lost participation
coverage, and returns only after a usable profile exists and this section's fair
preview rule succeeds. Solo instead uses the exact pause/resume contract.

## 13. Solo pause and resume

Solo pause freezes immediately:

- master song and encounter time;
- chart presentation and open judgment windows;
- hold progress;
- boss telegraphs, impacts, timers, and other clock-driven encounter behavior.

The pause surface hides upcoming notes and attacks and offers no timeline scrub.
Contact changes while paused do not judge notes. Resume runs a visible and
audible, phase-aligned beat countdown without advancing encounter time or
accepting Tap judgments, then continues from the exact frozen instant.

A held lane must be re-established by countdown end. Otherwise ordinary release
grace begins at resume. Pause itself adds or removes no contribution.

Multiplayer cannot freeze the shared song. Its menu explains that limitation;
stopped participation follows ordinary multiplayer inactivity rules.

## 14. Calibration

Calibration is saved per device/control profile by Player Data or Settings.
Rhythm applies its bounded input correction before note matching and applies its
visual-alignment correction against perceived audio. It retains raw and adjusted
timing evidence for private diagnostics.

Calibration never changes:

- the approved song or encounter clock;
- audio playback or chart event times;
- encounter/boss event times;
- judgment-window widths, grade factors, or passage budgets; or
- difficulty, progression, rewards, or equipment effects.

A profile change during an encounter applies to the next encounter and never
reinterprets resolved play. Extreme requested offsets prompt private
recalibration rather than expanding timing leniency or shifting the boss
timeline. Final bounds and calibration statistics require device playtesting.

## 15. Multiplayer synchronization and resilience

Each player performs against a synchronized local representation of the shared
encounter clock and receives immediate local judgment feedback. Stable content,
event, encounter, player, and result identities let the server validate, order,
and deduplicate shared consequences. The later technical architecture defines
the authority and transport implementation.

Ordinary round-trip delay does not alter or visibly roll back an already shown
grade. Duplicate or late copies cannot apply a combat effect twice. Every
player's chart/judgment stream is independent: another player's latency,
suspension, calibration, or errors do not shift local song timing or grades.

Small drift converges gradually outside active strike windows without changing
authored tempo, moving through an unresolved judgment window, or reinterpreting
history.

When clock, content-revision, or event-order confidence becomes unsafe:

1. Suspend scoring immediately.
2. Clear unresolved notes with no Miss and no contribution.
3. Show a clear, nonpunitive synchronization state.
4. Keep the shared multiplayer song/encounter running for unaffected players.
5. Restore play at the first eligible note with minimum preview after confidence
   returns.

Confirmed earlier results remain. The uncertain span records synchronization
suspension instead of poor execution. If confidence cannot recover within an
operational threshold, Rhythm marks the affected span/session timing-invalid and
hands control to Multiplayer rather than inventing results.

## 16. Accessibility invariants

- Lanes and judgments use shape and label as well as color.
- Critical timing and synchronization states have visual plus audio or optional
  haptic reinforcement.
- Haptics remain optional and restrained.
- Hold Assist changes only physical maintenance after the judged onset.
- Calibration corrects alignment rather than widening difficulty.
- Accessible presentation never changes chart contents, grade, normalized
  maximum, combat attribution, rewards, or public treatment.

## 17. Content Authoring reconciliation register

These entries were reconciled into
[`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md#14-cross-specification-handoffs-and-reconciliation)
on 2026-09-02. The table remains the Rhythm-owned source for implementation and
publication evidence; its support-status column describes work still required,
not an unresolved design boundary.

| Rhythm requirement | Semantic data | Required validation | Consumers | Compatibility/support status |
|---|---|---|---|---|
| Stable runtime identities | Immutable revision plus event, phrase, passage, role, and difficulty identity | Uniqueness, lineage, and no cross-revision collisions | Rhythm, Multiplayer, Combat, Results | Reconciled; implementation must preserve runtime granularity |
| Legal first-release patterns | Lane/action type, hold spans, repeat timing | Reject overlapping holds, events on a held lane, and physically infeasible release/repress density | Rhythm, authoring UI | Reconciled; explicit validators required |
| Equal passage budget | Per-event normalized weights for every playable role/difficulty | Weights sum to the common budget; no density or difficulty changes the ceiling | Rhythm, Combat, Results | Reconciled; export field and validator required |
| Boundary-crossing holds | Exact musical/exact-time span, phrase lineage, onset/duration allocation | Deterministic phrase closure and intent split at any effective beat boundary | Rhythm, Combat | Reconciled; scoring-field implementation required |
| Fair suspension re-entry | Eligible-note lookahead and minimum-preview evaluation facts | Every supported transition can find fair material without fake notes or arbitrary gaps | Rhythm, UI, encounters | Reconciled; runtime lookahead fields required |
| Dynamic recovery | Candidate identity, supported role/difficulty/roster, conflicts, maximum delay, challenge kind | Required recovery candidate coverage for all supported configurations | Rhythm, Survival, Cooperative Actions | Reconciled; no fixed revival timestamp permitted |

Global Early Miss association range, release grace, grade factors, hold allocation,
minimum preview lead, calibration bounds, drift thresholds, and resync limits are
playtest/operational configuration. They are not song-author-controlled values
unless a later approved rule explicitly changes ownership.

## 18. Deferred tuning and technical work

Design-complete but intentionally unfinalized numeric or implementation choices
include:

- final timing windows and grade factors;
- Early Miss association range and contact-bounce filter;
- hold onset/duration allocation and release grace;
- passage budget unit scale and UI rounding;
- minimum re-entry preview lead;
- calibration bounds/statistics;
- clock-confidence, drift, convergence, and resync thresholds;
- local/server validation and anti-cheat architecture; and
- event transport, retention, and analytics schemas.

These values may be tuned only within the behavioral rules in this document.
Technical architecture may choose representations and authority boundaries but
may not introduce artificial misses, alter equal passage ceilings, retime the
song, or make equipment affect musical judgment.

## 19. Approval and change control

The owner interview resolved RG-01 through RG-12 on 2026-08-21. This document is
the canonical Rhythm Gameplay design specification.

A material change to matching, judgment ownership, equal normalization,
suspension treatment, Hold Assist parity, intent attribution, solo pause,
calibration authority, or synchronization recovery requires an explicit design
amendment citing the superseded rule. Numeric playtest tuning inside the stated
boundaries does not require redesign, but approved values must remain versioned
and reproducible.
