# Bands Battle Boss Encounters and Tactical Positioning

- **Status:** Approved
- **Approved:** 2026-08-22
- **Parent systems:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#44-boss-encounters) and
  [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#45-tactical-positioning--movement)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Rhythm dependency:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)
- **Combat dependency:** [`COMBAT.md`](COMBAT.md)
- **Authoring dependency:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)
- **Decision source:** [`BOSS_ENCOUNTERS_WORKING.md`](BOSS_ENCOUNTERS_WORKING.md)
- **Interview plan:** [`BOSS_ENCOUNTERS_QUESTIONS.md`](BOSS_ENCOUNTERS_QUESTIONS.md)

## 1. Role and authority

This document defines how one approved song-shaped boss attempt runs from locked
deployment through an immutable outcome. It owns the encounter lifecycle,
ordered song functions, Resolve and Momentum state, finishing evaluation, boss
attack instances, target/geometry commitment, dynamic-event arbitration,
hazards, and exact outcome.

It also specifies Tactical Positioning & Movement as an integrated system owning
the arena graph, legal location/travel state, movement charges and recovery,
settling, shared occupancy, cover/hazard sampling, displacement, positional risk,
and authored graph transformations.

It does not own Rhythm judgments, combat conversion or Ward, matchmaking/party
membership, item/build/ability definitions, rewards, authoring approval, or
presentation implementation. Those systems provide versioned inputs or consume
the semantic facts defined here.

## 2. Governing invariants

1. **The full song remains the clock:** no Resolve break, movement, attack,
   recovery, group event, or nonterminal state skips, rewinds, pauses, or changes
   song speed.
2. **Functions follow the song:** Arrival, First Clash, Escalation, Climax, and
   Finishing Cadence are ordered authored regions, not equal fixed phases.
3. **Three layers, never skipped:** only the current Resolve layer is vulnerable.
4. **Finishing remains independent:** early performance and Momentum cannot
   replace the final authored performance requirement.
5. **Commit means commitment:** targets, geometry, effect, and impact time do not
   change after an attack commits.
6. **Authored variation only:** runtime chooses among approved candidates inside
   validated difficulty/roster envelopes.
7. **Required fairness is validated:** runtime never repairs bad content by
   shortening warnings, inventing notes, or forcing impossible overlaps.
8. **Locations are shared, not scarce:** players and acolytes never body-block or
   reserve exclusive tactical capacity.
9. **Movement is commitment, not immunity:** travel consumes opportunity and
   remains exposed to actual attack geometry.
10. **Risk requires performance:** merely occupying a dangerous location earns
    no Attack or reward value.
11. **Open state is immutable:** current Resolve thresholds, committed attacks,
    and confirmed outcome facts do not rescale or reroll retroactively.
12. **System failure is not player failure:** an outcome-critical runtime fault
    becomes Invalid / No Contest rather than fabricated Victory or Defeat.

## 3. Locked attempt configuration

Multiplayer hands Boss Encounters a validated deployment containing:

- exact content, schema, and balance revisions;
- encounter/boss/difficulty identity;
- deployment roster and connection identities;
- locked player loadouts and ability selections;
- approved master clock, song-function regions, Resolve openings, finishing
  cadence, Activity Map, candidates, conflicts, attacks, and arena graph; and
- supported difficulty/roster scaling envelopes.

Boss Encounters creates one immutable attempt identity and recorded selection
seed. No active attempt changes content/balance revision. Every runtime choice,
stage transition, effect, movement, and result refers to that identity.

## 4. Attempt lifecycle

The attempt states are:

1. **Preparing:** validate/load the locked package and confirm synchronized clock
   readiness.
2. **Countdown:** publish a future shared start boundary and complete deployment.
3. **Running:** advance the approved full-song clock and encounter state.
4. **Resolving:** accept no new ordinary gameplay and create one immutable result
   snapshot.
5. **Complete:** downstream presentation/reward/results flows consume that
   snapshot; gameplay cannot reopen.

Solo pause is an overlay on Running and freezes song/encounter time exactly under
`RHYTHM_GAMEPLAY.md`. Cooperative menus never freeze Running.

All-humans-down or exhausted solo recovery may move from Running to Resolving
early. Nonterminal attacks, breaks, movement, story beats, recovery, group events,
and player departures do not retime the song.

## 5. Song functions

Arrival, First Clash, Escalation, Climax, and Finishing Cadence are authored as
ordered musical regions with exact boundaries. They may differ greatly in length
and intensity, and transitions follow the selected song rather than a universal
timer.

- **Arrival:** deployment, reveal, approachable opening material, and initial
  arena readability.
- **First Clash:** core interaction and first Resolve pressure.
- **Escalation:** additional mechanics, pressure, movement, and layer progress.
- **Climax:** strongest structured pressure and final Resolve layer.
- **Finishing Cadence:** clearly previewed required final performance.

Quiet/recovery/repositioning moments may occur wherever the song supports them.
A Resolve break never changes these regions or forces a break in playable chart
material.

## 6. Resolve, Momentum, and post-break performance

Each of three ordered Resolve layers has stable identity, authored opening time,
difficulty/roster-scaled threshold, and world/UI presentation. Only the current
layer accepts Attack pressure; future layers remain visible and locked.

When a layer breaks before the next opening:

- same-packet overflow and later valid Attack enter Momentum;
- Momentum stops at its cap, initially around 20% of the next layer;
- excess records capped/discarded evidence rather than a hidden bank; and
- the song/function region continues unchanged.

At the next authored opening, Momentum applies as initial pressure but cannot by
itself break the new layer. It then resets. If the current layer breaks after the
next opening time, only the immediate next layer opens at the break boundary.
Even if later opening times passed, layers remain sequential and none is skipped.

After an early third-layer break, subsequent valid Attack enters a separate
capped post-break bank. That bank may improve result tier, reward potential, and
spectacle but never satisfies Finishing Cadence or produces early victory.

All pressure, overflow, Momentum, application, cap, discard, and post-break
values preserve individual causal attribution while supporting band totals
without public rankings.

## 7. Finishing and outcome

Victory is evaluated only after Finishing Cadence completes. It requires:

1. all three Resolve layers broken;
2. the selected difficulty's band finishing threshold met; and
3. at least one human Active or Return Protected after Combat's full
   same-boundary snapshot.

All humans down ends co-op when no Active or Return Protected human remains. A
solo attempt ends after failed emergency recovery or a later down. At a shared
final boundary, `COMBAT.md` first resolves player effects, committed impacts,
Ward, and downing; Boss Encounters then evaluates the atomic result.

Defeat reason priority is:

1. all humans down;
2. Resolve remaining at song end; then
3. finishing threshold missed.

Other true failure conditions remain supporting facts. Random effects, late
packets, reward rolls, and presentation cannot reverse or reroll the outcome.
Entering Resolving freezes it.

An outcome-critical runtime fault may instead produce **Invalid / No Contest**
under section 17. It is neither gameplay Victory nor Defeat.

## 8. Candidate selection and reproducible variation

At a scheduling boundary, selection begins only with human-approved candidates
from the locked revision. It filters by:

- exact current/future song position and song function;
- difficulty and active-roster variant;
- target eligibility;
- chart density, holds, silence, and role coverage;
- reaction time and movement/response feasibility;
- Finishing Cadence and current/future reservations;
- per-family cooldown and repetition limit;
- active intensity budget; and
- capacity required for later guaranteed events.

Among valid candidates, the recorded attempt seed, authored weight, recent-event
history, and intensity goals select reproducibly. Difficulty/roster can use only
validated variants and cannot silently shorten cues, add unsafe targets, or
change geometry outside their envelopes.

Each selected, filtered, deferred, and rejected candidate records identity and
reason. An empty valid set invokes the conflict/fallback rules; safety filters
are never bypassed for variety.

## 9. Boss attack lifecycle

Every attack has stable identity and four exact musical/exact-time stages:

1. **Telegraph:** boss pose/motion, sound motif, arena geometry, and compact
   shape/icon/text reinforcement reveal threat and response with validated lead.
2. **Commit:** target players, unsafe locations/routes, effect values/tags,
   impact time, cover interactions, and child hazards lock.
3. **Impact:** stable identified effects pass to Combat in authored order and
   child hazards activate.
4. **Recovery:** reservations close and an authored earned-advantage opportunity
   may follow successful response.

Critical cues never rely only on color or one sensory channel. Difficulty uses
only approved cue/combination variants.

After Commit, an attack cannot retarget, move its geometry, change effect values,
or shift impact time. Disconnect uses the committed snapshot. A committed attack
ends without impact only on terminal encounter resolution, explicit authored
cancellation, or critical safety failure. Safety cancellation visibly
dissipates, causes no harm, and never substitutes a surprise effect.

Persistent hazards are identified child instances with activation, occupancy
geometry, pulse effects/order, and end boundaries. They never become untracked
ambient damage.

## 10. Event reservation and conflict priority

Scheduling priority is:

1. immutable song/Finishing boundaries and already-committed windows;
2. urgent cooperative or solo recovery;
3. guaranteed authored events, including the required Crescendo and required
   story/encounter attacks;
4. accepted player requests such as Band Calls; and
5. optional attacks, bonus events, and presentation variation.

A lower-priority request cannot displace a higher reservation. It searches later
eligible candidates within its own maximum delay while preserving preview and
duration. If none remains, it follows its declared retain/cancel/wait/skip or
required-content-failure behavior.

Only urgent recovery may use the approved universal-beat fallback. Other events
cannot invent a boundary or substitute chart material.

Shipping validation proves required candidate coverage, nonoverlap, priority,
and maximum-delay success for every supported role, difficulty, and roster.
Runtime never compresses a warning or forces an unfair overlap to compensate for
bad content.

## 11. Arena graph and occupancy

Each location has stable identity, world and formation anchors, risk tier,
cover/hazard tags, enabled/corrupted state, and authored presentation. Each
directed edge has source/destination, route geometry, travel duration, legal
direction, and transformation constraints.

The familiar nine-position Near/Middle/Rear by left/center/right arrangement is
the baseline, not a universal requirement. An irregular graph is legal only when
validation and phone-scale review prove readable reachability and responses.

An encounter declares safe starting locations, ordinarily Middle. Assignment is
deterministic for readability but creates no exclusive slot. Humans and acolytes
share locations through formation offsets, cannot collide/body-block, and remain
subject to the same geometry, hazards, cover, and risk.

## 12. Graph transformations

Authored transformations may add, remove, disable, corrupt, restore, elevate, or
reconnect locations/edges only at musical boundaries with sufficient multimodal
preview. Every resulting graph preserves at least one validated response route
for relevant current/future attacks.

A mutation cannot silently destroy a committed travel edge. It waits for
landing or invokes an explicitly authored, telegraphed displacement. Removing an
occupied location follows an authored fallback, normally nearest reachable valid
Middle with stable tie-breaking. It never traps, overlaps, drops, or damages a
player merely because state changed.

Acolytes use the same valid graph/fallback facts at authored boundaries without
blocking human choices.

## 13. Voluntary movement

A dash is accepted only when the player is eligible, the edge/destination is
legal, and one movement charge is Ready. Acceptance commits the edge, spends the
charge, publishes route/timing, and suspends ordinary Rhythm.

Baseline travel is roughly 0.75 seconds per edge. Travel follows visible route
geometry and grants no invulnerability. Landing begins:

- a shorter Rhythm-settling interval before fair chart re-entry; and
- a separate two-beat charge recovery, initially clamped to roughly
  0.75–1.25 seconds.

Once rhythm-settled, a stationary player may perform and act while movement
remains unavailable.

A farther touch destination becomes a visible multi-edge route, not a teleport.
Every edge pays the same charge, travel, landing, and recovery cost. The next
edge begins on an advertised boundary after recovery; the stationary player may
cancel or change the remaining route between edges. If the remainder becomes
invalid, it stops at the last legal landing with an explanation and never
silently reroutes.

An input while unavailable is rejected with restrained feedback and never
queues. Equipment, builds, abilities, consumables, and difficulty do not alter
travel duration, charge count/recovery, settling, route cost, or invulnerability.

## 14. Cover, hazards, and displacement

At an impact/pulse, Positioning supplies exact settled location or committed
route segment/geometry at that logical time. Avoidance never uses eventual
landing or receipt time.

Cover applies only when location/route tags, attack direction/shape/tags, and
impact geometry meet the authored rule. It does not grant universal immunity or
retarget a committed attack.

Each persistent-hazard pulse samples current occupancy independently and sends
one identified effect through Combat.

Involuntary displacement follows authored route/fallback, spends no movement
charge, grants no hidden immunity, and does not reset existing voluntary charge
recovery. Rhythm suspension and settling still prevent artificial misses.

## 15. Position risk and banking

Starting risk-tier hypotheses are:

| Tier | Attack output | Incoming danger | Reward potential |
|---|---:|---:|---:|
| Near | +25% | +30% | +25% |
| Middle | Baseline | Baseline | Baseline |
| Rear | −20% | −25% | No Risk Bonus |

These values are versioned playtest data. Settled successful Attack uses the
location's Attack multiplier. Positioning separately publishes incoming-danger
facts for Combat and exposed-performance facts for Rewards.

Risk Bonus accrues only through successful normalized performance while Active
and settled at an eligible dangerous location. Travel, passive occupancy,
downed time, and NPC support earn nothing. Phrase completion banks its accrued
risk; movement or downing first loses only that phrase's unbanked value.
Previously banked value remains, subject to the Rewards-owned encounter cap.

Specific cover may stop one compatible threat without rewriting the location's
baseline risk/reward tier. Gear/builds cannot change those universal ratios.

## 16. Difficulty and active-roster scaling

Initial Resolve relationships are:

| Humans | Solo-player equivalents |
|---:|---:|
| 1 | 1.00 |
| 2 | 1.75 |
| 3 | 2.50 |
| 4 | 3.25 |
| 5 | 4.00 |
| 6 | 4.75 |

Each layer combines its difficulty base with active-human scaling when it opens,
then locks that threshold. A disconnected player remains counted during rejoin
grace. If grace expires, only still-unopened layers use the smaller roster at
their opening. A successful rejoin before opening remains included.

Departure never changes an open layer, banked Momentum, or prior attribution.
Acolytes/NPCs never count. Rejoin restores a deployment member; ordinary
join-in-progress is unavailable.

Difficulty/roster may select validated target-count, timing, geometry,
combination, and pressure variants. Larger target sets must remain individually
clear and within proven response capacity.

## 17. Runtime invalidation and No Contest

Invalidation follows lifecycle:

- **Before Telegraph:** discard privately and reselect.
- **During Telegraph before Commit:** visibly cancel, release reservations, and
  seek a later candidate inside allowed delay.
- **After Commit:** resolve as locked unless terminal state, authored
  cancellation, or critical safety failure prevents trustworthy resolution; a
  safety failure visibly dissipates with no harm.

Targets may change only before Commit and inside the validated variant. Future
graph mutations are delayed, safely replaced, or canceled before they can trap a
player or invalidate travel.

Noncritical presentation loss uses redundant accessible channels only when the
remaining cues still communicate the complete mechanic. Loss of a critical cue
cancels the affected uncommitted event.

Player-local clock/content uncertainty suspends only that player's Rhythm. A
global authoritative clock/content mismatch, corrupt/unavailable finishing
cadence, or other fault preventing fair outcome evaluation ends as **Invalid /
No Contest**. It grants no inferred fragment/story victory and does not count as
gameplay defeat. Rewards defines compensation/refund behavior.

Every invalidation records stage, reason, affected identities, fallback,
player-facing cue, and whether attempt validity remained intact.

## 18. Semantic output contract

Boss Encounters and Positioning expose identified facts for:

- attempt/configuration/countdown/Running/pause/Resolving/Complete/No Contest;
- song-function transitions;
- Resolve open/pressure/break/late-open, Momentum bank/apply/cap/reset, and
  post-break value;
- candidate filter/select/reserve/defer/cancel/skip/failure;
- attack Telegraph/Commit/Impact/Recovery, target/geometry, hazard, earned
  advantage, and safety cancellation;
- graph/location/edge/transformation/formation state;
- movement request/accept/reject/commit/travel/land/settle/recover/route change;
- avoidance, cover, hazard occupancy/pulse, displacement, and risk bank/loss;
- scaling snapshots and roster-change consequences; and
- finishing evaluation, outcome/reason/supporting facts, and invalidation.

As applicable, an event includes attempt, content/balance revision, seed,
logical musical and exact time, candidate/authored-event identity, stable order,
source/target/player/roster snapshot, causal identity, pre/post state, and
fallback/cap/discard evidence.

Rhythm, Combat, Abilities, Multiplayer, Rewards, UI, Audio, Results, and
Analytics consume these facts without redefining them. Accessible presentation
may change channels, scale, captions, effects, motion, or haptics; it cannot hide
a required cue or change timing, targeting, geometry, state, or outcome.

## 19. Content Authoring reconciliation register

These entries must be reconciled into Content Authoring after specifications 2
through 12 are complete.

| Encounter requirement | Semantic data | Required validation | Consumers | Compatibility/support status |
|---|---|---|---|---|
| Song-shaped lifecycle | Ordered function regions, Resolve openings, finishing identity/boundary, shared tie order | Regions cover intended song; three openings/finisher valid; ties deterministic | Encounter, UI, Audio, Results | Baseline exists; explicit global order requires confirmation |
| Momentum/post-break ranges | Eligibility boundaries and bank kind | No overlap/confusion; neither substitutes finishing | Encounter, Combat, Rewards | New explicit runtime distinction |
| Reproducible candidates | Family, seed weight, cooldown/repeat group, intensity cost, variants, preview/duration, priority, delay, fallback, future-capacity effect | Filter/selection reproducible; required future events remain feasible | Encounter, Abilities, Analytics | Candidate model exists; metadata expansion required |
| Attack contract | Stage times, target/geometry/effect/hazard identity, cue requirements, Commit locks, cancellation, Recovery advantage | Lead/response fair; targets/geometry stable; hazards fully bounded | Encounter, Combat, Positioning, UI/Audio | Stage baseline exists; runtime fields need reconciliation |
| Conflict coverage | Reservations, priority, max delay, required/optional classification | Full matrix passes for every role/difficulty/roster; only recovery has universal fallback | Encounter, Rhythm, Abilities | New comprehensive validator |
| Arena graph | Location/edge identity, geometry, anchors, starts/fallbacks, risk/cover/hazard tags, transformations | Reachability and readable response survive every authored graph state | Positioning, Encounter, UI | Baseline concept exists; schema required |
| Travel and displacement | Route geometry/duration, multi-edge boundaries, mutation constraints, fallback | Committed route stays legal; no trap/teleport/body-block; timings consistent | Positioning, Rhythm, Combat | New explicit runtime contract |
| Scaling variants | Difficulty and one-to-six-human Resolve/target/geometry envelopes | Open-layer snapshots deterministic; cue/response clarity retained | Encounter, Combat, Multiplayer | GDD values exist; authored variant coverage required |
| Critical validity | Critical cue/data classification and safe degradation/cancellation | Outcome-critical data survives export/load; failure produces No Contest | Encounter, UI/Audio, Results | New release/operational gate |

Formulae, difficulty numbers, movement constants, and reward caps remain versioned
balance/system configuration rather than private chart-note fields. Packages
reference compatible definitions and contain their authored geometry, timing,
candidates, and constraints.

## 20. Deferred tuning and technical work

Behavior is complete; these remain versioned playtest or architecture choices:

- Resolve thresholds, Momentum/post-break caps, finishing thresholds, and result
  tiers;
- candidate weights, family cooldowns, repetition limits, and intensity budgets;
- boss-specific attacks, effects, damage, target counts, warning/recovery time,
  and earned advantages;
- final arena graphs, formation offsets, travel/settling/recovery clamps;
- position multipliers and total Risk Bonus cap;
- invalidation detection, clock authority, seed generation, and event transport;
  and
- No Contest compensation, telemetry, and operational alert thresholds.

Tuning may not retime the song, skip layers, replace finishing, violate Commit,
weaken required cues, create impossible movement, alter position ratios through
gear, or turn system failure into player defeat.

## 21. Approval and change control

The owner interview resolved BE-01 through BE-12 on 2026-08-22. This document is
the canonical Boss Encounters and Tactical Positioning & Movement design
specification.

A material change to attempt lifecycle, Resolve/Momentum, victory requirements,
attack commitment, event priority, graph/occupancy, movement cost, position risk,
roster scaling, or No Contest behavior requires an explicit amendment citing the
superseded rule. Numeric tuning inside these boundaries creates a new balance or
content revision and never changes an active attempt.
