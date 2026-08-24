# Bands Battle Boss Encounters Working Record

- **Status:** Complete decision record; 12 of 12 questions reconciled
- **Started:** 2026-08-21
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#44-boss-encounters)
- **Included positioning system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#45-tactical-positioning--movement)
- **Interview plan:** [`BOSS_ENCOUNTERS_QUESTIONS.md`](BOSS_ENCOUNTERS_QUESTIONS.md)
- **Rhythm dependency:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)
- **Combat dependency:** [`COMBAT.md`](COMBAT.md)
- **Canonical result:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)

## 1. Role of this record

This document persists owner decisions while the Boss Encounters and Tactical
Positioning & Movement interview is in progress. It is not canonical until
reconciled into `BOSS_ENCOUNTERS.md`.

## 2. Inherited boundary

Boss Encounters owns active-attempt lifecycle, flexible song functions, Resolve
layers/openings, Momentum, finishing evaluation, boss attack instances, dynamic
event selection/conflict arbitration, targeting, hazards, and exact encounter
outcome. Tactical Positioning & Movement, specified in the same canonical
document, owns the arena graph, location/travel, movement charge/recovery,
settling, displacement, cover/hazard occupancy, risk tier, and authored graph
changes.

They do not own rhythm judgment, combat formulae or Ward state, multiplayer
membership/matchmaking, ability definitions, rewards, authoring workflow, or
presentation implementation.

## 3. Approved inputs

- Exact approved content, schema, and balance revisions with musical clock,
  encounter timeline, Activity Map, candidates, conflicts, and arena data.
- Rhythm musical boundaries/participation and Combat effect/survival snapshots.
- Multiplayer roster, connection, rejoin, and attempt-membership state.
- Ability/cooperative-action requests and candidate constraints.
- Difficulty configuration and validated player-count scaling.

## 4. Decision record

### Checkpoint A — Attempt, Resolve, and outcome

#### BE-01 — Attempt lifecycle and song functions

- **Status:** Approved.
- Multiplayer completes staging/readiness and hands over a locked roster,
  difficulty, exact content and balance revisions, and validated player
  loadouts. Boss Encounters creates an immutable attempt identity from that
  handoff; it never changes revisions mid-attempt.
- The attempt lifecycle is:
  1. **Preparing:** validate/load the locked encounter and confirm synchronized
     clock readiness.
  2. **Countdown:** publish one future shared start boundary and perform the
     deployment countdown.
  3. **Running:** advance the approved full-song clock and all encounter-owned
     events.
  4. **Resolving:** accept no new ordinary gameplay and construct the immutable
     outcome/result snapshot.
  5. **Complete:** presentation and downstream systems may consume the result;
     gameplay state cannot reopen.
- Arrival, First Clash, Escalation, Climax, and Finishing Cadence are ordered
  authored regions inside Running. Their lengths and exact transitions follow
  the song; a Resolve break does not advance or replace them.
- Solo pause overlays Running and freezes it under `RHYTHM_GAMEPLAY.md`.
  Cooperative menus do not change Running or the shared clock.
- A terminal all-humans-down or exhausted solo-recovery failure may transition
  early from Running to Resolving. All nonterminal events preserve song speed,
  position, and order.

#### BE-02 — Resolve, Momentum, and post-break performance

- **Status:** Approved.
- Three ordered Resolve layers have stable identity, authored opening time, and
  a threshold determined from the locked difficulty/roster configuration. BE-10
  will define the limited unopened-layer rescale allowed after departure.
- Only the current layer is vulnerable. Future layers remain visible and locked;
  Attack never reaches through the current layer.
- When a layer breaks before the immediate next opening, same-packet overflow
  and later valid Attack bank into Momentum up to its cap, initially targeted at
  roughly 20% of the next layer. Further value records as capped/discarded
  evidence rather than creating a hidden bank.
- At the next authored opening, banked Momentum applies as initial pressure to
  that new layer but cannot itself break the layer. The bank then resets, and
  subsequent early-break output may build Momentum for the following layer.
- If the current layer breaks after the next opening time, the immediate next
  layer becomes vulnerable at the completed break boundary. Even if additional
  authored openings have passed, layers still open and break sequentially; none
  is skipped or automatically damaged by elapsed time.
- After an early third-layer break, later valid Attack goes to one separate
  capped post-break performance bank. It may improve result tier, reward
  potential, and finishing spectacle but never counts toward the finishing
  threshold or creates early victory.
- Resolve, Momentum, post-break, cap, overflow, and discarded values retain
  individual causal packets for personal Results while also producing band
  totals without public ranking.

#### BE-03 — Victory, defeat, and exact reason

- **Status:** Approved.
- Victory cannot occur before the clearly previewed Finishing Cadence finishes.
  It requires:
  1. all three Resolve layers broken;
  2. the selected difficulty's band finishing threshold met; and
  3. at least one human in Active or Return Protected state after Combat's full
     same-boundary snapshot.
- All humans down ends co-op when the completed boundary snapshot contains no
  Active or Return Protected human. A solo attempt ends after failed emergency
  recovery or a later down, as reported by Combat/Survival.
- At the final cadence/song boundary, Combat first resolves player effects,
  committed impacts, Ward, and downing. Boss Encounters then evaluates one
  outcome from the resulting atomic snapshot.
- If multiple defeat conditions are true, the primary player-facing reason is:
  1. all humans down;
  2. Resolve remaining at the ending; then
  3. finishing threshold missed.
- All additional failed conditions are retained as supporting result facts. The
  exact primary reason never erases personal contribution or reward eligibility
  established elsewhere.
- Random effects, late packets, presentation timing, and reward rolls cannot
  reverse or reroll an outcome. Entering Resolving creates an immutable outcome
  snapshot and blocks new gameplay effects.

### Checkpoint B — Attacks, targeting, and dynamic-event arbitration

#### BE-04 — Candidate selection and encounter variation

- **Status:** Approved.
- At every scheduling boundary, the selector begins only with authored,
  human-approved candidates compatible with the locked content revision.
- It filters by exact current/future song position, function/phase, difficulty,
  active roster and target eligibility, chart activity, required reaction time,
  movement/response feasibility, silence/hold/finisher conflicts, active and
  reserved events, per-family cooldown and repetition limit, intensity budget,
  and capacity reserved for required future events.
- Difficulty and roster changes may select only an authored or parametrically
  validated variant. They cannot shorten a cue, add targets, combine patterns,
  or change geometry beyond the candidate's validated envelope.
- Among remaining candidates, the selector uses the attempt's recorded seed,
  authored weights, recent-family history, and intensity goals. This provides
  replay variation while making the entire decision reproducible from the same
  attempt state.
- Every selected, filtered, deferred, and rejected candidate records its identity
  and reason. An empty valid set invokes BE-06 fallback rules; it never permits a
  candidate that failed safety filtering.

#### BE-05 — Telegraph, Commit, Impact, and Recovery

- **Status:** Approved.
- Every attack instance receives stable identity and four exact musical/exact-
  time stage boundaries:
  1. **Telegraph:** preview boss pose/motion, sound motif, affected arena
     geometry, and compact shape/icon/text reinforcement with enough validated
     response time.
  2. **Commit:** lock target players, unsafe locations/routes, effect values and
     tags, impact time, cover interaction, and any child persistent hazard.
  3. **Impact:** emit identified effects to `COMBAT.md` in stable event order and
     activate any committed child hazard.
  4. **Recovery:** close the attack, release its reservation, and optionally
     expose an authored earned-advantage effect/opportunity for successful
     response.
- Critical information never relies on color or a single sensory channel.
  Difficulty may use only its validated cue/combination variant.
- After Commit, the attack cannot retarget, move geometry, change damage/tags,
  or silently shift impact time. Disconnect uses the committed target snapshot
  under Combat rules.
- A committed attack may end without impact only because the encounter became
  terminal, its definition has an explicit legal cancellation transition, or a
  critical runtime safety failure makes resolution untrustworthy. Safety
  cancellation visibly dissipates the attack, causes no player harm, records
  invalidation evidence, and never replaces it with a surprise effect.
- Persistent hazards remain identified children with their own activation,
  occupancy-sampling, pulse, and end boundaries. They do not become untracked
  environmental damage.

#### BE-06 — Conflict arbitration and required-event protection

- **Status:** Approved.
- Scheduling/reservation priority is:
  1. immutable song boundaries, Finishing Cadence, and already-committed event
     windows;
  2. urgent cooperative/solo recovery needs;
  3. guaranteed authored events, including the required Crescendo and any
     required story/encounter attack;
  4. accepted player-requested events such as a Band Call; and
  5. optional attacks, bonus opportunities, and presentation variation.
- A lower-priority request never displaces a higher-priority reservation. It
  searches chronologically later eligible candidates inside its event-specific
  maximum delay, preserving required preview and performance duration.
- If no valid candidate remains, the event follows its declared behavior:
  retain/cancel the player's charge where established, wait/skip a nonurgent
  event, or report required-content failure. It never compresses warning,
  overlaps silence or committed performance unfairly, or invents a boundary.
- Only urgent recovery may use the approved universal-beat fallback. Crescendos,
  Band Calls, boss attacks, and nonurgent events may not synthesize substitute
  chart material.
- Content cannot ship unless validators prove required candidate coverage,
  reservation compatibility, and maximum-delay success for every supported
  role, difficulty, and roster. Human in-Roblox review remains required.

### Checkpoint C — Arena graph and movement

#### BE-07 — Location graph, occupancy, and transformation

- **Status:** Approved.
- Every arena location has stable identity, world/formation anchors, risk tier,
  cover/hazard tags, enabled/corrupted state, and authored presentation. Every
  directed edge has stable identity, source/destination, route geometry, travel
  duration, legal movement direction, and transformation constraints.
- The familiar nine-location Near/Middle/Rear by left/center/right graph is the
  baseline, not a universal template. Different counts/irregular graphs are
  legal only when authoring validation and phone-scale review prove readable
  routes and fair responses.
- An encounter declares one or more safe starting locations, ordinarily Middle.
  Players are assigned deterministically across them for composition/readability,
  but no location is an exclusive gameplay slot.
- Humans and acolytes share locations using formation offsets, cannot collide or
  body-block, and do not consume a location's capacity. Everyone still shares
  its attack geometry, hazard occupancy, cover, and risk.
- Add, remove, disable, corrupt, restore, elevate, or reconnect transformations
  occur only at authored musical boundaries with sufficient multimodal preview.
  The resulting graph must retain at least one validated legal response path for
  every relevant committed/future attack state.
- A transformation cannot silently destroy a route while a player traverses it.
  It either waits for the committed edge to complete or invokes an explicitly
  authored, telegraphed displacement transition.
- Removing/invalidating an occupied location uses the authored fallback rule,
  normally the nearest reachable valid Middle location with stable tie-breaking.
  It never drops, traps, overlaps, or damages an occupant merely because data
  became invalid.
- Acolyte repositioning uses the same valid graph/fallback facts at authored
  musical boundaries but never blocks a human choice.

#### BE-08 — Voluntary movement and recovery

- **Status:** Approved.
- A voluntary edge begins only when the player is eligible, the edge/destination
  is legal, and one movement charge is Ready. Acceptance immediately commits the
  edge, spends the charge, publishes route/timing, and suspends ordinary Rhythm.
- Baseline travel is roughly 0.75 seconds per edge. The avatar follows the
  displayed route and gains no invulnerability. Boss impacts continue to test
  actual route/location exposure.
- Landing starts two independent intervals:
  - the shorter rhythm-settling interval after which fair chart re-entry may
    occur; and
  - movement-charge recovery of two beats, initially clamped to roughly
    0.75–1.25 seconds.
- Once rhythm-settled, a stationary player may perform, Defend, or use eligible
  actions even while the movement charge remains unavailable.
- Selecting a farther destination on touch creates a visible route across graph
  edges rather than teleporting. Every edge incurs the same charge, travel,
  landing, and recovery rules. The next edge begins only on its advertised
  boundary after recovery; while stationary between edges, the player may cancel
  or replace the remaining route.
- If graph/event state invalidates the uncommitted remainder, the route stops at
  the last valid landing and explains why. It never silently reroutes or begins
  an unsafe edge.
- Input received while no charge/edge is legal is rejected with restrained
  feedback and never auto-queues. Involuntary displacement is not a stored
  voluntary input.
- Equipment, builds, difficulty, consumables, and abilities cannot change edge
  travel, charge count, charge recovery, rhythm settling, route cost, or grant
  dash invulnerability unless a future explicit design amendment changes this
  boundary.

#### BE-09 — Cover, hazards, displacement, and positional risk

- **Status:** Approved.
- At each logical impact/pulse, Positioning supplies exact settled location or
  committed route segment/geometry. Avoidance and exposure use that snapshot,
  not the eventual landing location or network receipt time.
- Cover is effect-specific. It protects only when the location/route cover tags,
  attack direction/shape/tags, and impact geometry satisfy the authored rule.
  It never grants universal immunity or changes attack targeting after Commit.
- A persistent hazard is an identified attack child with activation, pulse,
  occupancy geometry, effect tags, and end boundary. Each pulse samples current
  occupancy separately and sends one identified effect through Combat.
- Involuntary displacement uses an authored route or deterministic fallback,
  costs no movement charge, grants no hidden invulnerability, and does not reset
  an existing voluntary recovery timer. Rhythm suspension/settling still applies
  so displacement never fabricates misses.
- Settled successful Attack contribution uses the location's Attack multiplier.
  Positioning publishes incoming-danger facts separately for Combat and
  successful exposed-performance facts separately for Rewards.
- Risk Bonus accumulates only from successful normalized performance while the
  player is active and settled at an eligible dangerous location. Merely
  occupying it, traveling through it, being downed there, or receiving NPC
  support earns nothing.
- Completing a phrase banks its accumulated risk value. Beginning movement or
  becoming downed before phrase completion discards only that phrase's unbanked
  risk; previously banked value remains. Total encounter reward upside remains
  bounded by Rewards.
- Specific cover may prevent one threat while the location retains its authored
  baseline risk/reward tier. Equipment/builds cannot rewrite those ratios.

### Checkpoint D — Scaling, resilience, and outputs

#### BE-10 — Difficulty and active-roster scaling

- **Status:** Approved.
- Initial Resolve relationships by active human count are:

  | Humans | Solo-player equivalents |
  |---:|---:|
  | 1 | 1.00 |
  | 2 | 1.75 |
  | 3 | 2.50 |
  | 4 | 3.25 |
  | 5 | 4.00 |
  | 6 | 4.75 |

- These are versioned playtest relationships. The difficulty-specific base and
  the active-human relationship determine a layer threshold when it opens. That
  threshold is immutable for the life of the open layer.
- A disconnected player remains in roster scaling during rejoin grace even
  though they are excluded from targeting and contribution. If grace expires,
  only still-unopened layers may use the smaller active roster, and each does so
  when it opens.
- A player returning successfully before a future layer opens remains included.
  Nobody joins an ordinary attempt without having belonged to its deployment
  roster; rejoin is restoration, not join-in-progress.
- Acolytes and other NPC support never count as humans for Resolve, target count,
  group thresholds, or departure scaling.
- Difficulty and roster may select only validated target-count, timing,
  geometry, combination, and pressure variants. More players may create more
  targets but cannot make ownership ambiguous, degrade cue reliability, or
  exceed validated response capacity.
- A departure never heals/reduces an open layer, changes banked Momentum, or
  erases prior source attribution. Its effect begins only at a future layer's
  opening snapshot.

#### BE-11 — Runtime invalidation and failure recovery

- **Status:** Approved.
- Candidate/event invalidation follows lifecycle:
  - **Before Telegraph:** discard privately and select another valid candidate.
  - **Telegraph before Commit:** visibly cancel/dissipate, release reservations,
    and seek a later candidate within allowed delay.
  - **After Commit:** resolve exactly as locked unless terminal state, explicit
    authored cancellation, or critical safety failure prevents trustworthy
    resolution. A safety cancellation visibly dissipates and causes no harm.
- A target invalid before Commit is removed/reselected only within the validated
  candidate envelope. After Commit, Combat's target snapshot rules apply.
- A future graph mutation that becomes invalid is delayed, replaced with an
  authored safe variant, or canceled before its boundary. It never traps players
  or invalidates a committed route.
- Loss of a noncritical presentation layer uses redundant accessible cues when
  the remaining channels still communicate the complete mechanic. Loss of a
  critical cue cancels the affected uncommitted event rather than testing an
  unreadable response.
- Player-local clock/content confidence loss suspends that player's Rhythm under
  its specification while the attempt continues. It does not invalidate other
  players' charts or the shared outcome automatically.
- A global authoritative clock/content mismatch, unavailable/corrupt Finishing
  Cadence, or other failure that makes fair outcome evaluation impossible ends
  as **Invalid / No Contest**. It is neither Victory nor Defeat and grants no
  fragment/story victory by inference. Rewards owns compensation/refund policy.
- Every invalidation records stage, reason, affected identities, chosen fallback,
  player-facing cue, and whether result validity was preserved.

#### BE-12 — Semantic outputs and content completeness

- **Status:** Approved.
- Boss Encounters and Positioning expose causally linked facts for:
  - attempt creation, configuration lock, countdown, Running, solo pause overlay,
    Resolving, Complete, and Invalid/No Contest;
  - Arrival/First Clash/Escalation/Climax/Finishing region transitions;
  - Resolve layer open/pressure/break/late-open, Momentum bank/apply/cap/reset,
    and post-third-break bank;
  - candidate filtering/selection/reservation/defer/cancel/skip/failure;
  - Telegraph/Commit/Impact/Recovery, target/geometry snapshot, persistent hazard,
    earned advantage, and safety cancellation;
  - arena graph/location/edge/transformation/formation state;
  - movement request/accept/reject/commit/travel/land/settle/recover/route change;
  - avoidance, cover, hazard occupancy/pulse, displacement, risk accrual/bank/loss;
  - roster/difficulty scaling snapshots and departure/rejoin effects; and
  - finishing eligibility/result, final outcome/reason/supporting facts, and
    invalidation evidence.
- As applicable, each event carries attempt/content/balance identity, seed,
  logical musical and exact time, authored-event/candidate identity, stable
  order, source/target/player/roster snapshot, causal event, pre/post state, and
  fallback/cap/discard evidence.
- Rhythm, Combat, Abilities, Multiplayer, Rewards, UI, Audio, Results, and
  Analytics consume the same semantic source at their required aggregation.
  Presentation and analytics never become encounter authority.
- Accessibility may change cue channels, scale, contrast, captions, haptics,
  effects, or motion. It cannot hide a required cue or change targeting,
  geometry, timing, state, or outcome.

## 5. Content Authoring reconciliation register

- Encounter content must expose exact ordered song-function regions, three
  stable Resolve layer openings, Finishing Cadence identity/boundaries, and
  deterministic stable order where events share a boundary.
- Runtime data must distinguish normal Momentum from post-third-break
  performance and declare their eligible time ranges without embedding private
  combat/reward formulae in the chart.
- Every candidate must declare selection weight, event family, cooldown/repeat
  group, intensity cost, supported variants, reaction/duration requirements,
  reservation priority, maximum delay, cancellation/skip behavior, and effect on
  required future capacity.
- Every attack must declare stable stage/target/geometry/effect/hazard identities,
  multimodal cue obligations, Commit locks, cancellation rules, and Recovery
  opportunity semantics.
- Conflict validation must exercise the full required-event priority matrix for
  every supported role, difficulty, and roster, including urgent recovery and
  the Finishing Cadence.
- Arena content must declare stable locations/edges, geometry, formation
  anchors, start/fallback rules, risk/cover/hazard tags, transformation
  boundaries, and reachability guarantees.
- Movement data must expose edge routes/durations and multi-edge boundaries;
  validators must prove graph mutations never silently invalidate committed
  travel or all legal responses.
- Cover/hazard/displacement definitions require exact geometry/tags, pulse and
  end boundaries, fallback behavior, and deterministic impact sampling.
- Resolve/difficulty/roster content must reference versioned scaling envelopes
  and contain validated target-count/geometry variants for one to six humans.
- Runtime packages must identify critical cue requirements and distinguish
  noncritical presentation degradation from state/outcome-invalidating failure.
- Content validation must prove that Finishing Cadence and all other outcome-
  critical data load and resolve consistently for every supported export.

## 6. Open handoffs

- `RHYTHM_GAMEPLAY.md` owns clock-aligned chart behavior and fair suspension.
- `COMBAT.md` produces Attack pressure, incoming-effect application, Ward/downed
  state, and completed same-boundary snapshots.
- `ABILITIES_AND_COOPERATIVE_ACTIONS.md` defines Signature, Band Call,
  Crescendo, and Solo Support behavior requested through encounter windows.
- `MULTIPLAYER.md` owns roster membership, staging, rejoin, and departure state.
- `REWARDS_AND_ECONOMY.md` consumes outcome, post-break, and banked risk facts.
- `UI_UX.md`, `AUDIO_PRESENTATION.md`, Results, and Analytics consume semantic
  encounter/position output without owning it.

## 7. Change log

- **2026-08-21:** Created the working record. Progress is 0 of 12 questions.
- **2026-08-22:** Approved BE-01 through BE-03. Progress is 3 of 12 questions.
  Established attempt, Resolve/Momentum, and final-outcome rules.
- **2026-08-22:** Approved BE-04 through BE-06. Progress is 6 of 12 questions.
  Established reproducible candidate selection, attack commitment, and event
  conflict priority.
- **2026-08-22:** Approved BE-07 through BE-09. Progress is 9 of 12 questions.
  Established graph/occupancy, voluntary movement, cover/hazards, displacement,
  and positional risk banking.
- **2026-08-22:** Approved BE-10 through BE-12 and reconciled all twelve
  decisions into canonical `BOSS_ENCOUNTERS.md`.
