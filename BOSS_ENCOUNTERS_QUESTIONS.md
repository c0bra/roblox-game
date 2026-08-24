# Bands Battle Boss Encounters Specification Questions

- **Status:** Interview complete; 12 of 12 questions resolved and reconciled
- **Started:** 2026-08-21
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#44-boss-encounters)
- **Included system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#45-tactical-positioning--movement)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Combat dependency:** [`COMBAT.md`](COMBAT.md)
- **Rhythm dependency:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)
- **Working record:** [`BOSS_ENCOUNTERS_WORKING.md`](BOSS_ENCOUNTERS_WORKING.md)
- **Canonical result:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)

## 1. Interview method

This interview uses four checkpoints of three questions. It inherits the song
clock, Combat/Survival order, and settled GDD encounter grammar. Defaults focus
on orchestration, authoring boundaries, and deterministic player-facing rules;
boss-specific numbers and attack catalogs remain content/balance work.

Answers are persisted after each checkpoint. When all twelve questions are
resolved, they are reconciled into canonical `BOSS_ENCOUNTERS.md`, with Tactical
Positioning & Movement as a major section.

## 2. Fixed inherited decisions

- A normal attempt uses one full three-to-seven-minute song as its clock. Resolve
  breaks never skip, rewind, speed up, or pause it.
- Arrival, First Clash, Escalation, Climax, and Finishing Cadence are flexible
  song functions, not equal-duration universal phases.
- A normal encounter has three sequential Resolve layers with authored openings.
  Difficulty and active human count change their requirements.
- Early breaks bank capped Momentum for the next layer. The finishing cadence is
  independently required for victory.
- Boss attacks use Telegraph, Commit, Impact, and Recovery. Targets cannot change
  after Commit and major responses must remain readable.
- Runtime variation selects only authored/validated Activity Map candidates and
  cannot create impossible overlaps.
- Baseline arenas use nine Near/Middle/Rear and left/center/right locations, but
  an encounter may author a readable different graph.
- Locations are shared zones with formation offsets; humans and acolytes cannot
  body-block or consume one another's gameplay capacity.
- A baseline voluntary dash takes roughly 0.75 seconds, consumes one movement
  charge, grants no invulnerability, and is followed by rhythm settling. Charge
  recovery begins after landing and is beat-based.
- Near/Middle/Rear risk hypotheses affect Attack, incoming danger, and earned
  reward potential. Risk Bonus requires successful exposed performance and banks
  by completed phrase.
- Empty Ward/downed state comes from Combat. All humans down ends co-op; solo
  ends only after its recovery rules do.
- Public attempts have no player host or ordinary join-in-progress. Multiplayer
  owns staging, membership, rejoin grace, and roster changes.

## 3. Question plan

### Checkpoint A — Attempt, Resolve, and outcome

#### BE-01 — Attempt lifecycle and song functions [Resolved]

- **Decision needed:** Which encounter states surround the running song, and how
  do flexible song functions relate to state rather than becoming a rigid phase
  template?
- **Must resolve:** Revision/roster lock, readiness handoff, synchronized start,
  running/pause distinction, function transitions, terminal interruption,
  completion, and immutable attempt identity.
- **Owner decision:** Multiplayer hands over a locked roster, difficulty,
  content revision, loadouts, and balance revision. Boss Encounters creates one
  immutable attempt and advances it through Preparing, synchronized countdown,
  Running, Resolving, and Complete. Arrival, First Clash, Escalation, Climax,
  and Finishing Cadence are ordered authored regions within Running rather than
  equal-duration runtime states. Only approved solo pause freezes Running.
  Nonterminal mechanics never retime the song; a terminal defeat may leave
  Running early for Resolving.

#### BE-02 — Resolve, Momentum, and post-break performance [Resolved]

- **Decision needed:** How do the three layer openings, early/late breaks,
  overflow, Momentum, and post-third-break value resolve over the fixed song?
- **Must resolve:** Active/locked layers, opening precedence, threshold snapshot,
  overflow, cap, carry/application, late opening, post-third-break value,
  group attribution, and no layer skipping.
- **Owner decision:** Only the current Resolve layer is vulnerable. An early
  break routes same-packet overflow and later valid Attack into capped Momentum.
  At the next layer's authored opening, Momentum applies as initial pressure but
  cannot break the new layer automatically, then its bank resets. If one or more
  later openings have passed before the current layer breaks, only the immediate
  next layer opens; no layer is skipped. After an early third-layer break,
  further valid Attack banks a separate capped post-break performance value for
  results, rewards, and spectacle, never finishing substitution. All group
  totals preserve source attribution.

#### BE-03 — Victory, defeat, and exact reason [Resolved]

- **Decision needed:** How are simultaneous song-end, finishing, Resolve, and
  all-humans-down facts converted into one final outcome?
- **Must resolve:** Finishing eligibility/threshold, early victory prohibition,
  terminal defeat, same-boundary precedence, exact reason priority, random
  effects, transition timing, and result snapshot.
- **Owner decision:** Victory is evaluated only after the finishing cadence and
  requires all three Resolve layers broken, the finishing threshold met, and at
  least one human Active or Return Protected after same-boundary Combat
  resolution. Co-op all-humans-down and exhausted/failed solo recovery may end
  the attempt earlier. When defeat conditions coincide, the primary reason
  priority is all humans down, Resolve remaining, then finishing threshold
  missed; every additional failed condition remains supporting evidence. Random
  effects cannot reverse the result. Entering Resolving freezes one immutable
  outcome snapshot.

### Checkpoint B — Attacks, targeting, and dynamic-event arbitration

#### BE-04 — Candidate selection and encounter variation [Resolved]

- **Decision needed:** How does runtime choose among authored attacks and event
  candidates without becoming random or unfair?
- **Must resolve:** Candidate filtering, deterministic/random seed, repetition,
  difficulty/roster adaptation, intensity budget, cooldowns, future feasibility,
  and replay variation.
- **Owner decision:** At each scheduling boundary, filter authored candidates by
  song position, difficulty, roster, valid targets, chart activity, reaction
  time, movement feasibility, current reservations, cooldown/repetition rules,
  intensity budget, and remaining capacity for required future events. Select
  among valid candidates with a recorded attempt seed and authored weights.
  Every choice and rejection is reproducible. Difficulty/roster may select only
  validated variants; randomness cannot weaken cue reliability or future
  feasibility.

#### BE-05 — Telegraph, Commit, Impact, and Recovery [Resolved]

- **Decision needed:** What exact lifecycle and guarantees apply to every boss
  attack instance?
- **Must resolve:** Stage boundaries, target/geometry lock, cancellation, impact
  packets, persistent hazards, earned recovery advantage, cue obligations,
  accessibility, and same-beat Combat handoff.
- **Owner decision:** Telegraph presents boss pose, sound, arena geometry, and a
  compact shape/icon cue. Commit locks targets, unsafe geometry, effect values,
  impact time, and child-hazard definition. Impact sends stable identified
  effects through Combat; Recovery ends the action and may issue an authored
  earned-advantage opportunity. A committed attack cannot retarget or silently
  change. Only terminal resolution, an explicit authored cancellation, or a
  critical runtime safety failure may cancel it; safety cancellation visibly
  dissipates without player harm and never substitutes a surprise hit.

#### BE-06 — Conflict arbitration and required-event protection [Resolved]

- **Decision needed:** Which events win when boss attacks, movement windows,
  recovery, Band Calls, Crescendos, finishing, silence, and chart commitments
  compete?
- **Must resolve:** Priority classes, reservation, urgent fallback, rescheduling,
  cancellation, required candidate budgets, maximum delay, and authoring failure.
- **Owner decision:** Priority is: immutable song/finishing boundaries and
  already-committed events; urgent recovery; guaranteed authored events such as
  the required Crescendo or story attack; accepted player requests such as Band
  Calls; then optional attacks/variation. A lower-priority conflict searches
  later valid candidates within its maximum delay, then follows its defined
  cancel/skip rule. Only urgent recovery may use the universal-beat fallback.
  Required-event coverage and nonoverlap are shipping validators; runtime never
  forces an unfair overlap to compensate for invalid content.

### Checkpoint C — Arena graph and movement

#### BE-07 — Location graph, occupancy, and transformation [Resolved]

- **Decision needed:** How are legal locations, routes, formation offsets, and
  authored graph changes represented to gameplay?
- **Must resolve:** Stable location/edge identity, shared occupancy, starting
  positions, invalid location evacuation, add/remove/corrupt/reconnect,
  reachability, acolytes, and presentation.
- **Owner decision:** Every location and directed edge has stable identity,
  travel time, risk tier, cover/hazard tags, and authored neighbors. Encounters
  declare safe starting locations, normally Middle, and assign them
  deterministically without exclusive occupancy. Humans/acolytes share locations
  through formation offsets and never body-block. Graph transformations happen
  only at authored musical boundaries, must preserve a valid response route,
  and cannot silently invalidate committed travel. Removing an occupied location
  requires a telegraphed authored displacement to a valid fallback.

#### BE-08 — Voluntary movement and recovery [Resolved]

- **Decision needed:** What state machine governs a one-edge or multi-edge dash?
- **Must resolve:** Input acceptance, charge spend, route locking, travel,
  exposure, landing, rhythm settling, beat/clamped charge recovery, multi-edge
  behavior, unavailable input, and no gear/difficulty changes.
- **Owner decision:** A legal dash commits only from a ready charge, immediately
  spends it, suspends Rhythm, and follows the displayed edge with no
  invulnerability. Landing begins the shorter Rhythm settling period and the
  separate two-beat clamped charge recovery. A farther touch destination creates
  a visible multi-edge route; every edge pays full travel/recovery cost, the next
  edge starts on an advertised boundary, and the player may cancel/change while
  stationary between edges. Invalid/unavailable input never queues. Gear and
  difficulty cannot alter travel, charge, settling, or recovery timing.

#### BE-09 — Cover, hazards, displacement, and positional risk [Resolved]

- **Decision needed:** How do location/route geometry and performance create
  avoidance, cover, hazards, incoming danger, Attack bonuses, and banked reward
  facts?
- **Must resolve:** Impact sampling, route intersections, cover tags, persistent
  hazard occupancy, involuntary displacement, dangerous-position attribution,
  phrase banking/loss, caps, and no passive rewards.
- **Owner decision:** An impact samples exact location or committed route
  geometry at its logical time. Cover applies only to declared attack tags and
  directions. Identified persistent hazards sample occupancy at authored pulse
  boundaries. Involuntary displacement spends no movement charge and does not
  reset existing recovery. Attack and Risk Bonus use successful settled
  performance at the location; completing a phrase banks its risk value, while
  moving/downing first loses only the current phrase's unbanked value. Cover
  does not rewrite the base risk/reward tier, and occupancy without successful
  performance earns nothing.

### Checkpoint D — Scaling, resilience, and outputs

#### BE-10 — Difficulty and active-roster scaling [Resolved]

- **Decision needed:** When and how may Resolve, target count, attack choice, and
  future pressure adapt to difficulty and one-to-six humans?
- **Must resolve:** Starting snapshot, sublinear Resolve, target clarity,
  disconnect/departure, unopened-layer rescale, current-layer immutability,
  rejoin, NPC exclusion, and no join-in-progress.
- **Owner decision:** Resolve begins at 1.00 solo equivalents for one human and
  adds 0.75 for each additional human through six. Each layer locks its scaled
  threshold when it opens. Disconnect during rejoin grace does not rescale; if
  grace expires before a future layer opens, that unopened layer uses the
  smaller active roster. An open layer never changes. A successful rejoin before
  opening remains included. Only humans count. Target counts and attack variants
  use validated difficulty/roster envelopes without weakening cue clarity, and
  no new player may join an attempt in progress.

#### BE-11 — Runtime invalidation and failure recovery [Resolved]

- **Decision needed:** What happens when a selected event, location, target,
  content asset, or clock assumption becomes invalid after the attempt starts?
- **Must resolve:** Pre-Commit replacement, post-Commit behavior, safe fallback,
  later candidate, critical versus noncritical failure, no impossible state,
  invalid attempt evidence, and player communication.
- **Owner decision:** An invalid candidate before Telegraph is discarded and
  reselected. During Telegraph but before Commit it visibly cancels and may seek
  a later candidate. After Commit it resolves as locked unless safe trustworthy
  resolution is impossible, when it visibly dissipates without player harm.
  Player-local timing failure suspends only that player. A global authoritative
  clock/content/finishing failure transitions to an identified Invalid / No
  Contest result, never gameplay victory or defeat; compensation is owned by
  Rewards. Every invalidation is visible where player action is affected and
  retains diagnostic evidence.

#### BE-12 — Semantic outputs and content completeness [Resolved]

- **Decision needed:** Which facts must Boss Encounters/Positioning expose, and
  which authored requirements must be added to the Content Authoring contract?
- **Must resolve:** Attempt/phase/layer/Momentum/finisher, attack, target,
  location/movement/cover/hazard/risk, outcome/reason, UI/Audio/Results/Analytics,
  stable identities, balance revision, validators, and completion audit.
- **Owner decision:** Boss Encounters/Positioning emits identified facts for
  attempt lifecycle, song functions, Resolve/Momentum/post-break state,
  candidates, attack stages/targets, graph/location/movement/cover/hazard/risk,
  roster scaling, finishing, outcome, and invalidation. Events carry logical
  musical time, content/balance revisions, attempt seed, causal identity, and
  relevant source/target snapshots. UI, Audio, Results, Rewards, Multiplayer,
  and Analytics consume the same semantics without redefining them. Accessible
  presentation never changes state or resolution.

## 4. Completion criteria

`BOSS_ENCOUNTERS.md` is complete only when:

- BE-01 through BE-12 are resolved;
- the song-shaped attempt lifecycle and exact outcome are deterministic;
- Resolve, Momentum, and finishing cannot skip or retime the song;
- attacks and dynamic events cannot violate Commit or authoring fairness;
- every legal movement, graph change, cover interaction, and risk value has one
  authoritative interpretation;
- roster/difficulty changes cannot create departure exploits or impossible
  encounters;
- runtime invalidation has a safe visible outcome; and
- every new authored-data need is registered for Content Authoring
  reconciliation.

## 5. Change log

- **2026-08-21:** Created the concise 12-question plan from the approved GDD,
  Systems Map, Rhythm, and Combat contracts.
- **2026-08-22:** Resolved BE-01 through BE-03, establishing attempt lifecycle,
  Resolve/Momentum progression, and deterministic outcome precedence.
- **2026-08-22:** Resolved BE-04 through BE-06, establishing reproducible event
  selection, attack commitment, and conflict/reservation priority.
- **2026-08-22:** Resolved BE-07 through BE-09, establishing arena graph,
  multi-edge movement, cover/hazard sampling, displacement, and banked risk.
- **2026-08-22:** Resolved BE-10 through BE-12 and reconciled all twelve answers
  into canonical `BOSS_ENCOUNTERS.md`.
