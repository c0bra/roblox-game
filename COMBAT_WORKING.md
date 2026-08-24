# Bands Battle Combat Working Record

- **Status:** Complete decision record; 12 of 12 questions reconciled
- **Started:** 2026-08-21
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#42-combat)
- **Included survival system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#43-player-survival--recovery)
- **Interview plan:** [`COMBAT_QUESTIONS.md`](COMBAT_QUESTIONS.md)
- **Rhythm dependency:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)
- **Canonical result:** [`COMBAT.md`](COMBAT.md)

## 1. Role of this record

This document persists owner decisions while the Combat and Player Survival &
Recovery interview is in progress. It is not canonical until reconciled into
`COMBAT.md`.

## 2. Inherited boundary

Combat owns intent selection/queuing, routing identified normalized performance,
post-score conversion and modifier rules, effect attribution, and semantic
effect output. Player Survival & Recovery, specified in the same canonical
document, owns Ward, application of final incoming/restorative effects, downed
state, cooperative revival state, solo emergency recovery, returned Ward, and
re-entry protection.

They do not own rhythm judgment, boss/Resolve/Momentum state, boss attack timing
or selection, movement/location state, individual ability definitions, item or
build definitions, encounter outcome, rewards, or presentation.

## 3. Approved inputs

- `RHYTHM_GAMEPLAY.md` identified normalized intent portions, coverage, and
  suspension facts.
- Boss Encounter threat/impact, active-target, phase, and finishing facts.
- Position/risk/cover state and legal movement outcomes.
- Selected equipment, build, ability, consumable, difficulty, and their
  versioned effect definitions.
- Multiplayer roster/connection state and identified cooperative actions.

## 4. Decision record

### Checkpoint A — Conversion and core intents

#### CM-01 — Effect pipeline and monotonicity

- **Status:** Approved.
- Combat accepts each identified Rhythm contribution portion at most once.
- The deterministic conversion order is:
  1. normalized earned contribution;
  2. versioned base conversion for its effective intent/effect kind;
  3. equipment and build modifiers whose declared conditions match;
  4. the position modifier applicable to that effect, such as dangerous-tier
     Attack output;
  5. encounter, target, or difficulty modifiers explicitly legal for that
     effect; and
  6. the effect-specific cap.
- Bonuses inside the same category combine additively. Distinct categories
  multiply once in the listed order. A source cannot enter more than one stage
  unless its definition explicitly contains separately budgeted effects.
- Internal calculation uses deterministic fixed precision. Rounding occurs only
  at the final display or discrete application boundary and never after every
  intermediate modifier.
- For identical non-rhythm state, increasing normalized performance cannot
  reduce the final effect. Ordinary zero performance remains zero regardless of
  multipliers or procs.
- The reliable base effect of an already-earned and armed Signature Special is
  the explicit zero-performance exception established by the GDD. It is not a
  general permission for equipment or build effects to trigger from no play.
- Difficulty ordinarily changes encounter targets, threat pressure, and
  explicitly declared recovery tuning rather than mutating Rhythm's normalized
  value. Every exception must be tagged at its legal pipeline stage.

#### CM-02 — Attack routing and Boss Encounter handoff

- **Status:** Approved.
- Combat converts an Attack-attributed Rhythm portion into one identified
  Attack-pressure packet containing source player, scoring group/intent segment,
  content revision, encounter instance, target opportunity, pre/post-modifier
  value, and applicable position/risk evidence.
- Boss Encounters supplies the current legal Resolve/Momentum destination and
  owns layer progress, opening/locking, break detection, Momentum state/cap, and
  finishing-cadence rules.
- Boss Encounters applies enough accepted pressure to the current layer, then
  routes overflow at a break and valid post-break Attack into Momentum according
  to its own state. Combat does not imitate or precompute that state.
- A future locked layer cannot receive Attack. When neither Resolve nor Momentum
  is a valid supplied destination, the contribution remains legitimate personal
  performance but creates no Attack-pressure application. UI should avoid
  presenting such material as if it damaged a locked target.
- Group aggregation sums accepted source packets without erasing their private
  attribution. Results may show each player's own Attack contribution and band
  total, but there is no public player damage ranking.

#### CM-03 — Defend targeting and mitigation

- **Status:** Approved.
- A player's defensive focus is automatic: the earliest unresolved telegraphed
  threat currently capable of affecting that player. No additional threat-
  selection control is introduced.
- When threats overlap, impact time determines priority; an exact tie uses the
  stable encounter-event order. Once a threat reaches Commit, its target set and
  any assigned mitigation are locked until that threat resolves or is canceled
  by its owning system.
- Defend contribution first fills the focused threat's personal mitigation
  capacity. Contribution beyond that cap converts at a weaker, bounded rate into
  Ward reinforcement. When no applicable threat exists, the entire contribution
  uses the weaker reinforcement path.
- Assigned mitigation is single-purpose. If movement, cover, cancellation, or
  another outcome means the threat no longer damages the player, unused value
  expires with that threat; it is not refunded, copied, or retroactively changed
  into Ward.
- Previously earned mitigation persists through subsequent intent selection and
  ordinary movement until its threat resolves. The incoming-effect rules in
  checkpoint B will decide its exact application order.
- UI and Audio consumers receive separate focused-threat, mitigation-fill,
  mitigation-cap, reinforcement, and expiration facts.

### Checkpoint B — Ward, damage, and return to play

#### CM-04 — Ward and incoming-effect resolution

- **Status:** Approved.
- Ward is the only first-release player-survival resource. A player begins an
  encounter with current Ward equal to their calculated normal maximum.
- Every incoming hit has stable encounter/source/target/effect identity. It
  resolves through these gates in order:
  1. owning-system cancellation or complete avoidance;
  2. active re-entry protection or other explicit immunity;
  3. position danger plus attack/encounter/difficulty scaling;
  4. applicable cover and other tagged reductions;
  5. committed mitigation assigned to that threat by Defend;
  6. available temporary Ward reinforcement; and
  7. current Ward.
- A gate that reduces the hit to zero prevents Ward loss. Used mitigation and
  reinforcement are consumed only by the rules of their own gate; one value
  cannot apply twice.
- Same-boundary hits retain separate identity and use stable encounter-event
  order. Ward clamps at zero and emits only one down transition for the boundary.
  CM-10 will place that batch relative to outgoing effects at the same musical
  time.
- Current/max Ward drives the Safe, below-50%, below-25%, and Empty presentation
  states. Exact numeric maximums and damage are balance data.
- A normal Perfect/Great/Good/Miss is never an incoming effect. A Miss matters
  by producing less offense, protection, and resources, not direct Ward damage.

#### CM-05 — Reinforcement and restoration

- **Status:** Approved.
- **Restoration** increases current Ward up to the current normal maximum.
- **Reinforcement** is a temporary, visually distinct segment on the same Ward
  meter. It is consumed before normal Ward, has a strict shared cap, and is not
  a second health resource or a way to avoid the down-at-zero rule.
- Multiple reinforcement sources add only to the cap. Reinforcement persists
  until consumed, the player is downed, or the encounter ends; additional
  applications at cap create no hidden reserve.
- Excess restoration is discarded. It becomes reinforcement only when an
  explicit, separately budgeted effect says so, and that conversion still obeys
  the reinforcement cap.
- A mid-combat maximum-Ward increase preserves the current absolute value unless
  the effect explicitly also grants current Ward. A decrease preserves the
  absolute value where legal, then clamps current Ward to the new maximum.
  Loadout swapping cannot occur during combat.
- Downed players cannot receive ordinary restoration or reinforcement. Revival
  and solo recovery use their dedicated return effects instead.
- Applied, capped, discarded, converted, consumed, and expired values keep
  source/effect/target attribution for feedback and Results.

#### CM-06 — Downing, revival, and solo recovery lifecycle

- **Status:** Approved.
- Reaching zero normal Ward after reinforcement and mitigation produces one
  down transition. The player remains at the prior location when legal, stops
  ordinary chart/combat contribution, and becomes ineligible for ordinary boss
  targeting.
- Prior accepted performance/combat history remains. Hype, unspent prepared
  resources, and already-spent-resource state persist through downing. Temporary
  reinforcement clears. Boss/position rules may substitute the nearest legal
  Middle location on return when the prior location is no longer valid.
- In cooperative play, revival may be initiated after any downing while at least
  one other human remains active. There is no arbitrary per-player or band
  revive-count limit; the finite song, valid candidate windows, and sacrificed
  combat contribution are its constraints.
- Each participant routes identified authentic-chart contribution exclusively
  into one revival target. That contribution cannot also become Attack, Defend,
  Special, or another revival. Multiple participants sum progress, accelerating
  completion; exact target and returned-Ward scaling remain tuning data.
- Revival progress belongs to the downed target and survives contributor changes
  until completed or invalidated by encounter end/all-humans-down. It never
  continues from fabricated or absent performance.
- Solo permits one emergency recovery attempt per encounter at a dynamically
  selected fair boundary. Success returns the tuned Ward amount. Challenge
  failure or any later down ends the solo attempt; there is no paid bypass or
  second recovery.
- A successful cooperative or solo return provides the configured Ward and
  roughly two beats of protection/settling. During it, the player is neither an
  ordinary target nor an ordinary chart contributor. Rhythm resumes afterward
  at the first fairly previewed note under `RHYTHM_GAMEPLAY.md`.
- Boss Encounters owns the all-humans-down and solo-attempt-ended defeat
  transition. Survival emits the authoritative player-state fact.

### Checkpoint C — Special, modifiers, and shared effects

#### CM-07 — Hype and Signature Special routing

- **Status:** Approved.
- Successful ordinary Attack or Defend performance creates its primary effect
  and also earns the separately budgeted slow passive Hype gain defined for the
  equipped Signature system.
- Selecting Special before Hype is full stores the prior Attack or Defend intent
  and redirects subsequent normalized contribution exclusively into the faster
  Hype conversion. It no longer produces the prior intent's primary effect.
- Reaching full Hype discards overflow, returns to the stored prior intent for
  the next playable material, and enters Ready. Hype stores exactly one charge,
  never fires automatically, and has no separate cooldown.
- Selecting Special while Ready arms the next ordinary scoring group as the
  activation performance. Selection alone does not spend Hype.
- If the player is downed or otherwise becomes invalid before that group begins,
  the arm cancels and full Hype remains. Once the group begins, the charge is
  committed. Its reliable base effect resolves at the following valid musical
  boundary even if execution is poor or participation ends mid-group.
- Normalized performance during the committed group scales only the effect's
  additional strength, duration, or utility. It cannot reduce the base below the
  ability definition's guarantee.
- Committed resolution consumes all Hype and returns to the prior intent. Hype
  persists through ordinary downing/revival but resets between encounters.
- Abilities & Cooperative Actions owns the equipped effect, valid resolution
  boundary, target/effect semantics, and balance values. Combat owns the charge
  state and performance routing contract described here.

#### CM-08 — Equipment, build, position, and difficulty modifiers

- **Status:** Approved.
- Every modifier definition is versioned and declares its source, effect tags,
  authoritative activation conditions, power-budget cost, legal pipeline stage,
  additive category, cap behavior, duration, and attribution.
- Modifiers in the same category add before independently budgeted categories
  multiply once in the CM-01 order. A definition cannot create recursive output
  that re-enters the pipeline as fresh normalized performance.
- Gear carries most direct power. Build rules emphasize bounded conditionals,
  tradeoffs, sidegrades, and behavioral interactions. All equipped effects share
  category caps and a build/loadout power budget so multiplicative synergy cannot
  escape the expected range or create one mandatory combination.
- Equipment and builds cannot change charts, timing windows, grades,
  calibration, Hold Assist value, note density, movement timing/charges,
  telegraph/reaction time, recovery-attempt counts, invulnerability, reward
  eligibility, or the authored position risk/reward ratios.
- Position supplies its universal Attack, incoming-danger, and reward facts at
  their owned stages. A build may trigger a separately budgeted, visibly
  attributable bonus while the player performs in a dangerous position; it may
  not rewrite the baseline position multiplier or remove its danger.
- Difficulty normally changes Resolve requirements, boss pressure, and
  explicitly declared survival/recovery values. It does not mutate Rhythm's
  normalized contribution or silently amplify every player effect.
- The typed category/power-budget model is extensible to later traits,
  sidegrades, sets, sockets, and advanced options without permitting new rhythm
  authority or required combat controls.

#### CM-09 — Multi-target, support, and cooperative routing

- **Status:** Approved.
- Every identified normalized contribution has exactly one primary destination:
  Attack, Defend, fast Hype, committed Signature performance, revival, Band Call,
  Crescendo, or another explicitly exclusive cooperative route.
- Separately budgeted passive readiness gain or triggered utility may observe an
  accepted contribution, such as slow Hype or Band Call readiness, but cannot
  copy its full primary effect or recursively become normalized performance.
- A multi-target/support definition declares one of two budget forms:
  - a fixed total divided deterministically among valid recipients; or
  - a per-recipient application constrained by one roster-aware group cap.
- Recipient eligibility and priority are part of the equipped/authored effect
  definition. The first release adds no manual teammate-target control merely to
  resolve support; invalid recipients are removed deterministically and the
  effect's fallback rule applies.
- Redirected revival and group-action contribution cannot also become the
  participant's ordinary intent output. Each participant's normalized share is
  calculated independently and then added, so weak execution never subtracts or
  cancels another participant's earned share.
- Duplicate instruments/roles do not change conversion or eligibility. Identity
  is player/effect based, not a unique-role slot.
- Acolytes create explicitly identified fixed NPC effect packets. They have no
  chart, grades, timing distribution, player performance/reward attribution, or
  positional risk multiplier. Their outputs obey solo-specific caps and
  restrictions, including the Vanguard's inability to perform a decisive layer
  break and no contribution to emergency recovery.
- Results may show a player's own support/group share and the band aggregate,
  never a public ranking of participants.

### Checkpoint D — Ordering, attribution, and completeness

#### CM-10 — Musical-boundary ordering and simultaneous events

- **Status:** Approved.
- Combat resolves events by logical musical time, never message arrival time.
  At one timestamp it uses this deterministic phase order:
  1. apply scheduled intent, participation, and eligibility changes;
  2. accept/finalize valid Rhythm contribution assigned to that boundary;
  3. resolve player, support, revival, and cooperative effects, including their
     Attack/Defend/restoration/protection consequences;
  4. resolve already committed boss impacts and hazards in their stable authored
     event order;
  5. apply resulting Ward thresholds and down transitions; and
  6. expose the final state snapshot for Boss Encounter outcome evaluation.
- Each phase reads the committed output of the prior phase and publishes one
  atomic result snapshot. Events inside a phase use stable event identity as the
  final tie-breaker.
- This order lets a defense, restoration, protection, or revival genuinely
  earned for that beat help before its impact. A layer break or outgoing effect
  on that beat does not cancel a boss attack that already passed Commit unless
  the attack definition contains an explicit legal cancellation behavior.
- A queued intent effective on the boundary applies before the playable event
  assigned to it. Previously resolved contribution never changes intent.
- Late transport may deliver a result after its logical boundary only inside the
  validation allowance. Accepted logical-time effects are inserted once into the
  authoritative history without treating receipt time as a new combat beat.
- Boss Encounters evaluates finishing success, all-humans-down, song-end state,
  and other encounter outcomes only after receiving the completed boundary
  snapshot. Combat does not choose between competing encounter outcomes.

#### CM-11 — Validation, invalidation, and exploit resistance

- **Status:** Approved.
- Before authoritative application, a packet validates immutable content and
  balance revisions; encounter, player, source-event, scoring-group, intent,
  target, and effect identity; logical time; eligibility/participation state;
  modifier sources; and recomputed value/cap evidence.
- Each accepted source/effect identity applies at most once. A duplicate is an
  idempotent no-op. Out-of-order dependencies may wait only within the bounded
  logical-time delivery allowance.
- Impossible, mismatched, negative, recursively generated, or wrong-state
  player effects are rejected rather than silently changed into a plausible
  value. Designed caps remain ordinary successful resolution and emit capped
  evidence.
- Server-confirmed history is immutable. Local immediate Rhythm feedback is not
  itself an authoritative combat application; technical architecture must
  preserve responsiveness without permitting unvalidated shared effects.
- Contribution logically completed before a connection loss may still be
  accepted within the delivery allowance. The disconnect boundary then blocks
  all absent-period contribution. Rhythm synchronization suspension likewise
  produces no inferred or synthetic effect.
- A boss impact already committed against the player resolves against the
  disconnect snapshot. After that committed set is processed, Multiplayer's
  disconnected state makes the performer ordinarily untargetable and unable to
  contribute until safe return.
- Rejection, duplicate suppression, cap, late acceptance, absence, and
  invalid-session evidence are private/system facts; they are not public blame
  labels.

#### CM-12 — Semantic outputs and balance boundary

- **Status:** Approved.
- Combat/Survival emits stable causally linked facts for:
  - intent queued, unavailable, effective, and automatic return;
  - contribution accepted, routed, split, capped, expired, or rejected;
  - Attack pressure created and Boss Encounter disposition received;
  - Defend focus, mitigation fill/use/expiry, and Ward reinforcement;
  - Hype gain, Ready, arm, commit, cancel, effect request, and consumption;
  - incoming effect gates, reductions, final damage, and avoidance;
  - current/max/reinforced Ward change and readable threshold crossings;
  - downing, revival targeting/progress/completion, solo recovery use/outcome,
    return protection, settling, and active-state restoration; and
  - multi-target, cooperative, NPC, source, recipient, and causal attribution.
- Each event identifies the content revision, balance revision, encounter,
  logical musical time, source, target, effect definition, causal source result,
  pre/post values, cap/discard evidence, and final state as applicable.
- Results consumers can derive personal Attack/Defend/Special contribution,
  Resolve/Momentum disposition, Ward loss/restoration/reinforcement, attacks
  avoided/defended/absorbed, revival help, group share, position modifiers, and
  personal combat history. They never need a public player ranking.
- UI and Audio receive the same semantic facts at immediate aggregation levels;
  Analytics observes them without becoming gameplay authority. Accessibility
  can add shape, label, audio, haptics, reduced motion, and private guidance but
  cannot alter resolution.
- Every encounter binds an immutable balance-data revision at staging lock.
  Live balance changes apply only to subsequently created encounters. Results
  retain the exact revision for reproducibility.
- Conversion rates, category caps, Ward values, reinforcement limits, Hype
  rates, mitigation budgets, restoration/revival amounts, delivery allowances,
  and thresholds remain versioned playtest data inside the approved behavioral
  rules.

## 5. Content Authoring reconciliation register

- Boss/encounter runtime data must expose stable threat identity, affected-player
  eligibility, Telegraph/Commit/Impact/Recovery boundaries, deterministic order
  for tied impacts, mitigation capacity, and cancellation/resolution state.
- Resolve/Momentum destination eligibility must be explicit at each scoring
  boundary so Attack is never applied to an inferred or locked target.
- Attack/hazard content must declare stable effect identity, affected targets or
  geometry, damage/effect tags, cancellation/avoidance state, cover interaction,
  difficulty scaling reference, and deterministic same-boundary order.
- Recovery candidate data must continue to support dynamic cooperative revival
  and one-use solo recovery without a fixed trigger timestamp.
- Signature activation, Band Call, and Crescendo data must expose valid musical
  resolution boundaries and effect identity without embedding private combat
  formulae in chart events.
- Authored multi-target/group effects must declare deterministic eligibility,
  distribution form, roster-aware cap reference, and invalid-recipient fallback.
- Every runtime package must provide stable same-boundary event order and enough
  logical-time identity to validate late delivery without using receipt order.

## 6. Open handoffs

- `BOSS_ENCOUNTERS.md` owns Resolve/Momentum, attacks, targeting, impacts,
  finishing evaluation, defeat, and encounter state.
- `ITEMS_AND_EQUIPMENT.md` and `BUILDS_AND_SPECIALIZATION.md` define legal
  modifier sources within the combination contract decided here.
- `ABILITIES_AND_COOPERATIVE_ACTIONS.md` defines Signature Specials, Band Calls,
  Crescendos, Solo Support, and effect-specific rules.
- `REWARDS_AND_ECONOMY.md` consumes banked risk and performance facts but never
  changes combat resolution retroactively.
- `MULTIPLAYER.md` supplies roster/connection authority and later technical
  architecture defines transport/server authority.
- `UI_UX.md`, `AUDIO_PRESENTATION.md`, Results, and Analytics consume semantic
  combat/survival output without owning it.

## 7. Change log

- **2026-08-21:** Created the working record. Progress is 0 of 12 questions.
- **2026-08-21:** Approved CM-01 through CM-03. Progress is 3 of 12 questions.
  Established modifier order, Attack handoff, and threat-focused Defend routing.
- **2026-08-21:** Approved CM-04 through CM-06. Progress is 6 of 12 questions.
  Established Ward resolution, reinforcement/restoration, downing, cooperative
  revival, and solo recovery.
- **2026-08-21:** Approved CM-07 through CM-09. Progress is 9 of 12 questions.
  Established Hype/Special commitment, modifier budgets, and multi-target/group
  effect routing.
- **2026-08-21:** Approved CM-10 through CM-12 and reconciled all twelve
  decisions into canonical `COMBAT.md`.
