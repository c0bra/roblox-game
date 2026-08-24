# Bands Battle Combat Specification Questions

- **Status:** Interview complete; 12 of 12 questions resolved and reconciled
- **Started:** 2026-08-21
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#42-combat)
- **Included system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#43-player-survival--recovery)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Rhythm input contract:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)
- **Working record:** [`COMBAT_WORKING.md`](COMBAT_WORKING.md)
- **Canonical result:** [`COMBAT.md`](COMBAT.md)

## 1. Interview method

This interview uses four checkpoints of three questions. It does not re-ask
settled GDD or Rhythm Gameplay rules. Defaults focus on deterministic effect
semantics; final numeric balance remains playtest data unless the owner chooses
to settle a number here.

Answers are persisted after each checkpoint. When all twelve questions are
resolved, they are reconciled into canonical `COMBAT.md` with Player Survival &
Recovery as a major section.

## 2. Fixed inherited decisions

- Rhythm supplies immutable identified Attack/Defend/Special contribution
  portions, execution quality, participation coverage, and suspension facts.
- Combat never regrades notes or uses raw note count as power.
- Attack advances the active Resolve layer; valid output after an early break is
  available to the Boss Encounter system for Momentum.
- Defend uses the ordinary instrument chart and protects against readable boss
  threats; there is no separate defense chart.
- Special routes performance to Hype or an armed Signature Special; ability
  definitions belong to Abilities & Cooperative Actions.
- Ward is the only first-release player survival bar. Every player starts full;
  empty Ward downs them.
- A Miss never directly damages Ward. Boss impacts, hazards, and other explicit
  effects do.
- Dangerous positions increase outgoing Attack, incoming danger, and potential
  earned rewards. Equipment cannot change the positional risk/reward ratio.
- Equipment/builds modify combat after normalized rhythm scoring and never
  alter charts, judgments, timing, calibration, movement timing, or recovery
  attempt counts.
- All-human-down co-op defeat belongs to Boss Encounters; Combat/Survival reports
  each player's state.
- Cooperative revival uses active teammates' real performance. Solo has one
  emergency recovery attempt and no paid bypass.

## 3. Question plan

### Checkpoint A — Conversion and core intents

#### CM-01 — Effect pipeline and monotonicity [Resolved]

- **Decision needed:** In what deterministic order does normalized rhythm value
  become a combat-effect value?
- **Must resolve:** Base conversion, additive versus multiplicative modifiers,
  difficulty/position/build ordering, caps, zero-performance behavior,
  monotonicity, and round-off.
- **Owner decision:** Each identified Rhythm contribution is processed once in
  this order: normalized value, intent conversion rate, equipment/build
  modifiers, applicable position modifier, encounter/target modifiers, then
  effect cap. Bonuses in one category add before distinct categories multiply.
  Calculation retains deterministic precision and rounds only final display or
  discrete application values. Under identical state, better performance cannot
  yield a weaker result. Zero performance produces zero ordinary effect; an
  already-charged Signature Special's guaranteed base effect is the explicit
  exception.

#### CM-02 — Attack routing and Boss Encounter handoff [Resolved]

- **Decision needed:** What does Combat produce from Attack contribution, and
  what state remains owned by Boss Encounters?
- **Must resolve:** Active-target identity, Resolve-pressure packet, early-break
  and locked-layer behavior, Momentum handoff, overkill, invalid/no target,
  group aggregation, and attribution.
- **Owner decision:** Combat converts Attack portions into identified,
  source-attributed Attack-pressure packets for Boss Encounters. Boss Encounters
  owns applying pressure to the active Resolve layer, detecting its break, and
  routing packet overflow or valid post-break output into capped Momentum. A
  locked or unavailable destination cannot be damaged. When Boss Encounters
  supplies no valid Resolve or Momentum destination, the performance remains in
  personal results but creates no Attack effect. Group totals preserve private
  individual attribution without a public damage ranking.

#### CM-03 — Defend targeting and mitigation [Resolved]

- **Decision needed:** Which threat receives Defend contribution and what
  happens when no applicable impact is pending?
- **Must resolve:** Threat association, commitment cutoff, overlapping threats,
  mitigation versus Ward reinforcement, excess value, expiration, movement,
  and player feedback.
- **Owner decision:** Defend contribution automatically focuses the earliest
  unresolved telegraphed threat capable of affecting that player. It fills the
  threat's mitigation up to its cap; additional contribution uses a weaker,
  bounded Ward-reinforcement conversion. With no applicable threat, all Defend
  contribution uses that reinforcement conversion. Committed mitigation stays
  associated with its threat until resolution. If movement or another outcome
  avoids the threat, unused mitigation expires rather than being spent twice or
  retroactively converted. UI must identify the focused threat and show
  mitigation/reinforcement separately.

### Checkpoint B — Ward, damage, and return to play

#### CM-04 — Ward and incoming-effect resolution [Resolved]

- **Decision needed:** How are incoming effects, mitigation, cover, position,
  and Ward applied deterministically?
- **Must resolve:** Max/current Ward, damage order, immunity/protection,
  simultaneous effects, zero crossing, readable thresholds, and the rule that a
  normal Miss never creates damage.
- **Owner decision:** An identified incoming hit resolves through cancellation
  or avoidance, re-entry protection, position/attack danger, cover/tagged
  reductions, threat-bound Defend mitigation, temporary Ward reinforcement, and
  finally current Ward. Ward clamps at zero and reaching zero creates one down
  transition. Same-boundary hits retain identity and use deterministic encounter
  order; exact boundary precedence is finalized in CM-10. Readable safe,
  below-50%, below-25%, and empty states follow current/max Ward. A normal
  Rhythm Miss never creates an incoming-damage packet.

#### CM-05 — Reinforcement and restoration [Resolved]

- **Decision needed:** How do Defend, support, authored recovery, abilities, and
  consumables preserve or restore Ward?
- **Must resolve:** Reinforcement versus restoration, maximum-Ward changes,
  overheal, caps, source stacking, downed-player eligibility, and attribution.
- **Owner decision:** Restoration refills current Ward only to its normal
  maximum. Reinforcement is a strictly capped temporary segment on that same
  Ward meter, consumed before current Ward and never treated as another health
  resource. It combines only to its cap and persists until consumed, downing, or
  encounter end. Excess restoration is discarded unless an explicit effect
  converts it to reinforcement. A mid-combat maximum change preserves current
  absolute Ward unless the effect explicitly grants Ward, then clamps if the
  maximum falls below it. Downed players cannot receive ordinary restoration or
  reinforcement; all applied/unused values retain source attribution.

#### CM-06 — Downing, revival, and solo recovery lifecycle [Resolved]

- **Decision needed:** What survival state machine and contribution rules govern
  downing and return?
- **Must resolve:** Down boundary, target eligibility, retained resources,
  revival routing and acceleration, returned Ward, solo attempt, challenge
  success/failure, re-entry protection, and second-down behavior.
- **Owner decision:** Downing preserves prior contribution, Hype, location when
  still legal, and already-spent resource state while removing the player from
  ordinary targeting and performance. Co-op revival may follow any downing when
  another human remains active, with no arbitrary per-player revive limit.
  Participants redirect authentic chart contribution away from other combat
  routes into one identified revival target; multiple participants accelerate
  progress. Solo has exactly one emergency recovery attempt: failure or a second
  down ends the attempt. Success returns a tuned Ward amount and provides about
  two beats of protection/settling before targeting and rhythm resume.

### Checkpoint C — Special, modifiers, and shared effects

#### CM-07 — Hype and Signature Special routing [Resolved]

- **Decision needed:** How does Combat route Special contribution while leaving
  individual ability behavior to its owning system?
- **Must resolve:** Slow/fast Hype generation, previous-intent return, arming,
  activation group, guaranteed base effect, performance scaling, consumption,
  downing persistence, and unavailable states.
- **Owner decision:** Successful Attack/Defend performance also earns slow
  passive Hype. Selecting unready Special redirects contribution exclusively to
  faster Hype gain. Full Hype discards overflow, restores the prior Attack or
  Defend intent, stores one charge, and never fires automatically. Selecting
  ready Special arms the next ordinary scoring group. Once that group begins,
  the charge is committed and its reliable base effect resolves at the valid
  boundary; execution scales only its bonus. Downing before group start cancels
  the arm but preserves Hype. Hype otherwise persists through downing, is
  consumed on committed resolution, and resets between encounters.

#### CM-08 — Equipment, build, position, and difficulty modifiers [Resolved]

- **Decision needed:** Which modifier classes are legal and how do they combine
  without changing musical skill or creating mandatory builds?
- **Must resolve:** Source categories, conditional activation, shared power
  budget, stacking/caps, dangerous-position ratios, difficulty scaling,
  extension points, and prohibited modifiers.
- **Owner decision:** Each versioned modifier declares one or more separately
  budgeted effect categories, authoritative conditions, power cost, pipeline
  stage, and cap. Same-category values add before distinct legal categories
  multiply. Equipment/builds cannot affect Rhythm, movement timing, recovery
  counts, or the authored positional risk/reward ratio. A build may create a
  separately capped effect triggered by dangerous-position play without
  rewriting the universal position multiplier. Difficulty ordinarily changes
  targets, incoming pressure, and explicitly declared recovery tuning rather
  than normalized player output. A shared power budget and category caps prevent
  mandatory multiplicative combinations while allowing later extension.

#### CM-09 — Multi-target, support, and cooperative routing [Resolved]

- **Decision needed:** How does one identified contribution create personal,
  teammate, revival, or group effects without double-spending?
- **Must resolve:** Recipient selection, splits versus copies, group caps,
  revival sacrifice, NPC support identity, duplicate roles, and public
  attribution without player ranking.
- **Owner decision:** Identified contribution has one primary destination; any
  secondary readiness gain or triggered effect must be separately budgeted and
  cannot repeat the full value. A multi-target effect declares either a fixed
  total split among valid recipients or a roster-capped group application, with
  deterministic effect-owned targeting rather than another combat control.
  Redirected revival/group contribution cannot also feed the player's ordinary
  intent. Participant shares remain independent so weak play never subtracts
  from another. Acolytes emit explicitly identified, fixed, capped NPC effects
  with no chart, judgment, player attribution, or risk multiplier.

### Checkpoint D — Ordering, attribution, and completeness

#### CM-10 — Musical-boundary ordering and simultaneous events [Resolved]

- **Decision needed:** What deterministic precedence applies when contribution,
  intent changes, impacts, breaks, downing, revival, and ability resolution
  share a boundary?
- **Must resolve:** Same-timestamp order, atomic state snapshots, late events,
  layer breaks, mitigation cutoffs, finishing performance, and no retroactivity.
- **Owner decision:** At one logical musical timestamp, scheduled intent and
  participation changes apply first, then accepted player contribution and
  player/group effects, then committed boss impacts, then Ward/downing state
  changes, and finally encounter-outcome evaluation. Thus a genuinely on-beat
  defense, restoration, or revival may help before impact. A same-beat Resolve
  break does not cancel an already committed attack unless that attack explicitly
  says so. Network arrival time never changes an event's authored logical time;
  no later event acts retroactively.

#### CM-11 — Validation, invalidation, and exploit resistance [Resolved]

- **Decision needed:** Which combat outputs are accepted, rejected, capped, or
  retained through disconnect and synchronization failure?
- **Must resolve:** Idempotency, content/player/state identity, impossible
  values, negative contribution, accepted-history immutability, absence,
  disconnect-impact handling, and anti-double-application.
- **Owner decision:** An authoritative combat effect is accepted once only after
  content revision, encounter, player, source event, target, state, logical time,
  and value validation. Duplicates are ignored; impossible or mismatched effects
  are rejected rather than silently clamped, while valid effects still use
  designed caps. Confirmed history is immutable. A contribution logically
  completed before disconnect may arrive within a bounded delivery allowance;
  absent or synchronization-suspended time produces nothing. Already committed
  boss impacts resolve against the disconnect snapshot before untargetability.

#### CM-12 — Semantic outputs and balance boundary [Resolved]

- **Decision needed:** Which exact facts must Combat/Survival expose and which
  values remain tunable rather than hard-coded design?
- **Must resolve:** Effect and state event catalog, source/target attribution,
  causal links, result statistics, UI/Audio/Analytics consumers, balance-table
  ownership/versioning, accessibility, and completion audit.
- **Owner decision:** Combat emits causally linked, fully attributed intent,
  routing, Attack, Defend, Hype/Special, Ward, incoming-effect, restoration,
  revival, protection, cap/expiration, rejection, and state-change facts for UI,
  Audio, Results, and Analytics. Every attempt binds the exact versioned balance
  data used. Balance changes begin only with a new encounter and never mutate an
  active song. Numeric rates, caps, Ward values, and thresholds remain playtest
  data constrained by this specification. Accessible presentation changes how
  facts are conveyed, not resolution.

## 4. Completion criteria

`COMBAT.md` is complete only when:

- CM-01 through CM-12 are resolved;
- every normalized performance unit has one deterministic route and cannot be
  spent twice;
- modifier order is monotonic, capped, and never changes rhythm judgment;
- Attack and Defend have explicit Boss Encounter effect contracts;
- Ward, downing, revival, solo recovery, and re-entry have a complete state
  model;
- simultaneous effects and disconnect/desync cases cannot produce ambiguous
  survival results;
- output supports Results without public damage ranking; and
- every new authored-data need is registered for Content Authoring
  reconciliation.

## 5. Change log

- **2026-08-21:** Created the concise 12-question plan from the approved GDD,
  Systems Map, and Rhythm Gameplay contract.
- **2026-08-21:** Resolved CM-01 through CM-03, establishing the modifier order,
  Attack-pressure/Boss Encounter boundary, and automatic Defend focus.
- **2026-08-21:** Resolved CM-04 through CM-06, establishing Ward damage order,
  temporary reinforcement, downing state, cooperative revival, and solo
  recovery.
- **2026-08-21:** Resolved CM-07 through CM-09, establishing Hype/Special
  commitment, legal modifier budgets, and non-duplicating multi-target routing.
- **2026-08-21:** Resolved CM-10 through CM-12 and reconciled all twelve answers
  into canonical `COMBAT.md`.
