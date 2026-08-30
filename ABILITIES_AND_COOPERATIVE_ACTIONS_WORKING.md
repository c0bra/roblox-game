# Bands Battle Abilities and Cooperative Actions Working Record

- **Status:** Completed; reconciled into canonical specification
- **Started:** 2026-08-24
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#46-abilities--cooperative-actions)
- **Included solo system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#47-solo-support)
- **Interview plan:** [`ABILITIES_AND_COOPERATIVE_ACTIONS_QUESTIONS.md`](ABILITIES_AND_COOPERATIVE_ACTIONS_QUESTIONS.md)
- **Combat dependency:** [`COMBAT.md`](COMBAT.md)
- **Encounter dependency:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Planned canonical result:** `ABILITIES_AND_COOPERATIVE_ACTIONS.md`

## 1. Role of this record

This document persists owner decisions while the Abilities, Cooperative Actions,
and Solo Support interview is in progress. It is not canonical until reconciled
into `ABILITIES_AND_COOPERATIVE_ACTIONS.md`.

## 2. Inherited boundary

Abilities & Cooperative Actions owns Signature, Band Call, and Crescendo
definitions; Hype; readiness/arming/consumption; Call request/lockout/invitation/
commit; Crescendo participation/tiers; group combination; and effect requests to
owning domains. Solo Support owns Vanguard/Warden/Herald state, fixed cadences,
suppression/recovery, formation requests, and capped NPC group contributions.

It does not own Special intent/input routing, raw Rhythm judgment, authored
candidate selection authority, Resolve/Ward/position application, revival,
multiplayer membership, item/build ownership, or presentation implementation.

## 3. Approved inputs

- Immutable loadout snapshot with Signature, Band Call, permitted modifiers,
  role, build, and balance revisions.
- Identified normalized Combat contribution and Special routing transitions.
- Boss Encounter Activity Map candidate/reservation decisions and current phase.
- Multiplayer roster/connection/eligibility state and player Join In actions.
- Rhythm scoring-group/musical boundary/participation facts.
- Combat/Survival/Positioning target and effect snapshots.

## 4. Decision record

### Checkpoint A — Signature Specials and Hype

#### AC-01 — Signature definition and effect contract

- **Status:** Resolved 2026-08-24.
- A Signature definition declares a stable definition identity and revision,
  effect family/tags, legal owner and target states, deterministic target and
  distribution policy, guaranteed base effect, performance-scaled bonus curve
  and cap, typed domain effects, valid musical resolution boundary, fallback,
  and multimodal semantic cues.
- Offense, Ward, support, and positioning results are requests to the owning
  Combat, Survival, or Positioning domain. A Signature cannot directly rewrite
  another system's state or bypass its validation and caps.
- Any role or instrument may equip any Signature unless an explicit thematic
  restriction still leaves equivalent valid choices and never creates a
  required party composition. Signatures are not class locks.
- A Signature cannot change Rhythm judgments or chart timing, movement timing,
  recovery counts, encounter rewards, or core control semantics. Any positional
  result uses the approved typed Positioning contract.
- Definitions are data-driven and revisioned so encounter snapshots, semantic
  events, results, replays, and tuning evidence identify the exact contract used.

#### AC-02 — Hype ownership and lifecycle integration

- **Status:** Resolved 2026-08-24.
- Abilities & Cooperative Actions is the authoritative owner of each player's
  Hype amount and `Accumulating`, `Ready`, `Armed`, `Committed`, and `Consumed`
  lifecycle states. `COMBAT.md` continues to own Special intent/input routing,
  normalized performance conversion, and the handoff back to the prior intent.
- Combat emits identified, idempotent eligible-contribution facts and marks
  whether the approved slow Attack/Defend or fast unready-Special gain applies.
  Abilities applies the configured gain and threshold exactly once.
- Hype stores at most one complete charge. Reaching the threshold enters Ready,
  signals Combat to restore the previously selected intent, and discards
  overflow. Ready Hype does not auto-fire and cannot begin another charge.
- Hype and its Ready/Armed state persist through downing and revival. It resets
  between encounters and may be consumed only through the committed Signature
  contract. State transitions carry encounter, player, definition revision,
  activation, and source identities for idempotency.
- Existing Combat wording that calls Combat the Hype owner must be reconciled
  when the canonical Abilities specification is published. This is an ownership
  correction only and does not change approved player behavior.

#### AC-03 — Activation scheduling, invalidation, and guaranteed base

- **Status:** Resolved 2026-08-24.
- Pressing a Ready Signature control arms the next ordinary scoring group for
  that player. Arming identifies the Signature revision, player, encounter, and
  selected group once it becomes known; repeated input cannot create a second
  activation.
- If the player becomes invalid before the scoring group begins, the arm is
  canceled and full Hype is preserved. Invalidity includes downing, leaving the
  encounter, or an encounter terminal/cancellation transition.
- Scoring-group start commits the activation and reserves the earliest valid
  clean resolution boundary after that group. From commitment onward, the
  guaranteed base cannot be canceled by downing, target changes, phase changes,
  or weak performance.
- Performance accepted before a scoring suspension or downing determines the
  bonus. Unperformed material adds nothing but cannot reduce the guaranteed
  base. The definition's deterministic fallback handles an invalid intended
  target or distribution at resolution.
- Hype is consumed only when the committed effect enters guaranteed resolution.
  A critical system failure that prevents resolution preserves or refunds the
  charge idempotently; normal weak play, downing after commitment, target
  fallback, or phase change is not a refund condition.
- Arm, cancel, commit, resolve, fallback, consume/refund, and restored-intent
  transitions emit distinct semantic feedback without exposing raw timing math.

### Checkpoint B — Band Call readiness and invitations

#### AC-04 — Band Call readiness, ownership, and lockout

- **Status:** Resolved 2026-08-24.
- Abilities owns personal Band Call readiness, Ready/Reserved/Committed/Used
  state, the encounter-wide pending-Call reservation, and the shared lockout.
- Any accepted successful human-played chart performance may build readiness
  once through a separately budgeted observer, regardless of whether its
  primary route is Attack, Defend, Special, recovery, or a cooperative action.
  Misses, absent material, fixed effects, and acolyte output provide none.
- Readiness is shown on the equipped Call control, caps at Ready, and discards
  overflow. Partial or Ready state persists through downing and same-encounter
  reconnection, then resets between encounters. A committed use sets Used and
  prevents recharging or committing a second Call in that encounter.
- Item and build hooks may adjust readiness rate or Call potency only within
  approved budgets and caps. They cannot grant another use, bypass a pending
  reservation, shorten the shared lockout below its global floor, or fabricate
  performance.
- One pending Call reservation blocks other requests before commitment. At Call
  start, that reservation becomes the shared lockout for a revisioned duration
  initially tuned around eight measures. Unavailable controls distinguish
  personal-not-ready, pending-Call, lockout, ineligible, and already-used states.
- Simultaneous valid requests use the authoritative server receipt sequence and
  stable player identity as a final tie-breaker. Only the winner reserves a
  candidate; other players keep readiness and receive explicit feedback.

#### AC-05 — Request, candidate scheduling, and invitation

- **Status:** Resolved 2026-08-24.
- An active, non-downed Ready player may request the equipped Call when no
  cooperative action, recovery state, pending Call, or shared lockout blocks it.
  The request carries encounter, player, Call definition/revision, and unique
  request identity.
- Boss Encounters filters the Activity Map and reserves the earliest valid
  authored ensemble candidate within the Call's maximum delay, subject to the
  approved event-priority and preview rules. A higher-priority conflict searches
  later candidates within that delay instead of displacing guaranteed content.
- During the queue, each eligible teammate receives a multimodal invitation
  identifying the initiator, Call name, effect, and beat-based countdown. The
  stable Join In control provisionally accepts; ordinary play continues, and an
  accepted player may withdraw until commitment.
- Movement or any state change that makes an invitee ineligible before the
  boundary removes only that invitee. Downing, disconnecting, or otherwise
  invalidating the initiator cancels the whole uncommitted request, releases its
  reservation, and preserves the initiator's readiness.
- If no candidate remains within the maximum delay, the request cancels and
  preserves readiness. Band Calls never synthesize a universal fallback beat or
  use an unvalidated chart passage.

#### AC-06 — Commit, participant performance, and effect resolution

- **Status:** Resolved 2026-08-24.
- At the scheduled boundary, Abilities spends the initiator's once-per-encounter
  use, locks the initiator plus remaining accepted participants, releases the
  pending reservation into the shared lockout, and snapshots the eligible-roster
  scaling inputs. Repeated boundary delivery is idempotent.
- Every committed participant, including the initiator, redirects authentic
  personal chart material exclusively to the Call for the definition's one- or
  two-measure duration. That material cannot also become ordinary Attack,
  Defend, Special, recovery, or another group action.
- The initiator guarantees the definition's base effect after commitment and
  may also earn a performance share. Each participant is independently
  normalized and capped; weak, missed, or absent play merely omits some positive
  share and cannot reduce the base or another participant's contribution.
- Definition and roster caps bound the combined bonus. Typed target and
  distribution policies determine application through the owning domains, with
  a deterministic fallback if the intended target is invalid at resolution.
- A participant downed or disconnected during the window retains already
  accepted contribution and adds nothing afterward. This does not cancel the
  committed Call or its guaranteed base.
- The Call resolves once at the first valid musical boundary after its
  performance window. Results and semantic events separately attribute the base,
  each human share, fixed solo support, fallback, final effect, and exact
  definition/balance revisions without ranking public players.

### Checkpoint C — Crescendos

#### AC-07 — Candidate budget, selection, and guarantee

- **Status:** Resolved 2026-08-24.
- A standard encounter authors and validates two to four Crescendo candidates
  for every supported instrument, difficulty, and roster. Easy, Normal, and
  Hard each guarantee exactly one normal activation.
- At encounter start, Boss Encounters filters and deterministically ranks viable
  candidates using the immutable roster/difficulty/content snapshot, musical-fit
  policy, and encounter seed. This permits authored variation while reproducing
  a selection from its inputs and revisions.
- The selected normal candidate is a required-event reservation. It outranks
  Band Calls and optional events but cannot displace immutable song/Finishing
  boundaries, an already committed window, or urgent recovery under the
  approved scheduling hierarchy.
- Candidate validation requires strong playable coverage, roughly two measures
  of readable reaction, sustained instrument activity, and sufficient separation
  from incompatible attacks, recovery, silence/quiet transitions, and the
  Finishing Cadence.
- If a reserved candidate becomes invalid before preview, the selector reserves
  the next valid unused candidate. A committed preview/window is protected by
  scheduling priority. Crescendos never synthesize a generic fallback passage.
- An exceptionally short song may ship with one candidate only through an
  explicit, recorded exception that still passes full coverage/conflict
  validation and human review. Standard and longer songs retain two to four.
- Missing required coverage blocks content publication. If every candidate is
  exhausted by an unexpected runtime defect, the encounter continues without a
  player reward penalty or fake event and records a critical content-integrity
  failure for immediate diagnosis.

#### AC-08 — Preview, opt-in, performance, and tiers

- **Status:** Resolved 2026-08-24.
- A prominent multimodal preview begins roughly two measures before the window
  and identifies Crescendo identity, effect, musical countdown, participation
  state, and the stable Join In control without obscuring ordinary rhythm play.
- Participation costs no resource and is optional. An eligible active,
  non-downed player may provisionally Join In or withdraw before commitment;
  declining or becoming ineligible leaves ordinary play unchanged.
- At the boundary, the valid participant roster and eligible-human-roster
  scaling snapshot lock. Participants redirect roughly two measures of their
  authentic personal chart exclusively to the Crescendo; nonparticipants retain
  normal intent routing.
- Each participant's result is normalized for instrument and difficulty before
  additive assembly. Independent nonnegative shares cannot reduce another
  player, and a down/disconnect during the window preserves only contribution
  already accepted.
- Revisioned thresholds map the combined capped total to Echo, Crescendo, or
  Full Crescendo. Thresholds scale from the eligible human roster rather than
  only accepted participants: one expert earns meaningful value but cannot stand
  in for an inactive full band.
- Echo is the minimum activated outcome. Near-zero performance remains Echo;
  individual mistakes cannot cancel the event. Values must keep declining and
  continuing ordinary play a legitimate choice rather than an exploit for a
  stronger redirected effect.
- Players cannot initiate, reschedule, or cancel the authored Crescendo itself;
  they control only their own provisional participation before commitment.

#### AC-09 — Crescendo effects and Easy recovery activation

- **Status:** Resolved 2026-08-24.
- The default candidate effect is a strong Resolve burst paired with modest Ward
  reinforcement. Boss-authored alternatives may use typed offense, defense,
  recovery, or positional effects when the preview clearly communicates the
  exact gameplay consequence.
- Every effect definition declares capped Echo, Crescendo, and Full Crescendo
  values, target/distribution rules, deterministic invalid-target fallback, and
  semantic cues. It resolves through the domain that owns the affected state.
- Crescendo effects cannot add a revival or solo-recovery attempt, directly
  complete encounter victory, bypass the Finishing Cadence, or otherwise
  guarantee success. Recovery-oriented effects create bounded help while
  leaving subsequent human performance necessary.
- Easy may evaluate one authored recovery decision only after the required
  Crescendo resolves. A revisioned deterministic behind-state rule combines song
  progress, remaining boss resistance against the expected range, and current
  living-player survival pressure.
- When substantially behind and an unused valid candidate remains, Easy may
  reserve at most one additional recovery Crescendo. It uses the same preview,
  opt-in, performance, tier, conflict, and no-fallback contracts as the normal
  event. The candidate becomes consumed whether the resulting tier is weak or
  strong.
- The extra activation grants no separate participation reward, does not change
  selected difficulty or reward eligibility, and cannot occur on Normal or Hard.
  If the behind condition is false or no valid candidate remains, nothing is
  scheduled.

### Checkpoint D — Solo Support and outputs

#### AC-10 — Vanguard, Warden, and Herald cadences

- **Status:** Resolved 2026-08-25.
- Vanguard observes each successfully completed ordinary human scoring group,
  regardless of active combat intent, and requests a small fixed Resolve packet
  after the corresponding human effect. Recovery, revival, Band Call, and
  Crescendo material cannot trigger it.
- Vanguard's packet belongs only to the resistance layer current for that
  scoring group, never spills forward, and clamps before the decisive break. A
  successful human Attack must break the layer.
- Warden uses encounter-authored clean pulse boundaries, initially targeted at
  roughly once every eight measures. Each pulse is previewed, reinforces Ward
  to the legal cap, and is neither banked nor retargeted when missed, suppressed,
  or unnecessary.
- Herald adds a bounded bonus only when authentic successful human performance
  earns Band Call readiness. It cannot passively fill readiness from inactivity,
  fixed effects, or acolyte output.
- An active Herald at a solo Band Call or Crescendo commitment supplies one
  small fixed, capped squad share. Vanguard and Warden may join presentation but
  do not create additional group shares.
- Acolyte values are fixed by revisioned encounter/difficulty configuration and
  produce no score, judgments, performance, risk, or reward credit. First-release
  player items and builds do not modify acolytes.

#### AC-11 — Suppression, positioning, and solo group participation

- **Status:** Resolved 2026-08-25.
- Each acolyte has an encounter-authored preferred tactical location. Positioning
  places or repositions it automatically at clean phase/musical boundaries.
  Visual formation offsets handle player co-location without swapping, blocking,
  reserving capacity, or changing gameplay location state.
- Acolyte location never earns a player risk multiplier. An ordinary committed,
  clearly telegraphed attack that affects the location suppresses an active
  acolyte at impact for a configured duration initially around four measures.
- Suppression does not stack, refresh, or extend. It pauses that acolyte's future
  support and displays its unavailable function plus beat-based countdown. The
  acolyte recovers automatically at the first clean boundary after the duration.
- Suppressed Vanguard triggers and Warden pulses are omitted rather than banked;
  Herald adds no readiness bonus. No catch-up effect occurs on recovery.
- Herald eligibility for a Band Call or Crescendo fixed share snapshots at group
  commitment. Later suppression cannot remove an already committed share; being
  suppressed at commitment supplies none.
- During solo emergency recovery all acolytes provide presentation only. They
  cannot score, shorten, complete, or rescue the challenge. Acolytes cannot be
  permanently downed, revived, individually equipped, commanded, or turned into
  escort objectives.

#### AC-12 — Semantic outputs and content completeness

- **Status:** Resolved 2026-08-25.
- Every Hype, Signature, Band Call, Crescendo, and acolyte transition emits an
  authoritative semantic fact with attempt, source, player/group, definition and
  balance revisions, logical musical time, state transition, target/fallback,
  contribution attribution, and result as applicable.
- UI and Audio receive presentation-neutral state, countdown, effect, warning,
  success/failure, and accessibility-cue facts. Results and Analytics receive
  human contribution separately from guaranteed bases and fixed solo support;
  public results never rank individual players.
- State-changing commands and facts are idempotent. Repeated gain, arm, request,
  Join In, commit, contribution, resolution, spend/refund, suppression, or
  recovery delivery cannot duplicate state or effects.
- Authoring supplies all revisioned Signature/Call/Crescendo contracts, candidate
  windows and conflicts, tier and roster rules, Hype/readiness values, acolyte
  cadences/locations/suppression rules, semantic cue keys, and accessibility
  alternatives.
- Publication validation covers every supported role, chart, difficulty, and
  roster; legal targets/fallbacks and effect-domain contracts; prohibited
  modifiers; required Crescendo coverage; and short-song exception evidence.
  Missing or inconsistent required data blocks publication.

## 5. Content Authoring reconciliation register

- Signature definitions require stable identity/revision, family/tags, legal
  states and targets, deterministic distribution/fallback, base and bonus
  contracts, musical resolution rules, typed effects, and semantic cue keys.
- Hype tuning requires revisioned slow/fast gain values and a one-charge
  threshold. Validation rejects definitions that alter prohibited Rhythm,
  movement-timing, recovery-count, reward, or control contracts.
- Band Call definitions require readiness tuning, maximum scheduling delay,
  preview lead, active duration, shared-lockout duration, legal participants,
  base effect, independent-share curve/caps, roster cap, typed target/
  distribution/fallback rules, and semantic cue keys. Activity Map validation
  must prove candidate coverage for supported instruments, difficulties, and
  rosters without relying on a universal fallback.
- Standard encounters require two to four Crescendo candidates with coverage,
  reaction, activity, conflict, priority, effect, and Easy-recovery metadata.
  A one-candidate short-song exception requires explicit approval. Definitions
  require eligible-roster-scaled tier thresholds/caps, typed tier effects,
  preview cues, deterministic selection inputs, and the Easy behind-state rule.
- `CONTENT_AUTHORING.md` scheduling language must be reconciled so Band Call and
  Crescendo invitation/performance windows are unambiguously mutually exclusive
  under the required-event priority contract.
- Solo encounter data requires preferred acolyte locations, Vanguard trigger and
  clamp values, previewed Warden cadence/pulse values, Herald readiness/group
  caps, suppression duration/vulnerability, formation/cue references, and full
  difficulty coverage. Validation rejects risk/reward attribution, fabricated
  performance, layer-breaking Vanguard output, or recovery assistance.
- Export validation must prove stable identities/revisions and complete semantic
  and accessibility cue coverage for every ability/group/solo state transition.

## 6. Open handoffs

- `COMBAT.md` owns intent, normalized effect conversion, domain effect
  application, and same-boundary order; Hype ownership wording will be reconciled.
- `BOSS_ENCOUNTERS.md` owns candidate selection/reservation/conflict priority and
  application of Resolve/position encounter effects.
- `ITEMS_AND_EQUIPMENT.md` owns equipped definition references and immutable
  loadout snapshots.
- `BUILDS_AND_SPECIALIZATION.md` owns permitted behavior-changing ability hooks.
- `MULTIPLAYER.md` owns roster/connection state and delivery of invitations/
  player inputs.
- Player Data owns durable equipped/unlocked definitions; UI, Audio, Results, and
  Analytics consume semantic ability/group/solo facts.

## 7. Change log

- **2026-08-24:** Created the working record. Progress is 0 of 12 questions.
- **2026-08-24:** Approved AC-01 through AC-03. Signature definitions, Hype
  authority/lifecycle, and reliable activation are resolved; progress is 3 of
  12 questions.
- **2026-08-24:** Approved AC-04 through AC-06. Band Call readiness, scheduling,
  invitation, commitment, contribution, and resolution are resolved; progress
  is 6 of 12 questions.
- **2026-08-24:** Approved AC-07 through AC-09. Crescendo selection, guaranteed
  activation, participation/tiers, typed effects, and Easy's bounded recovery
  activation are resolved; progress is 9 of 12 questions.
- **2026-08-25:** Approved AC-10 through AC-12. Solo acolyte functions,
  positioning/suppression, semantic outputs, and authoring completeness are
  resolved. All 12 questions were reconciled into the canonical specification.
