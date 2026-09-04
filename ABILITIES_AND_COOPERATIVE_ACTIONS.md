# Bands Battle Abilities, Cooperative Actions, and Solo Support

- **Status:** Approved
- **Approved:** 2026-08-25
- **Parent systems:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#46-abilities--cooperative-actions)
  and [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#47-solo-support)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Rhythm dependency:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)
- **Combat dependency:** [`COMBAT.md`](COMBAT.md)
- **Encounter dependency:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Authoring dependency:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)
- **Decision source:** [`ABILITIES_AND_COOPERATIVE_ACTIONS_WORKING.md`](ABILITIES_AND_COOPERATIVE_ACTIONS_WORKING.md)
- **Interview plan:** [`ABILITIES_AND_COOPERATIVE_ACTIONS_QUESTIONS.md`](ABILITIES_AND_COOPERATIVE_ACTIONS_QUESTIONS.md)

## 1. Role and authority

Abilities & Cooperative Actions owns Signature, Band Call, and Crescendo
definitions; Hype; personal Call readiness and use; Signature and group-action
arming, commitment, participation, combination, cancellation, consumption, and
resolution requests; Crescendo tiers; the shared Call reservation/lockout; and
presentation-neutral semantic ability facts.

Solo Support owns Vanguard, Warden, and Herald runtime state, fixed functions,
authored cadences, suppression/recovery, formation requests, and capped fixed
solo group contributions.

This document does not own Special intent/input routing, raw Rhythm judgments,
Activity Map candidate selection authority, Resolve/Ward/position application,
revival or solo recovery, multiplayer membership, item/build ownership, durable
unlocks, or presentation implementation. Owning domains receive typed requests
or provide identified versioned facts.

## 2. Governing invariants

1. **Authentic play only:** human performance comes from the player's approved
   chart and is normalized by Rhythm/Combat, never simulated by Abilities.
2. **One primary route:** group-action material cannot also become ordinary
   Attack, Defend, Special, revival, recovery, or another group action.
3. **One Hype charge:** no overflow, automatic firing, second charge, or separate
   cooldown.
4. **Committed Signatures are reliable:** the base effect survives weak play,
   downing, phase change, and target fallback after scoring-group start.
5. **One Call use per player:** canceled uncommitted requests preserve readiness;
   committed use cannot recharge in that encounter.
6. **The band cannot chain Calls:** one pending reservation and a shared musical
   lockout serialize player-requested Calls.
7. **Contribution is independently positive:** weak or absent play omits only
   that participant's possible share and never reduces another.
8. **One normal Crescendo is guaranteed:** required authored coverage exists on
   Easy, Normal, and Hard without a synthetic fallback.
9. **Easy help remains bounded:** at most one additional recovery Crescendo can
   help but never guarantee victory or add recovery attempts.
10. **Solo support is fixed and visible:** acolytes have no charts, judgments,
    score, player-performance credit, rewards, or risk multiplier.
11. **The human remains decisive:** Vanguard cannot break a Resolve layer and
    acolytes cannot help the solo emergency recovery challenge.
12. **Definitions and effects are exact:** every transition, target, fallback,
    revision, contribution, and application is deterministic and idempotent.

## 3. Locked inputs and definition contract

The encounter snapshot supplies exact content/schema/balance revisions, attempt
and musical-clock identities, roster/difficulty, validated player loadouts,
Signature and Band Call choices, legal item/build modifiers, Activity Map and
Crescendo candidates, encounter phase, charts, targets, and solo acolyte data.
Active state never silently adopts a later definition or balance revision.

Every Signature, Band Call, and Crescendo effect definition declares:

- stable identity and immutable revision;
- family/tags and legal owner, participant, target, and encounter states;
- guaranteed base where applicable plus normalized bonus/share curve and caps;
- deterministic target, distribution, and invalid-target fallback;
- typed Combat, Survival, or Positioning effect requests;
- legal musical scheduling and resolution boundaries;
- permitted item/build hooks and global budget/cap categories; and
- presentation-neutral semantic and multimodal cue keys.

Any role or instrument may equip any Signature or Call unless an explicit
thematic restriction leaves equivalent valid choices and never requires a party
composition. Definitions cannot alter charts, Rhythm timing/judgments, movement
timing, recovery/revival counts, encounter rewards, or core control semantics.

## 4. Hype ownership and gain lifecycle

Abilities authoritatively owns Hype amount and the `Accumulating`, `Ready`,
`Armed`, `Committed`, and `Consumed` states. Combat owns Special intent/input
routing, normalized conversion, identification of eligible contribution, and
automatic return to the previously stored Attack or Defend intent.

Combat emits each eligible contribution once and marks the approved route:

- successful ordinary Attack/Defend creates slow gain; or
- unready Special exclusively redirects contribution into fast gain.

Abilities validates the source identity and applies revisioned rates. Reaching
the one-charge threshold enters Ready, discards overflow, prevents more gain,
and tells Combat to restore the stored intent. Ready never auto-fires.

Partial, Ready, and Armed Hype persist through downing/revival and
same-encounter reconnection. Hype resets between encounters. State transitions
carry attempt, player, definition/balance revision, source, and activation
identities so repeated delivery cannot duplicate gain or consumption.

## 5. Signature arming and reliable resolution

Pressing the Ready Signature control arms the player's next ordinary scoring
group. Repeated input cannot create another activation. If downing, departure, or
an encounter terminal/cancellation state invalidates the player before the group
begins, the arm cancels and full Hype remains.

Scoring-group start commits the activation and reserves the earliest valid clean
boundary after that group. The guaranteed base can no longer be canceled by
weak play, downing, disconnection, phase/target change, or participation ending.
Only valid performance accepted before suspension scales the positive bonus.

At resolution, the definition's typed target/distribution contract applies. An
invalid intended target uses the deterministic fallback. Hype is consumed only
when the committed effect enters guaranteed resolution. A critical internal
failure that prevents resolution preserves or refunds the charge idempotently;
normal weak play, downing after commitment, or fallback is not refundable.

Arm, cancel, commit, fallback, resolve, consume/refund, and restored-intent
states produce distinct semantic feedback without exposing raw timing math.

## 6. Band Call readiness and shared availability

Abilities owns personal readiness, `Ready`, `Reserved`, `Committed`, and `Used`,
the encounter-wide pending-Call reservation, and the shared Call lockout.

Any accepted successful human-played chart performance may feed one separately
budgeted readiness observer regardless of its primary route. Misses, absent
material, fixed effects, and acolyte output provide no readiness. The equipped
Call control itself displays progress; no additional persistent meter is added.

Readiness caps at Ready and discards overflow. Partial/Ready state persists
through downing and same-encounter reconnection, then resets between encounters.
Committed use sets Used and prevents recharge or a second use.

Item/build hooks may adjust readiness rate or Call potency only within approved
budgets/caps. They cannot add a use, fabricate performance, bypass a pending
reservation, or shorten the shared lockout below its global floor.

One pending Call blocks other requests. At Call start it becomes a revisioned
shared lockout initially tuned around eight measures. Simultaneous valid requests
use authoritative server receipt sequence and stable player identity as final
tie-breaker; only the winner reserves a candidate and all others retain Ready.

## 7. Band Call request, scheduling, and invitation

An active, non-downed Ready player may request the equipped Call when no group
action, recovery state, pending Call, or shared lockout blocks initiation. The
request identifies attempt, player, exact Call revision, and unique request.

Boss Encounters filters the Activity Map and reserves the earliest valid
authored ensemble candidate inside the Call's maximum delay. Higher-priority
conflicts search later valid candidates without displacing guaranteed content.
Band Calls never synthesize a universal beat or use an unvalidated passage.

During the queue, each eligible teammate receives a multimodal invitation with
initiator, Call name/effect, and beat countdown. Join In provisionally accepts;
ordinary play continues and acceptance may be withdrawn until commitment.

An invitee becoming ineligible before commitment removes only that invitee. An
invalid initiator or no remaining candidate within maximum delay cancels the
whole request, releases the reservation, and preserves readiness.

## 8. Band Call commitment and effect

At the scheduled boundary, Abilities spends the initiator's use, locks the
initiator and valid accepted participants, snapshots eligible-roster scaling,
and converts the pending reservation into the shared lockout.

Each participant, including the initiator, redirects authentic personal chart
material exclusively to the Call for its defined one- or two-measure duration.
The initiator guarantees the base and may add an independent normalized share.
Every share is nonnegative and capped; weak, missed, or absent performance omits
only possible positive value.

Definition and roster caps bound the combined bonus. A participant downed or
disconnected mid-window keeps already accepted contribution and adds nothing
afterward. The Call and its guaranteed base continue.

The effect resolves once at the first valid boundary after the performance
window through its typed target/distribution/fallback contract. Results identify
the base, each human share, any solo fixed share, fallback, final effect, and
exact revisions without publicly ranking participants.

## 9. Required Crescendo candidates

A standard encounter authors two to four validated Crescendo candidates and
guarantees exactly one normal activation on Easy, Normal, and Hard. At attempt
start, Boss Encounters filters and deterministically ranks them using the locked
roster/difficulty/content snapshot, musical-fit policy, and encounter seed.

The selected candidate is a required-event reservation. Under Boss Encounters'
priority contract it sits below immutable/committed windows and urgent recovery,
but above Band Calls and optional events.

Each candidate requires strong playable coverage, roughly two measures of
reaction, sustained instrument activity, and separation from incompatible boss
attacks, recovery, silence/quiet transitions, group actions, and the Finishing
Cadence. If it becomes invalid before preview, the next valid unused candidate
is reserved. Once preview commits, priority protects the window.

Crescendos never use a generic fallback. An exceptionally short song may have
one candidate only through an explicit recorded exception with full validation
and human approval. Missing required coverage blocks publication. If an
unexpected runtime defect exhausts every candidate, play continues without a
reward penalty or fake event and emits a critical content-integrity failure.

## 10. Crescendo preview, participation, and tiers

A multimodal preview begins roughly two measures before the window and identifies
the Crescendo, authored effect, countdown, participation state, and stable Join
In control. Participation is free and optional.

An eligible active, non-downed player may accept or withdraw before commitment.
Declining/ineligibility leaves ordinary routing unchanged. At the boundary, the
valid participant roster and eligible-human-roster scaling snapshot lock.

Participants redirect roughly two measures of authentic personal chart
material exclusively to the Crescendo; nonparticipants continue ordinary play.
Each result is independently normalized for instrument and difficulty, then
added as a nonnegative capped share. Downing/disconnection preserves only
contribution already accepted.

Revisioned thresholds map the total to **Echo**, **Crescendo**, or **Full
Crescendo**. They scale from the eligible human roster, not only acceptors, so an
expert provides meaningful value but cannot represent a larger inactive band.
Echo is the minimum activated outcome. Weak play cannot cancel the event or
reduce another share. Values keep declining and continuing ordinary play a
legitimate choice rather than an exploit.

Players control only their own provisional participation. They cannot initiate,
reschedule, or cancel the authored event.

## 11. Crescendo effects and Easy recovery activation

The default effect is a strong Resolve burst plus modest Ward reinforcement.
Boss-authored alternatives may use clearly previewed typed offense, defense,
recovery, or positional effects. Each definition supplies capped values for all
three tiers plus deterministic target/distribution/fallback and cue contracts.

Effects cannot add revival/solo-recovery attempts, directly complete victory,
bypass the Finishing Cadence, or guarantee success. Recovery effects are bounded
help that leaves later human performance necessary.

After the required Crescendo resolves, Easy may perform one authored recovery
evaluation. A revisioned deterministic rule combines song progress, remaining
boss resistance against expected range, and current living-player survival
pressure. When substantially behind and an unused valid candidate remains, Easy
may reserve at most one additional recovery Crescendo.

The extra event uses the normal preview, opt-in, performance, tier, conflict,
and no-fallback rules. Its candidate is consumed at resolution regardless of
tier. It provides no separate reward, does not alter difficulty/reward
eligibility, and cannot occur on Normal or Hard.

## 12. Solo acolyte functions

Solo deploys visually distinct Vanguard, Warden, and Herald acolytes. Their
effects are fixed by encounter/difficulty revision. First-release player items
and builds do not modify them.

### Vanguard

After each successfully completed ordinary human scoring group, regardless of
intent, Vanguard requests a small fixed Resolve packet after the associated
human effect. Recovery, revival, Band Call, and Crescendo material cannot trigger
it. The packet belongs only to the layer current for that group, never spills
forward, and clamps before the decisive break. A successful human Attack must
finish the layer.

### Warden

Warden uses encounter-authored clean pulse boundaries, initially around once
every eight measures. A visible preview precedes each modest Ward reinforcement.
The pulse respects legal caps and is not banked or retargeted when suppressed,
missed, or unnecessary.

### Herald

Herald adds a bounded bonus only when authentic successful human performance
earns Call readiness. It creates no passive readiness from inactivity or fixed
support. When active at solo Band Call/Crescendo commitment, Herald supplies one
small capped fixed squad share. Vanguard/Warden may join presentation but add no
other group share.

Acolytes never receive charts, judgments, combos, score, player-performance or
reward credit, or positional risk multipliers.

## 13. Acolyte positioning, suppression, and recovery

Each acolyte has an authored preferred tactical location. Positioning places and
repositions it automatically at clean phase/musical boundaries. Formation
offsets handle player co-location without swapping, blocking, reserving gameplay
capacity, or changing tactical state.

A committed clearly telegraphed attack affecting an active acolyte's location
suppresses it at impact for a configured duration initially around four
measures. Suppression does not stack, refresh, or extend. It pauses future
support, identifies the unavailable function, and displays a beat countdown.
Automatic recovery occurs at the first clean boundary after the duration.

Suppressed Vanguard triggers and Warden pulses are omitted, not banked. Herald
adds no readiness bonus. No catch-up occurs. Herald's fixed group eligibility
snapshots at commitment: later suppression cannot remove an accepted fixed
share, while suppression at commitment supplies none.

During solo emergency recovery all acolytes provide presentation only. They
cannot score, shorten, complete, or rescue it. They cannot be permanently downed,
revived, equipped, commanded, or made into escort objectives.

## 14. Typed application, order, and idempotency

Abilities never mutates Resolve, Ward, movement, position, survival, or encounter
outcome directly. It sends a typed, revisioned, source-identified request to the
owning domain, which validates current target state, caps, and fallback before
application.

Logical musical time determines order. At the same boundary, associated human
performance resolves before Vanguard support so Vanguard cannot take break
credit. Committed group/Signature effects then follow the stable Combat and Boss
Encounter order. Network arrival does not reinterpret musical order.

Gain, arm, request, invitation response, commit, contribution, resolution,
spend/refund, suppression, and recovery commands/facts are idempotent. Duplicate
or stale identities cannot create extra resources, effects, contributions,
shares, acolyte pulses, or consumption.

## 15. Semantic outputs and accessibility

Authoritative facts include:

- Hype eligible input, gain, Ready, arm, cancel, commit, consume/refund, and
  Signature effect/fallback;
- Call readiness/unavailability, request, reservation, invitation, acceptance/
  withdrawal/removal, cancel, commit, lockout, contribution, and result;
- Crescendo ranking/reservation/invalidation, preview, participation, commit,
  tier, typed effect, Easy evaluation, and content-integrity failure; and
- acolyte placement, pulse/trigger/share, suppression/recovery, omitted support,
  and fixed attribution.

Facts carry the applicable attempt, source, player/group, definition/balance
revision, logical musical time, target/fallback, contribution, and result.

UI and Audio receive presentation-neutral state/countdown/effect/cue facts.
Critical readiness, invitations, commitment, outcomes, and suppression never
depend on color or sound alone. Results/Analytics distinguish human contribution,
guaranteed base, and fixed support; public results do not rank players.

## 16. Content Authoring reconciliation register

The following requirements were reconciled into
[`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md#14-cross-specification-handoffs-and-reconciliation)
on 2026-09-02. This table remains the Ability-owned publication contract.

| Contract | Required authored/configured data | Publication gate |
|---|---|---|
| Signature | Identity/revision, family, legal states/targets, base/bonus, boundary, typed effect, fallback, cues | No prohibited Rhythm/movement/recovery/reward/control effect; all targets/fallbacks valid |
| Hype | Slow/fast gain and one-charge threshold revisions | Every eligible source routes once; no overflow/second charge path |
| Band Call | Readiness, max delay, lead/duration, lockout, base/share/roster caps, effect/fallback/cues | Candidate coverage for every supported role/difficulty/roster; no generic fallback |
| Crescendo | Two-to-four candidates or approved short-song exception, coverage/conflicts, selection inputs, tiers/effects, Easy evaluation | One normal activation provable on every difficulty; incompatible windows excluded |
| Solo support | Locations/formations, Vanguard clamp, Warden cadence/pulse, Herald bonuses/shares, suppression and cue data | No fake performance/risk/reward, layer break, recovery help, or missing difficulty coverage |
| Semantic/accessibility | Stable keys for every state, countdown, warning, result, and alternative modality | Export/load preserves exact identity/revision and complete cue coverage |

Band Call invitation/performance and Crescendo preview/performance windows are
mutually exclusive. The required Crescendo wins reservation priority.

## 17. Deferred tuning and technical work

Behavior is complete; these remain versioned playtest, content, or architecture
work:

- Signature and Band Call catalogs, names, values, curves, caps, and cues;
- slow/fast Hype and Call-readiness rates and permitted modifier budgets;
- Call lead/duration/maximum delay and shared-lockout length;
- Crescendo candidate weights, exact lead/duration, thresholds, tier values, and
  Easy behind-state thresholds;
- acolyte values, cadence, suppression duration, choreography, portraits, and
  effects; and
- authoritative transport, persistence/rejoin, telemetry, and critical-failure
  alert implementation.

Tuning may not add a Hype charge, second Call use, synthetic event passage,
negative participant share, required composition, fake acolyte performance,
Vanguard layer break, acolyte recovery help, paid gameplay resource, or direct
victory guarantee.

## 18. Approval and change control

The owner interview resolved AC-01 through AC-12 on 2026-08-25. This document is
the canonical Abilities, Cooperative Actions, and Solo Support design
specification.

A material change to Hype ownership/lifecycle, Signature reliability, Call use
or lockout, group routing/shares, Crescendo guarantee/tiers/Easy recovery,
acolyte functions/suppression, or fixed-support attribution requires an explicit
amendment citing the superseded rule. Numeric tuning inside these boundaries
creates a new definition/balance revision and never changes an active attempt.
