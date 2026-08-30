# Bands Battle Abilities and Cooperative Actions Specification Questions

- **Status:** Completed; 12 of 12 questions resolved
- **Started:** 2026-08-24
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#46-abilities--cooperative-actions)
- **Included system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#47-solo-support)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Combat dependency:** [`COMBAT.md`](COMBAT.md)
- **Encounter dependency:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Working record:** [`ABILITIES_AND_COOPERATIVE_ACTIONS_WORKING.md`](ABILITIES_AND_COOPERATIVE_ACTIONS_WORKING.md)
- **Planned canonical result:** `ABILITIES_AND_COOPERATIVE_ACTIONS.md`

## 1. Interview method

This interview uses four checkpoints of three questions. It inherits settled
Hype/Special performance, event-priority, item/loadout, and group-contribution
rules and focuses on ability definitions, scheduling, cancellation, cooperative
combination, and predictable Solo Support.

Answers are persisted after each checkpoint. When all twelve questions are
resolved, they are reconciled into canonical
`ABILITIES_AND_COOPERATIVE_ACTIONS.md` and relevant Hype-ownership wording is
reconciled with `COMBAT.md` without changing approved player behavior.

## 2. Fixed inherited decisions

- Each player equips one Signature Special and one separate Band Call in a full
  spec preset. Any instrument may use any option unless an explicit thematic
  restriction never creates a required composition.
- Hype stores one charge, fills slowly through successful Attack/Defend and
  faster when unready Special exclusively redirects contribution. Full Hype
  returns to the prior intent and never auto-fires.
- Ready Special arms the next ordinary scoring group. Once that group begins,
  the reliable base effect is guaranteed; performance scales the bonus. Downing
  before group start cancels the arm but preserves Hype. Hype persists through
  downing and resets between encounters.
- Each player may earn/initiate at most one Band Call per encounter. A shared
  band lockout, initially around eight measures after start, prevents chaining.
- A Band Call request schedules the earliest valid Activity Map ensemble window,
  invites eligible teammates, and retains its charge if it cannot begin.
- At Band Call commit, participants redirect authentic ordinary chart material
  for roughly one or two measures. The initiator guarantees the base; each
  participant adds an independent accuracy-scaled share.
- A normal encounter has two to four authored Crescendo candidates and guarantees
  exactly one activation on every difficulty. Easy may add at most one recovery
  activation when substantially behind. Crescendos never use universal fallback.
- Crescendo participation is optional and uses each participant's authentic
  chart. Independent normalized shares combine into Echo, Crescendo, or Full
  Crescendo; one weak player cannot reduce another.
- Cooperative/group contribution has one primary route and cannot also become
  ordinary Attack/Defend/Special. Target/distribution policies are deterministic
  and roster-capped under `COMBAT.md`.
- Solo has visible Vanguard, Warden, and Herald acolytes with fixed predictable
  support, no charts/judgments/performance/reward credit, and no risk multipliers.
- Vanguard cannot make the decisive Resolve break; Warden pulses Ward support on
  an authored cadence; Herald improves Band Call readiness and contributes a
  small capped fixed share to Band Calls/Crescendos.
- Acolytes may be temporarily suppressed by telegraphed attacks, recover
  automatically at a musical boundary, cannot be downed/revived/equipped/
  commanded, and never help the solo emergency recovery challenge.

## 3. Question plan

### Checkpoint A — Signature Specials and Hype

#### AC-01 — Signature definition and effect contract

- **Status:** Resolved 2026-08-24.

- **Decision needed:** What must every Signature definition declare so different
  offense, Ward, support, and positional abilities remain deterministic?
- **Must resolve:** Stable identity, family/tags, base/bonus, target/distribution,
  valid states, musical resolution, typed domain effects, cues, thematic
  restrictions, fallback, prohibited effects, and no class lock.

#### AC-02 — Hype ownership and lifecycle integration

- **Status:** Resolved 2026-08-24.

- **Decision needed:** Which system owns Hype state while Combat continues to
  route normalized performance exactly as approved?
- **Must resolve:** Authority, slow/fast gain requests, cap/overflow, previous
  intent handoff, Ready/Armed/Committed/Consumed, downing persistence, encounter
  reset, idempotency, and cross-spec reconciliation.

#### AC-03 — Activation scheduling, invalidation, and guaranteed base

- **Status:** Resolved 2026-08-24.

- **Decision needed:** How does an armed Signature select its group/resolution
  boundary and remain reliable through target/state/event changes?
- **Must resolve:** Arm start, next scoring group, pre-start cancellation,
  commitment, downing mid-group, boundary reservation, target invalidity,
  effect-safe fallback, consumption timing, system failure, and player feedback.

### Checkpoint B — Band Call readiness and invitations

#### AC-04 — Band Call readiness, ownership, and lockout

- **Status:** Resolved 2026-08-24.

- **Decision needed:** How is one personal Call earned/used while a shared
  lockout prevents chaining?
- **Must resolve:** Readiness input, threshold/cap, one-per-encounter, item/build
  hooks, downing, persistence, shared lockout start/end, simultaneous requests,
  unavailable state, and reset.

#### AC-05 — Request, candidate scheduling, and invitation

- **Status:** Resolved 2026-08-24.

- **Decision needed:** How does a ready initiator request a Call and invite
  teammates without interrupting ordinary play?
- **Must resolve:** Eligibility, candidate filter/max delay, queue identity,
  preview/countdown, provisional accept/withdraw, movement/downing/disconnect,
  invalid initiator, charge retention, conflict priority, and no universal beat.

#### AC-06 — Commit, participant performance, and effect resolution

- **Status:** Resolved 2026-08-24.

- **Decision needed:** What locks at the boundary and how are base/participant
  shares combined/applied?
- **Must resolve:** Initiator spend, committed roster, ordinary-route suspension,
  group duration, independent normalization, guaranteed base, weak/absent share,
  roster cap, target/distribution, interruption, resolution, and attribution.

### Checkpoint C — Crescendos

#### AC-07 — Candidate budget, selection, and guarantee

- **Status:** Resolved 2026-08-24.

- **Decision needed:** How does the encounter choose exactly one viable authored
  Crescendo while preserving required-event fairness?
- **Must resolve:** Two-to-four candidates, guarantee, deterministic selection,
  coverage/reaction/conflicts, reservation priority, runtime invalidation, later
  candidate, no fallback, short-song exception, and authoring failure.

#### AC-08 — Preview, opt-in, performance, and tiers

- **Status:** Resolved 2026-08-24.

- **Decision needed:** How do players understand/accept the event and how do
  independent normalized shares create the three outcome tiers?
- **Must resolve:** Two-measure preview, effect disclosure, Join In, decline,
  eligibility/withdrawal, committed roster, own chart, duration, additive total,
  Echo/Crescendo/Full thresholds, near-zero result, and no player cancellation.

#### AC-09 — Crescendo effects and Easy recovery activation

- **Status:** Resolved 2026-08-24.

- **Decision needed:** Which effect contracts are legal and when may Easy use a
  second candidate as recovery?
- **Must resolve:** Default Resolve+Ward pattern, boss-authored alternatives,
  effect preview, tier scaling, target/distribution, behind-state detection,
  maximum one extra, no guaranteed victory, candidate consumption, and
  difficulty/reward neutrality.

### Checkpoint D — Solo Support and outputs

#### AC-10 — Vanguard, Warden, and Herald cadences

- **Status:** Resolved 2026-08-25.

- **Decision needed:** What exact event sources and limits make each acolyte
  predictable without simulating performance?
- **Must resolve:** Vanguard successful-group trigger/cap/no break, Warden
  cadence/preview/reinforcement, Herald readiness/group share, clock boundaries,
  difficulty scaling, player build interaction, fixed attribution, and no score.

#### AC-11 — Suppression, positioning, and solo group participation

- **Status:** Resolved 2026-08-25.

- **Decision needed:** How do acolytes occupy the arena, become suppressed/
  recover, and join Band Calls/Crescendos without becoming manageable party
  members?
- **Must resolve:** Automatic formation/location, attack targeting, suppression
  Commit/duration, overlapping suppression, support pause/recovery, group
  eligibility, fixed share, recovery-challenge exclusion, no equipment/commands,
  and feedback.

#### AC-12 — Semantic outputs and content completeness

- **Status:** Resolved 2026-08-25.

- **Decision needed:** Which ability/group/solo facts must be emitted and which
  authoring fields/validators are required?
- **Must resolve:** Hype/Signature, Call readiness/request/invite/commit/result,
  Crescendo candidate/preview/tier/effect, acolyte support/suppression, exact
  identities/times/revisions, UI/Audio/Results/Analytics, accessibility, Content
  Authoring register, and completion audit.

## 4. Completion criteria

`ABILITIES_AND_COOPERATIVE_ACTIONS.md` is complete only when:

- AC-01 through AC-12 are resolved;
- Hype has one authoritative owner without changing approved Combat behavior;
- every committed Signature keeps its reliable base without double-spending;
- Band Call charge, invitation, lockout, and independent participant shares are
  deterministic through cancellation/disconnect;
- every encounter can deliver its required fair Crescendo or fails authoring;
- Easy's extra recovery event cannot guarantee victory;
- acolytes remain predictable fixed support with no fabricated score; and
- every authored-data/semantic-output dependency is complete.

## 5. Change log

- **2026-08-24:** Created the concise 12-question plan from the approved GDD,
  Systems Map, Combat, Boss Encounter, and Items contracts.
- **2026-08-24:** Approved AC-01 through AC-03, completing Signature Specials
  and Hype checkpoint A. Progress is 3 of 12 questions.
- **2026-08-24:** Approved AC-04 through AC-06, completing Band Call checkpoint
  B. Progress is 6 of 12 questions.
- **2026-08-24:** Approved AC-07 through AC-09, completing Crescendo checkpoint
  C. Progress is 9 of 12 questions.
- **2026-08-25:** Approved AC-10 through AC-12, completing Solo Support and
  outputs checkpoint D. All 12 questions are resolved and the canonical
  specification was published.
