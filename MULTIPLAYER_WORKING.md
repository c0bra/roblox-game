# Bands Battle Multiplayer Working Record

- **Status:** Completed; reconciled into canonical specification
- **Started:** 2026-08-25
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#51-multiplayer-sessions-parties--matchmaking)
- **Included requirement:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#52-communication--safety)
- **Interview plan:** [`MULTIPLAYER_QUESTIONS.md`](MULTIPLAYER_QUESTIONS.md)
- **Encounter dependency:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Abilities dependency:** [`ABILITIES_AND_COOPERATIVE_ACTIONS.md`](ABILITIES_AND_COOPERATIVE_ACTIONS.md)
- **Planned canonical result:** `MULTIPLAYER.md`

## 1. Role of this record

This document persists owner decisions while the Multiplayer and Communication
& Safety interview is in progress. It is not canonical until reconciled into
`MULTIPLAYER.md`.

## 2. Inherited boundary

Multiplayer owns party membership/leadership/consent; public queue/matching;
staging, ready, roster/loadout lock and deployment handoff; encounter/active
roster; disconnect/grace/rejoin/departure; inactivity/resume; follow-up groups;
preset-ping delivery/rate/muting; and structural session safety.

It does not own loadout contents, campaign/difficulty eligibility, gameplay
domain state, encounter outcome/scaling formulae, rewards, durable progression,
Roblox moderation records, or presentation of domain-authored critical cues.

## 3. Approved inputs

- Player identity, onboarding/public-access state, boss/difficulty eligibility,
  and current session/party relationship.
- Shard/boss/difficulty selection and region/connection measurements.
- Validated Items spec/loadout snapshot and reserved prepared consumables.
- Boss Encounter deployment requirements, scaling envelopes, safe boundaries,
  active attempt/result, and No Contest facts.
- Rhythm participation coverage; Survival/Combat/Abilities/Positioning state
  needed for disconnect, targeting, rejoin, and group eligibility.
- Results completion and per-player Reward transaction facts.
- Roblox transport, filtering, account, block, report, and safety capabilities.

## 4. Decision record

### Checkpoint A — Entry, parties, and public matching

#### MP-01 — Shard entry modes, eligibility, and selection

- **Status:** Resolved 2026-08-25.
- The shard card presents boss identity, difficulty and eligibility, reward
  preview, and Solo, Current Party, and Public Band modes. Boss and difficulty
  are explicitly selected before mode; no transition silently substitutes or
  downgrades either choice.
- Server authority rechecks current campaign/difficulty eligibility, public-
  access onboarding state, and session compatibility at each transition. Public
  matching requires practice completion or explicit skip; private party play
  does not inherit that public-access gate.
- A player may occupy only one party proposal, public queue, staging/deployment,
  or active-attempt flow. Entering a conflicting flow requires an explicit leave
  from the current one and releases its temporary reservation/state.
- Stale card/progression data refreshes with an explanation instead of choosing
  another option. Cancellation before final deployment lock has no penalty.
  Session/service failure returns the player to the shard card or hub with the
  requested selection preserved where valid and no resource spend.

#### MP-02 — Current Party membership, leadership, and consent

- **Status:** Resolved 2026-08-25.
- A Current Party is a server-identified invited group of two to six humans. The
  creator is initial leader. Invitations target a player, expire, respect Roblox
  privacy/block state, and require acceptance; there is no open public join.
- Any member may leave. The leader may remove a member only while the party is
  outside final deployment lock and an active attempt; removal is explicit,
  notified, and carries no gameplay or reward penalty.
- If the leader leaves, leadership transfers to the longest-present eligible
  member, using stable player identity as a final tie-breaker. Leadership never
  grants combat, loadout, readiness, reward, resource, or follow-up authority
  over another player.
- A leader creates a revisioned boss/difficulty proposal. Every current member
  must be individually eligible and explicitly accept before staging. Decline,
  expiry, leader revision, or any membership change cancels the proposal and all
  acceptances without penalty; it never silently carries consent forward.
- The party persists through staging, encounter, and results unless members
  leave. Members see only safe party status such as selected role and readiness,
  not private inventory, quantities, performance history, or recommendations.

#### MP-03 — Public queue, matching inputs, and two-player choice

- **Status:** Resolved 2026-08-25.
- Public matching is server-owned and has no player host. Its key is the exact
  boss, difficulty, compatible content revision, and acceptable connection
  region. It targets three to six humans and permits two only through explicit
  consent.
- First release does not match or exclude by gear, spending, chosen instrument,
  prior performance, private readiness recommendation, or hidden skill rating.
  Duplicate roles/instruments are legal and never treated as a conflict.
- A queued player may move through the hub and use nonconflicting menus. Their
  loadout remains editable until staging lock. Canceling or entering another
  session flow explicitly leaves the queue.
- When exactly two players remain after an initial target around 45 seconds,
  both receive Start Together, Continue Waiting, Solo, and Leave choices. Start
  Together requires both acceptances. Timeout defaults only that player to
  Continue Waiting.
- Solo leaves the public queue and opens same-selection solo staging; Leave
  returns to the card/hub. Either action affects only that player, while the
  other remains queued. Failed matching or transport preserves valid selection,
  consumes nothing, and returns an identified retryable error.

### Checkpoint B — Staging, deployment, and follow-up

#### MP-04 — Staging validation, ready state, and final lock

- **Status:** Resolved 2026-08-25.
- Boss, difficulty, and compatible content revision are already locked when
  staging opens. Each player may still change role/instrument, equipment/full
  spec preset, Signature, Band Call, prepared consumables, and appearance.
  Duplicate roles remain explicitly valid.
- Ready requires complete authoritative Items/loadout validation for the selected
  encounter. All issues display together. Any edit or mutation that would alter
  the final encounter snapshot clears that player's Ready state; no stale Ready
  carries onto changed state.
- A viable public roster starts a revisioned short ready timer. Unready/invalid
  players at expiry are returned without penalty and may be replaced before
  final lock. Existing valid players remain in unlocked staging and reconfirm as
  required by the changed roster.
- A Current Party waits for every member's Ready. Cancellation, invalidation, or
  membership change returns it to the proposal state; a party member is never
  replaced from public matching or forced by the leader.
- When all required players are Ready, a final countdown initially around three
  seconds locks exact roster and loadout snapshots. Items atomically reserves
  prepared consumables at this lock. Leaving before lock releases temporary
  state and carries no penalty.

#### MP-05 — Deployment, active roster, and no join-in-progress

- **Status:** Resolved 2026-08-25.
- Final lock creates a unique deployment identity with exact player, content,
  schema, balance, loadout, consumable-reservation, and initial population-
  scaling snapshots. Repeated finalization cannot create another deployment.
- Every locked player must acknowledge successful load and exact revision
  readiness before Multiplayer hands Boss Encounters a shared future start
  boundary. No subset starts an encounter while another locked member is still
  loading or invalid.
- A disconnect, timeout, transport error, or invalid snapshot before attempt
  start cancels that deployment, releases consumable reservations idempotently,
  and returns reachable players to unlocked staging for refill and a new Ready/
  lock cycle. No gameplay result or reward is created.
- At song start, the immutable **deployment roster** establishes initial scaling.
  The **active roster** is a stateful subset used for targeting, group thresholds,
  and later-layer changes. Disconnect/inactivity may change active membership but
  cannot rewrite the deployment snapshot.
- There is no join-in-progress or mid-song refill. Only the same deployment
  identity may rejoin during its grace contract. Multiplayer hands exact roster,
  player snapshot, connection, and lifecycle identities to the owning gameplay
  domains without owning their state.

#### MP-06 — Results, rematch grouping, refill, and exit

- **Status:** Resolved 2026-08-25.
- After an immutable encounter result, each player's reward transaction finalizes
  independently before follow-up input. Follow-up choices cannot change, delay,
  pool, divide, or revoke that completed reward.
- Each player receives Retry Same Shard, Stay with Band, and Return to Hub.
  Timeout defaults only that player to Hub. No majority or leader choice binds
  another player.
- Public players choosing Retry or Stay form a temporary voluntary rematch group.
  Retry records that player's consent to the same boss/difficulty. Stay preserves
  the social group but requires explicit acceptance of the next proposal.
- Only mutually accepted/eligible members enter the next staging. Public matching
  may refill empty seats after the proposal is accepted; it cannot silently add
  a member to an unaccepted follow-up.
- Current Parties remain together unless members leave. Every same-shard retry or
  different-shard selection creates a fresh proposal, validation, staging Ready,
  lock, consumable reservation, and deployment cycle.
- Follow-up service failure preserves already granted rewards and returns players
  to the hub or voluntary group state with no forced queue or encounter.

### Checkpoint C — Disconnect, departure, and inactivity

#### MP-07 — Disconnect snapshot, grace, and safe rejoin

- **Status:** Resolved 2026-08-25.
- On authoritative connection loss, Multiplayer records the last confirmed Ward,
  tactical location, survival/downed state, Hype, Band Call state, consumable
  charges, immutable loadout, contribution, and participation identities without
  taking ownership of those domain states.
- Already committed boss impacts, consumable spends, Signature/group effects,
  and other guaranteed events resolve normally. Disconnect cannot cancel them or
  avoid their consequences.
- During grace, the missing player supplies no targets, chart judgments/misses,
  contribution, resource/readiness gain, or current group eligibility. Rhythm
  records identified absence so missing material cannot improve performance.
- Each deployment member receives one cumulative same-attempt grace budget,
  initially around 45 seconds. Disconnected time consumes it; a successful
  return pauses consumption, and later disconnects use only the remainder.
- Reconnect authenticates the same player/deployment/attempt and restores the
  immutable snapshot at Boss Encounters' next safe musical boundary. Prior legal
  location is preferred, otherwise the nearest valid Middle fallback applies.
- A standing return receives normal settling protection. A player disconnected
  while downed returns downed under existing revival rules. Spent resources,
  completed contribution, and committed state remain spent/completed/committed.
  The player cannot enter an already committed group window.

#### MP-08 — Grace expiry, active-roster change, and retained eligibility

- **Status:** Resolved 2026-08-25.
- Grace expiry atomically changes the deployment member to `Departed`. That
  player can no longer rejoin gameplay, but a later connection may reach the
  applicable results/hub flow. Repeated expiry delivery has no additional effect.
- Departure does not rewrite the open Resolve threshold, banked progress,
  committed attacks/effects, group snapshots, earned attribution, or encounter
  history. Pending invitations remove the player; future targets and group
  thresholds use the new active roster.
- Only still-unopened Resolve layers may use the smaller roster when they open.
  There is no join-in-progress/refill. Current Party leadership transfers under
  MP-02 without affecting combat or ending a session that still has humans.
- Completed meaningful contribution remains available to Results and Rewards.
  Network departure alone does not automatically erase eligibility; Rewards
  later defines thresholds using identified contribution, coverage, voluntary
  leave, inactivity, and encounter outcome evidence.
- If no Active or grace-eligible human remains, Multiplayer publishes the fact
  at the next safe boundary and Boss Encounters resolves Defeat with the explicit
  reason **all humans departed**. This is player/session state, not No Contest.
- Existing Boss Encounter outcome wording must be reconciled at canonical
  Multiplayer publication to include this exact reason without changing its
  system-failure/No Contest contract.

#### MP-09 — AFK warning, inactive state, resume, and reward consequence

- **Status:** Resolved 2026-08-25.
- AFK evidence counts only a connected, non-downed, unsuspended scoring group
  containing genuine playable material for that player's chart. Any plausible
  genuine gameplay attempt prevents an inactivity strike; accuracy and resulting
  contribution do not matter.
- Initial tuning privately warns after two consecutive ignored eligible groups
  and enters `Inactive` after two more. Exact thresholds remain revisioned
  operational tuning, but only consecutive eligible groups may drive them.
- Inactive players retain their snapshot but receive no future targets,
  judgments/misses, contribution, readiness, or group eligibility. Already
  committed effects/impacts remain valid.
- A connected inactive player may explicitly request one resume per encounter.
  Multiplayer returns them at the next safe boundary with normal re-entry/
  settling handling. The request cannot cancel an already committed consequence.
- Repeated inactivity after the one resume leaves the player inactive for the
  rest of the attempt and may remove participation-based reward eligibility.
  All evidence and notices remain private except safe active/inactive status
  required for group play.
- Quiet/rest passages, weak or mistimed play, safe positioning, declining an
  optional group action, downing, network absence, Rhythm suspension, or use of
  approved accessibility settings never creates a strike or misconduct label.
- If everyone is inactive but at least one player can resume, the attempt clock
  continues. When no Active/grace-eligible human remains and no inactive player
  can resume, Multiplayer publishes **all humans inactive/departed** for the same
  explicit Defeat handoff as MP-08.

### Checkpoint D — Communication, safety, and outputs

#### MP-10 — Protected cues, preset pings, rate limits, and muting

- **Status:** Resolved 2026-08-25.
- Domain-authored critical cues for attacks, targeting, movement, downing,
  revival, Band Calls, Crescendos, and phase changes are automatic. Muting a
  player or exhausting a ping limit can never hide, delay, or impersonate them.
- The first-release ping vocabulary is Move, Defend, Join Call, Revive, and
  Ready/Thanks. Each enum validates current context and any legal target/location;
  it never transmits arbitrary player text or creates a gameplay command.
- A per-sender token limit initially permits a burst of two and restores one
  allowance every four musical measures in encounters. Equivalent revisioned
  real-time limits apply outside a musical clock. Identical overlapping pings
  coalesce; rejection explains the remaining wait privately.
- A recipient may mute one sender's player-created pings for the current party/
  rematch flow and may undo it. Automatic domain cues remain protected. Delivery
  failure affects communication only and produces no inferred acceptance/action.
- Ping semantics map to localized text, icon/shape, optional sound/haptics, and
  accessible alternatives. Required meaning never depends on color or sound.
- First release adds no custom free-form text or voice. Roblox-provided chat/
  voice may remain available under platform/account controls, but gameplay,
  safety, readiness, and rewards never depend on it.

#### MP-11 — Structural anti-grief, moderation, and privacy

- **Status:** Resolved 2026-08-25.
- Server-owned rules make friendly fire, body blocking, negative contribution,
  spending another player's resource, forced role composition, vote-kick,
  player-host authority, leader combat authority, and binding follow-up votes
  impossible rather than relying on social enforcement.
- Weak accuracy, safe positioning, duplicate role choice, optional group decline,
  boss struggle, or accessibility use is not misconduct. Only the private,
  evidence-based inactivity contract changes participation state/reward evidence.
- Players can mute pings and use standard Roblox block/report/account surfaces.
  Where platform relationship data permits, blocks prevent new invitations and
  rematch grouping but never rewrite committed or active gameplay state.
- The game validates/rate-limits commands and records operational evidence, while
  Roblox owns account moderation/filtering/sanctions. There is no public report,
  accusation, AFK label, moderation history, or player-imposed punishment.
- Other players may see safe platform identity/appearance, selected role,
  readiness, connection/availability, downed/revival state, and necessary
  automatic cues. Inventory/quantities/purchases, private performance/history/
  recommendations, accessibility/calibration/settings, block/report state, and
  moderation facts remain private.
- Public Results expose the collective result and each viewer's own evidence,
  never an individual ranking, damage leaderboard, or blame attribution. Preset-
  only custom communication, safe defaults, and platform controls apply equally
  to minors and all other players.

#### MP-12 — Semantic outputs, service failures, and completeness audit

- **Status:** Resolved 2026-08-25.
- Multiplayer emits authoritative identified facts for party invitation/member/
  leader/proposal/consent; queue/match/two-player choice; staging validation/
  Ready/timer/lock; deployment acknowledgment/start/cancel; connection/grace/
  rejoin/departure; AFK/warning/inactive/resume; results/follow-up/rematch; and
  ping/rate/mute/delivery/failure.
- Facts carry applicable party, proposal, queue, match, staging, deployment,
  attempt, player, content/schema/balance revision, server/logical time, prior/
  next state, reason, source, and idempotency identity. Duplicate or stale input
  cannot repeat membership, consent, lock, reservation, rejoin, reward evidence,
  ping, or outcome handoff.
- UI/Audio receive presentation-neutral state/cue facts. Items receives lock/
  release; gameplay domains receive exact roster/connection/eligibility facts;
  Results/Rewards receive outcome-independent participation evidence; Player Data
  receives only required durable facts; Analytics receives privacy-reviewed
  operational semantics.
- Pre-start failures roll back/release safely. A player-local runtime failure uses
  connection grace or local Rhythm suspension. Global authoritative session,
  roster, or clock corruption preventing fair outcome evaluation hands an exact
  critical-failure fact to Boss Encounters for No Contest.
- Operational configuration revisions cover capacities, timers, region/latency,
  rate limits, retry policies, and privacy retention. Automated/network/manual
  tests cover every mode, one-to-six humans, supported difficulty/region/content,
  duplicate roles, membership/timing failures, repeat disconnects, roster
  changes, inactivity, pings/mutes, localization/accessibility, privacy, and
  duplicate/out-of-order delivery.

## 5. Content/configuration reconciliation register

- Operational configuration requires revisioned party capacity/invitation/
  proposal expiry, queue target/minimum/timeout, supported content-revision
  compatibility, region/connection limits, and two-player choice timing.
- Validation rejects hidden skill/gear/spending/composition matching and any
  route that silently changes boss, difficulty, or mode.
- Staging/deployment configuration requires ready and final-countdown durations,
  load/acknowledgment timeouts, revision compatibility, refill policy, and
  follow-up choice expiry/default. Validation proves atomic consumable release,
  no partial start, no join-in-progress, and a new consent/lock cycle for retry.
- Active-session configuration requires a cumulative grace budget, rejoin/load
  transport limits, safe-boundary handoff, AFK warning/inactive thresholds, and
  one-resume policy. Tests cover repeated disconnects, all roster transitions,
  open/unopened-layer behavior, group snapshots, and retained evidence.
- `BOSS_ENCOUNTERS.md` must add all-humans-departed/inactive as explicit gameplay
  Defeat reasons while preserving No Contest exclusively for critical system
  failure.
- Communication configuration requires revisioned localized ping semantics,
  legal context/targets, encounter and non-encounter token rates, coalescing,
  session mute, protected automatic-cue classification, and multimodal keys.
- Privacy/safety validation enforces the visibility allowlist, no custom chat/
  voice dependency, Roblox block/report handoff, and all structural anti-grief
  prohibitions. Operational testing covers every transition/failure matrix in
  MP-12 with privacy-reviewed retention.

## 6. Open handoffs

- `PROGRESSION.md` owns public matchmaking and boss/difficulty eligibility.
- `ITEMS_AND_EQUIPMENT.md` owns loadout validation, snapshots, reservations, and
  mutations.
- `BOSS_ENCOUNTERS.md` owns attempt lifecycle, population scaling, safe musical
  boundaries, outcome, and No Contest.
- `RHYTHM_GAMEPLAY.md` owns absence/participation coverage and fair re-entry.
- `COMBAT.md`/Survival/Positioning own gameplay state preserved or changed by
  disconnect/rejoin/departure.
- `ABILITIES_AND_COOPERATIVE_ACTIONS.md` owns group invitations/eligibility,
  Hype/Call state, and ability resolution.
- Rewards, Results, UI, Audio, Player Data, Analytics, and Roblox services
  consume or provide the identified facts defined by the canonical result.

## 7. Change log

- **2026-08-25:** Created the working record. Progress is 0 of 12 questions.
- **2026-08-25:** Approved MP-01 through MP-03. Shard entry, Current Party
  consent/leadership, and public queue/two-player choice are resolved; progress
  is 3 of 12 questions.
- **2026-08-25:** Approved MP-04 through MP-06. Staging validation/lock,
  all-acknowledged deployment, immutable/no-join roster, and voluntary follow-up
  are resolved; progress is 6 of 12 questions.
- **2026-08-25:** Approved MP-07 through MP-09. Cumulative disconnect grace, safe
  rejoin, roster departure/rescaling evidence, AFK/inactive state, and one resume
  are resolved; progress is 9 of 12 questions.
- **2026-08-25:** Approved MP-10 through MP-12. Protected cues/preset pings,
  structural safety/privacy, semantic outputs, failure handling, and completeness
  audit are resolved. All 12 questions were reconciled into the canonical spec.
