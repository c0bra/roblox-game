# Bands Battle Multiplayer and Communication Safety

- **Status:** Approved
- **Approved:** 2026-08-25
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#51-multiplayer-sessions-parties--matchmaking)
- **Included requirement:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#52-communication--safety)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Encounter dependency:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Items dependency:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)
- **Abilities dependency:** [`ABILITIES_AND_COOPERATIVE_ACTIONS.md`](ABILITIES_AND_COOPERATIVE_ACTIONS.md)
- **Decision source:** [`MULTIPLAYER_WORKING.md`](MULTIPLAYER_WORKING.md)
- **Interview plan:** [`MULTIPLAYER_QUESTIONS.md`](MULTIPLAYER_QUESTIONS.md)

## 1. Role and authority

Multiplayer owns party membership/leadership/consent; public queues/matching;
staging, Ready, final roster/loadout lock, and deployment handoff; immutable
deployment and changing active-roster membership; disconnect/grace/rejoin/
departure; inactivity/resume; voluntary follow-up groups; preset-ping delivery,
rate, and muting; and structural session safety/privacy output.

It does not own loadout contents, progression/difficulty eligibility, gameplay
domain state, population-scaling formulae, encounter outcome, reward calculation,
durable progression, presentation of domain-authored critical cues, or Roblox
moderation/filtering/account state. It validates and routes identified facts
between those owners.

## 2. Governing invariants

1. **Consent is individual:** no leader, majority, host, timeout, or rematch group
   can choose content, readiness, deployment, or follow-up for another player.
2. **Selections do not drift:** boss, difficulty, mode, and proposal revisions
   never change silently.
3. **Public matching has no player host:** server rules own queue, roster,
   replacement, and safety decisions.
4. **Composition is socially neutral:** duplicate instruments are legal; gear,
   spending, private performance, and hidden skill do not gate first-release
   matching.
5. **Ready describes exact state:** any snapshot-affecting edit clears Ready.
6. **Deployment is atomic:** either every locked member loads the same revisions
   and starts together or the pre-start deployment rolls back.
7. **No join-in-progress:** only a locked deployment member may rejoin its attempt.
8. **Absence is not performance:** disconnected/inactive chart material creates
   no judgments, misses, contribution, resource gain, or rating improvement.
9. **Commitment survives connection loss:** already committed gameplay effects
   and spends cannot be dodged or duplicated.
10. **Poor play is not misconduct:** only ignored eligible gameplay drives the
    private inactivity contract.
11. **Critical cues are protected:** muting/rate-limiting player pings cannot hide
    automatic gameplay information.
12. **Grief powers do not exist:** no friendly fire, body blocking, vote-kick,
    negative contribution, resource theft, leader combat power, or forced
    follow-up.

## 3. Authoritative identities and state

Every flow uses stable party, proposal, queue, match, staging, deployment,
attempt, player, content/schema/balance revision, and transition identities.
Server time orders service operations; the approved musical clock orders active
encounter transitions. Repeated/stale commands cannot duplicate consent,
membership, Ready, lock, reservations, rejoin, pings, or outcome evidence.

A player may occupy only one conflicting proposal, public queue, staging/
deployment, or active-attempt flow. Entering another requires an explicit leave
that releases temporary state. Current Party membership may persist around these
flows because it does not itself commit the player to content.

## 4. Shard selection and entry modes

The shard card presents boss, selected difficulty/eligibility, reward preview,
and Solo, Current Party, and Public Band. Boss and difficulty are selected before
mode. No failure or stale state silently substitutes a boss, difficulty, or mode.

The server rechecks current campaign/difficulty eligibility and session
compatibility at each transition. Public Band additionally requires practice
completion or explicit skip; private party play does not inherit that public-
access gate.

Stale progression/card data refreshes with an explanation. Cancellation before
final deployment lock carries no penalty. A service failure returns the player
to the card/hub, preserves the requested selection where still valid, and spends
no resource.

## 5. Current Parties and proposals

A Current Party is an invited server-identified group of two to six humans. The
creator is initial leader. Invitations target one player, expire, respect exposed
Roblox privacy/block relationships, and require acceptance; there is no open
public join.

Any member may leave. A leader may remove a member only outside final deployment
lock and an active attempt. Removal is explicit, notified, and carries no
gameplay/reward penalty. If the leader leaves, the longest-present eligible
member becomes leader, with stable player identity as final tie-breaker.

Leadership grants only proposal/party-management capability. It cannot control
another player's combat, role/loadout, Ready, resources, rewards, or follow-up.

A leader issues a revisioned boss/difficulty proposal. Every current member must
be individually eligible and explicitly accept before staging. Decline, expiry,
revision, or membership change cancels the proposal and all acceptances without
carrying consent forward.

The party persists through staging, encounter, and results unless members leave.
Members see safe role/readiness/session state, not private inventory, quantities,
performance history, or recommendations.

## 6. Public queue and two-player choice

Public matching is server-owned and keyed by exact boss, difficulty, compatible
content revision, and acceptable connection region. It targets three to six
humans and permits two only through explicit consent.

First release does not match or exclude by gear, spending, instrument, prior
performance, private readiness recommendation, or hidden skill rating. Duplicate
roles/instruments are valid.

Queued players may move through the hub and use nonconflicting menus; loadouts
remain editable until staging lock. Canceling or starting another session flow
explicitly leaves the queue.

When exactly two players remain after an initial target around 45 seconds, each
receives Start Together, Continue Waiting, Solo, and Leave. Start requires both.
Timeout defaults that player to Continue. Solo opens same-selection solo staging;
Leave returns to the card/hub. Either affects only that player, while the other
remains queued. Failure preserves valid selection and consumes nothing.

## 7. Staging, validation, and final lock

Boss, difficulty, and compatible content revision are locked before staging.
Each player may change role/instrument, equipment or full spec preset, Signature,
Band Call, prepared consumables, and appearance. Duplicate roles remain valid.

Ready requires complete server-side Items validation for the selected encounter.
All issues display together. Any edit or mutation that changes the eventual
snapshot clears that player's Ready.

A viable public roster starts a revisioned short ready timer. At expiry,
unready/invalid players return without penalty and may be replaced. Remaining
players stay in unlocked staging and reconfirm when required by roster changes.

A Current Party waits for all members. Cancellation, invalidation, or membership
change returns it to proposal state; it never receives an uninvited public
replacement.

When everyone is Ready, a final countdown initially around three seconds locks
the exact roster/loadout snapshots. Items atomically reserves prepared
consumables then. Leaving before lock has no penalty.

## 8. Atomic deployment and start

Final lock creates a unique deployment identity with exact player, content,
schema, balance, loadout, consumable-reservation, and initial population-scaling
snapshots.

Every member acknowledges successful load and exact-revision readiness before
Multiplayer hands Boss Encounters a shared future start boundary. No subset
starts while another locked member is loading or invalid.

A disconnect, timeout, transport failure, or invalid snapshot before attempt
start cancels the deployment, releases reservations idempotently, and returns
reachable players to unlocked staging for refill and a fresh Ready/lock cycle.
No gameplay result/reward exists.

At song start the deployment roster becomes immutable. It establishes initial
population scaling. A separate active-roster projection changes through the
rules below without rewriting deployment history.

## 9. Active roster and join prohibition

The active roster supplies current targeting, invitation, and future scaling
eligibility. States distinguish connected Active, connection Grace, Inactive
with or without Resume, and Departed. Domain-specific downed/return states remain
owned by Survival.

There is no mid-song replacement or refill. Only the same authenticated
deployment member may return during remaining grace. Active-roster changes do
not rewrite locked loadouts, initial scaling, committed effects, contribution,
or result evidence.

## 10. Disconnect snapshot and cumulative grace

Connection loss records the last confirmed Ward, location, survival/downed
state, Hype, Band Call state, consumable charges, loadout, contribution, and
participation identities without taking ownership of them.

Already committed boss impacts, resource spends, Signature/group effects, and
other guaranteed events resolve normally. During grace the player supplies no
new targets, judgments/misses, contribution, resource/readiness gain, or group
eligibility. Rhythm records absence so it cannot improve rating.

Each member receives one cumulative attempt grace budget, initially around 45
seconds. Disconnected time consumes it; successful return pauses consumption;
later disconnects use only the remainder.

A reconnect authenticates the same player/deployment/attempt and restores the
immutable snapshot at the next safe musical boundary. The prior location is used
when legal, otherwise the nearest valid Middle location. A standing return gets
normal settling protection. A downed return remains downed. Spent/completed/
committed state remains so, and the player cannot enter an already committed
group window.

## 11. Grace expiry, departure, and later scaling

Grace expiry transitions the member once to Departed. Gameplay rejoin is then
closed, though a later connection may reach results/hub.

Departure never rewrites an open Resolve threshold, banked progress, committed
attack/effect/group snapshot, earned attribution, or history. Pending invitations
remove the player; future targets/group thresholds use the smaller active roster.
Only unopened Resolve layers may use that roster when they open. No replacement
joins.

Meaningful completed contribution remains available to Results/Rewards. Network
departure alone does not erase eligibility; Rewards evaluates contribution,
coverage, voluntary leave/inactivity, and outcome evidence.

When no Active or grace-eligible human remains, Multiplayer publishes a terminal
safe-boundary fact. All Departed produces **all humans departed**. A terminal mix
of permanent Inactive/Departed produces **all humans inactive/departed**. Boss
Encounters treats these as explicit Defeat, not system-failure No Contest.

## 12. AFK warning, inactive state, and resume

AFK evidence counts only a connected, non-downed, unsuspended scoring group with
genuine playable material for that chart. Any plausible genuine gameplay attempt
prevents a strike; accuracy and output are irrelevant.

Initial tuning privately warns after two consecutive ignored eligible groups and
enters Inactive after two more. Inactive preserves its snapshot but gets no new
targets, judgments/misses, contribution, readiness, or group eligibility.
Committed effects/impacts remain valid.

The player may explicitly request one safe-boundary resume per encounter. Normal
re-entry/settling applies and cannot cancel a committed consequence. Repeated
inactivity after that resume lasts for the rest of the attempt and may remove
participation-based reward eligibility.

Quiet/rest passages, weak/mistimed play, safe positioning, optional decline,
downing, network absence, Rhythm suspension, and accessibility use never create
a strike or misconduct label.

If everyone is inactive but someone can resume, the clock continues. When no
Active/grace-eligible human remains and nobody can resume, Multiplayer publishes
the terminal inactive/departed Defeat fact.

## 13. Protected automatic cues and preset pings

Automatic domain cues communicate attacks, targeting, movement, downing,
revival, Band Calls, Crescendos, and phase changes. They cannot be hidden,
delayed, rate-limited, or impersonated by another player's ping/mute state.

First-release player pings are Move, Defend, Join Call, Revive, and Ready/Thanks.
Each validates current context and optional legal target/location. It transmits
no arbitrary text and creates no gameplay action.

Per sender, a token limit initially allows a burst of two and restores one token
every four encounter measures. Revisioned real-time equivalents apply outside a
musical clock. Identical overlapping pings coalesce; rejection privately explains
the wait.

A recipient may mute one sender's pings for the current party/rematch flow and
undo it. Automatic cues remain. Failed delivery creates no inferred acceptance
or action.

Each ping maps to localized text, icon/shape, optional audio/haptics, and
accessible alternatives. Meaning never depends on color or sound. First release
adds no custom free-form text/voice. Roblox communication may remain under
platform/account controls but is never required.

## 14. Structural safety, moderation, and privacy

The server makes friendly fire, body blocking, negative contribution, spending
another player's resources, forced composition, vote-kick, host/leader combat
authority, and binding follow-up impossible.

Weak accuracy, safe positioning, duplicate role choice, optional decline, boss
struggle, and accessibility use are not misconduct. Only the private evidence-
based inactivity contract affects participation state/reward evidence.

Players may mute pings and use Roblox block/report/account surfaces. Where
exposed relationship data permits, blocks prevent new invitations/rematch
grouping but never rewrite an active/committed encounter. The game validates and
rate-limits commands; Roblox owns account moderation/filtering/sanctions.

Other players may see safe platform identity/appearance, role, Ready,
connection/availability, downed/revival state, and required cues. Private
inventory/quantities/purchases, performance/history/recommendations,
accessibility/calibration/settings, block/report state, and moderation facts are
not exposed.

There is no public report/accusation/AFK label, moderation history, individual
ranking, damage leaderboard, or blame. Preset-only custom communication, safe
defaults, and platform protections apply to every player including minors.

## 15. Results and voluntary follow-up

After immutable outcome, each player's Reward transaction finalizes before
follow-up. Choices cannot delay, pool, divide, revoke, or condition that reward.

Each player chooses Retry Same Shard, Stay with Band, or Return to Hub. Timeout
defaults only that player to Hub. No leader/majority binds another.

Public Retry/Stay players form a voluntary temporary rematch group. Retry records
consent to the same boss/difficulty. Stay preserves the social group but requires
acceptance of the next proposal. Only mutually accepted eligible members stage;
public matching may then refill empty seats.

Current Parties persist unless members leave. Every retry/new selection repeats
proposal, validation, staging Ready, lock, consumable reservation, and deployment.
Failure preserves granted rewards and returns players to hub/group state.

## 16. Semantic outputs and service failure

Authoritative facts cover:

- party invitation/member/leader/proposal/consent;
- queue/match/two-player choice and cancellation;
- staging validation/Ready/timer/lock and deployment acknowledgment/start/cancel;
- connection/grace/rejoin/departure and active-roster projections;
- AFK evidence/warning/inactive/resume/terminal state;
- Results/follow-up/rematch/refill; and
- ping/rate/coalescing/mute/delivery/failure.

Facts carry applicable identities/revisions, server or logical musical time,
prior/next state, reason/source, and idempotency identity. Consumers receive only
what they own/need: UI/Audio presentation-neutral state, Items lock/release,
gameplay domains roster/eligibility, Results/Rewards participation evidence,
Player Data required durable facts, and Analytics privacy-reviewed semantics.

Pre-start failure rolls back safely. Player-local runtime failure uses grace or
local Rhythm suspension. Global authoritative session/roster/clock corruption
that prevents fair outcome evaluation sends an exact critical-failure fact to
Boss Encounters for No Contest. Noncritical communication/presentation failure
cannot fabricate consent or gameplay.

## 17. Operational configuration and verification

Revisioned configuration includes:

- party capacity, invitation/proposal expiry, and leader transfer;
- queue minimum/target/maximum, two-player timeout, region/latency, and content
  compatibility;
- public Ready, final countdown, loading/acknowledgment, follow-up, and refill;
- cumulative grace, rejoin transport, AFK warning/inactive, and one resume;
- localized ping enums/context/targets, musical/real-time token rates,
  coalescing/muting, and protected automatic-cue classification; and
- privacy allowlists, operational retention, failure codes, and alert thresholds.

Verification covers every mode, one-to-six humans, supported difficulty/region/
content, duplicate roles, party/membership/consent changes, all timeout/loading/
transport failures, repeated disconnects, open/unopened-layer changes, all-
departed/inactive states, pings/mutes, Roblox block/report handoff, localization,
accessibility, privacy, and duplicate/out-of-order delivery.

## 18. Deferred tuning and technical work

Behavior is complete; these remain versioned operational/playtest/architecture
work:

- exact invitation/proposal/queue/Ready/load/follow-up timeouts;
- region/latency measurement and queue-service implementation;
- initial three-second lock, 45-second grace, and AFK group counts;
- ping rate, localization, coalescing, and platform chat/voice integration;
- rejoin transport, authoritative state transport, privacy retention, abuse/
  failure telemetry, and operational alerts;
- Rewards participation thresholds and No Contest compensation; and
- final UI/audio presentation and manual network/safety test procedures.

Tuning may not introduce hidden skill/gear/spending matching, composition locks,
host authority, partial deployment, join-in-progress, resettable grace,
wall-clock AFK during rests, public blame, unprotected critical cues, custom chat
dependency, structural grief powers, forced follow-up, or player-failure results
for critical system corruption.

## 19. Approval and change control

The owner interview resolved MP-01 through MP-12 on 2026-08-25. This document is
the canonical Multiplayer and Communication Safety design specification.

A material change to consent, party/queue inputs, two-player choice, Ready/lock,
atomic deployment, no join-in-progress, cumulative grace, roster rescaling,
AFK/resume, ping protection/rate/mute, structural safety/privacy, or follow-up
requires an explicit amendment citing the superseded rule. Operational tuning
inside these boundaries creates a new configuration revision and never changes
an active deployment.
