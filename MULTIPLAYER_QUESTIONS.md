# Bands Battle Multiplayer Specification Questions

- **Status:** Completed; 12 of 12 questions resolved
- **Started:** 2026-08-25
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#51-multiplayer-sessions-parties--matchmaking)
- **Included requirement:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#52-communication--safety)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Encounter dependency:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Abilities dependency:** [`ABILITIES_AND_COOPERATIVE_ACTIONS.md`](ABILITIES_AND_COOPERATIVE_ACTIONS.md)
- **Working record:** [`MULTIPLAYER_WORKING.md`](MULTIPLAYER_WORKING.md)
- **Planned canonical result:** `MULTIPLAYER.md`

## 1. Interview method

This interview uses four checkpoints of three questions. It inherits settled
one-to-six-human scaling, loadout locks, encounter outcome, group-action,
progression eligibility, individual rewards, and Results boundaries. It focuses
on consent, queues, staging/deployment, active-roster failures, communication,
and structural safety.

Answers are persisted after each checkpoint. When all twelve questions are
resolved, they are reconciled into canonical `MULTIPLAYER.md` including the
Communication & Safety requirements.

## 2. Fixed inherited decisions

- A shard card offers Solo, Current Party, and Public Band after boss/difficulty
  selection; nobody enters content or difficulty they did not choose/qualify for.
- Current parties contain two to six humans. A leader proposes, every member
  consents, and the leader cannot force staging, deployment, or follow-up.
- Public matching is server-owned with no host, matches boss/difficulty/region,
  targets three to six humans, and offers both players a choice after an initial
  roughly 45-second target rather than forcing a two-player start.
- Staging permits loadout/role changes and duplicate instruments. Boss/difficulty
  lock before matching/staging; roster/loadouts lock at the final roughly
  three-second deployment countdown.
- Unready players are not dragged into battle. Ordinary encounters have no
  join-in-progress and use the deployment roster for initial scaling.
- Rewards grant individually before follow-up choices. Retry, Stay, and Hub are
  individual; Retry returns to staging and never bypasses readiness.
- Disconnect preserves confirmed gameplay/resource state, applies already
  committed impacts, then removes targeting/contribution/misses during grace.
- Rejoin grace initially targets roughly 45 seconds and returns the player at a
  safe musical boundary; downed returns downed and spent remains spent.
- Grace expiry removes the player from the active roster without shrinking the
  open Resolve layer; unopened layers may rescale at their next boundary.
- AFK is measured from ignored eligible gameplay, not quiet wall-clock time.
  Poor accuracy, safe positioning, optional decline, or struggle is not griefing.
- Coordination never depends on custom free-form text or voice. Protected
  automatic cues and rate-limited preset pings provide the core language.
- No friendly fire, vote-kick, body blocking, negative contribution, spending
  others' resources, forced follow-up, or public individual blame/ranking exists.

## 3. Question plan

### Checkpoint A — Entry, parties, and public matching

#### MP-01 — Shard entry modes, eligibility, and selection

- **Status:** Resolved 2026-08-25.

- **Decision needed:** How does one player choose a valid encounter path without
  creating ambiguous or overlapping session state?
- **Must resolve:** Card facts, Solo/Party/Public availability, boss/difficulty
  order, onboarding/progression eligibility, one active flow, cancellation,
  stale selection, service failure, and no silent substitution.

#### MP-02 — Current Party membership, leadership, and consent

- **Status:** Resolved 2026-08-25.

- **Decision needed:** How are two-to-six-person parties formed/changed and how
  does a proposal obtain unanimous consent without leader coercion?
- **Must resolve:** Invite/join/leave, capacity, leader assignment/transfer,
  proposal identity/expiry, member eligibility, accept/decline, membership
  change, ready reset, privacy, and party persistence.

#### MP-03 — Public queue, matching inputs, and two-player choice

- **Status:** Resolved 2026-08-25.

- **Decision needed:** How does server-owned matching build a fair regional
  roster and handle a long two-player queue?
- **Must resolve:** Queue key, region/latency, no host, target/min/max, skill or
  gear inputs, background queue, duplicate roles, roughly 45-second choice,
  unanimous two-player start, continue/solo/leave, replacement, and failures.

### Checkpoint B — Staging, deployment, and follow-up

#### MP-04 — Staging validation, ready state, and final lock

- **Status:** Resolved 2026-08-25.

- **Decision needed:** What may change in staging and what exactly locks before
  deployment?
- **Must resolve:** Editable loadout/role/spec, duplicate roles, validation,
  ready/unready, edit resets, ready timer, replacement/return, boss/difficulty
  lock, final countdown, roster/loadout snapshot, leaving, and consumables.

#### MP-05 — Deployment, active roster, and no join-in-progress

- **Status:** Resolved 2026-08-25.

- **Decision needed:** How does a locked roster become one immutable encounter
  deployment without partial starts or mid-song replacement?
- **Must resolve:** Loading acknowledgments, start boundary, timeout/failure,
  pre-start departure, initial population scaling, session/attempt identity,
  encounter roster versus active roster, no refill/join, state handoff, and
  rollback/refund.

#### MP-06 — Results, rematch grouping, refill, and exit

- **Status:** Resolved 2026-08-25.

- **Decision needed:** How do independent post-result choices create the next
  voluntary group?
- **Must resolve:** Immediate rewards, Retry/Stay/Hub, choice timer/default,
  public rematch pool, Current Party persistence, leader transfer, refill,
  staging return, revalidation, opt-out, failure, and no binding vote.

### Checkpoint C — Disconnect, departure, and inactivity

#### MP-07 — Disconnect snapshot, grace, and safe rejoin

- **Status:** Resolved 2026-08-25.

- **Decision needed:** Which state freezes at connection loss and how does a
  legitimate rejoin resume without dodging consequences or receiving unfair
  misses?
- **Must resolve:** Confirmed snapshot, committed impacts, absent targeting/chart
  handling, grace duration, reconnect identity, safe boundary, location fallback,
  settling, downed state, resources, group eligibility, and repeat disconnects.

#### MP-08 — Grace expiry, active-roster change, and retained eligibility

- **Status:** Resolved 2026-08-25.

- **Decision needed:** What changes when a missing player becomes Departed?
- **Must resolve:** Active-roster transition, current/open layer, later-layer
  rescaling, group thresholds, targets, all-players-gone, party leadership,
  completed contribution, result/reward eligibility, refill prohibition,
  notification, and idempotency.

#### MP-09 — AFK warning, inactive state, resume, and reward consequence

- **Status:** Resolved 2026-08-25.

- **Decision needed:** How does the system identify deliberate inactivity without
  punishing weak or interrupted play?
- **Must resolve:** Eligible-group evidence, quiet/rest exclusion, private
  warning, inactive threshold, targeting/group removal, one resume, repeated
  inactivity, coverage/performance, reward eligibility, all-inactive behavior,
  false-positive exclusions, and telemetry.

### Checkpoint D — Communication, safety, and outputs

#### MP-10 — Protected cues, preset pings, rate limits, and muting

- **Status:** Resolved 2026-08-25.

- **Decision needed:** What coordination language ships and which messages can
  never be hidden by another player's mute setting?
- **Must resolve:** Automatic-domain cues, Move/Defend/Join Call/Revive/
  Ready-Thanks pings, context/targets, cooldown/rate/burst, individual mute,
  critical-cue protection, localization, accessibility, custom/platform chat,
  spam response, and delivery failure.

#### MP-11 — Structural anti-grief, moderation, and privacy

- **Status:** Resolved 2026-08-25.

- **Decision needed:** Which harmful powers are impossible by design and which
  cases hand off to Roblox safety surfaces?
- **Must resolve:** No host authority/vote-kick/friendly fire/body blocking/
  negative contribution/resource spending/forced follow-up, leader limits,
  weak-play protection, block/report, platform controls, identity/data exposure,
  public results, sanctions boundary, and minors/safe defaults.

#### MP-12 — Semantic outputs, service failures, and completeness audit

- **Status:** Resolved 2026-08-25.

- **Decision needed:** Which authoritative facts/configuration/validators make
  the full flow deterministic and operable?
- **Must resolve:** Party/queue/staging/deployment/connection/AFK/follow-up/ping
  facts, identities/times/revisions, UI/Audio/domain/Results/Rewards/Analytics
  consumers, privacy, idempotency, timeout/service failure, No Contest handoff,
  configuration, test matrix, and completion audit.

## 4. Completion criteria

`MULTIPLAYER.md` is complete only when:

- MP-01 through MP-12 are resolved;
- every transition from shard choice to voluntary follow-up requires the right
  player's consent and has a safe cancellation/failure result;
- public matching has no host or hidden composition/skill coercion;
- staging/deployment produces one immutable roster/loadout snapshot without
  joining an active song;
- disconnect/rejoin/departure and AFK cannot create invulnerability, fabricated
  misses, retroactive scaling exploits, or false misconduct findings;
- critical cues remain protected while player pings are rate-limited/mutable;
- structural safety prohibits the specified grief vectors; and
- every state/output/failure path is identified, idempotent, and privacy-safe.

## 5. Change log

- **2026-08-25:** Created the concise 12-question Multiplayer and Communication
  & Safety interview from the approved GDD and canonical dependencies.
- **2026-08-25:** Approved MP-01 through MP-03, completing entry, party, and
  public-matching checkpoint A. Progress is 3 of 12 questions.
- **2026-08-25:** Approved MP-04 through MP-06, completing staging, deployment,
  and follow-up checkpoint B. Progress is 6 of 12 questions.
- **2026-08-25:** Approved MP-07 through MP-09, completing disconnect, departure,
  and inactivity checkpoint C. Progress is 9 of 12 questions.
- **2026-08-25:** Approved MP-10 through MP-12, completing Communication &
  Safety and outputs checkpoint D. All 12 questions are resolved and the
  canonical specification was published.
