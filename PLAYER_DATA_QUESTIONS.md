# Bands Battle Player Data Specification Questions

- **Status:** Interview complete; 12 of 12 questions resolved and reconciled
- **Started:** 2026-09-01
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#82-player-data)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Progression dependency:** [`PROGRESSION.md`](PROGRESSION.md)
- **Items dependency:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)
- **Rewards/Commerce dependency:** [`REWARDS_AND_ECONOMY.md`](REWARDS_AND_ECONOMY.md)
- **Builds dependency:** [`BUILDS_AND_SPECIALIZATION.md`](BUILDS_AND_SPECIALIZATION.md)
- **UI/settings dependency:** [`UI_UX.md`](UI_UX.md)
- **Working record:** [`PLAYER_DATA_WORKING.md`](PLAYER_DATA_WORKING.md)
- **Canonical result:** [`PLAYER_DATA.md`](PLAYER_DATA.md)

## 1. Interview method

This interview uses four checkpoints of three questions. It first inventories
durable facts and safe loading, then resolves transactions/concurrency, save and
recovery behavior, and finally Commerce/privacy/operational completeness.

Answers are persisted after each checkpoint. When all twelve questions are
resolved, they are reconciled into canonical `PLAYER_DATA.md`. Exact Roblox
service selection, storage keys, serialization, retry intervals, and capacity
budgets remain technical architecture unless a behavioral boundary is required
to prevent loss, duplication, stale overwrite, or unsafe play.

## 2. Fixed inherited decisions

- Player Data preserves facts and commits approved mutations; each domain owns
  what its data means, which mutation is legal, and which semantic facts result.
- Client state is never authority for progression, inventory, balances,
  purchases, Ready, rewards, or saved settings.
- Progression source facts are permanent and monotonic where specified: campaign
  nodes/fragments, difficulty access, General Progression/unlocks, mastery,
  milestones, personal records/archive, hub restoration, practice/public access,
  and migration/grandfathering evidence.
- Items persist unique instances/lineage/tier/rank, consumable stacks and durable
  reservations/spends/refunds, cosmetic entitlements/appearance, current
  loadout, three full spec presets, validation/missing state, locked references,
  and mutation/migration history.
- Builds persist draft/applied configurations, option/catalog/balance revisions,
  validation/migration/replacement state, and their place inside current/full
  spec presets. Builds do not create another preset store.
- Rewards/Economy persist permanent general/boss balances, pinned goals, frozen
  roll/guarantee history, immutable transaction ledger/journal, Pending/commit/
  rollback/compensation outcomes, and causal pre/post state.
- Commerce persists eligibility/notice dismissal where required, quotes only for
  their useful lifetime, verified receipt outcomes, exact granted purchase
  lineage, restoration, duplicate/recovery state, and no payment credential.
- UI/UX persists account-wide accessibility/teaching/language intent, explicit
  per-device/control/output profiles and overrides, calibration/bindings/touch/
  presentation/audio/caption preferences, onboarding checkpoints/skip, and
  contextual-prompt history.
- Transactions are atomic and idempotent. Same-identity retries return the
  established Pending/Committed/Rejected result and never repeat random rolls,
  spends, grants, unlocks, refunds, or compensation.
- Balances never become negative. Inventory, progress, entitlements, purchase
  history, and transaction correction are never silently deleted, substituted,
  clamped, rerolled, or rewritten.
- Uncertain persistence is visibly Pending or Save Unsafe, never false success.
  UI does not speculate or locally queue economic, Apply, Ready, consent,
  purchase, reward, or progression mutation while authority is unavailable.
- Active encounter gameplay state such as current Ward, position, Hype, rhythm
  history, roster/grace, boss state, and pending non-durable cues belongs to the
  live attempt/rejoin systems, not the permanent player profile. Only durable
  commitments and references required for value integrity cross that boundary.
- Active loadout snapshots retain exact revisions through the attempt. Catalog/
  schema migration never mutates an active gameplay snapshot or invents a
  player choice.
- Other players see only explicit safe summaries such as role, Ready,
  appearance, availability, survival/group cues. Inventory, balances, purchases,
  settings/accessibility, performance/history, private recommendations,
  migrations, and transaction failures remain private.
- Player Data stores no raw song audio, microphone data, payment credentials,
  raw analytics stream, public leaderboard, moderation accusation, or unnecessary
  private gameplay evidence.

## 3. Question plan

### Checkpoint A - Durable boundary, record model, and safe loading

#### PD-01 - Durable record inventory and ephemeral-state boundary

- **Status:** Resolved 2026-09-02.
- **Decision:** Persist source facts and durable commitments for Progression,
  Items, Economy, Commerce, builds/presets, settings, onboarding, and the bounded
  histories required for recovery or approved personal records. Do not persist a
  competing permanent copy of live encounter, party, queue, audio, cue, or raw
  analytics state.
- **Decision needed:** Which exact facts must survive sessions, and which live
  encounter/session facts must never become a competing permanent profile?
- **Must resolve:** Identity/envelope, Progression, Items/consumables/cosmetics,
  loadout/presets/builds, economy/transactions, Commerce, onboarding/settings/
  profiles, durable reservations/commitments, notification/dismissal history,
  ephemeral attempt/party/queue/analytics exclusions, retention class, and owner.

#### PD-02 - Logical record shape, source facts, revisions, and defaults

- **Status:** Resolved 2026-09-02.
- **Decision:** Treat the profile as one logical, versioned snapshot whose domain
  sections preserve source facts, stable identities, exact catalog revisions,
  causal lineage, and explicit empties. Derived values are recomputed. A complete
  deterministic starter record is created only after authoritative Not Found.
- **Decision needed:** How is one logical player snapshot organized so defaults,
  derived values, catalogs, and domain revisions cannot drift or overwrite truth?
- **Must resolve:** Stable player/record identity, domain sections/revisions,
  sequence/version, source versus derived facts, catalog references, timestamps,
  causal lineage, deterministic first-player defaults, valid starter setup,
  optional empties, no fake default on load failure, limits/overflow, and naming.

#### PD-03 - Load, migration, validation, session authority, and client projection

- **Status:** Resolved 2026-09-02.
- **Decision:** Acquire authoritative session ownership, load every required
  section, migrate, and validate before emitting a privacy-filtered Ready
  projection. Failed, partial, corrupt, or unknown-version loads never become a
  default profile or permit normal durable play.
- **Decision needed:** What must succeed before a player may enter normal durable
  play, and what happens when the initial record cannot be loaded safely?
- **Must resolve:** Authoritative load/not-found distinction, session ownership,
  schema/catalog migration, invariant validation, Ready state, client/private
  projection, first creation, retries/timeouts, partial/domain load, initial
  failure surface, no overwrite, settings access, and joining hub/queue/encounter.

### Checkpoint B - Mutations, cross-domain transactions, and concurrency

#### PD-04 - Mutation command envelope, ownership, and preconditions

- **Status:** Resolved 2026-09-02.
- **Decision:** Clients submit semantic action requests, never profile patches.
  The owning domain authorizes meaning and produces an exact, versioned,
  idempotent mutation plan; Player Data validates generic authority, integrity,
  and preconditions before applying that plan without reinterpretation.
- **Decision needed:** Which common request/result contract lets each domain
  authorize a mutation while Player Data safely commits it?
- **Must resolve:** Command/source/player/domain/action identity, expected
  versions, catalog/content/balance revisions, causal event, preconditions,
  requested patch versus domain result, validation authority, actor/service,
  timestamp/order, privacy, result/reason, idempotency, and prohibited client
  authority/arbitrary writes.

#### PD-05 - Atomic reward, progression, item, resource, and compensation commit

- **Status:** Resolved 2026-09-02.
- **Decision:** Freeze the result, calculations, and random outcomes before one
  transaction updates every affected journal, balance, item, unlock, record,
  refund, and downstream handoff. The entire result commits together or remains
  honestly Pending for recovery; corrections use linked compensation entries.
- **Decision needed:** How does one outcome/action commit every related ledger,
  balance, instance, unlock, record, and handoff without partial player-visible
  state?
- **Must resolve:** Frozen result, calculation ownership, transaction plan,
  cross-domain preconditions, append-only ledger, pre/post snapshot, atomic
  bundle, downstream handoff, Pending/recovery, rollback versus compensation,
  Results timing, Retry independence, No Contest refunds, and invariants.

#### PD-06 - Idempotency, ordering, session lease, and stale-write protection

- **Status:** Resolved 2026-09-02.
- **Decision:** One exclusive session epoch owns writes. Requests carry durable
  identities and expected versions; duplicates replay their established result,
  stale or out-of-order changes refresh or reject, and lease loss stops new
  mutation immediately. Rejoin or explicit takeover never creates two writers.
- **Decision needed:** How do duplicate delivery, server overlap, reconnect,
  timeout, and out-of-order work preserve one authoritative mutation history?
- **Must resolve:** Idempotency scope/retention, result replay, monotonic sequence,
  optimistic version, serialization, exclusive lease/session epoch, takeover/
  expiry, stale client/server response, concurrent device/server, retry safety,
  lock loss, split brain, and refresh/reject behavior.

### Checkpoint C - Save cadence, unsafe state, recovery, and migration

#### PD-07 - Dirty state, save cadence, checkpoints, shutdown, and budgets

- **Status:** Resolved 2026-09-02.
- **Decision:** Value, progression, inventory, consumable, applied-build, and
  loadout mutations require authoritative commit before success. Low-risk
  preferences may batch while visibly unsaved. Checkpoints and lifecycle flushes
  add protection, and write pressure prioritizes critical transactions without
  dropping dirty state or replacing the last confirmed version.
- **Decision needed:** When must state become durable, when may ordinary settings
  batch, and how do leave/shutdown/budget limits avoid false success or data loss?
- **Must resolve:** Immediate transactional commits, deferred low-risk writes,
  dirty domains, debounce/coalescing, periodic checkpoints, encounter boundaries,
  leave/teleport/shutdown, reservation lifecycle, write budget/backpressure,
  retry queue, flush deadline, last confirmed version, and UI save state.

#### PD-08 - Load/save failure, read-only or Save Unsafe play, and player recovery

- **Status:** Resolved 2026-09-02.
- **Decision:** A safe active attempt may finish from its locked snapshot during
  a persistence outage, with reservations recoverable and rewards Pending. Block
  new durable gameplay, Ready, queue, configuration, inventory, and Commerce
  actions until recovery; expose honest Save Unsafe state and safe player actions.
- **Decision needed:** Which actions remain safe during partial outage and how
  does the player recover without losing confirmed value or creating speculative
  progress?
- **Must resolve:** Initial-load block, mid-session read-only/unsafe transition,
  current encounter continuation, new encounter/queue/Ready block, settings
  drafts, rewards Pending, Commerce block, retry/backoff, reconnection, timeout,
  player messaging/actions, no local success, and operational escalation.

#### PD-09 - Schema/catalog migration, corruption, backup, rollback, and repair

- **Status:** Resolved 2026-09-02.
- **Decision:** Migrations are ordered, versioned, idempotent, backed up before
  destructive change, and validated before and after. Quarantine corruption and
  prefer verified last-known-good plus forward repair. Preserve exact retired
  references or use explicit equivalence, player choice, or compensation.
- **Decision needed:** How do records evolve and recover while preserving owned
  value, history, references, and explicit player choices?
- **Must resolve:** Schema and domain versions, migration registry/order,
  compatibility windows, idempotency, pre/post invariant checks, dry-run/evidence,
  backup/snapshot, corruption detection/quarantine, last-known-good, rollback,
  forward repair, retired identities/equivalence/compensation, active snapshots,
  and player/operation disclosure.

### Checkpoint D - Commerce, privacy, lifecycle, and completeness

#### PD-10 - Commerce receipts, purchase lineage, restoration, and recovery

- **Status:** Resolved 2026-09-02.
- **Decision:** Only a verified platform receipt authorizes fulfillment. Receipt,
  product mapping, exact grant, entitlement/item, balance effects, and purchase
  history commit atomically before acknowledgment. Duplicate delivery replays the
  original result; restoration and reversal preserve auditable lineage.
- **Decision needed:** How are verified platform receipts and exact grants made
  durable once without storing payment data or acknowledging uncertain value?
- **Must resolve:** Quote lifetime/storage, receipt identity/status, platform
  verification, product/catalog mapping, Pending/grant/Already Processed/cancel/
  Recovery Required, item/entitlement and purchase-history atomicity, duplicate/
  concurrent delivery, restoration, platform refund/support handoff, privacy,
  retention, and acknowledgment.

#### PD-11 - Privacy, minimization, access, retention, export, and deletion

- **Status:** Resolved 2026-09-02.
- **Decision:** Allowlist every stored field and consumer. Players, other players,
  game services, support, and Analytics receive only their minimum approved
  projection. Retention follows value/recovery need; export is player-readable,
  and deletion removes or anonymizes data while retaining only required protected
  tombstones or evidence under explicit policy.
- **Decision needed:** Which data may be stored or exposed, for how long, and how
  do account export/deletion/legal operations preserve integrity and child safety?
- **Must resolve:** Classification/allowlists, owner/service/other-player/admin/
  Analytics access, encryption/security handoff, payment/microphone/raw-evidence
  exclusions, operational ledgers, retention/compaction, audit access, export,
  deletion/tombstone, active purchase/refund/legal holds, backups, derived data,
  support correction, notification, and platform policy.

#### PD-12 - Semantic outputs, observability, disaster tests, and completion

- **Status:** Resolved 2026-09-02.
- **Decision:** Publish versioned semantic lifecycle and operation results. Use
  privacy-safe health metrics, alerts, and runbooks for every failure class, and
  test duplicate/order/concurrency, crash boundaries, outages, migrations,
  corruption/restore, receipts, privacy lifecycle, and cross-domain invariants.
- **Decision needed:** Which facts, metrics, alerts, runbooks, and test matrices
  make Player Data implementation-ready without inventing persistence behavior?
- **Must resolve:** Load/save/lease/version/migration/transaction/receipt/privacy/
  delete outputs, consumers, error taxonomy, correlation/idempotency identities,
  safe logs, SLO/budgets, alerts, dashboards/runbooks, failure injection,
  duplicate/order/concurrency tests, migration/corruption/restore drills,
  domain-invariant matrix, UI scenarios, Analytics boundary, and final audit.

## 4. Completion criteria

`PLAYER_DATA.md` is complete only when:

- PD-01 through PD-12 are resolved;
- every durable fact has one owner, retention class, logical location, revision,
  and mutation path while ephemeral encounter state stays outside the profile;
- first creation, loading, migration, validation, and client projection cannot
  confuse absence with failure or expose unsafe normal play;
- every mutation and cross-domain transaction is atomic, idempotent, ordered,
  versioned, and protected from client/stale/concurrent authority;
- save cadence and failure behavior distinguish immediate value commits from
  batchable preferences without false success;
- migration, corruption, backup, rollback, repair, retirement, and compensation
  preserve value and explicit choice;
- receipts, privacy, retention, export, deletion, and operational access obey the
  approved safety/economy boundaries; and
- outputs, observability, runbooks, budgets, and failure tests leave no
  implementation-agent design choice.

## 5. Change log

- **2026-09-02:** Resolved PD-01 through PD-03. Defined the durable/ephemeral
  boundary, the logical versioned record and first-create rules, and the safe
  authoritative load-to-Ready sequence. Progress is 3 of 12 questions.
- **2026-09-02:** Resolved PD-04 through PD-06. Defined semantic mutation plans,
  atomic cross-domain commit, result/roll freezing, idempotent replay, exclusive
  session authority, and stale-write protection. Progress is 6 of 12 questions.
- **2026-09-02:** Resolved PD-07 through PD-09. Defined critical versus batchable
  saves, Save Unsafe and active-attempt outage behavior, lifecycle flushing,
  migration evidence, corruption quarantine, restoration, and forward repair.
  Progress is 9 of 12 questions.
- **2026-09-02:** Resolved PD-10 through PD-12. Defined verified receipt commit
  and restoration, privacy/access/retention/export/deletion boundaries, semantic
  outputs, operational evidence, runbooks, and disaster testing. The Player Data
  interview is complete at 12 of 12 questions.
- **2026-09-02:** Reconciled all approved decisions into canonical
  `PLAYER_DATA.md` and completed the cross-specification audit.
- **2026-09-01:** Created the concise 12-question Player Data interview from the
  Systems Map and complete durable/persistence contracts of specifications 2
  through 12.
