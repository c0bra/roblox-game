# Bands Battle Player Data Working Record

- **Status:** Interview complete; 12 of 12 questions resolved
- **Started:** 2026-09-01
- **Question plan:** [`PLAYER_DATA_QUESTIONS.md`](PLAYER_DATA_QUESTIONS.md)
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#82-player-data)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Canonical result:** [`PLAYER_DATA.md`](PLAYER_DATA.md)

## 1. Role of this record

This file preserves the approved answers, refinements, inherited constraints,
and cross-system handoffs from the completed Player Data interview. It is
evidence for the canonical specification, not the final authority.

## 2. Inherited boundary

Player Data owns durable loading/saving/commit/recovery guarantees, logical
record/version/default contracts, concurrency/stale-write protection, schema
migration, privacy lifecycle, and player-visible save safety. Each domain owns
meaning, validation, calculation, and legal mutations. Live attempt/session
systems own ephemeral reconnect/gameplay state.

The complete inherited decision set is recorded in
[`PLAYER_DATA_QUESTIONS.md`](PLAYER_DATA_QUESTIONS.md#2-fixed-inherited-decisions).

## 3. Decision record

### Checkpoint A - Durable boundary, record model, and safe loading

#### PD-01 - Durable record inventory and ephemeral-state boundary

- **Status:** Resolved 2026-09-02.
- The durable profile is one logical record divided into owned domains. Every
  durable field must have one semantic owner, durability/retention class,
  privacy classification, revision context, and legal mutation path.
- Durable Progression source facts include campaign nodes and fragments,
  difficulty and general unlocks, mastery, milestones, personal bests, hub
  restoration, practice/public access, and migration or grandfathering evidence.
- Durable Items facts include unique item identity and lineage, tier/rank,
  consumable quantities and durable reservations/spends/refunds, cosmetic
  entitlements and appearance, current loadout, three complete spec presets,
  validation/missing/locked-reference state, and item mutation/migration history.
- Durable Builds facts include drafts and applied configurations, the source
  option/catalog/balance revisions, validation and replacement state, and their
  placement inside the current loadout or full spec presets. There is no second
  preset store.
- Durable Economy and Rewards facts include general and boss balances, pinned
  goals, frozen rolls and guarantees, immutable transaction history, and Pending,
  committed, rollback, refund, or compensation outcomes with causal lineage.
- Durable Commerce facts include verified receipt state, exact purchase/grant
  lineage, restoration and duplicate/recovery state, and required notice
  dismissal. Quotes persist only for their useful lifetime. Payment credentials
  never enter the record.
- Durable player-intent facts include account-wide accessibility, teaching, and
  language choices; explicit per-device/control/output profiles and overrides;
  calibration, binding, touch, presentation, audio, and caption preferences;
  onboarding checkpoints or skip; contextual-prompt history; and notification or
  dismissal state that must survive sessions.
- Frozen result facts persist only as long as required to finish transaction
  recovery, support a personal best, or provide an explicitly approved, bounded,
  private improvement history. Raw attempt events and raw analytics do not become
  permanent profile data.
- Active Ward, player position, Hype, boss and rhythm state, roster/grace, party,
  queue, live cues, audio streams, microphone input, and raw analytics are
  ephemeral. They remain owned by live systems and are not copied into a
  competing permanent player model.
- Durable commitments that cross the live boundary are limited to value-safety
  facts such as consumable reservations/spends/refunds, immutable locked
  references required for recovery, and frozen final results or transactions.

#### PD-02 - Logical record shape, source facts, revisions, and defaults

- **Status:** Resolved 2026-09-02.
- One logical snapshot does not require one physical storage object. Technical
  architecture may shard it, but loading, validation, and mutation must preserve
  one coherent player truth.
- The logical envelope includes stable player/record identity, overall schema
  version, and a monotonic overall commit sequence. Each domain section carries
  its own schema version, domain revision, last commit identity/time, and owned
  source facts.
- Saved references use stable identities plus the exact relevant catalog,
  content, option, or balance revision. Causal lineage links each durable change
  to its approved source event or transaction.
- Derived stats, compatibility, availability, projections, and display summaries
  are recomputed from source facts and current catalogs. They are not competing
  saved truth unless a frozen historical value is required as transaction or
  personal-record evidence.
- Optional absence is explicit. Required values are never silently fabricated,
  and unknown fields or versions are migrated or rejected rather than discarded.
- An authoritative Not Found result creates one deterministic, complete starter
  bundle, including a valid starter loadout, presets, balances, settings intent,
  and onboarding state. The entire bundle is validated and committed together.
- A timeout, error, partial response, corrupt record, or unknown version is never
  interpreted as Not Found and never causes default state to overwrite existing
  player data.
- Capacity or serialization overflow fails closed and is surfaced for recovery.
  Owned value and history are never silently truncated to make a record fit.

#### PD-03 - Load, migration, validation, session authority, and client projection

- **Status:** Resolved 2026-09-02.
- The authoritative load lifecycle is `Loading -> Migrating -> Validating ->
  Ready`, with explicit `Retrying` and `Blocked` branches. Normal durable play is
  available only in Ready.
- The server acquires exclusive session authority before accepting mutations,
  then loads every required logical section. Details of lease epochs, takeover,
  and stale-write rejection are resolved by PD-06.
- Ordered, idempotent schema migrations run before current-catalog compatibility
  checks and cross-domain invariant validation. A safe catalog retirement may
  produce an explicit Incomplete or replacement-required state inside an
  otherwise valid snapshot; structural corruption and unknown schema block Ready.
- First creation occurs only after an authoritative Not Found response and uses
  the complete starter transaction defined by PD-02.
- The client receives only the minimum privacy-filtered projection needed for its
  current surfaces, including the relevant overall/domain revisions. It cannot
  submit arbitrary patches or replacement records.
- Failed, partial, corrupt, unknown-version, or authority-conflicted loading does
  not enter the hub, queue, staging, store, or encounter using provisional
  defaults. It offers Retry or Leave and never writes defaults over uncertain
  state.
- The load screen may expose the minimal accessibility and device controls needed
  to use that screen. Until Ready, those temporary choices are clearly unsaved
  and do not imply profile authority.
- Retry preserves the same load intent. If another valid session owns the record,
  this session waits, retries, or performs an approved takeover rather than
  becoming a parallel writer.

### Checkpoint B - Mutations, cross-domain transactions, and concurrency

#### PD-04 - Mutation command envelope, ownership, and preconditions

- **Status:** Resolved 2026-09-02.
- The client submits a semantic intent such as equip an item, apply a build,
  change a setting, claim an eligible action, or begin a purchase. It never sends
  an arbitrary saved-field patch, replacement domain section, balance, result,
  entitlement, or revision.
- The semantic domain owns authorization and meaning. It validates the action
  against its rules and produces an exact mutation plan. Player Data owns generic
  identity, session authority, version, integrity, ordering, and commit checks;
  it does not recalculate or reinterpret domain meaning.
- Each request and resulting plan includes a unique request/transaction identity,
  player identity, actor/service, owning domain and action, causal event, expected
  overall and affected-domain revisions, exact relevant catalog/content/option/
  balance revisions, declared preconditions, and immutable ordered operations.
- The envelope also identifies privacy class, authoritative ordering context,
  creation time, and idempotency scope. Technical wire format and clock source
  remain architecture choices, but they must not weaken these semantics.
- Player Data rechecks the exclusive session epoch, expected versions, generic
  invariants, plan shape, allowed domain ownership, and referenced identities
  immediately before commit. Domain-specific legality remains the owner's job.
- The authoritative result is `Pending`, `Committed`, `Rejected`, `Conflict`, or
  `Recovery Required`, with stable reason, pre/post revisions where known, and
  the original request/transaction identity. A duplicate receives this recorded
  result rather than executing the action again.

#### PD-05 - Atomic reward, progression, item, resource, and compensation commit

- **Status:** Resolved 2026-09-02.
- Before committing an encounter or action outcome, its authoritative result,
  eligibility, calculations, random rolls, guarantees, item identities, catalog
  revisions, and permitted refunds or compensation are frozen. Retries never
  recalculate or reroll them.
- One transaction plan contains every related append-only journal entry, balance
  delta, item instance or stack change, Progression fact, unlock or entitlement,
  consumable reservation/spend/refund, personal record, purchase or reward
  lineage, and required downstream committed handoff.
- All affected domains validate their preconditions against the same input
  versions before commit. Player-visible domain state changes together; no
  consumer may observe or announce a partial reward as complete.
- When the implementation cannot perform one physical storage transaction, it
  must use a recoverable journal/state machine that exposes the outcome as
  Pending until every component reaches the exact frozen commit or the operation
  reaches its defined rollback state. Architecture chooses the mechanism, not a
  weaker behavior.
- The Results surface may show Pending and let the player Retry gameplay without
  waiting. It changes to Committed only from authoritative confirmation and
  offers recovery/support actions when required.
- A rollback reverses only uncommitted reversible work. Once an owned fact is
  durably committed or externally acknowledged, a correction uses an immutable,
  causally linked compensation entry rather than editing or deleting history.
- No Contest refunds remain capped to the exact consumables causally reserved or
  spent for that attempt. Recovery never invents extra value or removes unrelated
  player value.

#### PD-06 - Idempotency, ordering, session lease, and stale-write protection

- **Status:** Resolved 2026-09-02.
- Exactly one authoritative session lease and epoch may mutate a player's logical
  record at a time. Losing, expiring, or being revoked from that authority stops
  acceptance of new mutations immediately and enters the approved Save Unsafe or
  read-only behavior resolved by PD-08.
- Every mutation uses its stable idempotency identity, expected overall/domain
  revisions, owning session epoch, and a serialized monotonic commit sequence.
  Randomness and calculation outputs are part of the frozen original operation.
- Duplicate delivery or retry returns the previously recorded Pending,
  Committed, Rejected, or recovery result and never repeats a spend, grant,
  unlock, roll, refund, compensation, purchase, or other side effect.
- Irreversible economic, receipt, entitlement, and compensation identities are
  retained for their full recovery and duplicate-delivery lifetime. Lower-risk,
  reversible preference identities may use a bounded retention period defined by
  architecture only when an old retry cannot produce harmful duplicate effects.
- Stale or out-of-order requests fail their version/precondition check and return
  Conflict or refresh guidance. They are never silently applied over newer state.
  Only a domain may explicitly define a safe commutative merge for a narrow
  operation such as an independent preference.
- A new writer may acquire authority only after the former epoch is released,
  expires, or is validly revoked. Every later write from the old epoch is rejected
  even if it was delayed in transit.
- Reconnecting during an active attempt routes through the existing Multiplayer
  rejoin/session path instead of creating another writer. Outside an attempt, an
  explicit takeover may revoke and notify the older session before the new
  session becomes authoritative.
- Queued or delayed work must revalidate its expected version and session epoch
  at execution time. Split-brain ambiguity never resolves through last-write-wins
  or by merging unknown player state.

### Checkpoint C - Save cadence, unsafe state, recovery, and migration

#### PD-07 - Dirty state, save cadence, checkpoints, shutdown, and budgets

- **Status:** Resolved 2026-09-02.
- Purchases, receipts, currencies, rewards, Progression, item/entitlement changes,
  consumable reservations/spends/refunds, applied builds, loadouts, spec presets,
  and other gameplay-affecting choices require authoritative durable commit before
  their surface reports Saved, Applied, Granted, or otherwise complete.
- Low-risk preferences may update their local presentation immediately and use a
  short debounce or coalesced domain write. They remain marked dirty or unsaved
  until authoritative confirmation, and failure never silently changes that
  status. Architecture owns exact intervals.
- Each dirty domain tracks its base and intended revision, mutation identities,
  first/last dirty time, retry state, and last confirmed version. Coalescing must
  preserve the player's latest explicit intent and cannot absorb economic or
  other independently auditable transactions.
- Periodic checkpoints provide defense in depth for eligible dirty state. They do
  not replace immediate transaction commits or become a reason to delay critical
  player value.
- Encounter entry durably establishes any required consumable reservation and
  immutable locked references before the attempt can rely on them. Encounter end
  freezes and submits the final result transaction under PD-05.
- Leave, teleport, server shutdown, and orderly handoff trigger a bounded flush of
  eligible dirty state and record the last confirmed version. A deadline expiring
  does not create false success; uncertain critical work remains recoverable and
  visible as Pending or Save Unsafe.
- Under storage budget or backpressure, processing priority is receipts and
  Commerce recovery, economic/value transactions, consumable commitments,
  Progression and Items, gameplay configuration, then low-risk preferences.
  Priority changes timing, never correctness or atomicity.
- Dirty state is retried with bounded backoff and coalescing where semantically
  safe. It is never discarded, truncated, or overwritten with an older confirmed
  snapshot merely to satisfy a write budget.

#### PD-08 - Load/save failure, read-only or Save Unsafe play, and player recovery

- **Status:** Resolved 2026-09-02.
- Initial-load failure remains Blocked under PD-03; the player never enters normal
  play using provisional defaults.
- A mid-session persistence failure or loss of mutation authority enters an
  explicit `Save Unsafe` or read-only state. The client receives the last
  confirmed versions, affected domains, pending operation identities, safe
  actions, and recovery status without receiving private record internals.
- An already active encounter may continue from its immutable locked snapshot
  when the live game remains authoritative and continuation creates no new
  unrecoverable commitment. Existing consumable reservations remain tied to their
  original identities, and the frozen final outcome remains Pending until commit
  or recovery establishes its authoritative result.
- The player cannot Ready, join a new queue or encounter, purchase, claim, alter
  balances or inventory, apply loadouts/builds/presets, or initiate another
  durable gameplay action while mutation safety is unavailable.
- Accessibility, input, caption, audio, and presentation changes needed to keep
  using the current device may remain temporary local drafts. They are clearly
  unsaved and may be resubmitted once a valid authoritative snapshot returns.
- Automatic retry uses bounded backoff, stable operation identities, and refreshed
  lease/version checks. Recovery replays established operations; it never asks a
  player action to cause another roll, spend, grant, or transaction submission.
- The primary player surface states what is safe, what remains Pending, and
  whether recovery has succeeded. It offers Retry, view last confirmed state, or
  Leave as applicable. Recovered state refreshes the authoritative snapshot
  before durable interaction resumes.
- If retry exceeds its operational threshold, Player Data preserves correlation
  identities and routes the issue to the defined support/runbook path. The UI
  never promises background recovery unless the durable recovery mechanism has
  actually accepted ownership.

#### PD-09 - Schema/catalog migration, corruption, backup, rollback, and repair

- **Status:** Resolved 2026-09-02.
- The record envelope and every domain section have explicit versions. A migration
  registry defines the ordered path between supported versions, the owning
  domain, prerequisites, invariant checks, produced evidence, and compatibility
  window.
- Each migration is deterministic and idempotent. It validates structure and
  cross-domain invariants before and after transformation, supports dry-run or
  sampled validation before broad rollout, and emits privacy-safe outcome facts.
- Before a destructive or lossy transformation, preserve a recoverable snapshot
  with source version, revision, lineage, and integrity evidence. Backup retention
  and encryption are architecture/policy choices but must cover the approved
  rollback and incident window.
- Structurally corrupt or invariant-breaking data is quarantined without being
  overwritten by defaults or a partially migrated record. It enters the Blocked
  recovery path with diagnostic correlation safe for operations and support.
- Recovery may restore a verified last-known-good snapshot only when its lineage
  and transaction boundary are known, then replay committed journal facts and
  apply a forward repair. It must preserve later owned value or explicitly
  reconcile it; blind record downgrade is prohibited.
- A deployed bad migration is handled by stopping rollout and using the verified
  snapshot, journal, and forward repair or explicit compensation. Already
  committed history is not rewritten merely to match older code.
- Retired catalog identities remain stable references whenever possible. A
  formally versioned equivalence may replace them only when the owning domain has
  proved identical player meaning/value. Otherwise the record exposes Missing,
  asks for an explicit replacement choice, or grants defined compensation.
- Migration never changes an active encounter's locked snapshot and never
  invents a player selection. Player-facing disclosure is required when a choice,
  entitlement, value, or usable configuration changes; invisible structural
  maintenance need not interrupt the player.
- Restore, corruption, replay, and migration rollback/forward-repair paths require
  recurring operational drills rather than relying on untested backups.

### Checkpoint D - Commerce, privacy, lifecycle, and completeness

#### PD-10 - Commerce receipts, purchase lineage, restoration, and recovery

- **Status:** Resolved 2026-09-02.
- A client request or price quote may begin Commerce UI, but only a verified
  platform receipt authorizes fulfillment. The client cannot assert payment,
  receipt success, product mapping, eligibility, price, grant, or restoration.
- The durable receipt record includes privacy-classified platform receipt
  identity, player/account scope, mapped product and exact catalog revision,
  verification result, processing status, request/transaction identities, and
  exact grant or reversal lineage. It never stores payment credentials.
- Fulfillment uses one frozen PD-05 transaction. Receipt state, item or
  entitlement grant, balance effects, purchase history, and any required
  Progression facts commit atomically before the platform receipt is acknowledged
  as fulfilled.
- Processing states distinguish `Pending`, `Granted`, `Already Processed`,
  `Canceled`, and `Recovery Required`. A timeout or ambiguous platform/storage
  response never becomes success or denial by guesswork.
- Duplicate, delayed, concurrent, or replayed receipt delivery returns the
  established result for the same receipt identity and cannot produce another
  grant. Receipt idempotency survives server/session changes and normal history
  compaction for the full platform redelivery/recovery lifetime.
- Restoration reuses the original receipt, product, transaction, and exact grant
  lineage to re-establish missing presentation/access without granting duplicate
  value. An unexplained lineage mismatch enters Recovery Required.
- A platform-confirmed refund, revocation, or reversal follows the product's
  approved policy through an immutable linked compensation/reversal entry. It
  does not erase purchase history or silently remove unrelated player value.
- Quotes and eligibility notices retain only what their active lifetime requires
  and never substitute for receipt verification. Support receives correlation
  and safe lineage, while platform refund/account actions remain platform-owned.

#### PD-11 - Privacy, minimization, access, retention, export, and deletion

- **Status:** Resolved 2026-09-02.
- Every stored field and emitted projection is allowlisted by purpose, semantic
  owner, privacy class, permitted consumers, retention class, and deletion/export
  treatment. New fields are private and unavailable by default until reviewed.
- The player client receives only the safe projection required for current UI and
  action preconditions. Other players receive only approved public summaries such
  as role, Ready, appearance, availability, and necessary group/survival state.
- Each game service receives only its owned facts and required dependencies.
  Support access is role-limited, purpose-bound, and audited. Support correction
  uses an authorized immutable mutation or compensation path, not direct hidden
  record editing.
- Analytics receives privacy-reviewed semantic events or aggregates, never whole
  record snapshots, payment data, raw audio/microphone content, secrets, private
  settings, or unnecessary detailed history.
- Permanent owned value, receipts, transaction/idempotency evidence, and migration
  lineage remain while required for account correctness, recovery, duplicate
  prevention, refunds, security, or explicit policy. Quotes, transient retries,
  operational detail, and low-value histories expire or compact once they cannot
  affect correctness or an approved player feature.
- Compaction preserves authoritative totals, lineage, unresolved operations, and
  required duplicate-prevention identities. It cannot turn an audit/recovery need
  into an unverifiable summary.
- An account export provides the player-readable data associated with the account
  plus understandable transaction/purchase history, subject to approved security
  and third-party privacy exclusions. Export generation is authenticated,
  auditable, and does not expose internal secrets or other players' data.
- A verified deletion request revokes active session authority and prevents new
  durable actions while deletion proceeds. Data is deleted or irreversibly
  anonymized according to platform/policy requirements; only minimal protected
  tombstones or transaction evidence required for duplicate prevention, refunds,
  security, or legal hold may remain under explicit access and retention rules.
- Derived data and indexes are removed or anonymized with the source. Backups age
  out under the same approved lifecycle and cannot silently restore a deleted
  active profile. The player receives accurate request/status/completion or hold
  disclosure through the platform-approved surface.
- Player Data collects no unnecessary sensitive or demographic data to implement
  these systems. Encryption, key management, regional storage, exact time limits,
  and current platform/legal compliance are architecture and policy deliverables
  that must satisfy this semantic boundary.

#### PD-12 - Semantic outputs, observability, disaster tests, and completion

- **Status:** Resolved 2026-09-02.
- Player Data publishes identified, versioned semantic outputs for load lifecycle,
  Ready/projection refresh, save/dirty/confirmed state, lease acquisition/loss/
  takeover, mutation results, transaction/recovery results, migration/repair,
  receipt/restoration, export, deletion, and privacy-safe support correlation.
- Consumers receive stable operation identity, owning domain, authoritative state,
  relevant overall/domain versions, safe reason taxonomy, allowed next actions,
  and causal correlation. They do not infer success from timing, disappearance of
  a spinner, a local animation, or a generic network response.
- The error taxonomy distinguishes unavailable/retryable, authority or lease
  conflict, stale version, validation rejection, invariant/corruption, capacity,
  migration/compatibility, transaction recovery, receipt recovery, privacy/access,
  export/deletion, and terminal unsupported cases.
- Privacy-safe metrics and alerts cover load and save health, latency/budget
  pressure, lease conflicts/split-brain rejection, stale writes, dirty age,
  Pending transaction age, receipt replay/recovery, migration failure, quarantined
  corruption, restore/repair outcomes, and export/deletion job status.
- Logs and traces use approved correlation/idempotency identities and safe reason
  codes. They never contain complete profile dumps, credentials, raw receipts when
  a protected representation suffices, microphone/audio data, or unnecessary
  private player history.
- Operational dashboards and runbooks cover stuck initial load, mid-session Save
  Unsafe, lease loss or overlap, stale writes, stuck/partial transactions,
  receipt uncertainty, migration rollout/rollback, corruption quarantine,
  last-known-good restoration/forward repair, capacity pressure, and privacy
  lifecycle failure. Exact numerical SLOs and budgets remain architecture work.
- Required automated and failure-injection tests include duplicate and out-of-
  order request/receipt delivery; concurrent servers/devices; reconnect/takeover;
  lease loss; crash or timeout at every transaction boundary; storage outage and
  shutdown; schema/catalog migration; corruption/quarantine/restore/replay;
  capacity overflow; export/deletion/backup aging; and privacy projection tests.
- A cross-domain invariant matrix must prove all promises made by Progression,
  Items, Builds, Rewards/Economy/Commerce, Multiplayer, UI/UX, Audio settings,
  and encounter/result recovery. The test suite verifies no loss, duplication,
  negative balance, partial success, stale overwrite, private exposure, or
  invented player choice.
- Analytics receives only reviewed health/semantic outcomes. Operations and
  implementation may choose storage services, keys, serialization, retry timing,
  leases, and observability products, but may not weaken these player-visible and
  integrity guarantees.

## 4. Cross-spec reconciliation register

- PD-01 confirms that all durable source facts already promised by Progression,
  Items, Builds, Rewards/Economy/Commerce, and UI/UX belong in one logical player
  truth, while Multiplayer and encounter systems retain live state.
- PD-02 requires every referencing system to preserve stable identity and the
  exact relevant catalog/content/balance revision instead of saving derived
  display or compatibility results as competing truth.
- PD-03 requires UI/UX to present Loading, Migrating, Validating, Retrying,
  Blocked, and Ready honestly, and forbids every normal durable surface from
  operating on provisional defaults.
- PD-04 preserves every domain's semantic authority while giving Player Data one
  shared request/result contract. Domain specifications must emit exact plans,
  not ask persistence code to infer gameplay meaning.
- PD-05 confirms that Results, Rewards/Economy, Items, Progression, Commerce, and
  consumable recovery must share one frozen transaction identity and cannot
  advertise partial state as complete.
- PD-06 requires Multiplayer rejoin and multi-device behavior to preserve one
  writer. UI/UX must surface conflict, revocation, refresh, and Save Unsafe states
  without presenting a rejected local mutation as saved.
- PD-07 requires every domain surface to distinguish local intent from confirmed
  durable state. Encounter entry and Results must use durable reservations and
  frozen result transactions rather than lifecycle-save assumptions.
- PD-08 lets a safe active attempt continue from its locked snapshot while
  blocking every new durable action. Multiplayer, UI/UX, Rewards, and Commerce
  must share the same authority and Pending/Save Unsafe facts.
- PD-09 requires every catalog-owning specification to define stable retirement,
  exact equivalence, explicit replacement, or compensation behavior. No domain
  may silently reinterpret stored references during load.
- PD-10 closes the Rewards/Economy Commerce handoff: verified receipts, exact
  grants, purchase history, restoration, and reversal share one durable lineage
  and idempotency identity.
- PD-11 makes every public/private projection, support path, Analytics event,
  retention rule, export, and deletion behavior explicit across the other specs.
- PD-12 requires architecture and implementation plans to carry semantic outputs,
  privacy-safe evidence, runbooks, and the complete cross-domain failure matrix.
- The completed reconciliation verified these decisions against Progression,
  Items, Builds, Rewards/Economy/Commerce, Multiplayer, Onboarding, UI/Settings,
  Audio profiles, Results, and the final system-wide audit.
- Player Data must not absorb gameplay/domain semantics merely because a fact is
  stored, and no domain may keep a competing private durable source of truth.

## 5. Confirmed architecture handoffs

- Domain owners calculate/validate exact semantic mutation plans and consume
  committed facts. Player Data validates generic envelope/concurrency/integrity
  and commits the approved plan once.
- Multiplayer/live encounter services own ephemeral party/queue/roster/grace/
  combat state. Durable consumable commitments, locked references, and final
  transactions remain recoverable without persisting a second encounter model.
- UI owns presentation; Player Data emits exact Loading/Ready/Pending/Saved/
  Unsafe/Read-Only/Failed/Recovered states and permitted player actions.
- Analytics receives only privacy-reviewed semantic outcomes/health, never raw
  record snapshots, secrets, payment data, or unneeded private history.

## 6. Change log

- **2026-09-02:** Resolved PD-01 through PD-03. Progress is 3 of 12 questions.
- **2026-09-02:** Resolved PD-04 through PD-06. Progress is 6 of 12 questions.
- **2026-09-02:** Resolved PD-07 through PD-09. Progress is 9 of 12 questions.
- **2026-09-02:** Resolved PD-10 through PD-12. The Player Data interview is
  complete at 12 of 12 questions and ready for canonical reconciliation.
- **2026-09-02:** Published canonical `PLAYER_DATA.md` and completed its
  cross-specification reconciliation and audit.
- **2026-09-01:** Created the working record. Progress is 0 of 12 questions.
