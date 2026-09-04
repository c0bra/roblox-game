# Bands Battle Player Data

- **Status:** Approved
- **Approved:** 2026-09-02
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#82-player-data)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Progression dependency:** [`PROGRESSION.md`](PROGRESSION.md)
- **Items dependency:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)
- **Rewards/Commerce dependency:** [`REWARDS_AND_ECONOMY.md`](REWARDS_AND_ECONOMY.md)
- **Builds dependency:** [`BUILDS_AND_SPECIALIZATION.md`](BUILDS_AND_SPECIALIZATION.md)
- **UI/settings dependency:** [`UI_UX.md`](UI_UX.md)
- **Multiplayer dependency:** [`MULTIPLAYER.md`](MULTIPLAYER.md)
- **Decision source:** [`PLAYER_DATA_WORKING.md`](PLAYER_DATA_WORKING.md)
- **Interview plan:** [`PLAYER_DATA_QUESTIONS.md`](PLAYER_DATA_QUESTIONS.md)

## 1. Role and authority

Player Data preserves one durable, coherent history of player-owned facts and
commits approved cross-domain mutations without taking ownership of what those
facts mean. It owns:

- the logical player-record envelope, versions, source-fact rules, and defaults;
- authoritative first creation, loading, migration, validation, and Ready state;
- privacy-filtered projections of durable state;
- the common mutation envelope and generic integrity checks;
- atomic, idempotent, ordered cross-domain commit and recovery;
- exclusive mutation authority and stale-write protection;
- save cadence, dirty-state tracking, lifecycle flush, and Save Unsafe behavior;
- corruption quarantine, verified restoration, and forward repair;
- durable receipt/grant lineage and account-data lifecycle guarantees; and
- semantic outputs, operational evidence, and disaster-verification boundaries.

Each domain owns the meaning of its facts, legal actions, validation,
calculation, and exact semantic mutation plan. Player Data applies an approved
plan once; it does not infer Progression, Item, Build, Reward, Commerce,
Multiplayer, gameplay, or setting meaning from stored fields.

Player Data does not own active encounter or reconnect state, presentation,
Analytics interpretation, platform payment decisions, storage-service selection,
serialization, security implementation, or technical topology. Those consumers
and architecture choices must honor this document's integrity, privacy, and
player-visible behavior.

## 2. Governing invariants

1. **One logical truth:** every durable fact has one semantic owner and one
   authoritative logical location; no domain keeps a competing private profile.
2. **Source facts over cached conclusions:** store owned evidence and stable
   references, then recompute derived stats, compatibility, availability, and
   display summaries unless a frozen historical value is itself required.
3. **Absence is not failure:** only authoritative Not Found permits first
   creation; timeout, corruption, partial load, and unknown version never do.
4. **Ready means complete and valid:** normal durable play starts only after all
   required sections load, migrate, and pass current invariants.
5. **Clients request actions, never writes:** a client cannot submit balances,
   grants, results, revisions, arbitrary patches, or replacement records.
6. **Domains authorize meaning:** the owning domain freezes the exact legal
   mutation; Player Data validates generic authority/integrity and commits it.
7. **Related value changes are atomic:** one outcome cannot expose only part of
   its currency, item, Progression, entitlement, refund, or history effects.
8. **Retry is not repetition:** the same operation identity returns its
   established result and never rerolls, respends, regrants, or compensates twice.
9. **One writer:** exactly one valid session epoch may mutate a player's record.
10. **No last-write-wins:** stale, delayed, conflicted, or old-epoch writes reject
    or refresh; unknown changes are never silently merged.
11. **No false success:** uncertainty is Pending, Recovery Required, unsaved, or
    Save Unsafe, never a locally invented success state.
12. **Confirmed value is not silently lost:** balances never become negative and
    owned items, unlocks, entitlements, receipts, or correction history are not
    dropped, clamped, substituted, or rewritten.
13. **Explicit choice survives:** migration and repair never invent a replacement
    loadout, build, preset, item, or accessibility preference for the player.
14. **Active attempts are immutable:** catalog/schema work never mutates an
    encounter's locked snapshot or creates a second permanent encounter model.
15. **Verified receipts only:** client or quote state never authorizes a paid
    grant, and fulfillment is not acknowledged before durable atomic commit.
16. **Minimum exposure:** each client, service, support role, and Analytics
    consumer receives only its approved privacy-filtered projection.
17. **Recoverability is tested:** backup, replay, migration, transaction, receipt,
    corruption, and deletion behavior requires executable evidence and drills.
18. **Architecture may strengthen, not weaken:** technical choices may improve
    durability or security but cannot relax these semantics.

## 3. Logical record and durable inventory

The player profile is one logical snapshot divided into independently versioned,
semantically owned sections. It need not be one physical storage object. Any
sharding must preserve coherent load, validation, commit, recovery, export, and
deletion behavior.

The logical record includes the following durable categories.

### Record envelope

- stable player and logical-record identities;
- overall schema version and monotonic commit sequence;
- section presence and domain revisions;
- last confirmed commit lineage and integrity evidence; and
- the authority/version facts needed to reject stale or old-session writes.

### Progression

- campaign nodes, fragments, first-clear and difficulty-access facts;
- General Progression, unlocks, mastery, milestones, and non-expiring advancement;
- personal-best source facts and compatible historical categories;
- hub restoration and practice/public-access state; and
- migration, uplift, recommendation-source, or grandfathering evidence required
  to preserve earned value.

Progression owns these semantics. Player Data stores its exact mutation results.

### Items, configuration, and appearance

- unique item instances, stable lineage, tier, rank, and owned traits;
- consumable stack quantities and durable reservations, spends, and refunds;
- cosmetic entitlements and selected appearance;
- current loadout and its validation or missing-reference state;
- exactly three complete spec presets, including their item, role, ability, and
  build references;
- build drafts and applied configurations with option/catalog/balance revisions;
  and
- locked references and migration/replacement evidence required for recovery.

Builds do not create a second preset store. Applied builds and full spec presets
remain part of the same durable configuration truth.

### Rewards, economy, and Commerce

- general and boss-resource balances;
- pinned goals, frozen rolls, guarantee state, and deterministic-path evidence;
- immutable earned-transaction journal and causal pre/post state;
- Pending, committed, rollback, refund, and compensation outcomes;
- receipt status, exact purchase/grant lineage, restoration, duplicate, and
  recovery state; and
- notice dismissal or eligibility history only when required by the approved
  Commerce experience.

Quotes persist only for their useful lifetime. Payment credentials never enter
the profile.

### Settings, onboarding, and player intent

- account-wide language, accessibility, and teaching intent;
- explicit per-device, control, and output profiles and overrides;
- calibration, binding, touch-layout, presentation, audio, caption, motion,
  effects, and haptic preferences;
- onboarding checkpoints, completion, deliberate skip, and replay state; and
- contextual-prompt and notification/dismissal history needed to respect the
  player's choices.

### Bounded outcome history

Frozen result facts remain durable only while required to finish a transaction,
support an approved personal record, or provide a bounded private improvement
history. Raw attempt input, judgment streams, or Analytics events do not become
permanent profile history.

## 4. Ephemeral boundary

The permanent profile does not store a competing copy of:

- current Ward, Hype, position, movement, boss, Resolve, Momentum, or rhythm
  playback/judgment state;
- party, queue, roster, Ready, grace, AFK, ping, or live reconnect state;
- active audio layers, cue instances, transient captions, or haptic requests;
- raw microphone/audio data, payment credentials, or raw Analytics streams; or
- other players' private gameplay, profile, settings, or transaction data.

Live systems own those facts. Only commitments required to preserve durable
value cross the boundary: consumable reservations/spends/refunds, immutable
locked references, final frozen outcomes, and transaction/recovery identities.

## 5. Identity, revisions, source facts, and limits

Each domain section declares its schema version, domain revision, last commit
identity/time, owned source facts, and relevant privacy/retention class. The
overall record carries a monotonic commit sequence across domain mutations.

Saved references use stable identities plus the exact relevant catalog, content,
option, or balance revision. Every mutation preserves its approved cause and
lineage. A value copied only for display convenience does not become authority.

Derived stats, compatibility, availability, recommendations, validation
summaries, and presentation text are recomputed from current source facts and
catalogs. A frozen historical calculation may be retained when required to prove
a result, transaction, purchase, personal best, or compensation.

Optional absence is explicit. Required fields are not silently fabricated.
Unknown fields or versions are migrated or rejected, not discarded. Capacity,
serialization, or section overflow fails closed and enters recovery; owned value
and required history are never silently truncated to make a record fit.

## 6. First creation and authoritative loading

The load lifecycle is:

`Loading -> Migrating -> Validating -> Ready`

`Retrying` and `Blocked` are explicit non-Ready branches.

Before accepting mutations, the server acquires exclusive session authority and
loads every required logical section. It then runs ordered schema migrations,
current-catalog compatibility checks, and structural/cross-domain invariant
validation.

Only an authoritative Not Found result creates a new record. First creation
builds one deterministic, complete starter transaction containing valid starter
Progression, balances, inventory, loadout, three spec presets, settings intent,
and onboarding state. The whole starter snapshot is validated and committed
together.

A timeout, unavailable service, partial response, missing required section,
corruption, unknown schema, capacity failure, or conflicting authority does not
create defaults and does not enter normal play. The player receives Retry or
Leave and accurate status. Retrying preserves the original load intent.

A safe retired catalog reference may produce a specific Incomplete or
replacement-required fact inside an otherwise valid snapshot. Structural
corruption, unknown schema, or unresolved cross-domain invariants block Ready.

## 7. Ready projection and client boundary

After validation, Player Data emits a privacy-filtered Ready snapshot containing
only what the current player surface and permitted consumers require, including
the relevant overall and domain revisions. A projection is not an editable copy
of the stored record.

Before Ready, the load screen may expose the minimum local accessibility/device
controls required to use that screen. Those choices are clearly temporary and
unsaved until a valid authoritative profile accepts them.

Normal hub, queue, staging, encounter, inventory, build, progression, reward,
and store actions require Ready. A client cannot replace a section, claim a
revision, or infer saved state from a local animation or response timeout.

## 8. Semantic mutation contract

Clients submit meaningful intents such as equip an item, apply a build, save a
preference, claim an eligible action, or begin an approved purchase flow. The
owning domain authorizes the action and produces an immutable exact mutation
plan.

Every request and plan includes:

- unique request/transaction and player identities;
- actor/service, owning domain, and semantic action;
- causal source event and authoritative ordering context;
- expected overall and affected-domain revisions;
- exact relevant content, catalog, option, and balance revisions;
- declared preconditions and immutable ordered operations;
- privacy class, creation time, and idempotency scope; and
- the safe result/reason contract.

Immediately before commit, Player Data rechecks session epoch, expected
versions, allowed domain ownership, referenced identities, plan integrity, and
generic invariants. It does not recalculate the domain result.

The authoritative result is one of:

- `Pending`;
- `Committed`;
- `Rejected`;
- `Conflict`; or
- `Recovery Required`.

The result carries the original operation identity, stable safe reason, and
known pre/post versions. A duplicate receives the recorded result.

## 9. Atomic cross-domain transactions

Before an encounter or action outcome enters persistence, its authoritative
result, eligibility, calculations, random rolls, guarantees, item identities,
catalog revisions, and allowed refunds or compensation are frozen. Retry never
recalculates or rerolls the outcome.

One transaction plan includes every related:

- append-only journal entry;
- balance delta;
- item instance, entitlement, or stack change;
- Progression fact, unlock, milestone, or personal record;
- consumable reservation, spend, or refund;
- purchase/reward lineage and exact grant; and
- required downstream committed handoff.

All affected domains validate against the same input versions. Player-visible
state changes together. No consumer may announce a partial transaction as
complete.

If the implementation cannot provide one physical storage transaction, it must
use a recoverable journal/state machine that exposes Pending until every
component reaches the exact frozen commit or defined rollback state. Results may
show Pending and allow another gameplay Retry without waiting.

Rollback reverses only uncommitted reversible work. After a fact is durably
committed or externally acknowledged, correction uses an immutable causally
linked compensation entry. No Contest refunds are limited to consumables
causally reserved or spent for that attempt.

## 10. Idempotency, ordering, and exclusive authority

Exactly one session lease and epoch may mutate a player's logical record. Lease
loss, expiry, or revocation stops new mutation acceptance immediately.

Every mutation carries its stable idempotency identity, expected overall/domain
revisions, owning session epoch, and serialized monotonic commit position.
Duplicate or retried delivery replays the established result without repeating
any spend, grant, unlock, roll, refund, compensation, purchase, or side effect.

Irreversible economic, receipt, entitlement, and compensation identities remain
available for their full recovery and duplicate-delivery lifetime. A bounded
identity lifetime is permissible for reversible low-risk preferences only when
an old retry cannot cause harmful duplicate behavior.

Stale or out-of-order requests return Conflict or refresh guidance and never
overwrite newer state. Only an owning domain may define a narrow, safe,
commutative merge, such as an independent preference change.

A new writer gains authority only after the former epoch is released, expires,
or is validly revoked. Later writes from the old epoch remain rejected.
Reconnecting during an active attempt uses Multiplayer's existing rejoin path.
Outside an attempt, an explicit takeover may revoke and notify the older session
before the new session becomes authoritative.

Queued work rechecks version and epoch at execution. Split-brain ambiguity never
uses last-write-wins or unknown-state merging.

## 11. Save cadence, dirty state, and lifecycle flush

Purchases, receipts, currencies, Rewards, Progression, Items, entitlements,
consumable commitments, applied builds, loadouts, spec presets, and other
gameplay-affecting choices require durable commit before their surface reports
Saved, Applied, Granted, or complete.

Low-risk preferences may update locally and use a short debounce or coalesced
write. They remain dirty/unsaved until authoritative confirmation. Each dirty
domain retains its base and intended revision, operation identities, dirty age,
retry state, and last confirmed version. Coalescing preserves the latest explicit
intent and never absorbs independently auditable transactions.

Periodic checkpoints provide defense in depth. They do not replace immediate
value transactions. Encounter entry commits required consumable reservations and
locked references before relying on them; encounter end submits the frozen final
transaction.

Leave, teleport, orderly handoff, and server shutdown trigger a bounded flush of
eligible dirty state and record the last confirmed version. An expired flush
deadline cannot manufacture success. Accepted uncertain critical work remains
recoverable and visibly Pending or Save Unsafe.

Under storage pressure, priority is:

1. receipts and Commerce recovery;
2. economic/value transactions;
3. consumable commitments;
4. Progression and Items;
5. gameplay configuration; then
6. low-risk preferences.

Priority changes timing, never correctness or atomicity. Dirty state is retried
with bounded backoff and safe coalescing and is not dropped, truncated, or
replaced by an older confirmed snapshot.

## 12. Save Unsafe, outage, and player recovery

Initial-load failure remains Blocked under section 6. A mid-session persistence
failure or loss of mutation authority enters explicit `Save Unsafe` or read-only
state and publishes the affected domains, pending operation identities, last
confirmed versions, permitted actions, and recovery status.

An active encounter may continue from its immutable locked snapshot only while
the live game remains authoritative and continuation creates no new
unrecoverable commitment. Existing consumable reservations retain their original
identities. The frozen final outcome stays Pending until commit or recovery
establishes its result.

While mutation safety is unavailable, the player cannot:

- Ready or enter another queue/encounter;
- purchase, claim, or initiate another durable transaction;
- alter balances, inventory, entitlements, or consumable commitments; or
- apply loadouts, builds, presets, or other gameplay configuration.

Necessary accessibility, input, caption, audio, and presentation changes may
remain temporary local drafts and are clearly unsaved.

Automatic retry uses bounded backoff, stable operation identities, and refreshed
lease/version checks. The player receives accurate status plus Retry, view last
confirmed state, or Leave where applicable. Recovery refreshes the authoritative
snapshot before durable interaction resumes.

If an operational threshold is exceeded, Player Data preserves safe correlation
identities and enters the support/runbook path. The UI promises background
recovery only after a durable recovery mechanism has accepted responsibility.

## 13. Schema and catalog migration, corruption, and repair

The record envelope and every domain section carry explicit versions. A migration
registry defines the ordered supported path, owning domain, prerequisites,
compatibility window, invariant checks, and produced evidence.

Every migration is deterministic and idempotent. It validates before and after,
supports dry-run or sampled rollout evidence, and emits privacy-safe outcomes.
Before a destructive or lossy transformation, the system retains a recoverable
snapshot with source version, revision, lineage, and integrity evidence.

Corrupt or invariant-breaking data is quarantined. It is not overwritten by
defaults or a partial migration. Recovery may restore a verified last-known-good
snapshot only when its lineage and transaction boundary are known, replay later
committed journal facts, and apply a forward repair. Later owned value must be
preserved or explicitly reconciled; blind downgrade is forbidden.

A bad rollout stops and uses verified snapshot/journal evidence for forward
repair or explicit compensation. Already committed history is not rewritten to
match older code.

Retired catalog identities remain stable whenever possible. A versioned
equivalence may substitute only when the owning domain proves identical player
meaning and value. Otherwise the record exposes Missing, asks for explicit
replacement, or grants defined compensation. Migration never changes an active
encounter snapshot or invents a player choice.

The player is informed when migration changes usable configuration, entitlement,
value, or requires a choice. Invisible structural maintenance need not interrupt
normal play. Restore, corruption, replay, and repair paths require recurring
operational drills.

## 14. Commerce receipts and restoration

A request or price quote may open Commerce, but only a verified platform receipt
authorizes fulfillment. The client cannot assert payment, receipt success,
product mapping, eligibility, price, grant, or restoration.

The durable receipt record contains:

- privacy-protected platform receipt identity and account scope;
- mapped product and immutable catalog revision;
- verification and processing status;
- request/transaction identities; and
- exact item, entitlement, balance, Progression, grant, or reversal lineage.

It never contains payment credentials.

Receipt status distinguishes `Pending`, `Granted`, `Already Processed`,
`Canceled`, and `Recovery Required`. Receipt state, exact grant, entitlement or
item, balance effects, and purchase history commit in one frozen transaction
before fulfillment is acknowledged to the platform.

Duplicate, delayed, concurrent, or replayed delivery returns the original result
and cannot grant again. Ambiguous platform/storage response remains Pending or
Recovery Required rather than guessed success or denial.

Restoration reuses the original receipt/product/transaction/grant lineage. It
restores missing access or presentation without duplicating value. A lineage
mismatch enters Recovery Required.

A platform-confirmed refund, revocation, or reversal follows the approved
product policy through an immutable linked compensation/reversal record. It does
not erase purchase history or remove unrelated value. Quotes and eligibility
notices expire when no longer needed and never substitute for verification.

## 15. Privacy, access, retention, and compaction

Every stored field and emitted projection is allowlisted by purpose, owner,
privacy class, consumers, retention class, and export/deletion treatment. New
fields are private and unavailable by default until reviewed.

- The owner receives only the safe projection required by current UI/actions.
- Other players receive approved summaries such as role, Ready, appearance,
  availability, and necessary group/survival state.
- A game service receives only owned facts and required dependencies.
- Support access is role-limited, purpose-bound, and audited. Corrections use an
  authorized immutable mutation/compensation path, not hidden record editing.
- Analytics receives reviewed semantic events or aggregates, never full records.

Complete profiles, payment data, raw audio/microphone content, secrets, private
settings, unnecessary detailed history, and other players' information do not
enter logs, Analytics, or unrelated projections.

Permanent owned value, receipts, transaction/idempotency evidence, and migration
lineage remain while required for account correctness, recovery, duplicate
prevention, refunds, security, or explicit policy. Quotes, transient retry data,
operational detail, and low-value history expire or compact when they can no
longer affect correctness or an approved feature.

Compaction preserves authoritative totals, unresolved operations, lineage, and
required duplicate identities. It cannot replace recoverable evidence with an
unverifiable summary.

Player Data collects no unnecessary sensitive or demographic data. Encryption,
key management, regional storage, exact retention times, and current platform or
legal compliance remain architecture/policy deliverables that must satisfy these
minimum boundaries.

## 16. Account export and deletion

An authenticated account export provides understandable player-readable data
associated with the account, including appropriate transaction and purchase
history. It excludes internal secrets, protected security mechanisms, and other
players' data. Export generation is purpose-bound and audited.

A verified deletion request revokes active session authority and blocks new
durable actions while deletion proceeds. Data and derived indexes are deleted or
irreversibly anonymized under approved platform/policy requirements.

Only minimal protected tombstones or transaction evidence genuinely required
for duplicate prevention, refunds, security, or legal hold may remain, with
explicit access and retention rules. Backups age out under the same lifecycle and
cannot silently recreate an active deleted profile.

The player receives accurate request, processing, completion, or hold disclosure
through the platform-approved surface.

## 17. Semantic outputs and consumers

Player Data publishes identified, versioned outputs for:

- Loading, Migrating, Validating, Retrying, Blocked, and Ready;
- profile projection refresh and overall/domain revision;
- dirty, saving, confirmed, unsaved, read-only, and Save Unsafe state;
- lease acquisition, loss, revocation, conflict, and takeover;
- mutation, transaction, Pending, commit, rejection, rollback, compensation, and
  recovery;
- migration, compatibility, quarantine, restoration, and repair;
- receipt, grant, Already Processed, cancellation, restoration, and reversal;
  and
- export, deletion, hold, completion, and safe support correlation.

Each output carries its stable operation identity, owner, authoritative state,
relevant versions, privacy-safe reason, allowed next actions, and causal
correlation. Consumers do not infer success from elapsed time, spinner dismissal,
local animation, or a generic transport response.

The error taxonomy distinguishes unavailable/retryable, lease/authority conflict,
stale version, validation rejection, invariant/corruption, capacity, migration/
compatibility, transaction recovery, receipt recovery, privacy/access, export/
deletion, and terminal unsupported cases.

Progression, Items, Builds, Rewards/Economy/Commerce, Multiplayer, Onboarding,
Input/Settings, Results, UI, Audio Presentation, support, operations, and approved
Analytics consume only their required semantic facts without taking ownership.

## 18. Observability, runbooks, and verification

Privacy-safe health metrics and alerts cover:

- load/save latency, failure, and budget pressure;
- lease conflict, split-brain rejection, and stale writes;
- dirty-state and Pending-transaction age;
- receipt replay, restoration, and recovery;
- migration failure, corruption quarantine, and repair outcomes; and
- export/deletion job status and backup lifecycle.

Logs and traces use protected correlation/idempotency identities and safe reason
codes. They do not contain complete profile dumps, credentials, raw receipts when
a protected representation suffices, raw audio/microphone data, or unnecessary
private history.

Runbooks cover stuck initial load, mid-session Save Unsafe, lease overlap/loss,
stale writes, stuck or partial transactions, uncertain receipts, bad migration
rollout, corruption quarantine, verified restoration/replay/forward repair,
capacity pressure, and account-lifecycle failure.

Required automated, failure-injection, and operational-drill coverage includes:

- duplicate and out-of-order request/receipt delivery;
- concurrent servers/devices, reconnect, takeover, and old-epoch writes;
- crash or timeout at every transaction boundary;
- storage outage, shutdown, backpressure, and capacity overflow;
- schema/catalog migration, incompatibility, and failed rollout;
- corruption, quarantine, last-known-good restore, journal replay, and repair;
- receipt replay, grant restoration, reversal, and lineage mismatch;
- privacy projections, support access, export, deletion, and backup aging; and
- every cross-domain invariant in the canonical system specifications.

The cross-domain matrix must prove no loss, duplication, negative balance,
partial success, stale overwrite, private exposure, false acknowledgment, or
invented player choice.

## 19. Architecture and policy handoff

Technical architecture must define the storage services, key/topology model,
physical transactions or recovery journal, lease mechanism, serialization,
capacity limits, retry/backoff timing, shutdown deadlines, encryption and key
management, authenticated service access, backup/restore system, observability
products, and platform receipt integration.

Platform/policy work must define current privacy, retention, regional, export,
deletion, security, refund/reversal, legal-hold, and support-access requirements.

Balance/content/catalog owners must define their schemas, versions, migrations,
equivalence or replacement rules, and invariant validators. UI/UX must map every
semantic state to an honest accessible surface. None of these handoffs may weaken
the approved durable and player-visible contract.

## 20. Deferred technical and operational work

Behavior is complete. The following remain intentionally downstream:

- exact physical record/shard/key and indexing layout;
- data serialization, compression, capacity thresholds, and storage budgets;
- lease duration, heartbeat, takeover, retry, batching, checkpoint, and flush
  timing;
- concrete atomic transaction or recovery-journal implementation;
- backup frequency/retention and recovery-point/recovery-time targets;
- encryption, secrets, authentication, authorization, and regional controls;
- exact privacy/receipt/export/deletion/refund platform integrations;
- numerical SLOs, alert thresholds, dashboards, and on-call procedures; and
- full automated/chaos test implementation and ongoing restore drills.

These choices may not permit default-on-failure, partial visible success,
duplicate value, stale overwrite, parallel writers, silent data loss, invented
player choice, unverified receipt grants, or excessive private exposure.

## 21. Approval and change control

The owner interview resolved PD-01 through PD-12 on 2026-09-02. This document is
the canonical Player Data design specification.

A material change to the durable/ephemeral boundary, source-fact model, first
creation, Ready requirements, mutation/transaction semantics, session authority,
save cadence, Save Unsafe behavior, migration/repair, receipt fulfillment,
privacy/retention, export/deletion, or operational verification requires an
explicit amendment citing the superseded rule. Technical implementation choices
inside these boundaries do not require redesign but must remain versioned,
reviewed, and recoverable.
