# Bands Battle Technical Architecture

- **Status:** Approved architecture baseline
- **Approved:** 2026-09-10
- **Working-title notice:** `Bands Battle` is an internal project title, not the
  final public game title
- **Product authority:** [`GAME_VISION.md`](GAME_VISION.md) and
  [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **System authority:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md)
- **Detailed design inputs:** [`SYSTEM_SPECIFICATIONS_TRACKER.md`](SYSTEM_SPECIFICATIONS_TRACKER.md)
- **Numeric authority:** [`BALANCE_FRAMEWORK.md`](BALANCE_FRAMEWORK.md)
- **Evidence authority:** [`PLAYTEST_AND_ANALYTICS.md`](PLAYTEST_AND_ANALYTICS.md)
- **Language authority:** [`NAMING_AND_TONE.md`](NAMING_AND_TONE.md)
- **Visual authority:** [`ART_DIRECTION.md`](ART_DIRECTION.md)
- **Decision source:** [`TECHNICAL_ARCHITECTURE_WORKING.md`](TECHNICAL_ARCHITECTURE_WORKING.md)
- **Interview plan:** [`TECHNICAL_ARCHITECTURE_QUESTIONS.md`](TECHNICAL_ARCHITECTURE_QUESTIONS.md)

## 1. Role and authority

This document translates the approved game and system specifications into the
technical boundaries used to build, test, publish, and operate the Roblox game.
It owns runtime authority, component boundaries, concrete data and message
families, deployment topology, persistence mechanisms, source ownership,
verification layers, and the release process.

It does not redesign player-facing behavior, choose final balance values, approve
song or encounter content, select the public game title, or prove that an
unimplemented mechanism works. When a system specification and this document
appear to conflict, the system specification owns what the game must do and this
document owns how software realizes it. A real platform constraint that makes an
approved behavior infeasible must be raised as a design amendment rather than
silently changing the behavior.

The question plan and working record preserve interview history. This canonical
document supersedes their proposals as architecture authority.

## 2. Governing architecture principles

1. **Content before dependent gameplay.** Produce a complete real-song runtime
   package and prove its clock, layers, chart, and export in Roblox before building
   the wider combat stack.
2. **One semantic owner.** A gameplay fact has one authoritative domain. Other
   components consume typed facts and never recreate authority from UI, audio,
   animation, or totals.
3. **Server authority with immediate local response.** Clients collect input,
   run local presentation, and show immediate rhythm feedback. Servers validate
   and resolve every shared gameplay or durable consequence.
4. **Immutable attempt inputs.** An active attempt pins content, balance,
   catalogs, roster, Loadouts, and initial scaling. An update cannot rewrite it.
5. **Deterministic logical execution.** Authored and gameplay time uses integer
   microseconds. Frozen inputs, ordered rules, stable identities, and a server
   seed reproduce the same logical outcome.
6. **The song remains intact.** Calibration, recovery, performance settings,
   difficulty, or networking never changes source tempo, pitch, duration, or
   chart truth.
7. **Failure is an explicit state.** Pending, Conflict, Save Unsafe, Defeat, and
   No Contest remain distinct. A timeout or missing response never becomes
   success.
8. **Durable value commits atomically per player.** Related currency, item,
   Progression, reward, receipt, and Battle Supply changes commit from one frozen,
   idempotent transaction plan.
9. **Platform-neutral content is canonical.** Roblox exports may change
   representation but not the approved package semantics.
10. **Exact versions, never implicit latest.** Runtime packages, assets,
    dependencies, places, catalogs, and releases use reviewed immutable versions.
11. **Distinct boss levels share runtime code.** The common battle place is a
    host and loader. Every boss retains its own arena, atmosphere, rig, hazards,
    timeline, and presentation.
12. **Protected experience survives degradation.** Music, rhythm input, the
    musical clock, controls, critical cues, captions, and required alternatives
    remain intact while decorative work scales down first.
13. **Minimal operations first.** Use Roblox-managed runtime services and local
    creator tools. Add an external live backend only after a demonstrated gap.
14. **Observability cannot control gameplay.** Analytics and diagnostics consume
    facts. Failure to collect them cannot block or alter gameplay, rewards, or
    persistence.
15. **Evidence closes feasibility gates.** API documentation supports a choice;
    target-device, multi-client, published-game, and recovery tests prove it for
    this game.

## 3. End-to-end system and deployment map

```text
Creator workstation
  Song project directory
      -> existing audio processors and chart pipeline
      -> local TypeScript/Bun authoring web app
      -> reviewed platform-neutral song/encounter package
      -> Roblox exporter and asset publication
      -> immutable Roblox content revision

Roblox experience
  Hub place servers
      -> profile/session ownership
      -> parties, proposals, staging, and queue coordination
      -> MemoryStore deployment record
      -> group TeleportAsync to a reserved battle server

  Shared battle runtime place, one active band/attempt per server
      -> exact boss arena package or approved dedicated boss place
      -> exact content/balance/catalog revisions
      -> synchronized local song playback
      -> server-authoritative encounter reducers
      -> per-player confirmed result transactions
      -> Results and voluntary retry/stay/return flow

Roblox-managed services
  DataStore       durable player profiles, transactions, and receipts
  MemoryStore     expiring queue, deployment, and rejoin routing
  MessagingService best-effort wake signals only
  AnalyticsService reviewed aggregate events
  MarketplaceService server receipt boundary

Creator operations
  Git and immutable release records
  private staging experience
  Open Cloud publication, backup, export, and audited repair tools
```

There are separate development, private staging, and production Roblox
experiences. Their places, credentials, DataStores, MemoryStores, and traffic are
isolated. Shared creator-owned assets must be explicitly permissioned to every
environment that consumes them.

## 4. Technology and source ownership

### 4.1 Selected technology boundaries

| Surface | Selected direction | Reason |
|---|---|---|
| Roblox runtime | Strictly typed Luau organized through Rojo | Matches the existing project and Roblox execution model |
| Roblox toolchain | Rokit-pinned tools; introduce Wally with a committed lockfile for reviewed Luau dependencies | Gives reproducible tool and package resolution |
| Roblox UI | Pinned React Lua/ReactRoblox for stable screens and HUD composition | Supports declarative reusable interface state without putting frame-critical notes in broad rerenders |
| Frame-critical rhythm presentation | Game-owned pooled imperative renderers | Keeps moving notes and VFX independent from general UI reconciliation |
| Luau unit tests | Pinned Jest Lua in a dedicated test place/runner | Provides structured pure-rule and component tests inside Roblox |
| Local authoring application | TypeScript and Bun with a browser UI | Retains the current pipeline language and allows local directory/process access |
| Content validation | Zod at TypeScript boundaries plus generated schema/version fixtures for Luau | Makes exporter and runtime expectations comparable without pretending the languages share a type checker |
| Audio analysis | Existing useful Python and external processors | Preserves proven specialist tools behind explicit jobs |
| Roblox persistence | Standard DataStore through a game-owned `ProfileRepository`; pinned reviewed ProfileStore internally | Supplies session locking and ordered saves without delegating game semantics to a library |
| Ephemeral coordination | MemoryStore; optional MessagingService wake signals | Fits expiring queue/deployment state while preserving one authoritative record |
| Product analytics | Roblox AnalyticsService behind a game-owned adapter | Avoids a first-release external telemetry service |
| Publishing and administration | Human-supervised Studio/Rojo assembly, then scoped Open Cloud tools where proven | Preserves authored places and supports controlled automation |

Every third-party dependency records its exact version or commit, license,
source, checksum, approved use, and upgrade owner. Dependencies are installed
from a committed lockfile where supported. ProfileStore remains behind the
project adapter. React Lua and Jest Lua remain replaceable at their application
boundaries. No dependency receives authority over gameplay meanings.

### 4.2 Source-of-truth boundaries

| Source | Owns | Does not own |
|---|---|---|
| Git/Rojo source | Luau code, contracts, generated content data, configuration, localization source, tests | Hand-authored terrain and scene composition outside mapped paths |
| Studio place files | Terrain, lighting, hub and battle-runtime shells, scene-only instances | Canonical gameplay code or package semantics |
| Creator-owned Roblox assets/packages | Published boss arenas, models, audio, animations, images, reusable art | Unreviewed scripts or implicit latest versions |
| DCC source in Git | Blender files, textures, editable source, export settings | Runtime identity by filename alone |
| Authoring project directory | Editable musical, chart, timeline, review, and approval history | Live Roblox state |
| Platform-neutral runtime package | Approved content semantics and integrity | Creator-only raw suggestions or sensitive review material |
| Release record | Exact combination promoted to an environment | Mutable gameplay state |

The existing `roblox/Place1.rbxl` remains the Studio-authored scene baseline
until a deliberate migration creates named hub and battle-runtime place bases.
Rojo's current `$ignoreUnknownInstances` boundary stays in force so sync does not
erase Studio content. A clean `rojo build` proves the source-backed tree, but it
does not by itself reproduce the complete authored place.

### 4.3 Target repository layout

This is the intended ownership layout, not a requirement to move unrelated
legacy assets before their owning implementation task:

```text
tools/
  chart-pipeline/                 platform-neutral schemas and transformations
  content-authoring/              local Bun server and browser editor
  roblox-export/                  validated Roblox export and manifest creation

content/
  projects/<song-id>/             editable source projects and review history
  packages/<encounter>/<revision>/ immutable approved runtime packages

roblox/
  default.project.json
  rokit.toml
  wally.toml + wally.lock         pinned Luau runtime/dev dependencies
  places/                         approved Studio-owned place bases after migration
  releases/                       immutable release manifests
  src/shared/BandsBattle/
    Contracts/                    types, schemas, protocol registry
    Content/                      generated indexes and package readers
    Rules/                        pure deterministic domain rules
    ViewModels/                   presentation-neutral projections
  src/client/
    Bootstrap/ Input/ Rhythm/ Audio/ Presentation/ UI/
  src/server/
    Bootstrap/ Protocol/ Sessions/ Encounter/ Persistence/ Operations/
  tests/
    unit/ fixtures/ studio/ acceptance/
```

Generated files carry their generator version, source revision, and hash and are
never edited by hand. Generated outputs do not override their source package.

## 5. Content authoring, package, and Roblox export

### 5.1 Local authoring flow

The local Bun process receives an explicitly selected song project directory. It
alone reads and writes files and invokes external jobs. The browser communicates
with that process to:

1. inspect the declared master and any song-specific stem/role set;
2. run or rerun selected analysis stages with visible progress and errors;
3. play synchronized layers against the approved musical clock;
4. inspect and edit transcriptions, Easy/Normal/Hard charts, Activity Maps, and
   the encounter timeline;
5. distinguish generated suggestions from human edits;
6. identify stale dependent artifacts after reruns;
7. run structural, musical, compatibility, and publication validation; and
8. record explicit human review and approval.

A canceled or failed job keeps the last good artifact and records the failure. A
rerun never overwrites a human-approved edit without an explicit reviewed merge.
The app is an internal creator tool, not an in-game or public song editor.

### 5.2 Canonical runtime package contract

The serialized form may be split across files, but it represents one immutable
logical package with this minimum shape:

```text
SongEncounterPackage
  schemaVersion
  packageId
  songId
  encounterId
  contentRevision
  supersedesRevision?
  sourceFingerprints[]
  masterAudioRef
  fullMixFallbackRef
  roles[]
    roleId
    displayKey
    backingBedRef
    controllableLayerRefs[]
    neutralMixDefinition
    alignmentAndDuration
    responseCapabilities[]
  musicalClock
    durationUs
    tempoMeterSegments[]
    beats[]
    sections[]
    safeReturnPoints[]
  canonicalTranscriptionsByRole
  chartsByRoleAndDifficulty
    canonicalEventLineage
    events[]
    phrases[]
    passages[]
    availableWeight
  activityMapsByRole
  encounterTimeline
    songFunctions[]
    resolveOpenings[]
    bossCandidates[]
    fixedEvents[]
    globalTieOrder[]
  arena
    assetId
    assetVersion
    requiredAnchorsAndTags[]
    lightingProfile
    bossRigAndAnimations
    hazardsAndArenaGraph
    dedicatedPlaceId?
  presentation
    audioDefinitions
    cueDefinitions
    captionAndAlternativeDefinitions
    vfxAndCameraDefinitions
  dependencies
    balanceRevision
    catalogRevisions
    localizationRevision
    minimumRuntimeCapabilities[]
  validationSummary
  approvalRecord
  integrityManifest
```

Roles are stable data identities, not a four-value enum. Drums, vocals, guitar,
and bass are common starting categories; piano, synthesizer, percussion,
strings, winds, sampled parts, and role-specific composites are equally valid
when the real arrangement supports them. A package declares only its actual
roles. An instrumental song need not declare vocals.

Every timed value and boundary uses integer microseconds from the approved
master origin. Every event, phrase, passage, candidate, cue, asset, and revision
has a stable identity. Same-time events carry a stable global order. A content
revision never relies on file order, network receipt order, or an implicit
current catalog.

### 5.3 Roblox export and load contract

The exporter:

- validates the complete platform-neutral package before transforming it;
- maps legacy `medium` charts to Normal only through an explicit importer;
- converts legacy clip-relative timing to master-origin microseconds;
- produces Roblox-readable generated data and content indexes;
- publishes or binds exact creator-owned audio, model, image, and animation
  versions;
- records source-to-export hashes and an export-tool version;
- validates required capabilities and size/budget declarations; and
- rejects an export that would change semantic identity, order, duration, chart
  lineage, fallback, or approval meaning.

At runtime, the server selects one package revision from the release allowlist,
checks its schema, integrity, dependencies, capabilities, and asset permissions,
then exposes only the client projection needed for the attempt. A mismatch or
missing outcome-critical artifact fails before start. A critical failure after
start follows the No Contest path; it never substitutes different content.

## 6. Release identity and compatibility

Every environment publication produces an immutable record:

```text
ReleaseManifest
  releaseId
  environment
  gitCommit
  toolchainVersions
  dependencyLockHash
  hubPlaceId + placeVersion + baseHash
  battleRuntimePlaceId + placeVersion + baseHash
  dedicatedBossPlaces[]
  allowedContentRevisions[]
  balanceRevision
  catalogRevisions[]
  localizationRevision
  assetIdVersionPairs[]
  runtimeSchemaRange
  playerSchemaRange
  featureEnablement
  validationEvidenceRefs[]
  approver + approvedAt
  rollbackReleaseId?
```

Compatibility is declared, not inferred. Runtime, package, protocol, catalog,
and player schemas expose supported version ranges or exact versions as
appropriate. Breaking changes create new versions and explicit migration or
rejection behavior. Active attempts keep their pinned release inputs. New
servers and new attempts may adopt a newly enabled release only after approval.

## 7. Roblox runtime component layout

### 7.1 Shared modules

Shared modules contain no secrets and no authoritative mutable state.

| Component | Responsibility |
|---|---|
| `Contracts` | Branded identities, package/profile/protocol types, schema versions, enums, reason codes |
| `ProtocolRegistry` | Intent, event, and snapshot definitions plus visibility and size/rate metadata |
| `ContentReader` | Immutable package lookup and compatibility checks against generated indexes |
| `RhythmRules` | Press/hold matching, judgments, phrase/passages, normalization, suspension rules |
| `CombatRules` | Intent conversion, effects, Ward, damage, downing, Revive, same-boundary order |
| `EncounterRules` | Resolve, Momentum, candidate filtering, timeline, positioning, outcome rules |
| `AbilityRules` | Hype, Signature, Band Call, Crescendo, Solo Support semantics |
| `TransactionTypes` | Frozen mutation plan and result shapes, without DataStore access |
| `ViewModels` | Typed presentation-neutral public and owner-private projections |

Pure rule modules accept explicit state, content, clock time, input, and seed
state and return new state plus typed effects. They do not access remotes,
DataStores, Workspace state, audio playback, or UI instances.

### 7.2 Client components

| Component | Responsibility |
|---|---|
| `ClientBootstrap` | Composes client dependencies and owns encounter-scoped cleanup |
| `InputAdapter` | Maps touch, keyboard/mouse, and gamepad to semantic actions and timestamps them immediately |
| `ClientClock` | Maps local monotonic time, synchronized server estimate, and audio playhead to encounter time with confidence |
| `RhythmClient` | Runs immediate local matching against the frozen chart and sends evidence-bearing intents |
| `AudioGraph` | Wraps `AudioPlayer`/`Wire`, buses, preload, start, seek, pause, resync, and cleanup |
| `PresentationCoordinator` | Deduplicates semantic events and routes UI, notes, VFX, camera, captions, audio, and haptics |
| `UIRoot` | React Lua screen/HUD composition from view models and design tokens |
| `RhythmRenderer` | Pooled frame-critical staff, notes, timing feedback, and protected indicators |
| `SettingsDraft` | Local recoverable preview/draft state pending authoritative save |

The client may predict only approved local presentation. It cannot award damage,
spend Hype, choose boss outcomes, grant rewards, accept a purchase receipt, or
declare a durable mutation complete.

### 7.3 Server components

| Component | Responsibility |
|---|---|
| `ServerBootstrap` | Composes server dependencies; no domain logic beyond startup/lifecycle |
| `ProtocolGateway` | Authentic actor binding, decode, schema/size/rate/sequence validation, routing |
| `SessionCoordinator` | Hub session, party, proposal, queue, staging, deployment, and follow-up state |
| `EncounterHost` | One attempt lifecycle, frozen configuration, scheduler, reducers, snapshots, outcome |
| `ContentLoader` | Release allowlist, package integrity/capability checks, exact boss arena load |
| `ProjectionService` | Produces public, party-safe, and owner-private state without leaking private facts |
| `ProfileRepository` | ProfileStore adapter, lease/epoch ownership, `UpdateAsync`, confirmed saves, versions |
| `TransactionCoordinator` | Validates and commits frozen per-player mutation plans and receipt grants |
| `RecoveryCoordinator` | No Contest refunds, Pending work, migration, quarantine, and forward repair orchestration |
| `TelemetryAdapter` | Allowlisted semantic analytics, bounded logs, correlation, sampling, and safe failure |

These are logical components, not a command to create one Roblox service for
each design document. Related modules may share a package when authority and
lifecycle stay explicit. Startup files remain small composition roots.

## 8. Runtime protocol and state replication

### 8.1 Message families

There are exactly three authoritative protocol families:

```text
IntentEnvelope
  protocolVersion
  intentId
  kind
  clientSequence
  attemptOrSessionId
  revisionContext
  capturedEncounterTimeUs?
  syncEvidence?
  payload

SemanticEventEnvelope
  protocolVersion
  eventId
  ownerDomain
  kind
  sessionOrAttemptId
  logicalSequence
  encounterTimeUs?
  revisionContext
  causalIds[]
  visibility
  payload

SnapshotEnvelope
  protocolVersion
  snapshotId
  sessionOrAttemptId
  throughLogicalSequence
  serverTimeUs
  encounterTimeUs?
  revisionContext
  visibility
  projectedState
```

The Roblox connection supplies the authenticated player to the server; a client
payload never selects its trusted actor. Each intent definition declares valid
states, payload bounds, per-player rate, idempotency scope, required revisions,
and safe rejection reason. The server checks type, shape, size, rate, sequence,
revision, permission, current state, and domain preconditions before dispatch.

### 8.2 Transport rules

- Reliable remotes carry gameplay intents, semantic events, snapshots, consent,
  recovery, and any state needed for correct understanding.
- Unreliable transport may carry only disposable presentation details whose loss
  cannot affect timing, control, consent, danger, targeting, recovery, gameplay,
  rewards, or evidence qualification.
- Message definitions live in one protocol registry. Ad hoc remote payloads are
  prohibited.
- Public projections contain only band-visible facts. Owner-private projections
  contain that player's inventory, timing detail, recommendations, settings, and
  transaction state. Server-only data is never replicated.
- A client detects a logical-sequence gap or incompatible revision and requests a
  fresh projected snapshot. It does not invent missing events.
- Duplicate intent and event identities are idempotent. Stale requests return a
  stable Conflict/refresh result rather than overwriting newer state.

## 9. Deterministic encounter execution

### 9.1 Time and order

All authored and logical gameplay times are signed integer microseconds from the
package's master origin. Floating-point seconds exist only at Roblox audio and
engine boundaries. Conversion rounds through one shared utility and never
accumulates repeated frame deltas into logical song time.

Each attempt owns an ordered scheduler. Ordering keys are:

1. logical musical timestamp;
2. approved domain phase;
3. authored global tie order where present; then
4. stable event identity.

At one boundary, the fixed Combat order is:

1. scheduled intent, participation, and eligibility changes;
2. valid Rhythm contribution assigned to the boundary;
3. player, support, Revive, and cooperative effects;
4. committed boss impacts and hazards in authored order;
5. Ward thresholds and down transitions; then
6. the completed snapshot exposed for Boss Encounter outcome evaluation.

Network arrival, table traversal, frame order, or receipt order never breaks a
tie. The server owns the attempt seed and every random draw is named, ordered,
and captured in the resulting semantic fact or frozen transaction plan.

### 9.2 Reducer and effect rule

A logical reducer has this conceptual interface:

```text
reduce(frozenState, identifiedInput, ruleContext)
  -> newFrozenState
  -> semanticEffects[]
```

Effects request work from their owning boundary, such as publication, a private
projection, audio/VFX presentation, or a durable mutation. An effect is not
evidence that the requested external work succeeded. External acknowledgments
return as new identified inputs. This keeps retries and failures from executing
a domain transition twice.

## 10. Musical clock, input, judgment, and presentation

### 10.1 Start and playback lifecycle

```text
Package selected
  -> protected assets loading
  -> client exact-revision Ready
  -> server future start boundary
  -> synchronized countdown
  -> all local layers start from one master position
  -> Running observation and correction
  -> finish, safe suspension/re-entry, or No Contest
```

The server publishes an identified future start in server time. A ready client
maps `Workspace:GetServerTimeNow()` to its local monotonic clock, schedules the
audio graph, and observes actual playhead position. Required audio must report
ready before start. No locked subset starts while another member lacks exact-
revision readiness.

The clock controller filters small server-time estimate changes gradually and
outside open judgment windows. It never changes playback speed or pitch. If
measured audio/clock error exceeds the validated tolerance, the client:

1. stops accepting scored input for that player;
2. marks the interval as Rhythm suspension rather than misses;
3. keeps the shared encounter running in co-op;
4. seeks all local layers to one approved safe return point;
5. supplies the authored preview; and
6. resumes only after confidence and alignment are restored.

Solo pause freezes the server encounter time and all local layers at one point.
Resume uses a new future phase-aligned countdown. Cooperative menus never pause
the song.

Calibration modifies input-time and visual-alignment interpretation only. It
does not move audio, notes, boss events, or the authored clock.

### 10.2 Immediate judgment and authoritative consequence

The `InputAdapter` emits a semantic action with a local monotonic timestamp. The
client maps it to encounter microseconds, runs the shared Rhythm rules, and shows
the grade immediately. An attempt made too early to match an eligible note is an
identified Miss under `RHYTHM_GAMEPLAY.md`; absence during a valid suspension is
not.

The client sends:

- intent and local sequence identity;
- semantic action kind/lane;
- mapped encounter time;
- candidate chart-event identity when matched;
- frozen content/chart revision; and
- bounded sync evidence such as server estimate, audio playhead, and confidence.

It never sends trusted damage, Ward, Hype, reward, or contribution totals. The
server rechecks eligibility, revisions, sequence, rate, timing plausibility, and
the measured clock/latency envelope, then executes the same matching and
normalization rules once for shared consequences.

Ordinary network delay does not retract an already displayed grade. A duplicate
cannot apply twice. An isolated ambiguous result becomes a private sync issue;
repeated impossible patterns may enter private anti-abuse evidence without
public accusation or retroactively changing unrelated results.

### 10.3 Audio and multimodal presentation

`AudioGraph` wraps Roblox's modular `AudioPlayer`/`Wire` graph behind a project
interface. For a selected role it plays the role's neutral backing bed plus its
controllable layer or layers. Those references reconstruct the approved complete
song without simultaneously doubling a full mix. The full mix is an explicitly
approved fallback, not an extra normal layer.

Local buses are Song, Local Role, Protected Cues, Voices, Combat, Crowd, and
Ambience. Audio Presentation owns priority, headroom, voice, and degradation
rules. UI, VFX, captions, audio, camera, and optional haptics consume the same
identified semantic event through idempotent presentation instances. Stop,
pause, seek, rejoin, and cleanup apply to the whole encounter presentation scope.

Layer alignment, stable seeking, neutral reconstruction, and device performance
remain gates of the first real-song proof. If the selected Roblox audio graph
cannot satisfy them, only the engine-facing adapter changes before dependent
gameplay proceeds.

## 11. Experience topology, boss levels, and multiplayer

### 11.1 Places and arena loading

One Roblox experience contains:

- a hub place for progression, preparation, parties, queueing, and shard entry;
- a shared battle runtime place used by reserved encounter servers; and
- optional dedicated battle places for exceptional bosses.

The shared runtime place is not a generic player-visible level. It loads the
selected boss revision's unique arena geometry, scenery, lighting and
post-processing, boss rig, animations, camera/spawn anchors, ambience, VFX,
arena graph, hazards, music, charts, and encounter timeline. A boss uses a
dedicated place only when terrain, streaming, memory, or safe authoring makes a
versioned dynamic scene package unsuitable. Both routes use the same runtime
contracts and gameplay modules.

One reserved battle server hosts one active band/attempt at a time. It may retain
consenting players through Results and a new staging cycle. Practice uses the
same runtime in a reserved solo session so timing and gameplay code are shared
and exact solo pause remains possible.

### 11.2 Party, queue, deployment, and attempt flow

```text
Hub
  -> shard + boss/difficulty/mode selection
  -> Current Party proposal, Public Band queue, or Solo
  -> unlocked staging and validation
  -> every player Ready
  -> final roster/Loadout/Battle Supply lock
  -> durable reservations confirmed
  -> expiring server-side deployment record
  -> group teleport to reserved battle server
  -> every exact-revision load acknowledgment
  -> Preparing -> Countdown -> Running -> Resolving -> Complete
  -> Results -> Retry, Stay with Band, or Return to Hub
```

MemoryStore owns expiring party-flow, queue, match, deployment, and rejoin route
records. MessagingService may wake cooperating hub servers, but a missing signal
cannot lose or create authoritative state. Durable eligibility, Loadout,
reservation, and revision facts remain in Player Data.

Teleport data contains only nonsecret display hints and an opaque one-time
deployment identity. The destination consumes the corresponding server-held
record and authenticates each exact member and revision. Any disconnect,
timeout, invalid snapshot, or load failure before attempt start cancels the
deployment, releases reservations idempotently, and returns reachable players to
unlocked staging. No gameplay result exists yet.

### 11.3 Active attempt and rejoin

At song start, the deployment roster and initial population scaling become
immutable. There is no mid-song replacement. Only an authenticated member of
that roster may rejoin during remaining cumulative grace.

A returning member receives a new one-time admission identity, reconnects to the
same reserved server, receives an authoritative snapshot, seeks all audio layers,
and resumes at a safe musical boundary after adequate preview. Previously
committed effects remain; absence creates no inputs or contribution.

Grace expiry closes gameplay rejoin. All humans voluntarily Departed or
terminally Inactive produces the approved Defeat. Loss of the authoritative
battle server or irrecoverable global attempt state produces No Contest. The
first release does not reconstruct a partial song on another server. A durable,
idempotent recovery transaction refunds only eligible causally linked Battle
Supplies.

## 12. Player profile and durable transactions

### 12.1 Storage and one-writer authority

Each player's complete logical profile lives under one standard DataStore key so
one `UpdateAsync` can atomically change all related domains. A pinned reviewed
ProfileStore copy sits behind `ProfileRepository` for session locking, load,
ordered save, and version access. Project code owns the actual schema, writer
epoch, revisions, validation, transaction plans, capacity rules, and Save Unsafe
behavior.

```text
PlayerProfile
  schemaVersion
  profileId
  overallRevision
  writerEpoch
  domainRevisions
  progression
  inventoryAndItemInstances
  battleSuppliesAndReservations
  battleStylesAndThreeSavedLoadouts
  currentConfiguration
  currenciesAndEntitlements
  settingsOnboardingAndTeachingHistory
  boundedOutcomeHistory
  transactions
    idempotencyIndex
    pendingAndCommittedJournal
    compensations
  commerce
    permanentReceiptIndex
    grantsAndRestoration
  lifecycle
    dirtyDomains
    recoveryAndQuarantineState
    retentionClasses
```

Exactly one server epoch may mutate the profile. A hub-to-battle handoff first
confirms required reservations and frozen references, then releases the hub
session so the destination can acquire a new epoch. Lease loss or ambiguity
immediately stops new mutation acceptance. It never falls back to last-write-
wins.

### 12.2 Mutation and transaction contract

Owning domains authorize semantic client requests and produce an immutable plan:

```text
TransactionPlan
  transactionId
  playerId
  ownerDomain
  semanticAction
  sourceAndCausalIds[]
  expectedOverallAndDomainRevisions
  writerEpoch
  contentCatalogBalanceRevisions
  preconditions[]
  frozenCalculationsAndRandomRolls
  orderedOperations[]
  privacyClass
  createdAtServerUs
  idempotencyScope

TransactionResult
  transactionId
  status: Pending | Committed | Rejected | Conflict | RecoveryRequired
  safeReason
  preAndPostRevisions
  confirmedProfileVersion?
```

The plan includes every causally related journal append, balance delta, item or
entitlement change, Progression fact, Battle Supply reservation/spend/refund,
receipt/reward lineage, and downstream committed handoff. `ProfileRepository`
rechecks epoch, versions, domain ownership, references, plan integrity, and
generic invariants immediately before applying it. It does not reroll or
recalculate the domain outcome.

A critical action reports Saved, Applied, Granted, or complete only after the
exact transaction identity appears in confirmed saved data. A duplicate returns
the recorded result. A Conflict refreshes state. Uncertainty remains Pending or
Recovery Required. Low-risk settings may debounce but remain visibly unsaved
until confirmation.

### 12.3 Results and purchase receipts

A band's outcome creates separate atomic per-player result plans. One player's
storage delay never revokes another player's committed reward, and Results may
show Pending without blocking a gameplay Retry.

`MarketplaceService.ProcessReceipt` is handled only on the server:

1. derive a stable receipt transaction identity;
2. read or acquire the owning player profile;
3. return the previously recorded outcome if already processed;
4. freeze the exact quoted grant and earnable-equivalent lineage;
5. write receipt identity and grant in the same profile transaction;
6. wait for the exact confirmed save; then
7. return `PurchaseGranted`.

An ambiguous load or save returns `NotProcessedYet`. The permanent receipt index
outlives ordinary transaction compaction. Storage capacity nearing its safe
ceiling blocks new value growth honestly rather than accepting a purchase that
might be lost.

### 12.4 Save Unsafe, migration, and recovery

Initial profile load failure blocks Ready gameplay. Mid-session persistence
failure or writer-epoch loss enters Save Unsafe/read-only state with affected
domains, pending operation identities, last confirmed version, and permitted
actions. A running encounter may continue only from its already durable locked
snapshot when doing so creates no new unrecoverable commitment. New encounters,
purchases, value mutations, and applied Loadout changes remain blocked.

Migrations form an ordered deterministic idempotent registry with pre/post
invariant checks. They are tested on copied or mock profiles and staged before
production. Corrupt data enters quarantine. Recovery may select a verified prior
DataStore version, replay later committed journal facts, and write a new forward
repair. It never silently resets the player or reverses the schema.

Retired value-relevant catalog identities remain stable where possible. A
replacement needs approved exact equivalence, player choice, or an identified
compensation transaction.

Roblox key-version history supports immediate recovery. Encrypted Open Cloud
exports are required before destructive or high-risk migrations and on a
schedule once public purchases are enabled. A small authenticated admin tool may
read, export, delete, and submit idempotent compensation/repair operations. It
does not edit raw values. Every operation is least-privilege, audited, reasoned,
and attributable.

## 13. UI, input, settings, localization, and accessibility

React Lua/ReactRoblox composes menus, preparation, settings, onboarding, Results,
and stable HUD elements from typed semantic view models and shared design tokens.
Components request semantic actions; they do not query unrelated services or
decide legality. Frame-critical notes and VFX use pooled specialized renderers.

One repository-owned action catalog drives:

- semantic action identity;
- `InputAction`/`InputBinding` hardware mappings;
- fixed touch controls;
- keyboard/mouse and gamepad defaults;
- prompt glyph/text selection;
- remapping and conflict validation;
- active-device change; and
- unavailable-input recovery.

The adapter timestamps a rhythm action before UI or network processing. Touch,
keyboard/mouse, and gamepad expose the same gameplay opportunity. Losing a usable
input profile suspends that player's Rhythm safely; it does not invent misses,
change the co-op clock, or change roster membership.

Settings edit a recoverable local draft. Presentation-only settings may preview
immediately; the UI says Saved only after the correct profile confirmation.
During initial-load failure, essential accessibility/device choices can remain a
clearly temporary local draft.

Repository-owned stable localization keys generate the Roblox localization
tables. Layouts permit approximately 30 percent text expansion and support phone
safe areas, scale, focus, navigation order, captions, non-color labels/shapes,
contrast, reduced motion, and optional haptics. Accessibility choices remain
private and never change difficulty, maximum contribution, matchmaking, rewards,
or public Results treatment.

## 14. World, model, animation, and asset lifecycle

Boss arenas and reusable art are creator-owned, versioned Roblox packages or
models without executable gameplay scripts. Their content manifests pin exact
asset versions. Runtime never requests automatic latest versions for an active
revision.

Before approval, publication validates:

- creator/group ownership and experience permissions;
- moderation and availability status;
- declared asset ID, version, source hash, and license/source record;
- required arena anchors, tags, collision groups, graph nodes, and bounds;
- boss rig, animation, camera, spawn, lighting, and effect references;
- absence of scripts or other prohibited executable instances;
- supported device budget and optional quality variants;
- dynamic load and cleanup in the shared runtime; and
- the previous exact-version rollback path.

`ContentLoader` loads only an allowlisted creator-accessible exact model version.
Failure before countdown cancels deployment safely. An unexpected critical arena
failure after start becomes No Contest. A boss whose scene cannot meet dynamic
load, streaming, memory, or authoring constraints moves to a dedicated place
without changing content or gameplay contracts.

## 15. Security, privacy, and abuse resistance

### 15.1 Trust boundaries

| Boundary | Rule |
|---|---|
| Client to server | Treat all payloads, timestamps, candidate matches, and claimed revisions as untrusted evidence |
| Shared replicated code/data | Assume players can inspect it; store no secrets or server-only selection state there |
| Teleport data | Treat as client-visible hints; authorize through consumed server-held deployment records |
| Loaded arena/model | Allowlist exact owned versions and reject executable scripts or malformed required structure |
| DataStore mutation | Permit only through the active profile epoch and validated immutable transaction plan |
| Marketplace receipt | Accept only through the server receipt handler and confirmed idempotent profile commit |
| Admin/Open Cloud | Use separate least-privilege credentials outside Git, audited operations, and environment scoping |
| Analytics | Send only allowlisted minimum facts; never complete profiles, raw audio, chat, or continuous input replay |

The server validates remote structure, size, numeric bounds, rate, revision,
state, authorization, and timing plausibility. Gameplay remotes use bounded
queues and stable reason codes. Rate limiting cannot silently discard an already
accepted value transaction.

Anti-abuse signals remain private. One latency or clock anomaly triggers sync
recovery, not public blame. Impossible aggregate patterns may be reviewed, but
the system never publishes a cheater label or lets an automated analytics rule
alter another player's result.

Privacy, child-safety, commerce, retention, account deletion, and current Roblox
policy review are explicit production gates. Each stored or collected field has
an owner, purpose, access class, retention/deletion behavior, and export rule.
Secrets are never committed to the repository or embedded in client code.

## 16. Performance, capacity, and degradation

### 16.1 Supported profiles and evidence status

The standard presentation target is 60 frames per second, equivalent to about
16.7 ms per frame. The supported lower-device profile is a stable 30 frames per
second, about 33.3 ms per frame. Rhythm judgments use captured input time rather
than rendered frames in both profiles.

The first real-song slice must measure a representative lower-end phone,
ordinary phone/tablet, and desktop/gamepad profile. The following caps become
versioned release configuration from that evidence rather than invented values:

| Budget | Architecture rule | Current evidence |
|---|---|---|
| Client frame CPU/GPU and spikes | Numeric target per supported profile; critical input and staff work isolated | Not yet measured |
| Client and server memory | Per-place baseline plus per-boss and per-player increment | Not yet measured |
| Decoded/compressed audio and voices | Must fit complete neutral song, local role, and protected cues | Not yet measured |
| Arena instances, geometry, textures, effects | Each boss package declares measured use and quality variants | Not yet measured |
| Hub and battle load time | Required assets have explicit readiness horizon and timeout behavior | Not yet measured |
| Server step work | Event-driven scheduler with measured one-to-six-player worst case | Not yet measured |
| Reliable/unreliable remote traffic | Per-player and per-attempt caps; no per-frame authoritative replication | Not yet measured |
| DataStore/MemoryStore operations | Expected concurrency plus tested throttle margin | Not yet measured |
| Profile/package size | Safe ceiling below platform limits with explicit growth behavior | Not yet measured |

These missing measurements are implementation evidence, not unresolved
architecture. The first boss package cannot be approved until its target numbers
and measured results are recorded. Every later package publishes its budget use
and fails validation when it does not fit a supported profile.

### 16.2 Degradation order

Under device or network pressure, reduce in this order:

1. crowd density and crowd variation;
2. optional ambience and decorative spatial/reverb detail;
3. optional UI/world variation;
4. particles and purely decorative VFX;
5. shadows, texture detail, and other cosmetic quality; then
6. ordinary combat presentation variation that has a protected semantic form.

Never reduce, drop, retime, or obscure song/role playback, rhythm input, the
musical clock, fixed controls, note eligibility, committed danger/target/
recovery/group cues, captions, or required alternatives. Overload that prevents
those guarantees suspends the affected player safely or ends the attempt as No
Contest; it does not continue with silently unreliable outcomes.

## 17. Analytics, diagnostics, and operations

### 17.1 Telemetry contract

`TelemetryAdapter` accepts only registered semantic facts. A concrete event has:

```text
TelemetryEvent
  telemetrySchemaVersion
  eventId + eventVersion
  ownerDomain
  collectionClass
  authoritativeOrObservedSource
  serverTimeUs
  encounterTimeUs?
  releaseId
  serverJobId
  sessionDeploymentAttemptIds?
  contentChartBalanceCatalogRevisions?
  causalAndIdempotencyIds?
  allowlistedDimensions
  numericValue?
  correctionOrCancellationRef?
```

Production, Private Study, QA/Synthetic, and Prohibited collection classes follow
`PLAYTEST_AND_ANALYTICS.md`. Production player identity is no wider than the
named use requires. Raw continuous input, complete permanent judgment streams,
voice, video, microphone/audio, chat content, complete profiles, private
accessibility history, and unnecessary purchase details remain prohibited.

Servers emit authoritative product/playtest facts to AnalyticsService in
published environments. Clients may submit only registered, rate-limited,
bounded observations; the server labels them as observed rather than
authoritative. High-frequency events are aggregated or sampled according to the
metric definition. Adapter deduplication uses event/idempotency identity.
Collection failure drops or delays telemetry within a bounded buffer and never
changes gameplay or rewards.

### 17.2 Initial health views and runbooks

Initial dashboards or reviewed reports cover:

- package/asset load and exact-revision readiness failures;
- unsafe clock and audio-resynchronization rate;
- deployment cancellation, rejoin failure, Defeat, and No Contest reasons;
- profile load failure, Save Unsafe, Pending age, migration quarantine, and
  recovery outcome;
- receipt Pending age and grant/duplicate/recovery outcomes;
- remote rejection/rate anomalies without public accusation; and
- supported-device frame, memory, audio, and load evidence.

Short runbooks exist before release for content-load failure, teleport/deployment
failure, unsafe timing, authoritative-server loss, Save Unsafe, stuck Pending
transactions, pending receipts, corrupt/migration-quarantined profiles, and bad
publication. Repair uses the audited recovery and compensation paths from
section 12. Raw DataStore edits are prohibited.

The first playable release uses Roblox analytics, structured server output,
private staging traces, and the approved lightweight playtest records. An
external telemetry backend requires a documented failed investigation need,
privacy/retention review, cost approval, and an architecture amendment.

## 18. Verification strategy

### 18.1 Three verification layers

| Layer | Required coverage | What it cannot prove alone |
|---|---|---|
| Local deterministic tests | Bun pipeline/export tests, Zod schemas, golden fixtures, Jest Lua pure reducers, migration and transaction models | Roblox service, device, audio, or teleport behavior |
| Studio tests | Client/server boundaries, virtual input, screen/device profiles, snapshots, fault injection, up to eight simulated clients | Published-only services, real network, physical audio output |
| Private published staging | Reserved teleports, MemoryStore/DataStore, receipts through safe test paths, permissions, layered audio, multi-device timing, performance | Production policy approval or future scale without measured evidence |

Golden fixtures contain a small platform-neutral package and its Roblox export,
including expected clock conversion, event order, chart judgments, normalized
contribution, encounter transitions, protocol encode/decode, and migration
results. Cross-language fixtures use the same source JSON and expected semantic
hashes so TypeScript and Luau cannot silently interpret a revision differently.

Fault fixtures cover clock offset/drift, early input, duplicate and out-of-order
messages, latency and packet loss, client disconnect/rejoin, missing assets,
invalid content, profile lease loss, throttled/ambiguous saves, duplicate
receipts, migration failure, and server loss.

### 18.2 Current and future commands

The existing maintained checks remain:

```bash
cd tools/chart-pipeline
bun run check
bun test

cd roblox/web
bun run check
bun test
```

The browser demo checks remain reference/prototype evidence only. Roblox
implementation adds pinned dependency installation, Jest Lua test-place
execution, Rojo source-tree build, package validation, and Studio acceptance
commands as the relevant phases land. Their exact commands must be committed in
`roblox/README.md` before an implementation task can claim that gate.

### 18.3 Smallest end-to-end acceptance path

The mandatory release scenario is:

1. open one real song project in the local authoring app;
2. validate and approve its exact platform-neutral revision;
3. export and bind Roblox assets with matching hashes/revisions;
4. load its distinct boss level in private staging;
5. deploy at least two clients through staging into a reserved battle server;
6. complete the aligned full-song attempt with immediate local judgments and
   server-authoritative consequences;
7. produce one immutable outcome and per-player transaction plans;
8. confirm reward persistence;
9. reconnect each test profile and verify the exact saved result; and
10. inspect required phone, desktop/gamepad, audio, accessibility, performance,
    and telemetry evidence.

At least one representative physical phone is mandatory for the earliest audio
slice. Full release coverage follows the device and participant matrix in the
approved specs. A unit-test pass never substitutes for physical timing/listening
or human playtest evidence.

## 19. Build, publication, rollout, and rollback

### 19.1 Build and promotion

The initial release process is intentionally human-supervised:

1. select an exact clean Git commit and locked tool/dependency set;
2. validate approved content packages and generate Roblox data/localization;
3. validate every referenced asset ID/version, permission, hash, and package
   budget;
4. open the approved versioned Studio place base and sync the exact Rojo source;
5. run place, structure, script-absence, schema, and test validators;
6. save the immutable release-candidate place artifacts and hashes;
7. publish those same inputs to the private staging experience;
8. run staging acceptance and record its evidence;
9. create the final `ReleaseManifest` and obtain explicit human approval;
10. publish the approved artifacts to production;
11. smoke-test a newly started production server; then
12. enable new content only for new attempts.

Open Cloud place publication may automate a proven artifact upload. It does not
replace complete-place assembly or validation, and its documented unsupported
instance types must be checked against each place. Credentials are scoped per
environment and are never available to game clients or committed files.

### 19.2 Rollout and active compatibility

Development may use synthetic packages and mock stores. Staging uses safe test
profiles and creator-owned release assets. Production uses only approved
manifests and migrations.

A server records its `releaseId` at boot. Content and feature enablement applies
to new sessions/attempts only. An active attempt retains its frozen revisions.
If a critical compatibility or integrity failure prevents safe completion, the
attempt becomes No Contest and follows recovery; it never silently changes to a
different package.

### 19.3 Rollback

Rollback selects a prior approved place/code/content/balance/asset manifest and
starts new servers on it. DataStore schemas never move backward. Any data issue
uses a forward-compatible reader, deterministic forward repair, or a staged new
migration. A rollback record names the reason, affected release, replacement,
active-attempt treatment, player-value impact, evidence, and approver.

Previously approved content remains addressable by exact revision for compatible
active attempts, personal records, audit, and recovery. Removing a broken
revision from new selection does not erase its history.

## 20. Content-first implementation sequence

The [Implementation Plan](IMPLEMENTATION_PLAN.md) translates this sequence
into milestones and tasks. The [execution record](IMPLEMENTATION_PROGRESS.md)
tracks results and outstanding checks.

| Phase | Bounded outcome | Exit gate |
|---:|---|---|
| 0 | Pin tools/dependencies; establish contracts, generated fixtures, test place, and target source layout without deleting prototypes | Reproducible local checks and source-tree build documented |
| 1 | Build the minimum local authoring app around the existing pipeline and publish one complete immutable real-song package | Human-approved package passes structural, musical, compatibility, and export validation |
| 2 | Build the narrow Roblox audio/rhythm slice with package loading, clock, aligned layers, Normal chart, input adapter, immediate feedback, and two-client start | Published staging proof on desktop and a representative phone |
| 3 | Add typed protocol, deterministic server scheduler/reducers, semantic projections, and encounter lifecycle around that slice | Replay/fault fixtures and client/server Studio tests pass |
| 4 | Add distinct boss scene loading, reserved deployment, rejoin, one-writer profiles, confirmed transactions, and No Contest recovery | End-to-end staging scenario persists and reloads exact Results |
| 5 | Add full first-release Combat, abilities, progression, Loadouts, hub, UI/accessibility, and the remaining approved boss content | Three-boss content-complete readiness and playtest gates pass |
| 6 | Add commerce only after receipt/recovery/storage proof, then production telemetry, backups, runbooks, release automation, and policy review | Production release checklist and explicit approval complete |

Later phases may begin isolated preparation only when they do not assume an
unproved interface from an earlier gate. In particular, broader gameplay cannot
choose a different song-clock or package model while phase 2 remains unresolved.

### 20.1 Implementation task contract

An implementation task must name:

- one observable objective and the authoritative specification sections;
- owned files/modules and interfaces it may change;
- exact input/output contracts and revision assumptions;
- excluded neighboring systems and mutations;
- failure, cleanup, and idempotency behavior;
- targeted automated checks;
- required Studio, published-game, physical-device, or human QA scenario;
- artifact/evidence location; and
- rollback or safe-disable behavior.

Tasks are sized to one boundary or vertical behavior. “Build the boss system” is
not a valid task. A valid task resembles: implement the server-owned attempt
lifecycle transitions for one frozen package contract, without boss attacks or
rewards, and demonstrate the specified reducer and Studio scenarios.

Agents may not invent product behavior to complete an interface, edit generated
content directly, broaden telemetry, bypass a failed gate, or declare a platform
mechanism proven from source inspection alone.

## 21. Selected but unverified gates

All 24 architecture questions are resolved. The following are required evidence
or values, not permission to choose a different architecture silently:

1. Prove `AudioPlayer`/`Wire` aligned multilayer start, full-song drift, seek,
   stop, and reconstruction on target devices.
2. Measure the `GetServerTimeNow()` mapping, client clock confidence, judgment
   envelope, and resynchronization thresholds under realistic network conditions.
3. Establish numeric device, memory, audio, server-step, remote, service, load,
   package, and profile budgets from the real-song slice.
4. Prove exact-version dynamic boss scene loading and cleanup; route a boss to a
   dedicated place if it cannot meet the gate.
5. Prove reserved deployment, one-time admission, rejoin grace, and No Contest
   recovery in a published private experience.
6. Pin and audit React Lua, Jest Lua, ProfileStore, Wally, and any later
   dependency before production use.
7. Prove profile capacity, lease handoff, confirmed-save detection, migration,
   receipt retry, backup, and forward repair with failures injected.
8. Turn the current Studio/Rojo authored-place workflow into a recorded,
   repeatable release assembly before automating publication.
9. Complete current Roblox privacy, child-safety, commerce, retention, account-
   deletion, asset-permission, and publication-policy review before public use.
10. Select the final public game title before publication; this does not block
    implementation under the internal project identity.

A failed gate changes the smallest replaceable adapter or returns to the owning
design decision. It does not authorize silent fallback that changes player
behavior, fairness, music, value, or accessibility.

## 22. Cross-specification reconciliation

| Approved input | Architecture realization |
|---|---|
| Content Authoring | Local Bun/browser tool, immutable package contract, flexible roles, exporter, exact asset/revision load, content-first gate |
| Rhythm Gameplay | Master-origin microseconds, shared pure matching, immediate client feedback, server consequence, safe suspension/re-entry |
| Combat | Deterministic reducers, fixed same-boundary phase order, server-owned effects, identified semantic facts |
| Boss Encounters | Attempt scheduler/lifecycle, distinct boss levels, shared runtime host, frozen outcome, No Contest |
| Progression | Profile domain revisions and atomic result plans from exact attempt facts |
| Items & Equipment | Server-validated Loadout snapshots, locked references, item identities, Battle Supply reservation/spend/refund lineage |
| Abilities & Cooperative Actions | Shared deterministic rules with server-owned Hype/group action state and idempotent presentation |
| Multiplayer | MemoryStore flow records, reserved deployment, exact load barrier, locked roster, rejoin route, voluntary follow-up |
| Rewards & Economy | Frozen random/value plans, confirmed per-player commits, durable receipt lineage, compensation rather than history rewrite |
| Builds & Specialization | Three saved Loadouts and active configuration in one profile, semantic validation, immutable attempt snapshot |
| UI/UX | React Lua view-model boundary, semantic action catalog, responsive focus/state, truthful pending/failure presentation |
| Audio Presentation | Layered audio adapter, semantic buses/events, protected cue priority, complete lifecycle cleanup and fallback gate |
| Player Data | One-key logical profile, writer epoch, ProfileRepository, transaction journal, Save Unsafe, migration/recovery/admin paths |
| Balance Framework | Exact balance revision in package/release/attempt, deterministic calculation inputs, measured tuning evidence |
| Playtest & Analytics | Typed event registry, collection classes, revision correlation, lightweight human evidence, readiness gates |
| Naming & Tone | Stable localization keys, generated tables, approved terms, internal identity separated from player label |
| Art Direction | Studio/DCC/asset ownership, boss-specific scenes, versioned visual assets, measured quality variants |

No authoritative gameplay fact, durable value, content truth, player-facing term,
or evidence decision gains a competing owner here. Every first-release system in
`SYSTEMS_MAP.md` has an implementation boundary, consumer path, and verification
route. Quests, achievements, guilds, broad live operations, player trading, PvP,
user-authored songs, and the other declared non-goals remain outside this
architecture until their design is explicitly approved.

## 23. Primary platform references

These sources were consulted on 2026-09-10. They document available mechanisms;
the evidence gates above determine whether those mechanisms work for this game.

- [Rojo project format](https://rojo.space/docs/v7/project-format/) and
  [sync details](https://rojo.space/docs/v7/sync-details/)
- [Roblox client/server security](https://create.roblox.com/docs/scripting/security/client-server-boundary)
  and [security tactics](https://create.roblox.com/docs/scripting/security/security-tactics)
- [Remote events and callbacks](https://create.roblox.com/docs/scripting/events/remote)
- [`Workspace:GetServerTimeNow()`](https://create.roblox.com/docs/reference/engine/classes/Workspace#GetServerTimeNow)
- [Roblox audio objects](https://create.roblox.com/docs/audio/objects) and
  [`AudioPlayer`](https://create.roblox.com/docs/reference/engine/classes/AudioPlayer)
- [`ContentProvider`](https://create.roblox.com/docs/reference/engine/classes/ContentProvider)
- [Teleporting between places](https://create.roblox.com/docs/projects/teleporting)
- [Data stores and memory stores](https://create.roblox.com/docs/cloud-services/data-stores-vs-memory-stores),
  [DataStore](https://create.roblox.com/docs/cloud-services/data-stores), and
  [MemoryStore](https://create.roblox.com/docs/cloud-services/memory-stores)
- [`MessagingService`](https://create.roblox.com/docs/reference/engine/classes/MessagingService)
- [`MarketplaceService.ProcessReceipt`](https://create.roblox.com/docs/reference/engine/classes/MarketplaceService#ProcessReceipt)
- [Open Cloud](https://create.roblox.com/docs/cloud) and
  [place publishing](https://create.roblox.com/docs/cloud/guides/usage-place-publishing)
- [React Lua](https://github.com/jsdotlua/react-lua),
  [Jest Lua](https://github.com/jsdotlua/jest-lua),
  [ProfileStore](https://github.com/MadStudioRoblox/ProfileStore), and
  [Wally](https://github.com/UpliftGames/wally)
- [`InputAction`](https://create.roblox.com/docs/reference/engine/classes/InputAction)
  and [`InputBinding`](https://create.roblox.com/docs/reference/engine/classes/InputBinding)
- [Roblox localization](https://create.roblox.com/docs/production/localization)
- [Roblox packages](https://create.roblox.com/docs/projects/assets/packages) and
  [`InsertService:LoadAssetVersion`](https://create.roblox.com/docs/reference/engine/classes/InsertService#LoadAssetVersion)
- [Performance optimization](https://create.roblox.com/docs/performance-optimization)
- [Studio testing modes](https://create.roblox.com/docs/studio/testing-modes) and
  [`StudioTestService`](https://create.roblox.com/docs/reference/engine/classes/StudioTestService)
- [Roblox custom analytics events](https://create.roblox.com/docs/production/analytics/custom-events)

## 24. Approval and change control

The owner interview resolved TA-01 through TA-24 on 2026-09-10. This document is
the approved implementation baseline. A selected mechanism that still requires
evidence remains selected until its named gate passes or a documented failure
justifies an amendment.

A material change to client/server authority, canonical package semantics,
protocol families, deterministic order, place topology, profile atomicity,
receipt acknowledgment, environment isolation, protected degradation, or the
content-first implementation sequence requires an explicit architecture
amendment. Routine module names, private helper structure, measured thresholds,
and tool patch versions may evolve through reviewed implementation when they
preserve these contracts and update their manifests, evidence, and documentation.
