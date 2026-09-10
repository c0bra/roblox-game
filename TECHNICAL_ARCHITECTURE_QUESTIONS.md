# Technical Architecture Questions

- **Status:** Complete; 24 of 24 questions resolved
- **Started:** 2026-09-10
- **Current checkpoint:** Complete
- **System authority:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md)
- **Design inputs:** [`SYSTEM_SPECIFICATIONS_TRACKER.md`](SYSTEM_SPECIFICATIONS_TRACKER.md)
- **Working record:** [`TECHNICAL_ARCHITECTURE_WORKING.md`](TECHNICAL_ARCHITECTURE_WORKING.md)
- **Canonical result:** [`TECHNICAL_ARCHITECTURE.md`](TECHNICAL_ARCHITECTURE.md)

## 1. Interview method and scope

This interview translates the approved design into an implementable Roblox
architecture. It covers the local content tools, game client, game servers,
platform services, data contracts, and development/release workflow.

There are eight checkpoints of three related questions. Present each checkpoint
in plain language with a recommended approach and its practical tradeoffs. Split
a checkpoint into smaller pieces when needed. The owner decides meaningful
product, workflow, cost, and maintenance tradeoffs; the assistant researches APIs
and supplies the engineering detail behind those decisions.

Do not ask the owner to select every library, field name, or timeout. Resolve
routine engineering choices within accepted constraints and record their
rationale. Follow up only where an answer leaves a consequential choice open.
User answers, selected defaults, rejected alternatives, and remaining evidence
needs are persisted in the working record at each meaningful checkpoint.

A proposal is not an approved decision. Existing design requirements, observed
repository facts, recommended architecture, and verified platform capabilities
must remain distinguishable. Consult current primary documentation before
committing to platform APIs, limits, or third-party dependencies. An API reference
does not prove that the game meets its timing or durability requirements.

The final document must name components, interfaces, state transitions, data
representations, failure behavior, and verification requirements clearly enough
for the subsequent implementation plan to produce bounded tasks. Detailed code,
complete implementation tickets, finished catalogs, and actual release evidence
are separate deliverables.

## 2. Inherited decisions

These are approved inputs, not questions to reopen:

- The playable product targets Roblox; mobile and PC behavior must follow the
  approved UI, Input, and Accessibility contracts. The final public game title
  is deferred; `Bands Battle` remains an internal project name.
- All thirteen system specifications, the Balance Framework, Playtest &
  Analytics, and Naming & Tone are approved. Art Direction remains the visual
  authority. Architecture implements their rules and semantic ownership.
- Content Authoring runs outside Roblox in a lightweight internal web app around
  existing processing tools. Establish usable song data before dependent
  gameplay implementation. There is no public song-creation platform in scope.
- The human-approved musical clock, transcription, charts, Activity Maps, and
  encounter timeline produce an immutable platform-neutral runtime package.
  Roblox exports preserve its meaning. Automated analysis cannot approve it.
- Instruments are extensible and song-specific. Four common stems are an early
  tool assumption, not a mandatory runtime roster. Difficulty is Easy, Normal,
  and Hard; the canonical transcription is separate from difficulty charts.
- One to six humans can play, with the defined solo support. Active encounters
  admit only rejoining members of their locked deployment roster. Ready, player
  consent, loading, and start behavior follow Multiplayer's existing contract.
- Attempts freeze exact content, balance, roster, and Loadout references as
  specified. Catalog changes cannot rewrite active attempts or confirmed facts.
- Local rhythm feedback is immediate. Shared consequences require authoritative
  validation; ordinary network delay cannot visibly revise an already shown
  grade. Unsafe timing causes nonpunitive suspension and fair re-entry.
- Song timing, pitch, and structure remain intact. Presentation, equipment,
  calibration, accessibility, and other players cannot change chart truth or
  the equal maximum contribution rules.
- Every gameplay fact has one semantic owner. Shared events describe actual
  facts; presentation and Analytics do not invent or authorize gameplay.
- Player Data preserves one coherent logical profile with exclusive mutation
  authority. Value changes are atomic and idempotent; uncertain results remain
  Pending or Save Unsafe. Verified purchase receipts precede paid fulfillment.
- The three saved Loadouts, Battle Style hierarchy, free configuration changes
  in allowed locations, and locked active configuration are already designed.
- Protected cues, accessible controls, honest recovery/status displays,
  localization keys, private projections, and minimal telemetry are required.
- Balance numbers retain their approved or hypothesis status. Architecture
  cannot turn an untested timing, performance, or capacity target into evidence.
- System-map entries do not require one Roblox service or module apiece. Choose
  a practical implementation structure while preserving semantic ownership.

## 3. Question plan

### Checkpoint A - Development foundation

#### TA-01 - Existing prototypes and production boundaries

- **Status:** Resolved 2026-09-10.
- **Decision:** Continue in the existing repository and evolve the Roblox
  project. Keep the browser demo as an experiment/reference surface rather than
  a second shipping product. Reuse only components that satisfy the approved
  specifications, and extend the separate chart pipeline for canonical content.
- **Question:** How should the existing Roblox prototype, browser demo, and chart
  pipeline contribute to the production game?
- **Recommended starting point:** Keep this repository. Evolve the Roblox
  project, retain the browser demo as an experiment/reference surface, and reuse
  components that satisfy the approved specs. Extend the separate chart pipeline
  for canonical content. Do not require a second shipping browser game.
- **Must establish:** Source-of-truth boundaries, selective reuse versus
  replacement, prototype gaps, and whether any existing experiment must remain
  supported as a product. Record a migration path without promising wholesale
  reuse or an unnecessary rewrite.

#### TA-02 - Languages, tools, and dependency approach

- **Status:** Resolved 2026-09-10.
- **Decision:** Keep Luau/Rojo for Roblox, TypeScript/Bun for the local authoring
  app and pipeline coordination, and useful Python/external audio processors.
  Prefer explicit typed modules and introduce dependencies for demonstrated
  implementation or maintenance value.
- **Question:** What development stack will be straightforward to maintain with
  AI-assisted implementation?
- **Recommended starting point:** Continue the existing Luau/Rojo workflow for
  Roblox and TypeScript/Bun for local web tools and pipeline coordination. Keep
  useful Python and external audio processors. Use explicit typed modules and
  add a framework or library only for a demonstrated need.
- **Must establish:** Language/toolchain ownership, version pinning, dependency
  selection criteria, code/Studio workflow, and how cross-language data contracts
  remain consistent. Specific UI and persistence library choices follow in their
  relevant checkpoints.

#### TA-03 - Hosting, operating burden, and cost constraints

- **Status:** Resolved 2026-09-10.
- **Decision:** Prefer Roblox-managed runtime services and a locally run
  authoring application. Add an external live backend only when a documented
  requirement cannot reasonably be met on Roblox.
- **Question:** How much infrastructure should we operate ourselves?
- **Recommended starting point:** Prefer Roblox-managed runtime services and a
  local authoring application. Introduce an external live backend only for a
  documented requirement Roblox cannot reasonably satisfy.
- **Must establish:** Local versus hosted responsibilities, owner maintenance and
  spending constraints, any existing service commitments, and the evidence that
  would justify an added service. This is a direction, not a claim that every
  persistence/telemetry requirement is already proven feasible on Roblox.

### Checkpoint B - Content foundation and publication

#### TA-04 - Local authoring application and processing jobs

- **Status:** Resolved 2026-09-10.
- **Decision:** Run a TypeScript/Bun application locally with its UI in a web
  browser. The local process owns directory access and pipeline jobs; the UI
  provides synchronized playback, timeline inspection, editing, validation, and
  explicit rerun/cancel/approve actions. Generated suggestions remain separate
  from human edits, and reprocessing marks dependent work stale without silently
  overwriting it.
- **Question:** How will selecting a song directory, reviewing its timeline, and
  rerunning analysis work in the local application?
- **Must establish:** Browser UI/local process boundary; directory access;
  explicit job execution, progress, cancellation, and error reporting; reuse of
  pipeline commands; persistence of human edits; invalidation of stale derived
  data; review/approval state; and the minimum editor needed for the first boss.
  Reprocessing must not silently replace approved edits or revisions.

#### TA-05 - Schemas, catalogs, and Roblox export

- **Status:** Resolved 2026-09-10.
- **Decision:** Publish one immutable versioned package per song/encounter
  revision. Its manifest binds all approved audio, musical, chart, Activity Map,
  encounter, catalog, localization, asset, approval, and integrity data. Generate
  validated Roblox-readable data and uploaded asset references from this package,
  while retaining the platform-neutral package as source of truth. Roles remain
  extensible data; legacy `medium` and clip-relative timing require explicit
  import conversion.
- **Question:** What exactly is published, and how does Roblox know that all of
  its song, gameplay, and asset data belongs together?
- **Must establish:** Concrete package and catalog schema families; stable IDs;
  exact-time/musical coordinates; event lineage and contribution weights; role
  extensibility; schema/content/balance revisions; validation; checksums and
  dependency binding; audio/model/localization asset references; export/import
  and asset-access checks; and immutable publication/rollback. Map legacy charts
  explicitly, including `medium` versus Normal and clip offsets versus the
  master-origin clock.

#### TA-06 - Early feasibility proof and content-first milestone

- **Status:** Resolved 2026-09-10.
- **Decision:** Make one real approved song package and a narrow Roblox vertical
  slice the first architecture milestone. It must demonstrate real master and
  controllable-role audio, the approved clock, a Normal chart with immediate
  timing feedback, safe seek/resynchronization, two-client alignment in a
  published Roblox test, and a representative phone check before dependent
  gameplay implementation proceeds.
- **Question:** What must the first approved song package prove in Roblox before
  we build the wider gameplay systems around it?
- **Must establish:** A small content-to-Roblox proof covering real audio asset
  access, loading, aligned layers, seeking, musical time, timing feedback,
  representative phones, and multiplayer synchronization. State evidence and
  failure criteria, protected-audio fallback boundaries, the minimum tool/export
  work that precedes it, and which later work depends on its outcome. A planned
  experiment is not a completed feasibility result.

### Checkpoint C - Runtime components and contracts

#### TA-07 - Client, server, and shared component layout

- **Status:** Resolved 2026-09-10.
- **Decision:** The server owns authoritative encounter state, shared combat and
  cooperative results, session decisions, rewards, and durable mutations. The
  client owns input, local audio/clock playback, immediate rhythm feedback, UI,
  camera, VFX, captions, and haptics. Shared modules contain typed contracts,
  immutable definitions, and pure deterministic rules, without secrets or
  authoritative mutable state. Organize implementation by runtime side and
  cohesive domain rather than mirroring every design heading as a service.
- **Question:** Which parts run on the player's device, which run on the server,
  and how are they organized without duplicating authority?
- **Must establish:** Concrete component/directory map; semantic owners;
  client/server/shared boundaries; platform adapters versus testable domain
  logic; dependencies; startup/readiness/shutdown; encounter-scoped cleanup; and
  where pure rules may be shared safely. Client-visible code/data must not carry
  secrets or grant mutation authority.

#### TA-08 - Commands, events, and replicated state

- **Status:** Resolved 2026-09-10.
- **Decision:** Use typed client-intent, server-confirmed-event, and versioned-
  snapshot message families. Required gameplay and authoritative state use
  reliable transport; only disposable presentation details may use unreliable
  delivery. Every client request is narrow, bounded, revisioned, sequenced,
  rate-limited, and validated by the server. Projections expose only the state
  each recipient needs, and detected gaps recover through a compatible snapshot.
- **Question:** What information moves between components and over the network?
- **Must establish:** Typed command/event/snapshot envelopes and representative
  payloads; identity/revision/time/sequence fields; acknowledgments and safe
  errors; snapshot versus delta transport; reliability choices; validation and
  rate limits; duplicate/stale/out-of-order handling; resubscription/resync; and
  public, owner-private, server-only, and telemetry projections.

#### TA-09 - Deterministic rule execution and effects

- **Status:** Resolved 2026-09-10.
- **Decision:** Use integer microseconds at authored-data and runtime-protocol
  boundaries, converting platform audio seconds only at playback edges. Server
  encounters process identified events through pure state transitions in a
  stable approved order with stable identity tie-breakers. The server owns seeded
  random rolls and shared state advancement. Clients may evaluate the same
  rhythm rules for immediate feedback without authorizing shared effects.
- **Question:** How do simultaneous notes, attacks, movement, abilities, and
  Battle Style effects resolve consistently?
- **Must establish:** Scheduler and ordering model; time units and numeric
  precision; phrase/passage boundaries; typed effects and modifier allowlists;
  hook order, caps, and recursion prevention; random-seed/roll ownership; result
  finalization; frozen version inputs; and reproducible reference cases. Preserve
  the resolution orders already defined by the gameplay owners.

### Checkpoint D - Rhythm timing and presentation

#### TA-10 - Musical clock, calibration, and resynchronization

- **Status:** Resolved 2026-09-10.
- **Decision:** The server publishes an identified future start boundary for the
  frozen content revision. Ready clients derive expected encounter time from the
  synchronized server-time estimate while observing their local monotonic clock
  and audio playhead. Filter small estimate errors outside judgment windows
  without changing playback speed or pitch. Unsafe audio/playhead alignment
  suspends local scoring and seeks every layer to a safe authored return point.
  Calibration affects only input-time and visual alignment. Solo pause freezes
  server encounter time and all local layers; resume uses a newly scheduled
  phase-aligned countdown. Exact thresholds remain measured configuration.
- **Question:** How does each device stay aligned with the song and encounter
  while preserving fair timing?
- **Must establish:** Authoritative start/time mapping; local audio and monotonic
  time relationship; calibration application; clock-confidence measurements;
  bounded drift correction; solo pause; device/background interruptions;
  suspension and minimum-preview re-entry; and operational thresholds with
  explicit hypothesis/evidence status.

#### TA-11 - Responsive judgment and server validation

- **Status:** Resolved 2026-09-10.
- **Decision:** The client immediately judges a raw semantic action against its
  frozen chart and displays the result. It sends the bounded action, sequence,
  mapped encounter time, candidate event identity, and synchronization evidence,
  without trusted damage or contribution. The server validates state, revision,
  eligibility, order, rate, timing plausibility, and clock/latency evidence, then
  runs the same matching and normalization rules once for shared consequences.
  Normal delay never retracts the displayed grade; unsafe confidence suspends
  future scoring and returns fairly. Anti-abuse treatment remains private and
  accounts for the server's inability to observe the physical input instant.
- **Question:** How can notes feel immediate while the server validates their
  shared effects?
- **Must establish:** Local judgment versus server acceptance; timestamp and
  sequence validation; bounded delay allowances; early/empty Miss and hold
  handling; duplicates and fabricated/replayed inputs; validation-state
  presentation; uncertainty versus abuse response; and data retention. Explain
  the remaining trust limits honestly and avoid punishing ordinary latency or
  accessibility settings.

#### TA-12 - Audio, VFX, and presentation lifecycle

- **Status:** Resolved 2026-09-10.
- **Decision:** Use Roblox's modular `AudioPlayer`/`Wire` graph behind a game-owned
  adapter as the production candidate, gated by the real-song proof. Play the
  selected role's approved neutral backing bed plus controllable role layers so
  they reconstruct the song without doubling the full mix. Presentation systems
  consume the same semantic events through idempotent instances; critical assets
  load before their scheduling horizon and optional effects degrade first. The
  complete presentation scope owns coordinated pause, seek, rejoin, stop, and
  cleanup. The adapter permits replacing the engine-facing audio implementation
  if target-device verification fails.
- **Question:** How will audio and visual feedback react to gameplay without
  becoming a competing version of the game state?
- **Must establish:** Concrete audio-engine/asset strategy, full mix and
  controllable role rendering without unintended doubling; buses and protected
  cue priority; gain/effect ramps; loading and readiness; safe seek/rejoin/stop;
  semantic UI/VFX/caption/haptic consumers; resource cleanup; and critical versus
  optional failure handling. Bind this selection to TA-06's verification plan.

### Checkpoint E - Multiplayer and encounter hosting

#### TA-13 - Hub, practice, and encounter server topology

- **Status:** Resolved 2026-09-10.
- **Decision:** Use one Roblox experience with a hub place and a shared battle
  runtime place. Every boss package supplies a distinct level, including its
  arena, scenery, lighting, boss rig, animations, anchors, ambience, VFX,
  hazards, music, charts, and timeline. Reserved battle servers host one active
  band/attempt at a time and may retain consenting players through a new staging
  cycle. Practice uses the same runtime in a reserved solo server. A content
  manifest may route an exceptional arena to a dedicated place for terrain,
  streaming, memory, or authoring reasons while preserving common code and
  contracts.
- **Question:** Where do the hub and individual battles live, and what happens
  to a server between battles?
- **Must establish:** Roblox experience/place/server arrangement; public and
  private/reserved sessions; hub and practice capacity; one-to-six-player
  isolation; distinct boss-specific levels and asset ownership; shared runtime
  code; solo pause; repeat attempts and voluntary regrouping; startup and
  termination; and development versus production topology. A shared battle place
  means a common runtime host/loader, never a visually generic arena. Verify
  platform constraints before selecting the mechanism.

#### TA-14 - Parties, matching, and reliable deployment

- **Status:** Resolved 2026-09-10.
- **Decision:** Store expiring party, proposal, queue, match, deployment, and
  reserved-server routes in partitioned MemoryStore structures. Durable player
  eligibility, Loadouts, revisions, and Battle Supply reservations remain in
  Player Data. MemoryStore is authoritative for matchmaking state; best-effort
  MessagingService may only wake cooperating servers. Server-initiated group
  teleport uses an opaque single-use deployment identity. The destination
  authenticates the exact roster/revisions and waits for required load
  acknowledgments before scheduling start. Pre-start failure cancels safely and
  releases reservations idempotently.
- **Question:** How do players form a band and arrive in the same ready battle?
- **Must establish:** Party/queue/proposal state location and expiry; compatible
  matching and region handling; consent and Ready revisions; snapshot lock;
  durable reservations; admission/handoff credentials; destination validation;
  all-member readiness barrier; teleport/loading failure and rollback; and
  duplicate/canceled requests. Never trust client-supplied handoff state.

#### TA-15 - Disconnect, rejoin, and server failure

- **Status:** Resolved 2026-09-10.
- **Decision:** An expiring server-only route permits the same deployment member
  to rejoin the reserved server during remaining cumulative grace. Rejoin
  restores an authoritative snapshot and resumes at a fairly previewed musical
  boundary after audio alignment. Absence supplies no new input or contribution,
  while committed effects remain. Grace expiry closes gameplay rejoin. Terminal
  voluntary departure/inactivity remains Defeat. Loss of the authoritative
  battle server or irrecoverable global state produces No Contest and recovers
  eligible Battle Supply refunds by durable attempt/reservation identity. The
  first release does not reconstruct a partial song on another server.
- **Question:** How does the technical recovery path preserve the already
  approved player experience?
- **Must establish:** Rejoin routing and roster identity; cumulative grace;
  active-roster changes versus local input/timing suspension; return snapshot
  and audio alignment; inactivity handling; accepted actions during disconnect;
  all-departed cleanup; lost-server disposition; No Contest ownership; and
  shutdown/recovery responsibility for unfinished durable work.

### Checkpoint F - Durable data and transactions

#### TA-16 - Profile storage and one-writer authority

- **Status:** Resolved 2026-09-10.
- **Decision:** Store each complete logical player profile in one versioned
  standard DataStore key to retain per-player atomic updates. Use a pinned,
  reviewed ProfileStore dependency behind a game-owned repository interface for
  session locking, ordered saves, and version access. The game owns the schema,
  epochs, revisions, validation, mutation plans, capacity limits, and Save Unsafe
  state. One server session owns the write lease; critical mutations are not
  reported successful until the exact transaction appears in confirmed saved
  data. Teleport transfers ownership through save/end/acquire, while low-risk
  preferences may debounce.
- **Question:** How will we store one reliable player profile across servers and
  devices?
- **Must establish:** Physical services/keys/sections and size budgets; typed
  serialization; load/create/migrate/Ready states; lease/epoch protocol; write
  serialization and revision checks; teleport/takeover ownership handoff;
  save/debounce/retry priorities; capacity handling; and precise Save Unsafe
  behavior. Evaluate libraries against these requirements before selecting one.

#### TA-17 - Rewards, purchases, and recoverable transactions

- **Status:** Resolved 2026-09-10.
- **Decision:** Every durable action produces an immutable, identified plan with
  causal source, expected revisions, frozen rolls, and exact ordered operations.
  Apply it once to the player's profile, store its result in the audit/
  idempotency journal, and confirm the saved version before announcing success.
  Related Battle Supply, reward, currency, item, Progression, and purchase facts
  commit together per player; a group result is not a cross-player transaction.
  Purchase receipt identity and exact grant persist before `PurchaseGranted`;
  uncertainty returns `NotProcessedYet`. Preserve value-relevant identities and
  block new value growth honestly before a profile-capacity limit is exceeded.
- **Question:** How do we prevent lost or duplicated value when a save, receipt,
  or server fails halfway through?
- **Must establish:** Physical atomicity or journal/recovery state machine;
  exact frozen mutations, reward rolls, and identities; receipt verification and
  acknowledgment; Battle Supply reservation/spend/refund; cross-domain commit;
  retry ownership; deduplication retention and storage growth; compensations;
  Results/Pending and subsequent Retry behavior; and crash-point examples.
  Distinguish per-player atomic commits from a band-wide transaction.

#### TA-18 - Migration, recovery, privacy, and account lifecycle

- **Status:** Resolved 2026-09-10.
- **Decision:** Use ordered deterministic/idempotent migrations with invariant
  checks, mock/copy testing, staged rollout, and explicit compatibility rules.
  Quarantine corrupt records and restore through verified prior versions, journal
  replay, and forward repair rather than default replacement. Use DataStore key
  history plus encrypted Open Cloud backups before high-risk migrations and on a
  schedule once public purchases are active. A protected audited admin tool
  supports read/export/deletion and issues corrections as compensation
  transactions. Credentials stay outside the repository; every field declares
  purpose, retention, and lifecycle handling, subject to current pre-release
  platform/policy review.
- **Question:** How will we safely change saved data and handle recovery or
  account-data requests as the game evolves?
- **Must establish:** Versioned migration paths; compatible retired catalog
  references; corruption quarantine; backup/restore and forward repair;
  recovery targets; audited owner/support access; field retention and deletion;
  export and account deletion mechanisms; credential handling; and platform
  responsibilities. Identify current policy dependencies and named release gates
  without claiming an architecture decision is legal approval.

### Checkpoint G - Client experience, assets, and performance

#### TA-19 - UI, input, settings, and localization implementation

- **Status:** Resolved 2026-09-10.
- **Question:** How will the game present one consistent interface across phones
  and PCs?
- **Must establish:** UI technology and components; design tokens and semantic
  view models; screen/focus/navigation lifecycle; action bindings and touch
  layouts; calibration and device profiles; settings drafts versus saved state;
  localization keys; captions/haptics; onboarding and Results; accessibility
  alternatives; and handling unavailable input without changing gameplay rules.

#### TA-20 - World assets and Studio/source ownership

- **Status:** Resolved 2026-09-10.
- **Question:** How will models, arenas, animations, effects, and code move from
  authoring into a reproducible game build?
- **Must establish:** Studio-owned versus filesystem-owned trees; scene and
  asset source files; Rojo import/sync boundaries; stable asset/animation/audio
  references; versions, permissions, and manifest validation; loading/streaming;
  optional quality variants; and how builds combine code with authored places.
  Preserve the existing scene until a deliberate migration is defined.

#### TA-21 - Device, network, and operating budgets

- **Status:** Resolved 2026-09-10.
- **Question:** What measurable limits will keep the game responsive and practical
  to operate?
- **Must establish:** Representative device/network profiles; frame-time,
  memory, audio-layer, asset, loading, server-step, bandwidth, and storage
  budgets; service capacity/cost estimates; measured instrumentation; quality
  reduction order; and overload behavior. Label unmeasured thresholds as targets
  and protect rhythm input, essential cues, and outcome correctness under load.

### Checkpoint H - Verification, operations, and delivery

#### TA-22 - Automated tests and real Roblox verification

- **Status:** Resolved 2026-09-10.
- **Question:** What checks will let implementation agents demonstrate that a
  component works and that the complete game still behaves correctly?
- **Must establish:** Test tools and commands; pure-rule/reference simulations;
  cross-language schema/export fixtures; clock/network fault injection;
  persistence/receipt failure cases; Studio and published multi-client tests;
  physical-device/audio/accessibility checks; performance evidence; and a small
  content-to-encounter-to-saved-result acceptance scenario. Reuse the lightweight
  human playtest process already approved.

#### TA-23 - Analytics, diagnostics, and practical operations

- **Status:** Resolved 2026-09-10.
- **Question:** What minimal tools will tell us when something is broken and
  support a useful playtest or recovery investigation?
- **Must establish:** Exact event schemas and approved projections; collection
  tools; timestamps/revision correlation; sampling/batching/retry/deduplication;
  retention and access; useful queries and alerts; diagnostic builds; support
  actions; and short runbooks for the actual failure paths. Collection failure
  cannot alter gameplay or durable rewards.

#### TA-24 - Build, deployment, rollback, and implementation handoff

- **Status:** Resolved 2026-09-10.
- **Question:** How do we turn this architecture into manageable implementation
  work and safely deliver changes?
- **Must establish:** Development/test/production isolation; reproducible build
  inputs and commands; code/content/balance/asset compatibility; publication
  approval; rollout and rollback, including forward repair of migrated data;
  active-attempt compatibility; feature enablement gates; content-first build
  sequence; agent task/interface ownership; and final cross-spec reconciliation.
  Separate unfinished implementation evidence from unresolved architecture.

## 4. Coverage and final deliverables

| Approved input | Main architecture coverage |
|---|---|
| Content Authoring | TA-01, TA-04 through TA-06, TA-20, TA-24 |
| Rhythm Gameplay | TA-05, TA-08 through TA-12, TA-21, TA-22 |
| Combat; Boss Encounters | TA-07 through TA-09, TA-13 through TA-15 |
| Abilities & Cooperative Actions; Builds & Specialization | TA-05, TA-07 through TA-09, TA-16, TA-22 |
| Progression; Items & Equipment | TA-05, TA-14, TA-16 through TA-18 |
| Multiplayer | TA-08, TA-10, TA-13 through TA-15, TA-21 |
| Rewards & Economy; Player Data | TA-03, TA-16 through TA-18, TA-22 through TA-24 |
| UI/UX; Audio Presentation | TA-06, TA-10 through TA-12, TA-19 through TA-21 |
| Balance Framework | TA-05, TA-09, TA-21 through TA-24 |
| Playtest & Analytics | TA-06, TA-18, TA-21 through TA-24 |
| Naming & Tone; Art Direction | TA-05, TA-12, TA-19, TA-20, TA-24 |

The canonical document is ready only when all 24 questions are resolved and it
contains:

- a concrete component/deployment map and repository ownership layout;
- selected tools/services with current primary references and reasons;
- package, catalog, runtime-message, profile, and transaction contracts with
  enough fields and examples to implement against;
- ordered normal/failure/recovery flows for timing, encounters, deployment,
  durable mutations, and receipt fulfillment;
- explicit security, privacy, capacity, and lifecycle boundaries;
- performance targets distinguished from observed evidence;
- a content-first feasibility plan and test/release gates, with unresolved
  platform feasibility dependencies clearly identified;
- a phased handoff suitable for the later implementation plan and bounded agent
  tasks; and
- reconciled references in the systems map, tracker, READMEs, and `AGENTS.md`.

An unknown choice that changes interfaces, storage correctness, or required
player behavior remains open. A chosen approach awaiting its prescribed runtime
test is recorded as selected but unverified and gates dependent implementation
or release; it is never described as proven.

## 5. Change log

- **2026-09-10:** Created the 24-question plan and working record from the approved
  design inputs and existing repository tooling. Opened checkpoint A.
- **2026-09-10:** Approved TA-01 through TA-03, retaining the repository and
  existing language/tool boundaries while preferring Roblox-managed runtime
  services and local content tooling. Progress is 3 of 24; opened checkpoint B.
- **2026-09-10:** Approved TA-04 through TA-06, establishing the local authoring
  application, immutable package/export boundary, and real-song Roblox proof as
  the first implementation milestone. Progress is 6 of 24; opened checkpoint C.
- **2026-09-10:** Approved TA-07 through TA-09, establishing runtime authority,
  typed communication families, and deterministic event resolution. Progress is
  9 of 24; opened checkpoint D.
- **2026-09-10:** Approved TA-10 through TA-12, establishing the synchronized
  musical clock, immediate/client and validated/server judgment flow, and modular
  layered presentation candidate. Progress is 12 of 24; opened checkpoint E.
- **2026-09-10:** Approved TA-13 through TA-15 with the explicit distinct-level/
  shared-runtime boundary, establishing matchmaking, atomic deployment, rejoin,
  and server-failure behavior. Progress is 15 of 24; opened checkpoint F.
- **2026-09-10:** Approved TA-16 through TA-18, establishing one-writer player
  profiles, confirmed atomic per-player transactions, and migration/recovery/
  privacy operations. Progress is 18 of 24; opened checkpoint G.
- **2026-09-10:** Approved TA-19 through TA-21, establishing native client UI,
  semantic cross-device input, exact-version scene/asset ownership, and measured
  performance budgets. Progress is 21 of 24; opened checkpoint H.
- **2026-09-10:** Approved TA-22 through TA-24, completing the interview at 24
  of 24. Published and reconciled canonical `TECHNICAL_ARCHITECTURE.md`.
