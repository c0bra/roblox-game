# Technical Architecture Working Record

- **Status:** Complete; 24 of 24 questions resolved
- **Started:** 2026-09-10
- **Current checkpoint:** Complete
- **Question plan:** [`TECHNICAL_ARCHITECTURE_QUESTIONS.md`](TECHNICAL_ARCHITECTURE_QUESTIONS.md)
- **System authority:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md)
- **Design inputs:** [`SYSTEM_SPECIFICATIONS_TRACKER.md`](SYSTEM_SPECIFICATIONS_TRACKER.md)
- **Canonical result:** [`TECHNICAL_ARCHITECTURE.md`](TECHNICAL_ARCHITECTURE.md)

## 1. Role of this record

This file preserves the architecture interview as it progresses. Approved design
rules remain in their owning specifications. Candidate technical choices remain
proposals until accepted or selected within an accepted engineering constraint.
Resolved entries are approved architecture decisions; open entries remain
proposals until the owner accepts or revises them.

At each checkpoint, record the owner's answer, concrete decision, rationale,
implementation consequences, open dependencies, and validation needed. Preserve
the distinction between repository observations, documented platform capability,
and an approach actually tested in this game.

## 2. Repository observations

Inspected 2026-09-10:

| Surface | Observed foundation | Architecture consequence |
|---|---|---|
| [`roblox/README.md`](roblox/README.md) and [`roblox/default.project.json`](roblox/default.project.json) | Existing Luau source under client/server/shared trees; Rojo sync preserves unknown Studio instances | There is an existing Roblox project to evolve; production architecture is not yet established by this skeleton |
| [`roblox/rokit.toml`](roblox/rokit.toml) | Rojo is pinned to 7.7.0 | Evaluate the existing toolchain before changing it; a checked-in version is not a claim that it is the latest |
| [`roblox/web/package.json`](roblox/web/package.json) | Browser prototype uses TypeScript, Bun/Vite, and Babylon.js and links the chart pipeline | It can provide experiments and reference assets/behavior; production reuse requires spec-level review |
| [`tools/chart-pipeline/README.md`](tools/chart-pipeline/README.md) and [`package.json`](tools/chart-pipeline/package.json) | Separate TypeScript/Bun pipeline, validation commands, platform-neutral bundles, and external audio tools | Extend the maintained content foundation; local authoring need not live inside Roblox |
| Chart pipeline documented output | Four fixed role categories, `easy/medium/hard`, clip-relative chart time with a source offset | TA-05 must map these to extensible roles, Easy/Normal/Hard, canonical transcription, and the approved master-origin musical clock |
| [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md) | Existing Blackened Crown charts are derived runtime examples, not complete editable source packages | A schema/export migration is required; generated examples do not supersede the approved content contract |
| Existing Luau runtime skeleton | Rhythm input, judgment, scoring, UI, and audio currently run in one client script; the server entry point only announces startup | Treat it as prototype evidence and split production authority, pure rules, and presentation through TA-07 through TA-09 |
| Root README | Its former current-reality sentence denied the now-existing Roblox project | Corrected the entry-point description during this interview setup |

These are observations from manifests, file inventory, and documentation. No
runtime code audit, audio proof, performance benchmark, or persistence guarantee
has been completed as part of this setup.

## 3. Checkpoint progress

| Checkpoint | Questions | State | Resolved |
|---|---|---|---:|
| A - Development foundation | TA-01 through TA-03 | Approved 2026-09-10 | 3 of 3 |
| B - Content foundation and publication | TA-04 through TA-06 | Approved 2026-09-10 | 3 of 3 |
| C - Runtime components and contracts | TA-07 through TA-09 | Approved 2026-09-10 | 3 of 3 |
| D - Rhythm timing and presentation | TA-10 through TA-12 | Approved 2026-09-10 | 3 of 3 |
| E - Multiplayer and encounter hosting | TA-13 through TA-15 | Approved 2026-09-10 | 3 of 3 |
| F - Durable data and transactions | TA-16 through TA-18 | Approved 2026-09-10 | 3 of 3 |
| G - Client experience, assets, and performance | TA-19 through TA-21 | Approved 2026-09-10 | 3 of 3 |
| H - Verification, operations, and delivery | TA-22 through TA-24 | Approved 2026-09-10 | 3 of 3 |

## 4. Decision record

### Checkpoint A - Development foundation

#### TA-01 - Existing prototypes and production boundaries

- **Status:** Resolved 2026-09-10.
- **Proposal:** Continue in the existing repository and Roblox project. Keep the
  browser prototype available for experiments and reference, without committing
  to ship and maintain a parallel browser game. Reuse components after checking
  them against the approved specs. Extend the separate chart pipeline.
- **Rationale:** The repository already separates these surfaces. This preserves
  useful work and leaves room to replace prototype assumptions that conflict
  with the approved design.
- **Owner answer:** The proposed boundary makes sense. Continue with the existing
  repository and Roblox project, keep the browser demo as a reference/experiment,
  and extend the chart pipeline.
- **Follow-through:** Identify code/asset reuse and migration boundaries during
  subsequent component and publication decisions. This proposal does not
  authorize deleting prototypes or unrelated work.

#### TA-02 - Languages, tools, and dependency approach

- **Status:** Resolved 2026-09-10.
- **Proposal:** Keep Luau and Rojo for Roblox, TypeScript/Bun for the local web
  tools and pipeline coordination, and the useful Python/external audio
  processors. Prefer explicit typed modules and select libraries where they
  reduce real implementation or maintenance work.
- **Rationale:** These technologies already have distinct jobs in the repository;
  keeping those boundaries reduces avoidable migration work.
- **Owner answer:** The proposed tool foundation makes sense. Retain Luau/Rojo,
  TypeScript/Bun, and the useful audio-processing tools in their stated roles.
- **Follow-through:** Specify source ownership, type/schema consistency, pinned
  tools, UI choice, persistence adapters, and test harnesses in later checkpoints.
  No new framework is selected by this proposal.

#### TA-03 - Hosting, operating burden, and cost constraints

- **Status:** Resolved 2026-09-10.
- **Proposal:** Prefer Roblox-managed runtime services and run the authoring app
  locally. Add an external live backend only for a demonstrated requirement.
- **Rationale:** Roblox exposes durable and temporary server-side storage
  services. They are candidates for the required design, not proof that every
  transaction or recovery mechanism has already been solved.
- **Owner answer:** The proposed low-operations direction makes sense. Prefer
  Roblox-managed live services and a local authoring app; no external backend or
  paid-service commitment is approved by this answer.
- **Follow-through:** Verify storage/queue limits, physical transaction design,
  data lifecycle, observability, and operating costs at their checkpoints. Raise
  a concrete gap if these require revisiting the hosting direction.

### Checkpoint B - Content foundation and publication

#### TA-04 - Local authoring application and processing jobs

- **Status:** Resolved 2026-09-10.
- **Proposal:** Run a TypeScript/Bun application locally and open its UI in a web
  browser. The local process owns directory access and invokes the existing
  pipeline; the browser provides synchronized playback, timeline inspection,
  editing, validation, and explicit rerun/cancel/approve actions. Keep human
  edits separate from generated suggestions so reprocessing can mark dependent
  work stale without overwriting it.
- **Practical consequence:** The owner starts one local tool and selects a song
  project directory. It can load any declared stem/role set, display job progress
  and errors, and rerun individual processing stages as needed.
- **Owner answer:** Approved as proposed.

#### TA-05 - Schemas, catalogs, and Roblox export

- **Status:** Resolved 2026-09-10.
- **Proposal:** Publish one immutable versioned package per song/encounter
  revision. Its manifest binds the master and controllable audio roles, musical
  clock, canonical transcriptions, difficulty charts, Activity Maps, encounter
  timeline, catalog/balance revisions, localization references, assets, approval,
  and integrity hashes. Generate validated Roblox-readable data plus uploaded
  asset references from that package; the platform-neutral package remains the
  source of truth.
- **Practical consequence:** A Roblox battle loads one exact compatible revision
  rather than gathering unrelated current files. Add a legacy importer that maps
  `medium` to Normal and explicitly converts clip-relative timing to the approved
  master-origin clock; role names remain data rather than a fixed four-stem enum.
- **Owner answer:** Approved as proposed.

#### TA-06 - Early feasibility proof and content-first milestone

- **Status:** Resolved 2026-09-10.
- **Proposal:** Before wider gameplay implementation, complete enough of the
  authoring/export path to publish one real song package, then exercise it in a
  narrow Roblox vertical slice. The slice loads the real master and controllable
  roles, follows the approved musical clock, renders a Normal chart with local
  timing feedback, seeks/resynchronizes safely, and aligns two clients in a
  published Roblox test. Verify at least one representative phone as well.
- **Practical consequence:** This is the first architecture milestone and a gate
  for systems that depend on live song timing. Failure sends the architecture
  back to audio/package/timing choices before a larger combat stack is built.
- **Owner answer:** Approved as proposed.

### Checkpoint C - Runtime components and contracts

#### TA-07 - Client, server, and shared component layout

- **Status:** Resolved 2026-09-10.
- **Proposal:** Put authoritative encounter state, combat resolution, cooperative
  actions, rewards, session decisions, and durable mutations on the server. The
  client owns input collection, local audio/clock playback, immediate rhythm
  feedback, UI, camera, VFX, captions, and haptics. Shared modules contain typed
  data contracts, immutable content definitions, and pure deterministic rules
  that both sides may evaluate; they never contain secrets or authoritative
  mutable state.
- **Practical consequence:** Replace the current all-in-one client prototype
  gradually. Organize production code by domain and runtime side, with small
  startup/composition roots and encounter-scoped cleanup rather than creating one
  Roblox service for every design-document heading.
- **Owner answer:** Approved as proposed.

#### TA-08 - Commands, events, and replicated state

- **Status:** Resolved 2026-09-10.
- **Proposal:** Define a typed protocol registry with three distinct shapes:
  client intents, server-confirmed semantic events, and versioned snapshots for
  initial load or recovery. Use reliable remotes for gameplay commands and
  authoritative state. Permit unreliable transport only for disposable
  presentation details whose loss cannot change gameplay, consent, recovery, or
  player understanding.
- **Practical consequence:** Every client request has a narrow action, bounded
  data, identity, sequence, and revision context. The server validates type,
  size, rate, timing, permissions, and current state before applying it. Clients
  receive only public or owner-private projections and request a new snapshot
  when a gap or incompatible revision is detected.
- **Owner answer:** Approved as proposed.

#### TA-09 - Deterministic rule execution and effects

- **Status:** Resolved 2026-09-10.
- **Proposal:** Represent authored and logical gameplay time as integer
  microseconds at data/protocol boundaries, converting Roblox audio seconds only
  at the playback edge. Each server encounter uses an ordered event queue and
  pure state transitions: frozen state plus one identified input produces a new
  state and typed semantic effects. Ties use the approved domain phase order,
  followed by stable source/event identity. The server owns seeded random rolls.
- **Practical consequence:** The client may run the same rhythm calculation for
  instant feedback, while only the server advances shared combat. Replaying the
  same frozen inputs produces the same result, enabling tests, deduplication,
  failure investigation, and fair simultaneous-action handling without building
  the entire game as an event-sourced database.
- **Owner answer:** Approved as proposed.

### Checkpoint D - Rhythm timing and presentation

#### TA-10 - Musical clock, calibration, and resynchronization

- **Status:** Resolved 2026-09-10.
- **Proposal:** The server publishes an identified future start boundary and the
  frozen content revision. Each ready client derives expected encounter time from
  Roblox's synchronized server-time estimate while observing its local monotonic
  clock and audio playhead. Begin only after protected audio is ready. Filter
  small clock-estimate errors gradually outside open judgment windows; never
  change song playback speed or pitch. Actual audio/playhead misalignment beyond
  the validated tolerance suspends that player's scoring and seeks all layers to
  a safe authored return point before fair preview resumes.
- **Practical consequence:** Calibration adjusts only input-time and visual-
  alignment interpretation. It does not move audio, chart, or boss events. A solo
  pause freezes server encounter time and all local layers at one point; resume
  uses a newly scheduled phase-aligned countdown. Exact tolerances remain
  versioned hypotheses until the real-song, network, and phone tests measure
  them.
- **Owner answer:** Approved as proposed.

#### TA-11 - Responsive judgment and server validation

- **Status:** Resolved 2026-09-10.
- **Proposal:** The client immediately matches a raw semantic action against its
  frozen chart and shows the result. It sends action kind/lane, local sequence,
  mapped encounter timestamp, candidate event identity, and sync evidence, but
  never sends damage or a trusted contribution total. The server validates the
  player's state, content revision, event eligibility, sequence, rate, timing
  plausibility, and measured clock/latency envelope, then runs the same matching
  and normalization rules once for shared consequences.
- **Practical consequence:** Normal delay does not retract the displayed grade.
  Duplicates cannot apply twice. Isolated ambiguous timing is treated as a sync/
  validation issue rather than public misconduct; unsafe confidence suspends
  future scoring and returns fairly. Aggregate impossible patterns can inform
  private anti-abuse review. This acknowledges that a server cannot directly
  observe the physical instant a player touched their screen.
- **Owner answer:** Approved as proposed.

#### TA-12 - Audio, VFX, and presentation lifecycle

- **Status:** Resolved 2026-09-10.
- **Proposal:** Use Roblox's modular `AudioPlayer`/`Wire` audio graph behind a
  small game-owned adapter as the production candidate, subject to the TA-06
  real-song proof. For a selected role, play its approved neutral backing bed
  plus controllable role layer(s), which reconstruct the song without doubling a
  full mix. The full mix remains a package reference and explicitly approved
  fallback, not an additional simultaneously playing layer. Build local buses for
  Song, Local Role, protected cues, voices, combat, crowd, and ambience, with
  priority, voice, headroom, and degradation rules from Audio Presentation.
- **Practical consequence:** UI, VFX, audio, captions, and haptics consume the
  same semantic event stream through idempotent presentation instances. Critical
  assets load before their scheduling horizon; optional effects degrade first.
  Stop, pause, seek, rejoin, and cleanup apply to the whole encounter presentation
  scope. If the modular audio API cannot demonstrate stable aligned layers and
  safe seeking on target devices, the adapter permits changing the engine-facing
  implementation before dependent gameplay is built.
- **Owner answer:** Approved as proposed.

### Checkpoint E - Multiplayer and encounter hosting

#### TA-13 - Hub, practice, and encounter server topology

- **Status:** Resolved 2026-09-10.
- **Owner clarification:** Every boss encounter should have its own distinct
  level, aesthetics, geometry, scenery, lighting, boss, hazards, and authored
  design. A shared place must not imply a generic player-visible arena.
- **Proposal:** Use one Roblox experience with a hub place and a shared battle
  runtime place. The shared place is a host/loader for common gameplay code, not
  the arena itself. It loads the selected immutable boss package, including that
  boss's distinct arena scene, lighting/post-processing profile, boss rig,
  animations, camera and spawn anchors, ambience, VFX, arena graph, hazards,
  music, charts, and encounter timeline. Run each active band in a reserved
  battle server with one to six humans. Practice uses the same runtime place in a
  reserved solo session, preserving identical timing and gameplay code while
  permitting exact solo pause.
- **Practical consequence:** A reserved battle server hosts one active attempt at
  a time and may retain consenting players through Results and a new staging
  cycle for Retry or Stay with Band. It may accept replacements only while
  staging; the active-song roster stays locked. Development and test places
  remain access-restricted from production. The content manifest may route an
  exceptional boss to a dedicated place when terrain, streaming, memory, or
  authoring constraints make dynamic scene loading unsuitable; that place still
  uses the same runtime contracts and shared gameplay code.
- **Owner answer:** The clarified shared-runtime/distinct-level arrangement makes
  sense. Approve the proposal with the dedicated-place escape hatch.

#### TA-14 - Parties, matching, and reliable deployment

- **Status:** Resolved 2026-09-10.
- **Proposal:** Keep ephemeral party, proposal, queue, match, deployment, and
  reserved-server routing records in partitioned Roblox MemoryStore structures
  with expiry. Use durable Player Data only for required Loadout, eligibility,
  reservation, and revision facts. Hub servers perform or cooperate in matching;
  best-effort MessagingService signals may wake them, while MemoryStore state
  remains the authority. A server calls `TeleportAsync` for the locked group into
  a reserved battle server.
- **Practical consequence:** Teleport data carries only nonsecret display hints
  and an opaque, single-use deployment identity. The destination consumes the
  server-held deployment record, authenticates each exact member and revision,
  waits for all required client-load acknowledgments, and then schedules the
  shared start. Pre-start failure expires or cancels the deployment,
  idempotently releases Battle Supply reservations, and returns reachable players
  to staging.
- **Owner answer:** Approved as part of the clarified multiplayer set.

#### TA-15 - Disconnect, rejoin, and server failure

- **Status:** Resolved 2026-09-10.
- **Proposal:** Store a short-lived server-only rejoin route and deployment/
  attempt record in MemoryStore. A returning player receives a new one-time
  admission identity, rejoins the same reserved battle server during remaining
  cumulative grace, receives a current authoritative snapshot, seeks audio, and
  resumes only at a safe musical boundary with adequate preview. The server keeps
  already committed effects and applies no inputs or contribution during absence.
- **Practical consequence:** Grace expiry closes gameplay rejoin. All players
  voluntarily departed or terminally inactive produces the already-approved
  gameplay Defeat. Loss of the authoritative battle server or irrecoverable
  global attempt state produces No Contest; a durable recovery operation uses
  attempt and reservation identities to refund eligible Battle Supplies. The
  first release does not reconstruct and continue a partially completed song on
  another server. Pending value transactions recover separately and cannot be
  acknowledged twice.
- **Owner answer:** Approved as part of the clarified multiplayer set.

### Checkpoint F - Durable data and transactions

#### TA-16 - Profile storage and one-writer authority

- **Status:** Resolved 2026-09-10.
- **Proposal:** Keep each player's complete logical profile in one versioned
  standard DataStore key so one `UpdateAsync` can commit changes across its
  progression, inventory, configuration, settings, currency, transaction, and
  receipt sections atomically. Use a pinned, reviewed copy of the Apache-2.0
  [ProfileStore](https://github.com/MadStudioRoblox/ProfileStore) module behind a
  game-owned `ProfileRepository` adapter for loading, session locking, ordered
  saves, and version access. Our own record schema, epoch, revisions, mutation
  plans, validation, capacity rules, and Save Unsafe state remain authoritative;
  the dependency does not define game meaning.
- **Practical consequence:** Exactly one server session owns a player's write
  lease. Hub-to-battle teleport saves the reservation/snapshot, ends the hub
  session, and lets the battle server acquire the next epoch. Critical mutations
  request an immediate save and wait until the exact transaction identity appears
  in confirmed saved data before the UI reports success. Low-risk settings may
  debounce. Loss of the session or repeated storage failure immediately blocks
  further durable actions and exposes Save Unsafe.
- **Owner answer:** Approved as proposed.

#### TA-17 - Rewards, purchases, and recoverable transactions

- **Status:** Resolved 2026-09-10.
- **Proposal:** Every durable action first produces an immutable transaction plan
  containing its ID, causal attempt/receipt, expected revisions, frozen random
  rolls, and exact ordered operations. Apply the plan once to the single player
  profile, append its result to the audit/idempotency journal, save, and confirm
  that saved version before announcing completion. Battle Supply reservation,
  spend/refund, reward, currency, item, progression, and purchase lineage commit
  together for that player.
- **Practical consequence:** A band's Results screen comprises separate atomic
  per-player commits; one player's storage delay never revokes another player's
  reward. A duplicate returns the recorded result. Roblox purchase fulfillment
  stores the permanent receipt identity and exact grant in the same commit, then
  returns `PurchaseGranted` only after confirmed storage; uncertainty returns
  `NotProcessedYet`. Journals compact only facts that can no longer affect value,
  while receipt identities remain. Near a safe profile-size ceiling, new
  purchases/value growth become honestly unavailable until migration expands
  capacity rather than risking loss.
- **Owner answer:** Approved as proposed.

#### TA-18 - Migration, recovery, privacy, and account lifecycle

- **Status:** Resolved 2026-09-10.
- **Proposal:** Use an ordered registry of deterministic, idempotent schema
  migrations with pre/post invariant checks. Test migrations against copied or
  mock profiles and stage rollout before production. Corrupt records enter
  quarantine; recovery selects a verified prior DataStore version, replays later
  committed journal facts, and writes a forward repair rather than silently
  resetting the player. Retired catalog references require an approved exact
  equivalence, player choice, or compensation.
- **Practical consequence:** Use Roblox key-version history for immediate
  recovery and make encrypted Open Cloud exports before destructive/high-risk
  migrations, plus a scheduled backup once public purchases are enabled. A
  small authenticated admin tool supports audited read/export/deletion and
  submits corrections as ordinary idempotent compensation transactions rather
  than editing raw values. Secrets remain outside the repo; each stored field has
  an owner, purpose, retention class, and deletion/export behavior. Current
  Roblox privacy, child-safety, commerce, retention, and deletion requirements
  remain explicit pre-release gates.
- **Owner answer:** Approved as proposed.

### Checkpoint G - Client experience, assets, and performance

#### TA-19 - UI, input, settings, and localization implementation

- **Status:** Resolved 2026-09-10.
- **Proposal:** Use pinned React Lua/ReactRoblox for native Roblox screen, menu,
  preparation, settings, onboarding, Results, and stable HUD composition. Keep
  high-frequency note-lane animation and VFX in specialized pooled renderers so
  they do not rerender the whole UI every frame. UI components consume typed
  semantic view models and design tokens; they do not query unrelated services or
  decide gameplay legality. Use Roblox `InputAction`/`InputBinding` through a
  game-owned semantic input adapter for touch, keyboard/mouse, and gamepad.
- **Practical consequence:** One action catalog drives bindings, displayed
  prompts, remapping, active-device changes, and fixed touch controls. Timestamp
  rhythm input at the adapter boundary. Settings edit a recoverable local draft,
  preview presentation-only values, and display Saved only after the appropriate
  profile confirmation. Generate stable localization keys and Roblox localization
  tables from repository-owned string definitions; permit approximately 30%
  expansion and keep captions, shapes, labels, focus, scale, contrast, reduced
  motion, haptics, and safe areas as component acceptance states.
- **Owner answer:** Approved as proposed.

#### TA-20 - World assets and Studio/source ownership

- **Status:** Resolved 2026-09-10.
- **Proposal:** Let Git/Rojo own Luau code, schemas, generated content data,
  localization source, and configuration. Let versioned Studio place files own
  hand-authored terrain, lighting, and the hub/battle runtime shell. Retain DCC
  source such as Blender files and textures in the repository. Publish each
  boss arena and reusable art set as creator-owned Roblox packages/models without
  gameplay scripts, and pin exact approved asset/package versions in the content
  manifest; never use automatic latest-version selection for an active revision.
- **Practical consequence:** Extend the existing `$ignoreUnknownInstances` Rojo
  boundary instead of allowing source sync to overwrite Studio scenes. A release
  record binds Git commit, place version, content/balance revisions, and every
  audio/model/animation/image asset ID/version. Publication validates ownership,
  experience permissions, moderation state, hashes, required anchors/tags, script
  absence, loading, and rollback. Dynamic arena loading uses owned exact-version
  assets behind a loader adapter and falls back to the already-approved dedicated
  boss-place route when an arena cannot be packaged safely.
- **Owner answer:** Approved as proposed.

#### TA-21 - Device, network, and operating budgets

- **Status:** Resolved 2026-09-10.
- **Proposal:** Treat 60 fps as the standard presentation target and a stable
  30 fps as the supported lower-device profile, while keeping judgment based on
  captured input time rather than render frames. The real-song vertical slice
  measures representative lower-end phone, ordinary phone/tablet, and desktop/
  gamepad profiles and establishes numeric caps for frame/CPU spikes, memory,
  decoded/compressed audio, concurrent audio players/voices, instances, graphics,
  initial/battle load time, server work, remote traffic, and DataStore/
  MemoryStore use before the first boss package is approved.
- **Practical consequence:** Packages publish measured budget use and fail
  validation when they do not fit a supported profile. Replication remains
  event-driven rather than per-frame; service-capacity estimates cover expected
  concurrency and a tested throttle margin. Under pressure, reduce crowd,
  ambience, decorative spatial/reverb detail, optional UI/world variation,
  particles, shadows, textures, and ordinary-combat variation in that order.
  Never reduce or retime song/role playback, rhythm input, the musical clock,
  committed danger/target/recovery/group cues, captions, or required alternatives.
- **Owner answer:** Approved as proposed.

### Checkpoint H - Verification, operations, and delivery

#### TA-22 - Automated tests and real Roblox verification

- **Status:** Resolved 2026-09-10.
- **Proposal:** Use three complementary verification layers. Fast local tests
  cover pure Luau rules, TypeScript authoring/export code, schema validation,
  migrations, and shared golden fixtures. Studio tests cover client/server
  boundaries, virtual input, device profiles, and up to eight simulated clients.
  A private published staging experience covers the platform behavior Studio
  cannot prove, including teleports, reserved servers, DataStore/MemoryStore,
  asset permissions, layered audio, and physical-device timing.
- **Practical consequence:** Maintain deterministic fault cases for clock drift,
  latency, duplicate/out-of-order messages, disconnect/rejoin, failed content
  loading, throttled saves, and uncertain receipts. The release acceptance path
  takes one approved real-song source through export, boss deployment, completed
  encounter, confirmed reward save, reconnect, and profile verification. Mobile,
  desktop/gamepad, audio, accessibility, and performance checks remain explicit
  human evidence rather than being inferred from unit tests.
- **Owner answer:** Approved as proposed.

#### TA-23 - Analytics, diagnostics, and practical operations

- **Status:** Resolved 2026-09-10.
- **Proposal:** Put a typed, game-owned telemetry adapter behind the semantic
  events already approved by Playtest & Analytics. Servers emit authoritative
  facts to Roblox AnalyticsService in published environments and write bounded
  structured diagnostic records; clients may submit only validated, rate-limited
  diagnostic observations. Do not add an external telemetry backend for the
  first playable release unless real investigations show that Roblox's tools and
  the local playtest workflow are insufficient.
- **Practical consequence:** Correlate records with release, server, deployment,
  attempt, content, balance, and transaction revisions while avoiding raw audio,
  chat, detailed input streams, and unnecessary identity. Collection is sampled
  or aggregated where appropriate, deduplicated at the adapter, and safely
  dropped when unavailable; it never gates gameplay or rewards. Initial
  dashboards watch content-load failure, No Contest, unsafe timing, Save Unsafe,
  pending receipts, and error rates. Short runbooks cover those same failures and
  route any repair through the audited tools and transactions approved in TA-18.
- **Owner answer:** Approved as proposed.

#### TA-24 - Build, deployment, rollback, and implementation handoff

- **Status:** Resolved 2026-09-10.
- **Proposal:** Keep separate development, private staging, and production Roblox
  experiences with separate credentials and data. Build a release from an exact
  Git commit, approved Studio place base, pinned dependencies, and immutable
  code/content/balance/asset manifests. Begin with a human-supervised Studio/Rojo
  assembly and validation step because the existing authored place is not yet a
  reproducible source-only Rojo build; automate publication through Open Cloud
  only after that assembly path is proven and any unsupported instance types are
  accounted for. Production publication always requires explicit human approval.
- **Practical consequence:** Promote the same validated release inputs through
  staging and production, smoke-test a new production server, and enable new
  content only for new attempts. Rollback selects a previously approved place,
  code, and content release; saved-data schemas only move forward and use repair
  migrations rather than rollback. An active attempt remains pinned to its
  compatible revision or ends as No Contest. Implementation starts with the
  content package/export contract and real-song vertical slice, then proceeds in
  bounded tasks that name owned interfaces, excluded work, tests, and observable
  acceptance evidence.
- **Owner answer:** Approved as proposed.

## 5. Platform reference notes

Consulted 2026-09-10 for the initial candidate direction:

- [Rojo project format](https://rojo.space/docs/v7/project-format/) and
  [sync details](https://rojo.space/docs/v7/sync-details/) document filesystem to
  Studio mappings. The checked-in project determines this repository's current
  mapping; publishing a source tree is not automatically a complete scene build.
- [Roblox script types and locations](https://create.roblox.com/docs/scripting/locations)
  documents client/server execution and shared ModuleScript locations.
- [Roblox client/server security guidance](https://create.roblox.com/docs/scripting/security/client-server-boundary)
  requires server validation of client-triggered state, including context,
  structure, values, and rate. The separate
  [security tactics guide](https://create.roblox.com/docs/scripting/security/security-tactics)
  warns that replicated client and ModuleScript code is inspectable.
- [Roblox remote events and callbacks](https://create.roblox.com/docs/scripting/events/remote)
  describes unreliable remotes as appropriate for continuously changing or
  noncritical data, supporting the boundary that required semantic state remains
  on reliable transport.
- [`Workspace:GetServerTimeNow()`](https://create.roblox.com/docs/reference/engine/classes/Workspace#GetServerTimeNow)
  supplies a smoothed monotonic client estimate of server time intended for
  synchronized experiences. Its documented approximation still requires game-
  specific drift/confidence measurement rather than becoming the audio clock by
  assumption.
- [Roblox audio objects](https://create.roblox.com/docs/audio/objects) and the
  [`AudioPlayer` reference](https://create.roblox.com/docs/reference/engine/classes/AudioPlayer)
  provide modular playback, routing, readiness, position, and scheduled play/stop
  capabilities. TA-06 must verify aligned multi-layer behavior in this game.
- [`ContentProvider`](https://create.roblox.com/docs/reference/engine/classes/ContentProvider)
  supports preload/fetch status for essential assets while Roblox recommends
  against preloading an entire world. Architecture still needs explicit required
  audio readiness and timeout behavior.
- [Roblox teleport guidance](https://create.roblox.com/docs/projects/teleporting)
  supports server-initiated group teleports and reserved servers, and explicitly
  classifies teleport data as visible to clients and unsuitable for secure state.
- [Roblox's cloud-service comparison](https://create.roblox.com/docs/cloud-services/data-stores-vs-memory-stores)
  identifies MemoryStore as temporary cross-server state suitable for
  matchmaking. Its state expires and remains subject to capacity and throttling,
  so durable reservations stay in Player Data.
- [`MessagingService`](https://create.roblox.com/docs/reference/engine/classes/MessagingService)
  is best-effort cross-server notification, not authoritative queue or deployment
  storage. Matchmaking must remain correct when a signal is late or absent.
- [Roblox cloud-service comparison](https://create.roblox.com/docs/cloud-services/data-stores-vs-memory-stores),
  [Data stores](https://create.roblox.com/docs/cloud-services/data-stores), and
  [Memory stores](https://create.roblox.com/docs/cloud-services/memory-stores)
  distinguish durable from temporary storage and server access. Concrete leases,
  atomicity, capacity, receipt, and recovery choices remain open.
- [Roblox DataStore guidance](https://create.roblox.com/docs/cloud-services/data-stores)
  documents per-key `UpdateAsync`, key versions, and the risk of multi-server
  `SetAsync` writes. Atomicity in this plan therefore stays inside one player's
  profile key until an explicit journaled sharding migration is required.
- [ProfileStore](https://github.com/MadStudioRoblox/ProfileStore) is an Apache-2.0
  DataStore wrapper with session locking. Its
  [API](https://madstudioroblox.github.io/ProfileStore/api/) exposes immediate
  saves and confirmed `LastSavedData`; the game adapter must still enforce the
  stronger schema, epoch, transaction, and Save Unsafe contract.
- [`MarketplaceService.ProcessReceipt`](https://create.roblox.com/docs/reference/engine/classes/MarketplaceService#ProcessReceipt)
  is the server receipt boundary. Roblox's purchase guidance makes the developer
  responsible for durable user-specific purchase history, so acknowledgment is
  gated on a confirmed receipt/grant transaction.
- [Roblox Open Cloud](https://create.roblox.com/docs/cloud) permits authenticated
  operational access to DataStores. Backup, export, deletion, and support tools
  must use narrowly scoped credentials and audited operations.
- [React Lua](https://github.com/jsdotlua/react-lua) provides the maintained
  React/ReactRoblox renderer candidate under MIT terms. Pin its exact reviewed
  revision and keep frame-critical rhythm rendering outside broad UI rerenders.
- Roblox [`InputAction`](https://create.roblox.com/docs/reference/engine/classes/InputAction)
  and [`InputBinding`](https://create.roblox.com/docs/reference/engine/classes/InputBinding)
  provide semantic actions with hardware and touch bindings. The game adapter
  retains one action contract and enables capability testing or replacement.
- [Roblox localization](https://create.roblox.com/docs/production/localization)
  supports cloud tables, programmatic lookup, and CSV exchange. Repository-owned
  stable keys remain the review/source contract before upload.
- [Roblox packages](https://create.roblox.com/docs/projects/assets/packages)
  support asset grouping and version history. Runtime content pins approved
  versions; package auto-update cannot change an active content revision.
- [`InsertService:LoadAssetVersion`](https://create.roblox.com/docs/reference/engine/classes/InsertService#LoadAssetVersion)
  can load a creator-accessible exact model version. The boss loader must treat
  failure as a content-readiness failure and validate that loaded content contains
  no executable scripts.
- [Roblox performance guidance](https://create.roblox.com/docs/performance-optimization)
  identifies frame time, memory, join time, server heartbeat, and network traffic
  as measurable constraints. Exact game budgets remain target-device evidence,
  not values inferred from documentation.
- [Studio testing modes](https://create.roblox.com/docs/studio/testing-modes)
  provide separate client/server simulation, multi-client sessions, persistent
  device profiles, virtual input, and Studio-only programmable testing services.
  Published-platform behavior and physical audio timing still require staging and
  device evidence.
- [Roblox custom events](https://create.roblox.com/docs/production/analytics/custom-events)
  are server-emitted, published-game signals with bounded names and fields. They
  support aggregate playtest and health views, not authoritative gameplay state
  or lossless incident traces.
- [Open Cloud place publishing](https://create.roblox.com/docs/cloud/guides/usage-place-publishing)
  can publish an existing binary or XML place artifact under scoped credentials,
  but documents instance-type limitations. It is a publication mechanism, not a
  substitute for proving how this repository assembles the complete place.

Recheck the specific primary documentation when selecting each implementation
mechanism. A source reference establishes documented behavior, not measured game
behavior or an assurance about future platform policy.

## 6. Completion

All 24 architecture questions are resolved. The approved decisions are
reconciled into [`TECHNICAL_ARCHITECTURE.md`](TECHNICAL_ARCHITECTURE.md), which
now governs implementation. This record remains as interview history.

## 7. Change log

- **2026-09-10:** Created the interview record and 24-question plan. Recorded
  existing tooling, known content-contract gaps, and three foundation proposals.
  Progress remained 0 of 24; no canonical architecture was published.
- **2026-09-10:** Approved TA-01 through TA-03. Recorded the retained repository,
  toolchain boundaries, and Roblox-managed/local-first service direction, then
  opened checkpoint B with three content-foundation proposals. Progress is 3 of
  24.
- **2026-09-10:** Approved TA-04 through TA-06. Made the local authoring tool,
  immutable platform-neutral package, Roblox exporter, and real-song proof the
  content foundation, then opened checkpoint C. Progress is 6 of 24.
- **2026-09-10:** Approved TA-07 through TA-09. Recorded the server/client/shared
  split, typed reliable protocol, deterministic time/order/effect model, and
  private projections, then opened checkpoint D. Progress is 9 of 24.
- **2026-09-10:** Approved TA-10 through TA-12. Recorded shared-start/local-audio
  clock handling, immediate judgment with validated shared consequences, and the
  modular layered-audio candidate, then opened checkpoint E. Progress is 12 of
  24.
- **2026-09-10:** Approved TA-13 through TA-15 after clarifying that the shared
  battle runtime loads a distinct level for every boss. Recorded MemoryStore
  matchmaking/deployment, reserved-server rejoin, and No Contest recovery, then
  opened checkpoint F. Progress is 15 of 24.
- **2026-09-10:** Approved TA-16 through TA-18. Recorded ProfileStore behind a
  game-owned adapter, confirmed per-player transactions, durable receipt handling,
  migrations, recovery, backups, and audited account-lifecycle tooling, then
  opened checkpoint G. Progress is 18 of 24.
- **2026-09-10:** Approved TA-19 through TA-21. Recorded native UI and semantic
  input boundaries, version-pinned boss assets within distinct levels, and
  measurable device/performance budgets, then opened checkpoint H. Progress is
  21 of 24.
- **2026-09-10:** Approved TA-22 through TA-24. Recorded the three verification
  layers, minimal Roblox-first diagnostics, environment-isolated release and
  rollback process, and content-first implementation handoff. Progress is 24 of
  24; published canonical `TECHNICAL_ARCHITECTURE.md`.
