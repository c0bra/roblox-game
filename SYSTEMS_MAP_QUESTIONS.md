# Bands Battle Systems Map Interview Plan

- **Status:** Interview complete; 18 of 18 questions resolved
- **Created:** 2026-08-18
- **Parent design:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Higher authorities:** [`GAME_VISION.md`](GAME_VISION.md) and
  [`ART_DIRECTION.md`](ART_DIRECTION.md)
- **Working record:** [`SYSTEMS_MAP_WORKING.md`](SYSTEMS_MAP_WORKING.md)
- **Canonical result:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md)

## Purpose

This is the finite reference plan for turning the approved game design into a
map of the major systems that make Bands Battle work. It controls the order and
scope of the owner interview. Answers and resulting decisions are recorded in
`SYSTEMS_MAP_WORKING.md` as the interview proceeds.

The interview identifies stable responsibility boundaries and relationships. It
does not design Roblox services, remotes, modules, schemas, persistence formats,
file layouts, or implementation tasks. Those decisions belong in later system
specifications and `TECHNICAL_ARCHITECTURE.md`.

## Required result

Every retained system in `SYSTEMS_MAP.md` must state:

- **Purpose:** why the system exists in the player experience or production
  workflow;
- **Owns:** the concepts, state, rules, outcomes, or authoring responsibilities
  for which it is the primary source of truth;
- **Does not own:** nearby responsibilities deliberately assigned elsewhere
  when the boundary could otherwise be ambiguous;
- **Depends on:** other systems whose outputs or guarantees it requires;
- **Used by:** the direct consumers of its outputs or guarantees;
- **First-release role:** first-release runtime, first-release-supporting
  production capability, deferred future system, or cross-cutting requirement;
- **Needs detailed spec?:** yes or no, with the proposed document and priority
  when yes;
- **Major unresolved decisions:** design questions that remain after the GDD,
  with an explicit destination rather than an invitation for implementation to
  improvise; and
- **Source references:** the governing `GAME_DESIGN.md` sections.

The final document must also contain a compact dependency view, a prioritized
list of detailed system specifications, explicit first-release exclusions, and
an ownership-gap check.

## Interview protocol

- Ask exactly one numbered top-level question at a time.
- Use the indented prompts only to clarify the current question. They are not
  separate required questions.
- Begin each question with a concrete proposal inferred from the approved GDD
  whenever the source material supports one. The owner may accept, amend,
  merge, split, rename, or reject it.
- After each material answer, record the owner's words or a faithful summary,
  the resulting map decision, boundary consequences, dependency consequences,
  detailed-spec implications, and any explicitly deferred decisions in
  `SYSTEMS_MAP_WORKING.md`.
- At the start of each new question, update the progress count and mark the
  preceding question `[x]`, `[~]` if explicitly deferred, or `[>]` if resolved
  by another answer. Never silently skip a question.
- If one answer resolves a later question, record a cross-reference instead of
  asking for the same decision again.
- Follow-up prompts do not receive new identifiers unless they materially expand
  the bounded scope.
- Treat `GAME_DESIGN.md` as authoritative for player-facing behavior. If an
  answer would change a settled rule, identify it as a proposed GDD amendment
  instead of quietly encoding the conflict in the systems map.
- System ownership describes design responsibility and source-of-truth
  boundaries. It does not prescribe one code module, service, class, datastore,
  or network boundary per system.
- Record uncertain numbers as tuning questions, not new systems. Record a boss,
  song, item, screen, or hub location as content or a surface unless it owns an
  independent reusable ruleset and lifecycle.

## Completion rule

The interview is complete when all `SM-` questions are answered, explicitly
deferred with a destination, or resolved by another recorded answer, and all of
the following are true:

- every first-release responsibility in `GAME_DESIGN.md` has one clear primary
  owner or is explicitly identified as a cross-cutting requirement;
- every retained system has all fields from **Required result** completed;
- system boundaries do not contradict the GDD or duplicate ownership without an
  explained reason;
- dependencies and reverse consumers agree with one another;
- deferred and excluded features cannot be mistaken for first-release
  commitments;
- the detailed-spec backlog is finite, prioritized, and limited to documents
  that prevent downstream design improvisation; and
- remaining open decisions have named destinations in a system spec, content
  brief, balance sheet, UI specification, playtest plan, or technical
  architecture document.

## Source boundaries that this interview does not reopen

- Rhythm directly controls combat through three fixed inputs and a compact
  moving staff during active performance passages.
- A full song is the encounter clock; resistance breaks and gameplay events do
  not pause, skip, rewind, or retime it.
- Attack, Defend, and Special route the same authored performance into different
  combat outcomes.
- Boss victory requires all Resolve layers plus the finishing performance.
- Tactical positions, Ward, downing, recovery, personal Specials, Band Calls,
  Crescendos, and solo acolytes follow the rules in `GAME_DESIGN.md`.
- Solo and three-to-six-player co-op are both first-release requirements.
- Instruments are identities rather than locked combat classes; equipment and
  specialization create builds without altering musical judgment fairness.
- The first shippable product includes the Order hub, onboarding and practice,
  three replayable bosses, basic progression, equipment, rewards, economy,
  accessibility, and safe multiplayer flows.
- PvP, user-authored songs, free-roaming worlds, deep crafting, player trading,
  paid recovery, paid randomness, and multi-song raids are not first-release
  systems.
- The naming-and-tone pass and the song-pipeline upgrade remain required
  follow-up work, not permission to change settled mechanics here.

## Phase 1: Map frame and scope

- [x] **SM-01: What is the correct top-level system inventory and level of
  granularity for Bands Battle?**
  - Is the following GDD-derived starting inventory divided at useful
    responsibility boundaries, or should any candidates be merged, split,
    renamed, removed, or added?
  - **Core battle:** Rhythm Gameplay; Combat; Boss Encounters; Player Survival &
    Recovery; Abilities & Cooperative Actions; Solo Support; Difficulty &
    Scaling.
  - **Multiplayer:** Multiplayer Sessions, Parties & Matchmaking; Communication &
    Safety.
  - **Progression and meta:** Items, Equipment & Loadouts; Builds &
    Specialization; Rewards, Loot & Economy; Player, Campaign & Boss Mastery;
    Commerce.
  - **Experience shell:** Order Hub & Navigation; Onboarding, Practice &
    Calibration; Results & Feedback; UI, Input, Settings & Accessibility; Audio
    Presentation.
  - **Content and platform:** Song, Chart & Encounter Authoring; Player Data;
    Analytics & Playtest Evidence.
  - Which entries are true systems, which are cross-cutting requirements or
    production workflows, and which are merely player-facing surfaces over
    another system?
  - Decision: approved the GDD-derived starting inventory as proposed. The five
    working domains are Core Battle, Multiplayer, Progression and Meta,
    Experience Shell, and Content and Platform. All named entries remain in the
    inventory; later questions will classify them as systems, cross-cutting
    requirements, supporting production capabilities, or surfaces and will
    refine their internal ownership boundaries without silently removing them.

- [x] **SM-02: Which scope and status rules should determine what appears in the
  systems map?**
  - Which labels should distinguish first-release runtime systems,
    first-release-supporting production capabilities, cross-cutting
    requirements, deferred future systems, and explicit non-goals?
  - Should a deferred feature receive a full system entry, a short boundary
    note, or only an exclusion entry?
  - Should Roblox/platform capabilities appear as dependencies while only the
    game-owned policy and behavior appear as Bands Battle systems?
  - Which uncommitted categories—such as quests, achievements, daily or
    retention mechanics, live operations, trading, guilds, or multi-song
    raids—need explicit exclusion so later agents cannot infer them from genre
    convention?
  - What threshold makes a system complex or consequential enough to require a
    separate detailed design specification?
  - Decision: use six entry types: First-Release Runtime System,
    First-Release-Supporting Production System, Cross-Cutting Requirement,
    Orchestrated Experience/Surface, Deferred Future System, and Explicit
    Non-Goal. “First-release” replaces the ambiguous word “launch” and means the
    first publicly playable version defined by `GAME_DESIGN.md`. Roblox-provided
    capabilities appear as external dependencies; only Bands Battle's own rules
    and policies appear as its systems. Deferred systems receive full records
    only when already committed and sufficiently defined; other deferred or
    excluded areas remain in compact registers. A detailed spec is required
    when independent rules or state, multiple consumers, material fairness,
    progression or safety risk, or unresolved design would otherwise force
    implementation to invent behavior. The GDD's explicit exclusions remain
    excluded, while quests, achievements, dailies, live operations, and guilds
    are recorded as not assumed for the first release.

## Phase 2: Core battle boundaries

- [x] **SM-03: What exactly does the Rhythm Gameplay system own, and where does
  its responsibility stop?**
  - Does it own the authoritative musical clock, chart playback, note and hold
    judgment, phrase/scoring groups, timing grades, performance contribution,
    latency offsets, and pause/resume countdown behavior?
  - Which responsibilities instead belong to content authoring, input and
    settings, audio presentation, combat, results, or analytics?
  - What guarantees does Rhythm Gameplay provide to combat, encounters, group
    actions, recovery, audio, and results?
  - Which direct dependencies does it require at runtime and during content
    production?
  - Does it require the first and highest-priority detailed system spec, and
    which unresolved decisions must that spec settle?
  - Decision: Rhythm Gameplay is a First-Release Runtime System and the design-
    level source of truth for current song time, musical boundaries, runtime
    chart playback, input-to-note matching, judgments, holds, scoring groups,
    pre-combat normalized performance contribution, chart suspension and
    resumption, application of calibration/accessibility profiles, and solo
    pause/resume timing. It consumes authored chart data, encounter
    configuration, input/settings/accessibility profiles, instrument, and
    difficulty. It provides outputs to Combat, Boss Encounters, Survival &
    Recovery, Abilities & Cooperative Actions, Multiplayer, Audio Presentation,
    Results & Feedback, and Analytics. It does not own content authoring, input
    bindings or settings storage, combat conversion, encounter rules, audio
    mixing, results presentation, or analytics collection.
    `RHYTHM_GAMEPLAY.md` is required as the first gameplay specification after
    the initial `CONTENT_AUTHORING.md` contract and will settle precise matching,
    aggregation, normalization, transition, pause/resume, calibration-
    application, and output-contract decisions.

- [x] **SM-04: Should Combat and Player Survival & Recovery be separate systems,
  and what does each own?**
  - Does Combat own intent selection and queuing, contribution routing, Resolve
    damage, mitigation, combat modifiers, and outcome calculation?
  - Does Player Survival & Recovery own Ward, damage intake, state thresholds,
    downing, revival, re-entry protection, and the solo emergency challenge?
  - Which system owns Defend's boundary between performance contribution and
    protection, and which system owns encounter victory and defeat conditions?
  - Where do consumables, risk bonuses, boss impacts, equipment modifiers, and
    results attribution cross these boundaries?
  - Does each system need its own detailed spec, or can survival remain a major
    section of `COMBAT.md` without losing a clear owner?
  - Decision: Combat and Player Survival & Recovery are separate First-Release
    Runtime Systems documented as distinct major sections of `COMBAT.md`, the
    second gameplay specification after Content Authoring. Combat owns intent
    state and queuing, routing normalized rhythm
    contribution, calculating combat effects and permitted modifiers, effect
    attribution, and the rule that a miss does not directly damage Ward. Player
    Survival & Recovery owns Ward and survival states, applying post-mitigation
    damage and restoration, downing, cooperative revival state, the solo
    emergency recovery opportunity, and re-entry state. Rhythm produces
    performance; Combat calculates effects; Survival applies player-directed
    effects; Boss Encounters applies boss-directed Resolve effects and owns
    encounter outcomes. Results and Analytics consume facts from both systems.
    A separate `PLAYER_SURVIVAL.md` is unnecessary unless the shared spec later
    becomes unmanageably large.

- [x] **SM-05: How should Boss Encounters, boss attacks, and tactical positioning
  be divided?**
  - Does Boss Encounters own the encounter lifecycle, five song-shaped
    functions, Resolve-layer openings, Momentum timing, finishing cadence,
    victory/failure orchestration, and boss-specific event timeline?
  - Are the Telegraph/Commit/Impact/Recovery attack grammar and legal attack
    selection part of Boss Encounters, a reusable Boss Mechanics subsystem, or
    part of Combat?
  - Are tactical locations, directional graph, movement charges, settling,
    cover, hazards, displacement, and risk tiers a separate Positioning &
    Movement system or a responsibility within Combat/Boss Encounters?
  - What belongs to reusable rules versus an individual boss content package?
  - Which dependencies and consumers make these boundaries useful, and which
    detailed specification or specifications are required?
  - Decision: Boss Encounters and Tactical Positioning & Movement are separate
    First-Release Runtime Systems documented as major sections of the priority-
    three `BOSS_ENCOUNTERS.md`. Boss Encounters owns the active attempt
    lifecycle, five song-shaped functions, Resolve layers, Momentum, finishing
    evaluation, boss-directed effect application, attack/event legality and
    progression, targeting, hazards, group/recovery opportunities, and shared
    outcome. Tactical Positioning & Movement owns the arena graph, player
    location and travel state, legal destinations, movement charges and
    recovery, settling, multi-edge travel, displacement, shared-location rules,
    cover/hazard occupancy, risk-tier state, and application of authored graph
    changes. Positioning emits exposure and risk facts; Combat applies combat
    modifiers, and Rewards & Economy will own the eventual reward bonus. Boss
    attacks remain an internal Boss Encounters subsystem. Individual boss
    packages contain authored configuration; Boss Encounters executes them, and
    Song, Chart & Encounter Authoring creates and validates them. Tactical
    Positioning & Movement is an explicit refinement added to the SM-01
    inventory because it owns independent state used by several systems.

- [x] **SM-06: How should personal abilities, cooperative actions, and solo
  support be represented?**
  - Should Hype and Signature Specials, player-initiated Band Calls,
    song-authored Crescendos, revive routing, Join In, and Order acolytes share
    one Abilities & Cooperative Actions system or remain distinct systems?
  - Which system owns ability definitions, equipped choices, charge/readiness,
    invitations, participant eligibility, musical-boundary scheduling, effect
    resolution, and presentation contracts?
  - Does Solo Support own only acolyte state and authored support cadence, or
    also solo-only scaling and the emergency recovery challenge?
  - How do these responsibilities depend on Rhythm Gameplay, Combat, Activity
    Maps, Multiplayer, and Builds?
  - Which areas need detailed specs rather than sections inside Combat,
    Multiplayer, or Boss Encounters?
  - Decision: Abilities & Cooperative Actions and Solo Support are separate
    First-Release Runtime Systems. Abilities & Cooperative Actions owns
    Signature Special, Band Call, and Crescendo definitions and runtime state;
    Hype; readiness, initiation, invitation, Join In, eligibility, cancellation,
    scheduling, contribution combination, group tiers, and effect resolution
    through Combat, Survival, and Boss Encounters. Combat retains Special intent
    and contribution routing; Boss Encounters opens authored opportunities; and
    Player Survival & Recovery retains cooperative revival. Solo Support owns
    the three acolytes' runtime functions, musical cadences, suppression and
    recovery, formation requests, fixed group contributions, and prohibition on
    fabricated performance or independent Resolve breaks. It does not own solo
    emergency recovery, scaling, positions, or judgments. Both systems will be
    specified in separate major sections of
    `ABILITIES_AND_COOPERATIVE_ACTIONS.md`; no separate `SOLO_SUPPORT.md` is
    initially required. Final specification priority remains for SM-17.

- [x] **SM-07: Is Difficulty & Scaling a system with its own source of truth or
  a cross-cutting policy applied by other systems?**
  - Who owns the Easy/Normal/Hard rules, normalized maximum passage
    contribution, timing-window profiles, boss pressure, recovery allowances,
    reward modifiers, and Hard unlock requirement?
  - Who owns one-to-six-human scaling, duplicate-instrument handling, target
    counts, aggregate pressure, Cohesion Bonus, and solo-equivalence goals?
  - Who owns position risk/reward ratios and the bounded performance bonuses
    that affect both combat and rewards?
  - How are global invariants enforced without allowing several systems to
    define conflicting scaling values?
  - Does scaling require a dedicated design spec, a balance specification, or
    named sections in Rhythm, Combat, Encounter, Multiplayer, and Economy specs?
  - Decision: Difficulty & Scaling is a Cross-Cutting Requirement with one
    canonical design policy rather than an independent runtime system. It owns
    the Easy/Normal/Hard profiles, normalized-contribution rules, one-to-six-
    human invariants, duplicate-instrument neutrality, allowed scaling
    dimensions, solo/co-op equivalence goals, Cohesion Bonus principles,
    accessibility/reward invariants, and positional risk/reward constraints.
    Domain systems apply their own fields; Multiplayer supplies roster facts,
    Progression owns difficulty unlocks, Positioning owns current risk tier,
    Combat applies combat modifiers, and Rewards & Economy owns banked and
    unbanked Risk Bonus plus the final reward bonus. Each affected detailed spec
    must contain a Difficulty & Scaling section. A later shared
    `BALANCE_FRAMEWORK.md` will hold canonical matrices, curves, caps, and
    playtest-adjustable values rather than creating a separate gameplay-system
    specification.

## Phase 3: Multiplayer and authored content

- [x] **SM-08: What does the Multiplayer system own from party formation through
  post-encounter continuation?**
  - Which responsibilities cover parties, public matchmaking, queues, roster
    formation, encounter transport, ready state, session membership, rematch or
    leave choices, and return to the hub?
  - Which system owns player-count changes, disconnect and reconnect behavior,
    all-humans-down defeat, participation credit, and the boundary between
    individual results and shared encounter outcome?
  - Should communication pings, invitations, mute/block/report entry points, and
    age-appropriate defaults belong here or in a distinct Communication &
    Safety system?
  - Which Roblox/platform services are dependencies rather than game-owned
    systems?
  - Does the complete first-release multiplayer flow require one detailed
    `MULTIPLAYER.md` specification or several narrower specifications?
  - Decision: Multiplayer Sessions, Parties & Matchmaking is a First-Release
    Runtime System. It owns party membership/leadership/consent, public matching,
    queue and two-player choices, ready and staging state, lock timing, the no-
    join-in-progress rule, encounter and active-roster membership, disconnect/
    rejoin/AFK/resume state, rematch/refill/leave actions, leader transfer, and
    preset-ping delivery/rate limiting/muting. Domain systems retain their own
    restored state. Multiplayer owns connection and participation-status facts;
    gameplay systems emit contribution, while Rewards and Progression decide
    eligibility and grants. Communication & Safety is a Cross-Cutting
    Requirement covering safe coordination, ping policy, anti-coercion, and
    structural anti-grief invariants. Roblox filtering, blocking, reporting,
    privacy, age controls, matchmaking infrastructure, and transport are
    external dependencies. Both boundaries will be detailed in one
    `MULTIPLAYER.md`; final priority remains for SM-17.

- [x] **SM-09: Is Song, Chart & Encounter Authoring one supporting system or a
  family of production capabilities?**
  - Who owns ingestion, rights/provenance records, beat and structure analysis,
    chart editing, difficulty derivation, Activity Maps, encounter-event tracks,
    validation, preview, export, approvals, and content version compatibility?
  - Where is the boundary between source content, automated or AI suggestions,
    human-authored decisions, exported runtime data, and the runtime systems
    that consume it?
  - Should the lightweight first-release pipeline and a future polished
    authoring tool be mapped as different stages of the same system?
  - Which validators are owned here versus in Rhythm, Encounters, Multiplayer,
    Accessibility, or Analytics?
  - Which detailed schema, pipeline, tooling, and content-production documents
    are required, and in what order relative to the gameplay specs?
  - Decision: Song, Chart & Encounter Authoring is one First-Release-Supporting
    Production System with internal intake/analysis, chart/difficulty authoring,
    encounter-timeline authoring, validation/approval, and runtime-export
    capabilities. It is explicitly an offline, platform-neutral toolchain and
    must not live inside or ship with the Roblox client/server runtime. It extends
    the existing root-owned TypeScript tool in `tools/chart-pipeline/`, whose
    platform-neutral bundle and validation contract already separates processing
    from game renderers. Roblox consumes an approved exported package through an
    adapter/export step and cannot edit or reinterpret authored meaning.
    Domain systems define semantic content requirements; the authoring system
    runs their validators, assembles evidence, and blocks invalid approval.
    Individual songs and bosses remain content packages, not systems. A
    lightweight first-release workflow and later polished offline tooling are
    maturity stages of the same production system. `CONTENT_AUTHORING.md` is
    required after the Rhythm and Boss Encounter content contracts; schemas,
    file formats, tool architecture, and Roblox adaptation remain later
    technical specifications.

## Phase 4: Items, economy, and progression

- [x] **SM-10: What belongs to Items, Equipment & Loadouts, and should inventory
  or consumables be separate systems?**
  - Who owns item and cosmetic definitions, fixed stats and traits, ownership,
    the three first-release gear slots, equipped Signature Special and Band Call,
    prepared consumables, appearance unlocks, and pre-battle validation?
  - Does this system own inventory capacity and saved loadouts, while Builds &
    Specialization owns saved build presets and behavioral rules?
  - Which responsibilities belong instead to Player Data, Rewards & Economy,
    Commerce, Combat, or UI?
  - Is a standalone Inventory system justified by an independent lifecycle, or
    is it a responsibility inside Items & Equipment for the first release?
  - What must `ITEMS_AND_EQUIPMENT.md` settle before architecture and item
    content production can begin?
  - Decision: Items, Equipment & Loadouts is one First-Release Runtime System.
    Inventory and consumables are responsibilities within it rather than
    separate systems because the first release has no trading, capacity-
    management game, or independent inventory lifecycle. It owns item,
    consumable, and cosmetic definitions; owned collections; fixed stats,
    traits, tier, and rank state; power, action-reference, consumable, and
    cosmetic slots; equip and validation rules; staging/encounter locks;
    prepared quantities and use authorization; resolved equipment modifiers;
    and prohibited-modifier enforcement. It does not own earning, drops,
    crafting, salvage, purchases, upgrade transactions, currencies, ability
    behavior, specialization, combat calculation, persistence implementation, or
    UI. Player Data persists records while Items defines their meaning and valid
    mutations. Items authorizes consumable use; Combat, Survival, or Abilities
    resolves its effect. `ITEMS_AND_EQUIPMENT.md` is required; final priority is
    confirmed in SM-17.

- [x] **SM-11: Is Builds & Specialization a separate system from items and
  abilities, and what is its source of truth?**
  - Who owns the major behavior-changing rule, three supporting rules, universal
    functional categories, role presets, unlock requirements, free respec,
    synergy caps, and saved presets?
  - Which system owns the actual effect definitions and validation when a build
    modifies intent, position, Specials, Band Calls, support, or consumables?
  - How does the required naming-and-tone pass relate to mechanics that may be
    prototyped under working names?
  - Which systems consume the resolved build, and which systems may never be
    modified by it because of the GDD's fairness boundaries?
  - Does this require its own detailed specification or a bounded section of
    Items & Equipment and Combat?
  - Decision: Builds & Specialization is a separate First-Release Runtime System
    with its own later `BUILDS_AND_SPECIALIZATION.md`. It owns universal
    functional categories, one major and three supporting slots, cross-category
    mixing, beginner role presets, the advanced-editor gate, three saved build
    presets, free out-of-combat respec, encounter locking, compatibility,
    stacking, power budgets, synergy caps, build-modifier definitions, resolved
    modifier output, and enforcement of fairness prohibitions. Progression owns
    option unlocks; Player Data persists presets; Items owns gear; Abilities owns
    base abilities; and Combat, Survival, Positioning, and Abilities apply only
    the resolved modifiers permitted by their contracts. Working terminology may
    support prototypes but cannot ship without the required naming-and-tone
    pass. Final specification priority remains for SM-17.

- [x] **SM-12: How should Rewards, Loot & Economy, upgrades/crafting, and
  Commerce be separated?**
  - Who owns encounter reward calculation, guaranteed and random grants,
    currencies and boss materials, deterministic crafting progress, salvage,
    upgrade ranks, consumable costs, loot pools, and exact transaction results?
  - Is reward calculation distinct from the economy ledger and item ownership,
    and which system performs the final idempotent grant?
  - Is first-release crafting a bounded economy capability rather than the
    deferred deep-crafting system?
  - Should Commerce be a separate safeguard boundary that owns catalog offers,
    purchase eligibility, earnable equivalents, tier ceilings, purchase
    validation, and prohibited surfaces?
  - Which detailed economy tables, item catalogs, reward specs, and commerce
    rules must exist before implementation or content balancing?
  - Decision: Rewards, Loot & Economy and Commerce are separate First-Release
    Runtime Systems documented as major sections of one
    `REWARDS_AND_ECONOMY.md`. Rewards, Loot & Economy owns eligibility and
    calculation, Risk and Cohesion Bonus reward state/effects, resource and boss-
    material balances, loot pools, deterministic progress, earned random drops,
    guaranteed/first-clear/signature rules, salvage, bounded first-release
    crafting, upgrades, consumable costs, idempotent reward orchestration, and
    economy prohibitions. It creates a once-only transaction plan; Items and
    Progression validate/apply their domain mutations, and Player Data durably
    commits the result. Deep crafting remains deferred. Commerce owns the paid
    catalog, store eligibility, purchase/confirmation/receipt lifecycle,
    duplicate protection, earnable-equivalent mappings, tier/stat validation,
    current-tier grants, product-category safeguards, and prohibited prompt
    surfaces, using Roblox Marketplace as an external dependency. Exact tables
    and catalogs remain downstream balance/content artifacts; receipt security
    and atomic persistence belong to technical architecture.

- [x] **SM-13: What does Player, Campaign & Boss Mastery progression own, and
  should those tracks be separate systems?**
  - Who owns general progression, campaign destinations, Shattered Song
    fragments, first-clear state, difficulty unlocks, boss mastery ranks,
    personal bests by instrument and difficulty, recipes, and system unlocks?
  - Which system decides meaningful participation and progression amounts, and
    which system merely receives the encounter result and grants them?
  - Who owns current-tier relevance, old-item uplift eligibility, recommended
    power, and the hub restoration state driven by campaign progress?
  - Are player progression, campaign/world progression, and boss mastery one
    coherent system with tracks, or separate systems with different consumers?
  - What must `PROGRESSION.md` settle, and which numeric questions belong in
    balance tables instead?
  - Decision: one First-Release Runtime System named Progression owns three
    coordinated tracks: general player progression, campaign progression, and
    boss mastery/personal bests. It owns system unlocks, destinations, first
    clears, fragments, difficulty availability, mastery ranks/milestones,
    personal bests, unlock eligibility, current campaign tier, recommended
    power, old-item uplift eligibility, progress from victory/failure, hub-
    restoration state, and non-expiring progression policy. Gameplay systems
    emit outcome/participation evidence; Rewards & Economy owns the canonical
    meaningful-participation result and once-only transaction; Progression
    calculates and validates its domain mutations; Player Data persists them;
    and downstream systems consume unlock/tier state. The tracks remain unified
    because they consume the same encounter result and jointly determine
    unlocks. `PROGRESSION.md` is required; exact rates and thresholds belong in
    the balance framework.

## Phase 5: Experience shell and platform responsibilities

- [x] **SM-14: Are the Order hub, onboarding/practice, and results/post-battle
  flow systems, orchestrated experiences, or UI surfaces over other systems?**
  - Who owns shard discovery and activation, hub functional anchors, campaign-
    driven restoration, fast access, and transitions into matchmaking or solo
    play?
  - Who owns onboarding checkpoints, contextual teaching state, practice,
    calibration entry, skip/replay rules, prompt history, and the store unlock
    gate?
  - Who owns immediate outcome explanation, performance breakdowns, improvement
    suggestions, reward presentation, adaptive next action, retry, stay-with-
    band, and return-to-hub flow?
  - Which responsibilities are persistent game rules versus compositions of
    UI, Rhythm, Encounters, Multiplayer, Progression, and Rewards?
  - Which of these areas need detailed product-flow or UI specifications even if
    they are not independent runtime systems?
  - Decision: Order Hub & Navigation and Results & Feedback are Orchestrated
    Experiences/Surfaces, while Onboarding & Practice is a First-Release Runtime
    System. The hub owns spatial/navigation composition, shard and functional-
    anchor interaction routes, fast access, and application of Progression's
    visible restoration state, but not the domain rules behind its destinations.
    Onboarding owns tutorial sequence/checkpoints, completion/skip/replay, safe
    practice state, contextual teaching and prompt history, the public-
    matchmaking completion gate, the onboarding portion of Commerce eligibility,
    calibration/settings entry, and non-pausing contextual instruction. Results
    owns the derived summary, outcome-versus-performance separation, exact reason
    display, already-granted reward/unlock presentation, detail views, private
    suggestions, adaptive next action, and routing without claims, rankings,
    blame, or paid prompts. It does not calculate or grant domain outcomes. All
    three receive major sections in `UI_UX.md`; they do not initially need
    separate system specs. Hub physical composition may later receive a content/
    world brief without becoming another system.

- [x] **SM-15: How should UI, input, settings, accessibility, calibration, and
  responsive audio presentation be mapped?**
  - Is there one Player Experience platform system, separate Input & Settings,
    UI Presentation, Accessibility, and Audio Presentation systems, or a mix of
    systems and cross-cutting requirements?
  - Who owns device detection, binding profiles, touch layout, safe areas,
    control remapping, UI scale, visual scroll speed, Hold Assist, reduced
    motion/flashing, captions, subtitles, audio buses, haptics, and saved
    calibration?
  - Do gameplay systems own the semantic state and cues while presentation
    systems own only how those cues are rendered, mixed, and controlled?
  - How are accessibility invariants applied and validated across every boss,
    input device, difficulty, and reward path without becoming optional per-
    system polish?
  - Which detailed `UI_UX.md`, input-map, accessibility, and audio
    specifications are required before architecture or content production?
  - Decision: UI Presentation, Input, Settings & Calibration, and Audio
    Presentation are separate First-Release Runtime Systems. Accessibility is a
    Cross-Cutting Requirement and mandatory design scaffold rather than an
    optional subsystem. UI owns responsive composition, hierarchy, component/
    navigation states, rendering of semantic cues, device labels/references,
    and visual/caption equivalents. Input/Settings/Calibration owns semantic
    action mapping, input modes, device/profile selection, remapping, touch
    configuration, settings values, guided/manual calibration, and exposing
    profiles; Rhythm/UI/Audio apply their relevant results, and Player Data
    persists them. Accessibility owns multimodal/non-color invariants, scaling,
    independently reducible effects, difficulty/reward/privacy neutrality, and
    prohibition on public labeling or shaming; every system must expose the
    semantic state needed to comply. Audio owns stable song/stem presentation,
    responsive local instrument treatment, gameplay/crowd/ambience mixing, cue
    priority and ducking, buses/dynamic range/mono/caption metadata, aggregate
    band response, and restrained feedback requests while following Rhythm's
    clock. `UI_UX.md` covers UI, Input/Settings/Calibration, Accessibility, Hub,
    Onboarding, and Results. `AUDIO_PRESENTATION.md` is separate. No standalone
    input or accessibility spec is initially required.

- [x] **SM-16: What do Player Data, Communication & Safety, and Analytics &
  Playtest Evidence own?**
  - Which player-owned facts require durable records: progression, inventory,
    purchases, builds, loadouts, mastery, personal bests, settings, calibration,
    onboarding, prompt history, and unlocks?
  - Do domain systems own the meaning and mutation rules while Player Data owns
    durable storage, versioning, migration, loading, saving, and recovery
    guarantees?
  - Which communication behavior and safety policy belongs to Bands Battle, and
    which filtering, reporting, privacy, or parental capability is supplied by
    Roblox?
  - Who owns analytics event meaning, consent-conscious collection boundaries,
    playtest segmentation, readiness reports, and the evidence gates from
    GD-34?
  - Which of these require detailed design documents now, and which belong only
    in technical architecture, security/privacy review, or a playtest plan?
  - Decision: Player Data is a First-Release Runtime Platform System. It owns
    load/save/recovery guarantees, durable cross-domain commits, record
    versioning/migration, default profiles, concurrency/stale-write protection,
    retry/rollback/failure policy, durable storage of all approved player-owned
    facts, and player-visible unavailable/unsafe-save behavior, while domain
    systems retain semantic mutation ownership. Ephemeral reconnect state stays
    with Multiplayer and gameplay owners. Analytics & Playtest Evidence is a
    First-Release-Supporting Production System with runtime instrumentation. It
    owns the event/metric catalog, collection/segmentation rules, evidence
    synthesis, GD-34 readiness reports, data-quality checks, research consent/
    safeguarding/retention/access boundaries, and the rule that evidence cannot
    automatically change difficulty, rewards, matchmaking, or public labels.
    Domain systems own emitted facts; Analytics collects and reports them.
    Roblox persistence, temporary storage, analytics transport, privacy, and
    account capabilities are external dependencies. `PLAYER_DATA.md` and
    `PLAYTEST_AND_ANALYTICS.md` are required; exact schemas, storage, telemetry,
    security, and transport remain technical. Communication & Safety remains the
    SM-08 cross-cutting requirement documented in `MULTIPLAYER.md`.

## Phase 6: Specification backlog and final reconciliation

- [x] **SM-17: Which systems need separate detailed design specifications, and
  in what dependency order should those documents be created?**
  - Are `RHYTHM_GAMEPLAY.md`, `COMBAT.md`, `BOSS_ENCOUNTERS.md`,
    `PROGRESSION.md`, and `ITEMS_AND_EQUIPMENT.md` the correct first five?
  - Which additional documents—such as `MULTIPLAYER.md`, `REWARDS_AND_ECONOMY.md`,
    `CONTENT_AUTHORING.md`, `PLAYER_DATA.md`, or `UI_UX.md`—are required before
    technical architecture, and which can wait?
  - When should several related systems share one document, and when would that
    force an implementation agent to invent design decisions?
  - What exact scope and prerequisites should each document have?
  - Which system-map open questions are assigned to each spec, and which go to
    a naming pass, content brief, balance sheet, playtest plan, or technical
    architecture instead?
  - Owner direction, 2026-08-18: `CONTENT_AUTHORING.md` must be created before
    the gameplay specifications because the gameplay design and implementation
    need actual song data and an agreed exported-data contract as their starting
    point. The existing offline `tools/chart-pipeline/` is the maintained
    foundation. Content Authoring should establish the initial source/bundle/
    validation contract first, then receive a reconciliation pass after Rhythm
    Gameplay, Combat, Boss Encounters, and related specs refine their downstream
    requirements. SM-17 remained open at that point and was resolved by the
    approved decision below.
  - Decision: the approved specification order is `CONTENT_AUTHORING.md`,
    `RHYTHM_GAMEPLAY.md`, `COMBAT.md`, `BOSS_ENCOUNTERS.md`, `PROGRESSION.md`,
    `ITEMS_AND_EQUIPMENT.md`, `ABILITIES_AND_COOPERATIVE_ACTIONS.md`,
    `MULTIPLAYER.md`, `REWARDS_AND_ECONOMY.md`,
    `BUILDS_AND_SPECIALIZATION.md`, `UI_UX.md`, `AUDIO_PRESENTATION.md`, and
    `PLAYER_DATA.md`. Content Authoring first establishes the offline song-data
    and exported-bundle contract. After the gameplay and presentation specs
    identify their complete data needs, the same document receives a mandatory
    reconciliation pass before `TECHNICAL_ARCHITECTURE.md` becomes canonical.
    `BALANCE_FRAMEWORK.md` and `PLAYTEST_AND_ANALYTICS.md` are supporting
    documents developed alongside later specs. The approved parent-document
    grouping means no standalone specs are initially required for Survival,
    Positioning, Solo Support, Difficulty, Communication & Safety, Commerce,
    Inventory, consumables, Hub, Onboarding, Results, Input, or Accessibility.

- [x] **SM-18: Does the completed map form a coherent dependency and ownership
  model with no missing first-release responsibility?**
  - Does every `Depends on` relationship have the corresponding `Used by`
    relationship, and are any apparent cycles unexplained or harmful at the
    design level?
  - Does every major state transition and outcome have one primary owner,
    especially encounter start/end, note judgment, combat contribution, Ward
    damage, Resolve breaks, victory, participation credit, rewards, inventory
    grants, unlocks, saves, and result presentation?
  - Are any GDD sections orphaned, assigned twice, or incorrectly treated as
    content, a surface, a policy, or a technical concern?
  - Are all first-release, first-release-supporting, deferred, and excluded
    entries labeled so future planning cannot confuse them?
  - Are the remaining open decisions narrow, routed, and safe to defer before
    approving `SYSTEMS_MAP.md` as the bridge to detailed system design?
  - Decision: the final audit is approved. All 25 retained entries have a
    classification and primary responsibility; all 34 GDD sections are covered;
    every major state, outcome, transaction, and presentation responsibility has
    one owner; apparent dependency cycles are documented handshakes; the
    thirteen-spec sequence and Content Authoring reconciliation gate are
    coherent; exclusions are explicit; and remaining design, content, balance,
    naming, research, and technical questions have named destinations. The
    interview is complete and [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md) is the canonical
    result.

## Deferred technical documents

These outputs may be named or prioritized by the systems map, but their internal
technical questions do not add questions to this interview:

- `TECHNICAL_ARCHITECTURE.md` for Roblox runtime boundaries, authority,
  networking, persistence, security, observability, and deployment concerns;
- rhythm chart, encounter timeline, Activity Map, item, progression, and player-
  data schemas;
- remote/API contracts and client/server state synchronization;
- analytics event schema, privacy review, test plans, and release gates;
- production schedules, asset manifests, song licensing records, and boss/song
  content packages; and
- implementation plans and agent-sized engineering tasks.

## Plan change log

- **2026-08-18:** Created the 18-question baseline from the approved
  `GAME_DESIGN.md`, its required follow-up work, and its deferred technical
  specifications. No owner questions have yet been added, removed, or resolved.
- **2026-08-18:** Approved SM-01's proposed five-domain candidate inventory
  without additions, removals, renames, merges, or splits.
- **2026-08-18:** Approved SM-02's classification and inclusion rules. Replaced
  “launch” with “first-release” throughout the systems-map documents to mean the
  first publicly playable version defined by the GDD.
- **2026-08-18:** Approved SM-03's Rhythm Gameplay boundary and designated
  `RHYTHM_GAMEPLAY.md` as the first gameplay specification after the initial
  Content Authoring contract.
- **2026-08-18:** Approved SM-04's separation of Combat from Player Survival &
  Recovery, with both documented in the second gameplay specification,
  `COMBAT.md`.
- **2026-08-18:** Approved SM-05's separation of Boss Encounters from Tactical
  Positioning & Movement, with both documented in the third gameplay
  specification, `BOSS_ENCOUNTERS.md`.
- **2026-08-18:** Approved SM-06's separate Abilities & Cooperative Actions and
  Solo Support systems and their shared later detailed specification.
- **2026-08-18:** Resolved SM-07 using the recommended model after the owner
  delegated the decision: Difficulty & Scaling is a cross-cutting requirement
  backed by a later shared balance framework.
- **2026-08-18:** Approved SM-08's Multiplayer runtime boundary and
  Communication & Safety cross-cutting policy, to be detailed together in
  `MULTIPLAYER.md`.
- **2026-08-18:** Approved SM-09's single authoring production system with the
  explicit amendment that it remains an offline, platform-neutral extension of
  `tools/chart-pipeline/` rather than living in Roblox.
- **2026-08-18:** Approved SM-10's unified Items, Equipment & Loadouts runtime
  system, with inventory and consumables retained as internal responsibilities.
- **2026-08-18:** Approved SM-11's separate Builds & Specialization runtime
  system and later `BUILDS_AND_SPECIALIZATION.md` specification.
- **2026-08-18:** Approved SM-12's separate Rewards, Loot & Economy and Commerce
  systems, with both documented in `REWARDS_AND_ECONOMY.md`.
- **2026-08-18:** Approved SM-13's unified Progression runtime system containing
  player, campaign, and boss-mastery tracks.
- **2026-08-18:** Approved SM-14's Order Hub and Results classifications as
  orchestrated surfaces and Onboarding & Practice as a runtime system, all to be
  specified within `UI_UX.md`.
- **2026-08-18:** Approved SM-15's separate UI, Input/Settings/Calibration, and
  Audio runtime systems and Accessibility cross-cutting scaffold, documented in
  `UI_UX.md` and `AUDIO_PRESENTATION.md`.
- **2026-08-18:** Approved SM-16's Player Data runtime/platform system and
  Analytics & Playtest Evidence supporting-production system, with separate
  `PLAYER_DATA.md` and `PLAYTEST_AND_ANALYTICS.md` specifications.
- **2026-08-18:** SM-17 owner direction moved `CONTENT_AUTHORING.md` ahead of all
  gameplay specifications so an agreed song-data contract exists first. The
  question remained open until the revised complete order was approved.
- **2026-08-18:** Approved SM-17's revised thirteen-spec order, with Content
  Authoring first and a mandatory contract-reconciliation pass before technical
  architecture is finalized.
- **2026-08-18:** Approved SM-18's final ownership/dependency audit, completed
  the 18-question interview, and produced canonical `SYSTEMS_MAP.md`.
