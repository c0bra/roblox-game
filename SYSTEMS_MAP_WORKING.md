# Bands Battle Systems Map Working Record

- **Status:** Interview complete; archived decision record; 18 of 18 resolved
- **Started:** 2026-08-18
- **Design baseline:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Higher authorities:** [`GAME_VISION.md`](GAME_VISION.md) and
  [`ART_DIRECTION.md`](ART_DIRECTION.md)
- **Interview plan:** [`SYSTEMS_MAP_QUESTIONS.md`](SYSTEMS_MAP_QUESTIONS.md)
- **Canonical result:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md)

## 1. Role of this document

This is the progressive decision record for the Bands Battle systems-map
interview. It preserves owner answers, derived system boundaries, ownership and
dependency consequences, detailed-spec decisions, deferred work, and change
history as each question is resolved.

This document is the archived decision record for the completed interview.
[`SYSTEMS_MAP.md`](SYSTEMS_MAP.md) is the canonical systems map;
`GAME_DESIGN.md` continues to govern player-facing behavior. This record
preserves how the map was reached but does not override either canonical
document.

## 2. Fixed source constraints

The interview begins from these approved boundaries rather than reopening them:

- The full song is the encounter's master clock, and authored rhythm performance
  directly creates combat contribution.
- Attack, Defend, and Special route the same performance into distinct combat
  outcomes.
- Boss encounters combine Resolve layers, Momentum, a finishing performance,
  readable attack phases, tactical positions, Ward, downing, recovery,
  abilities, group actions, and solo support.
- Solo and public or preformed three-to-six-player co-op are first-release
  requirements.
- Equipment and specialization create builds without altering charts, timing
  judgments, movement fairness, essential telegraphs, or reward eligibility.
- The first-release loop includes the Order hub, onboarding and practice, three
  bosses, results, rewards, progression, mastery, items, upgrades,
  accessibility, and a tightly safeguarded voluntary store.
- Authoring must produce reviewed charts, difficulty variants, Activity Maps,
  encounter timelines, validation evidence, and runtime-ready content.
- Player-facing behavior comes from `GAME_DESIGN.md`; implementation structure
  is deferred to system specifications and technical architecture.

## 3. Recording format

Each resolved question receives:

- **Owner answer:** the owner's words or a faithful summary;
- **Map decision:** the approved system inventory, boundary, ownership,
  dependency, first-release-role, or specification decision;
- **Consequences:** effects on neighboring systems, dependency direction,
  source-of-truth ownership, and later documents;
- **Detailed-spec impact:** whether a spec is required, its likely scope, and
  its priority; and
- **Deferred:** unresolved matters and their named destination.

Each proposed system record accumulated here will eventually contain purpose,
owns, does not own, depends on, used by, first-release role, detailed-spec need,
open decisions, and GDD source references.

## 4. Decision record

### SM-01: System inventory and granularity

- **Owner answer:** The proposed inventory and granularity are accepted as
  presented: “Yes these are fine.”
- **Map decision:** Use five working domains and retain every proposed candidate
  entry:
  - **Core Battle:** Rhythm Gameplay; Combat; Boss Encounters; Player Survival &
    Recovery; Abilities & Cooperative Actions; Solo Support; Difficulty &
    Scaling.
  - **Multiplayer:** Multiplayer Sessions, Parties & Matchmaking; Communication &
    Safety.
  - **Progression and Meta:** Items, Equipment & Loadouts; Builds &
    Specialization; Rewards, Loot & Economy; Player, Campaign & Boss Mastery;
    Commerce.
  - **Experience Shell:** Order Hub & Navigation; Onboarding, Practice &
    Calibration; Results & Feedback; UI, Input, Settings & Accessibility; Audio
    Presentation.
  - **Content and Platform:** Song, Chart & Encounter Authoring; Player Data;
    Analytics & Playtest Evidence.
- **Consequences:** This inventory is the completeness baseline for the rest of
  the interview. Later questions may classify an entry as a runtime system,
  supporting production capability, cross-cutting requirement, or surface and
  may refine internal ownership, but they must not silently remove an approved
  responsibility. A material merge, split, addition, or removal will be
  recorded as an explicit amendment to SM-01.
- **Detailed-spec impact:** Approval of an inventory entry does not yet require
  a separate document. Specification need and priority remain open through the
  boundary questions and are finalized in SM-17.
- **Deferred:** System status and scope labels are assigned in SM-02. Exact
  ownership, dependencies, consumers, and detailed-spec boundaries are assigned
  in SM-03 through SM-17 and reconciled in SM-18.

### SM-02: Scope, status, and inclusion rules

- **Owner answer:** The owner first requested clarification of “launch.” After it
  was defined as the first publicly playable version described by the GDD, the
  owner approved using the clearer term **first-release** throughout: “This is
  fine.”
- **Map decision:** Classify entries with six explicit types:
  1. **First-Release Runtime System:** operates in the first publicly playable
     version.
  2. **First-Release-Supporting Production System:** required to create,
     validate, or operate first-release content but is not itself a player-facing
     runtime system.
  3. **Cross-Cutting Requirement:** a shared invariant or policy that several
     systems must enforce rather than an independent owner of their state.
  4. **Orchestrated Experience/Surface:** composes several systems into a player
     flow without owning the underlying domain rules.
  5. **Deferred Future System:** intended after the first release. It receives a
     full record only when already committed and defined enough to have stable
     responsibilities; otherwise it receives a short boundary entry.
  6. **Explicit Non-Goal:** recorded in an exclusion register rather than given
     a full system entry.

  Roblox-provided capabilities appear as external dependencies. Bands Battle
  system entries cover only game-owned behavior, policy, state meaning, and
  guarantees layered on those capabilities.
- **Consequences:** Replace “launch” with “first-release” throughout the systems-
  map plan and working record. The first-release scope is the GDD's first
  shippable product: the Order hub and onboarding, three replayable bosses, solo
  and three-to-six-player co-op, rhythm combat, basic equipment/rewards/upgrades/
  progression, accessibility, safety, and the required supporting content
  pipeline. PvP, user-authored songs, free-roaming worlds, deep crafting,
  trading, paid recovery, paid randomness, and multi-song raids remain explicit
  non-goals for that release. Quests, achievements, daily or retention systems,
  live operations, and guilds are not assumed for the first release and cannot
  be introduced by genre convention alone.
- **Detailed-spec impact:** Require a separate detailed specification when an
  entry has substantial independent rules or state, supplies multiple
  consumers, carries material fairness/progression/safety risk, or contains an
  unresolved design choice that implementation would otherwise have to invent.
  An orchestrated surface or small subordinate responsibility may instead be a
  bounded section of its owning system's spec.
- **Deferred:** Each candidate's type is assigned while its boundary is resolved
  in SM-03 through SM-16. SM-17 confirms the complete spec backlog, and SM-18
  verifies that classification did not hide an ownership gap.

### SM-03: Rhythm Gameplay boundary

- **Owner answer:** The proposed Rhythm Gameplay system record is approved as
  presented: “Yes.”
- **Map decision:** **Rhythm Gameplay** is a **First-Release Runtime System**.
  - **Purpose:** Turn authored instrument charts and player inputs into musically
    aligned judgments and normalized performance contribution while keeping the
    full song as the encounter clock.
  - **Owns:** The design-level source of truth for current song time and musical
    boundaries; runtime chart playback for the selected instrument and
    difficulty; tap, hold, repeat, alternate, and rest interpretation; input-to-
    note matching; Perfect, Great, Good, and Miss judgments; early/late feedback;
    maintained-hold contribution; phrase and scoring-group aggregation;
    difficulty-normalized musical contribution before combat modifiers;
    suspending and resuming chart participation during movement, downing, and
    re-entry; applying calibration offsets and assists such as Hold Assist; solo
    pause and beat-counted resumption; and performance outputs consumed by other
    systems.
  - **Does not own:** Chart creation, Activity Maps, or content validation;
    device bindings, calibration interfaces, or saved settings; Attack, Defend,
    or Special conversion and other combat effects; boss phases, attacks, or
    encounter outcomes; audio mixing and responsive instrument effects; results
    presentation; or analytics collection.
  - **Depends on:** Authored chart data; encounter configuration; input,
    settings, and accessibility profiles; selected instrument; and selected
    difficulty.
  - **Used by:** Combat; Boss Encounters; Player Survival & Recovery; Abilities &
    Cooperative Actions; Multiplayer; Audio Presentation; Results & Feedback;
    and Analytics & Playtest Evidence.
  - **First-release role:** First-Release Runtime System.
  - **Needs detailed spec?:** Yes—`RHYTHM_GAMEPLAY.md`, the first gameplay
    specification after the initial `CONTENT_AUTHORING.md` contract.
  - **Major unresolved decisions:** Exact input-matching rules; phrase
    aggregation; contribution normalization; overlapping holds and rapid-input
    behavior; movement, downing, and re-entry transitions; pause/resume rules;
    calibration application; and the semantic output contract provided to other
    systems.
  - **Source references:** GD-02, GD-04 through GD-10, GD-12, GD-14 through
    GD-19, GD-29, GD-33, and GD-34.
- **Consequences:** Rhythm Gameplay owns musical interpretation and produces a
  pre-combat result; it does not decide what that performance does to the boss,
  player, band, rewards, or presentation. The detailed spec may define semantic
  contracts and timing guarantees but must leave client/server clock authority,
  networking, and concrete module structure to technical architecture.
- **Detailed-spec impact:** `RHYTHM_GAMEPLAY.md` is the first gameplay-system
  specification after the initial Content Authoring song-data contract because
  Combat, Encounters, group actions, recovery scheduling, responsive audio, and
  results all consume its guarantees.
- **Deferred:** Numeric timing windows already approved as starting values remain
  playtest-tunable. Network clock synchronization, anti-cheat, replication, and
  recovery from technical desynchronization belong to
  `TECHNICAL_ARCHITECTURE.md`.

### SM-04: Combat and Player Survival & Recovery boundaries

- **Owner answer:** The proposed separation and shared specification are
  approved: “Yes I like this separation.”
- **Map decision:** **Combat** and **Player Survival & Recovery** are separate
  **First-Release Runtime Systems**.

  **Combat**
  - **Purpose:** Convert rhythm performance, combat intent, position, equipment,
    and build effects into combat consequences.
  - **Owns:** Attack, Defend, and Special intent selection, queuing, and boundary
    application; routing normalized rhythm contribution by active intent;
    calculating Resolve pressure, mitigation, Ward reinforcement, restoration,
    support, and ability contribution; applying permitted equipment, build,
    position, and difficulty modifiers after musical scoring; combat-effect
    attribution; and the rule that missed notes do not directly damage Ward.
  - **Does not own:** Rhythm judgments; boss state, Resolve-layer state, or
    encounter outcomes; player Ward state; ability definitions; rewards; or
    persistent items.
  - **Depends on:** Rhythm Gameplay; encounter and target state; position/risk
    state; equipped items and resolved build effects; abilities; and applicable
    difficulty/scaling policy.
  - **Used by:** Boss Encounters; Player Survival & Recovery; Abilities &
    Cooperative Actions; Solo Support; Results & Feedback; Audio Presentation;
    Rewards, Loot & Economy; and Analytics & Playtest Evidence.

  **Player Survival & Recovery**
  - **Purpose:** Manage whether each player remains active, becomes downed, and
    returns to play.
  - **Owns:** Current and maximum Ward; stable, warning, critical, empty, downed,
    and re-entry states; applying damage after Combat calculates mitigation;
    Ward restoration and reinforcement; downing and targeting-ineligibility
    state; cooperative revival state and progress; the one-use solo emergency
    recovery opportunity; revived Ward levels; temporary re-entry protection;
    and player-state facts used to determine all-humans-down defeat.
  - **Does not own:** Boss attack timing; Defend performance; rhythm judgments
    during recovery phrases; encounter defeat; consumable definitions; or
    revival presentation.
  - **Depends on:** Combat effects; Rhythm Gameplay for recovery performance;
    Boss Encounters and authored Activity Maps for eligible recovery boundaries;
    Abilities & Cooperative Actions; Multiplayer roster state; consumable
    effects; and applicable difficulty/scaling policy.
  - **Used by:** Combat; Boss Encounters; Multiplayer; Solo Support; Abilities &
    Cooperative Actions; Results & Feedback; Audio Presentation; and Analytics &
    Playtest Evidence.

  The approved semantic handoff is: Rhythm Gameplay produces musical
  performance; Combat converts it into an effect; Player Survival & Recovery
  applies player-directed effects to Ward and survival state; Boss Encounters
  applies boss-directed effects to Resolve and determines encounter victory or
  defeat. Results and Analytics consume facts emitted by both systems.
- **Consequences:** Combat calculation is separated from mutable player-survival
  state and from encounter lifecycle state. Defend performance is interpreted by
  Combat; its resulting mitigation or reinforcement is applied by Survival.
  Survival emits all-human survival state but does not declare encounter defeat.
  Consumable/item definitions and ability definitions remain outside both
  systems, while their resolved combat effects pass through the same calculation
  and application boundaries.
- **Detailed-spec impact:** Both systems require detailed specification and will
  be major, explicitly separated sections of `COMBAT.md`, the second gameplay
  specification after the Content Authoring contract. A separate
  `PLAYER_SURVIVAL.md` is not required unless the shared document later becomes
  unmanageably large.
- **Deferred:** Precise formulae, stat curves, caps, Ward values, damage values,
  restoration values, and contribution tables belong to later balance sheets
  informed by playtesting. Client/server authority and anti-cheat enforcement
  belong to `TECHNICAL_ARCHITECTURE.md`. Risk Bonus ownership is finalized by
  SM-05, SM-07, and SM-12.

### SM-05: Boss Encounters and Tactical Positioning & Movement boundaries

- **Owner answer:** The proposed separation, content boundary, and shared
  specification are approved as presented: “Yes.”
- **Map decision:** **Boss Encounters** and **Tactical Positioning & Movement**
  are separate **First-Release Runtime Systems**. Tactical Positioning &
  Movement is an explicit refinement added to the approved SM-01 inventory
  because it has independent state and several consumers.

  **Boss Encounters**
  - **Purpose:** Orchestrate a complete song-shaped boss attempt and determine
    its shared outcome.
  - **Owns:** The active encounter lifecycle from initialization through victory
    or defeat; the five flexible encounter functions aligned to the song;
    current Resolve layer, authored layer-opening boundaries, Momentum banking
    and application, and finishing-cadence evaluation; applying boss-directed
    effects calculated by Combat; Telegraph, Commit, Impact, and Recovery attack
    progression; choosing only authored attacks and events that are currently
    legal; boss targeting, hazards, phase events, recovery windows, and group-
    action opportunities; shared victory or defeat and its exact reason; and
    execution of reusable boss rules from an individual boss content package.
  - **Does not own:** Rhythm judgments; combat-effect formulae; Ward state;
    multiplayer membership; reward calculation or grants; or creation of boss
    content.
  - **Depends on:** Rhythm Gameplay and musical boundaries; Combat; Player
    Survival & Recovery; Tactical Positioning & Movement; Abilities &
    Cooperative Actions; Solo Support; Multiplayer roster state; difficulty and
    scaling policy; and authored song/chart/encounter packages including
    Activity Maps.
  - **Used by:** Rhythm Gameplay for lifecycle configuration; Combat; Player
    Survival & Recovery; Tactical Positioning & Movement; Abilities &
    Cooperative Actions; Solo Support; Multiplayer; Results & Feedback; Rewards,
    Loot & Economy; Audio Presentation; and Analytics & Playtest Evidence.

  **Tactical Positioning & Movement**
  - **Purpose:** Maintain each player's legal arena location and movement state
    while exposing tactical risk, cover, and attack geometry to other systems.
  - **Owns:** The arena's directional location graph; current location, travel
    state, and legal destinations; movement charges, beat-based recovery, and
    rhythm-settling periods; multi-edge travel and involuntary displacement;
    shared-location and no-body-blocking rules; cover and hazard occupancy;
    Near/Middle/Rear risk-tier state; application of authored graph changes
    requested by Boss Encounters; and position/movement facts consumed by other
    systems.
  - **Does not own:** Input bindings; boss attack selection or impact timing;
    combat modifier calculations; Ward damage; reward calculation or grants; or
    authoring an arena graph.
  - **Depends on:** Boss Encounters and authored arena configuration; Rhythm
    Gameplay for beat boundaries; normalized movement requests from Input;
    Player Survival & Recovery for active/downed state; difficulty/scaling
    invariants; and Solo Support for acolyte formation behavior.
  - **Used by:** Boss Encounters; Combat; Player Survival & Recovery; Solo
    Support; Abilities & Cooperative Actions; Rewards, Loot & Economy; Results &
    Feedback; UI Presentation; Audio Presentation; and Analytics & Playtest
    Evidence.

  Individual boss packages contain their song references, arena graph, phase
  markers, permitted attacks, hazards, events, and presentation references.
  Boss Encounters owns how those packages execute; Song, Chart & Encounter
  Authoring owns creating and validating them. Boss attacks remain an internal
  Boss Encounters subsystem rather than another top-level system.
- **Consequences:** Encounter outcome now has a single owner distinct from
  Combat calculations and survival state. Positioning state is reusable across
  bosses and can supply consistent location, exposure, cover, movement, and
  graph guarantees without Boss Encounters or Combat owning duplicate copies.
  Positioning emits exposure and risk facts; Combat applies combat consequences,
  while Rewards, Loot & Economy will own eventual reward-bonus calculation and
  grants.
- **Detailed-spec impact:** Both systems require detailed specification and will
  be major, explicitly separated sections of `BOSS_ENCOUNTERS.md`, the third
  gameplay specification after the Content Authoring contract. A separate
  `POSITIONING_AND_MOVEMENT.md` is not required unless the shared document later
  becomes unmanageably large.
- **Deferred:** Exact encounter state transitions, event-conflict arbitration,
  attack-selection rules, target-selection details, graph-transformation rules,
  and movement request semantics belong in `BOSS_ENCOUNTERS.md`. Numeric Resolve,
  Momentum, movement, exposure, cover, and hazard values remain playtest-tunable
  or belong in balance tables. Risk Bonus banking and reward ownership are
  finalized in SM-07 and SM-12. Runtime authority, replication, and anti-cheat
  belong to `TECHNICAL_ARCHITECTURE.md`.

### SM-06: Abilities, cooperative actions, and Solo Support boundaries

- **Owner answer:** The proposed two-system separation and shared specification
  are approved as presented: “Yes.”
- **Map decision:** **Abilities & Cooperative Actions** and **Solo Support** are
  separate **First-Release Runtime Systems**.

  **Abilities & Cooperative Actions**
  - **Purpose:** Manage music-aligned personal powers and coordinated band
    performances after Rhythm Gameplay and Combat determine each player's
    contribution.
  - **Owns:** Signature Special, Band Call, and Crescendo definitions; Hype
    charge, readiness, arming, consumption, and encounter reset; Band Call
    readiness, initiation allowance, shared lockout, invitations, and
    cancellation; Join In acceptance and participant eligibility; scheduling
    accepted actions at eligible musical boundaries; combining contributions so
    weak play cannot reduce another player's result; Band Call and Crescendo
    result tiers; and resolving defined effects through Combat, Player Survival
    & Recovery, and Boss Encounters.
  - **Does not own:** Special intent selection or contribution routing; authored
    Crescendo opportunity selection; cooperative revival state or progress;
    item/loadout ownership; build rules; rhythm judgments; boss Resolve state;
    Ward state; or presentation.
  - **Depends on:** Rhythm Gameplay; Combat; Boss Encounters and authored
    Activity Maps; Player Survival & Recovery; Multiplayer roster state; selected
    abilities from Items, Equipment & Loadouts; resolved build effects; Input;
    and applicable difficulty/scaling policy.
  - **Used by:** Combat; Boss Encounters; Player Survival & Recovery; Solo
    Support; Multiplayer; Results & Feedback; Audio Presentation; and Analytics &
    Playtest Evidence.

  **Solo Support**
  - **Purpose:** Make solo encounters complete through visible, predictable
    assistance without fabricating rhythm performance.
  - **Owns:** Vanguard, Warden, and Herald runtime state; their fixed support
    functions and authored musical cadences; active, suppressed, and recovering
    states; automatic repositioning and formation requests; their capped, fixed
    Band Call and Crescendo contributions; and the rule that acolytes never play
    charts, receive judgments, earn performance credit, or independently break
    Resolve.
  - **Does not own:** The solo emergency recovery challenge; general solo
    difficulty or player-count scaling; tactical-location state; musical
    judgments; or the player-directed combat effects it requests.
  - **Depends on:** Rhythm Gameplay and musical boundaries; Combat; Boss
    Encounters; Tactical Positioning & Movement; Player Survival & Recovery;
    Abilities & Cooperative Actions; authored Activity Maps; and applicable
    difficulty/scaling policy.
  - **Used by:** Combat; Boss Encounters; Abilities & Cooperative Actions;
    Results & Feedback; UI Presentation; Audio Presentation; and Analytics &
    Playtest Evidence.

  Combat retains Special intent selection and contribution routing. Boss
  Encounters opens authored Crescendo and group-action opportunities. Player
  Survival & Recovery retains cooperative revival state and progress.
- **Consequences:** Personal abilities and opt-in group performances can share
  readiness, musical-boundary scheduling, effect-definition, and contribution-
  combination concepts without absorbing unrelated revival state. Solo Support
  remains independently legible and testable while contributing only the fixed
  support explicitly permitted by the GDD.
- **Detailed-spec impact:** Both systems require detailed specification and will
  be separate major sections of `ABILITIES_AND_COOPERATIVE_ACTIONS.md`. A
  separate `SOLO_SUPPORT.md` is not initially required. Final priority is
  assigned in SM-17 after Multiplayer and content-authoring dependencies are
  mapped.
- **Deferred:** Exact ability catalogs, final names, numeric Hype/Call rates,
  effect strengths, tier thresholds, shared-lockout values, candidate-window
  delays, and acolyte values belong to the detailed specification, naming pass,
  content catalogs, balance tables, and playtesting as appropriate. Runtime
  authority and replication belong to `TECHNICAL_ARCHITECTURE.md`.

### SM-07: Difficulty & Scaling classification and ownership

- **Owner answer:** The owner delegated the decision: “Do what you think is
  best.” The recommended cross-cutting-policy model is therefore adopted.
- **Map decision:** **Difficulty & Scaling** is a **Cross-Cutting Requirement**
  with one canonical design policy rather than an independent runtime system.
  - **Purpose:** Ensure difficulty and human-player count change challenge
    appropriately without changing song speed, encounter identity, musical
    fairness, accessibility rights, or maximum available contribution.
  - **Owns:** The canonical Easy, Normal, and Hard profiles; rules for normalized
    maximum contribution across chart densities; one-to-six-human scaling
    invariants; duplicate-instrument neutrality; allowed scaling dimensions for
    timing windows, Resolve requirements, pressure, damage, telegraphs,
    recovery, target counts, and reward modifiers; solo/co-op completion-
    equivalence goals; Cohesion Bonus principles and limits; the rule that
    accessibility options never reduce rewards; and positional risk/reward
    ratios, caps, and invariants.
  - **Does not own:** Mutable gameplay state; selected-difficulty unlock state;
    current roster; current position; runtime combat, encounter, or reward state;
    or final numeric tuning inside another system's permitted dimensions.
  - **Depends on:** The GDD's fairness rules; the capabilities and parameters
    exposed by Rhythm Gameplay, Combat, Boss Encounters, Tactical Positioning &
    Movement, Player Survival & Recovery, Multiplayer, Rewards & Economy, and
    Progression; authored content constraints; and segmented playtest evidence.
  - **Used by:** Rhythm Gameplay; Combat; Boss Encounters; Tactical Positioning &
    Movement; Player Survival & Recovery; Abilities & Cooperative Actions; Solo
    Support; Multiplayer; Song, Chart & Encounter Authoring; Rewards, Loot &
    Economy; Progression; Results & Feedback; and Analytics & Playtest Evidence.
  - **First-release role:** Cross-Cutting Requirement.
  - **Needs detailed spec?:** No separate gameplay-system specification. Every
    affected detailed spec must contain a Difficulty & Scaling section. A shared
    `BALANCE_FRAMEWORK.md` will later collect the canonical matrices, curves,
    caps, and playtest-adjustable values.
  - **Major unresolved decisions:** Exact per-difficulty parameter tables,
    player-count curves, Cohesion thresholds, positional-risk values and caps,
    reward modifiers, and acceptable solo/co-op variance remain for the balance
    framework and playtesting within the approved invariants.
  - **Source references:** GD-07, GD-08, GD-13 through GD-21, GD-24 through
    GD-28, GD-31, GD-33, and GD-34.
- **Consequences:** Domain systems apply only their respective parts of the
  canonical policy: Rhythm applies timing and chart normalization; Combat
  applies contribution and damage modifiers; Boss Encounters applies Resolve,
  pressure, telegraph, target-count, and recovery profiles; Multiplayer supplies
  roster facts; Progression owns difficulty unlock state; Tactical Positioning
  owns current risk tier; and Rewards, Loot & Economy owns banked and unbanked
  Risk Bonus and the final reward bonus. No domain may silently define a
  conflicting local difficulty model.
- **Detailed-spec impact:** `RHYTHM_GAMEPLAY.md`, `COMBAT.md`,
  `BOSS_ENCOUNTERS.md`, `MULTIPLAYER.md`, `PROGRESSION.md`, and the later rewards/
  economy specification must each state which canonical scaling fields they
  consume and apply. `BALANCE_FRAMEWORK.md` follows the behavioral specs rather
  than preceding them.
- **Deferred:** Technical representation, configuration loading, versioning, and
  runtime resolution of the canonical profiles belong to
  `TECHNICAL_ARCHITECTURE.md`. Numeric tuning remains playtest-driven.

### SM-08: Multiplayer and Communication & Safety boundaries

- **Owner answer:** The proposed runtime and cross-cutting boundaries are
  approved as presented: “Yes.”
- **Map decision:** **Multiplayer Sessions, Parties & Matchmaking** is a
  **First-Release Runtime System**. **Communication & Safety** is a
  **Cross-Cutting Requirement** implemented through Multiplayer and every other
  affected domain.

  **Multiplayer Sessions, Parties & Matchmaking**
  - **Purpose:** Move consenting players from shard selection into a stable
    encounter roster and then into individually chosen follow-up actions.
  - **Owns:** Current-party membership, leadership, proposals, and individual
    consent; public matching by boss, difficulty, and appropriate connection
    region; server-owned public groups without a player host; queue state and
    the two-player launch/wait/leave choice; ready and staging state; roster and
    loadout lock timing; the no-join-in-progress rule; encounter membership and
    active-roster facts; disconnect, rejoin grace, safe-boundary return, AFK
    warning, inactivity, and one permitted resume; rematch groups, Stay with
    Band, refill, leave, and return-to-hub actions; party-leader transfer; and
    preset-ping delivery, rate limiting, and per-player ping muting.
  - **Does not own:** Loadout contents; boss outcomes; musical judgments; combat,
    Ward, position, Hype, Call, consumable, or downed state; reward or progression
    calculation; or persistent player progression. It coordinates state
    preservation and restoration while each gameplay system remains authoritative
    for the meaning and value of its state.
  - **Depends on:** Order Hub & Navigation for shard entry; Items, Equipment &
    Loadouts for valid staging choices; Player Data for party-relevant persistent
    facts; Boss Encounters; Rhythm Gameplay; Combat; Player Survival & Recovery;
    Tactical Positioning & Movement; Abilities & Cooperative Actions; Difficulty
    & Scaling; Results & Feedback; Communication & Safety policy; and external
    Roblox matchmaking, transport, connection, block, and report capabilities.
  - **Used by:** Boss Encounters; Difficulty & Scaling consumers; Abilities &
    Cooperative Actions; Player Survival & Recovery; Results & Feedback;
    Rewards, Loot & Economy; Player/Campaign/Boss Mastery; UI Presentation; Audio
    Presentation; and Analytics & Playtest Evidence.

  **Communication & Safety**
  - **Purpose:** Make core cooperation understandable and safe without requiring
    voice or unrestricted text and without relying on punitive player policing.
  - **Owns:** The allowed preset-ping vocabulary and behavior policy; the rule
    that automatic critical cues cannot be muted with player pings; safe defaults;
    anti-coercion policy; and structural protections including no friendly fire,
    vote-kick, body blocking, negative contribution, forced follow-up, or spending
    another player's resources.
  - **Does not own:** Runtime party/session state; platform moderation records;
    Roblox filtering, block, report, privacy, or age-control behavior; or the
    presentation of automatic gameplay cues owned semantically by domain
    systems.
  - **Depends on:** Roblox platform safety capabilities; Multiplayer; UI, Input,
    Settings & Accessibility; Audio Presentation; Results & Feedback; and every
    gameplay system responsible for enforcing a structural invariant.
  - **Used by:** Multiplayer; Combat; Boss Encounters; Tactical Positioning &
    Movement; Abilities & Cooperative Actions; Results & Feedback; UI
    Presentation; Audio Presentation; and Analytics & Playtest Evidence.

  Multiplayer owns connected, absent, inactive, and active-roster facts. Rhythm,
  Combat, and Boss Encounters emit performance and contribution facts. Rewards
  and Progression determine participation eligibility and grants from those
  facts. Difficulty & Scaling defines roster-change invariants; Boss Encounters
  applies them at permitted boundaries.
- **Consequences:** Public sessions cannot acquire accidental host authority or
  force follow-up choices. Disconnect and AFK handling have one lifecycle owner
  without transferring ownership of gameplay state. Safety requirements remain
  enforceable across domains, while Roblox-provided capabilities remain explicit
  external dependencies rather than being misrepresented as game-owned systems.
- **Detailed-spec impact:** `MULTIPLAYER.md` is required, with distinct sections
  for parties, public matchmaking, staging, encounter membership, reconnect/AFK
  behavior, roster changes, rematching, preset communication, safety invariants,
  and participation evidence. Its final priority is assigned in SM-17.
- **Deferred:** Exact queue, ready, grace, AFK, refill, and ping-rate values;
  matchmaking-region and skill inputs; service failure behavior; reconnect
  transport; active-roster rescaling mechanics; localization; moderation
  integration; and reward-eligibility thresholds belong to the detailed spec,
  balance framework, technical architecture, platform integration, and testing
  as appropriate.

### SM-09: Song, Chart & Encounter Authoring boundary

- **Owner answer:** The single-system proposal is approved with an explicit
  deployment and reuse boundary: “Yes but it shouldn't live in Roblox. There's
  already significant tooling for processing songs.”
- **Map decision:** **Song, Chart & Encounter Authoring** is one
  **First-Release-Supporting Production System** implemented as an offline,
  platform-neutral toolchain outside the Roblox client/server runtime.
  - **Purpose:** Turn approved music and encounter concepts into reviewed,
    validated, versioned content packages that runtime systems can execute
    without inventing missing behavior.
  - **Owns:** Song intake records for masters, stems, provenance, rights status,
    lyrics, duration, and arrangement context; automated suggestions for tempo,
    beats, onsets, holds, rests, dropouts, energy, and structure; human-authored
    beat grids, instrument charts, three-input mappings, holds, rests, phrases,
    and passages; difficulty derivation and normalized-contribution metadata;
    per-instrument/per-difficulty Activity Maps; ensemble eligibility and roster-
    coverage data; encounter timelines containing functions, Resolve openings,
    boss events, movement/recovery moments, group opportunities, and finishing
    cadence; preview, review, validation, approval, versioning, and runtime
    export; and the rule that automation/AI may suggest but never approve or
    publish content.
  - **Internal capabilities:** Song intake and analysis; chart and difficulty
    authoring; encounter timeline authoring; validation and approval; and runtime
    package export. These are one content lifecycle rather than five top-level
    systems.
  - **Does not own:** Roblox runtime chart playback; rhythm judgments; boss or
    encounter execution; runtime item/reward behavior; game-client authoring UI;
    runtime reinterpretation of approved data; or the domain semantics that
    gameplay system specs require its validators to enforce.
  - **Depends on:** Approved source music and provenance; encounter/content
    briefs; `RHYTHM_GAMEPLAY.md`; `BOSS_ENCOUNTERS.md`; Player Survival &
    Recovery, Abilities & Cooperative Actions, Multiplayer, Difficulty &
    Scaling, and Accessibility requirements; human musical/design/technical
    review; and external processing tools.
  - **Used by:** Rhythm Gameplay; Boss Encounters; Tactical Positioning &
    Movement; Player Survival & Recovery; Abilities & Cooperative Actions; Solo
    Support; Difficulty & Scaling validators; Multiplayer validation; Audio
    Presentation; and Analytics & Playtest Evidence.
  - **First-release role:** First-Release-Supporting Production System.
  - **Needs detailed spec?:** Yes—`CONTENT_AUTHORING.md`, after the Rhythm and
    Boss Encounter content contracts are established.
  - **Major unresolved decisions:** The expanded runtime package semantics;
    Activity Map and encounter-timeline contract; version/approval/rework
    lifecycle; validator ownership and evidence format; Roblox adaptation/export
    boundary; and the minimum authoring/review surface needed for the first three
    bosses.
  - **Source references:** GD-02, GD-04 through GD-10, GD-14 through GD-19,
    GD-21, GD-29, GD-31, GD-33, and GD-34.

  The starting implementation asset is the existing root-owned TypeScript
  [`tools/chart-pipeline/`](tools/chart-pipeline/README.md), which already builds
  and validates a deterministic, platform-neutral song bundle independent of the
  web game or Roblox. The authoring system extends that maintained toolchain and
  its bundle contract rather than replacing it. Song-specific processing scripts
  and experiments may supply inputs or proven techniques, but generated artifacts
  are not the source of truth when the maintained pipeline or approved specs
  disagree.
- **Consequences:** Roblox receives only approved exported runtime packages
  through an adapter/export step. No analysis dependencies, desktop/CLI
  authoring surfaces, source stems, or approval workflow are required to ship in
  the Roblox experience. Runtime systems consume authored semantics and reject
  incompatible packages; they do not edit or silently reinterpret them. A future
  polished offline creator is a maturity stage of this system rather than a new
  game system.
- **Detailed-spec impact:** `CONTENT_AUTHORING.md` must describe the end-to-end
  offline workflow, extension of the existing chart pipeline, ownership of
  source versus generated data, human approval gates, validator aggregation,
  runtime-package semantics, and Roblox export boundary. Exact schemas, file
  formats, CLI/UI architecture, storage, and adapter implementation remain for
  technical specifications after the gameplay content contracts are settled.
- **Deferred:** The GDD's pipeline upgrade remains a separate future
  implementation task. This interview authorizes design mapping only, not tool
  changes. Exact dependencies, migration from the current bundle schema, source-
  asset storage, licensing operations, and Roblox asset publication belong to
  later technical and production planning.

### SM-10: Items, Equipment & Loadouts boundary

- **Owner answer:** The proposed unified system boundary is approved as
  presented: “Yes.”
- **Map decision:** **Items, Equipment & Loadouts** is one **First-Release
  Runtime System**. Inventory and consumables remain responsibilities within it
  rather than separate first-release systems.
  - **Purpose:** Represent what a player owns and what they bring into an
    encounter, producing one validated loadout that other systems can consume.
  - **Owns:** Item, consumable, and cosmetic definitions; player-owned item and
    appearance collections; fixed item stats, traits, tier, and upgrade-rank
    state; Instrument, Ward Core, and Resonator slots; equipped Signature
    Special and Band Call references; two prepared-consumable references;
    separate stat-free appearance slots; equip, unequip, and loadout-validation
    rules; staging and encounter loadout locking; the rule that the full
    inventory cannot change during combat; consumable quantities, prepared
    charges, and consumption authorization; resolved equipment modifiers
    consumed after rhythm scoring; and enforcement of the GDD's prohibited-
    modifier list.
  - **Does not own:** Earning, drops, crafting, salvage, purchases, upgrade
    transactions, currencies, or costs; Signature Special or Band Call runtime
    behavior; build-specialization rules; combat-effect calculation; persistence
    implementation; or inventory/loadout presentation.
  - **Depends on:** Player Data for durability; Rewards, Loot & Economy for
    acquisition and upgrade transactions; Commerce for purchases; Player/
    Campaign/Boss Mastery for unlock eligibility; Builds & Specialization for
    the complete resolved player configuration; Abilities & Cooperative Actions
    for valid action definitions; content catalogs; and UI for player-directed
    editing.
  - **Used by:** Rhythm Gameplay for selected instrument/chart identity; Combat;
    Player Survival & Recovery; Boss Encounters; Abilities & Cooperative Actions;
    Multiplayer staging and lock state; Builds & Specialization; Rewards, Loot &
    Economy; Commerce; Results & Feedback; UI Presentation; Audio Presentation;
    and Analytics & Playtest Evidence.
  - **First-release role:** First-Release Runtime System.
  - **Needs detailed spec?:** Yes—`ITEMS_AND_EQUIPMENT.md`; final dependency
    priority is confirmed in SM-17.
  - **Major unresolved decisions:** Item identity and catalog-versus-instance
    semantics; slot and equip validation; loadout lock transitions; consumable
    lifecycle; modifier resolution/allowlist; cosmetic ownership; extension
    points; and the exact boundary of item mutation requests from Economy and
    Commerce.
  - **Source references:** GD-01, GD-17, GD-18, GD-24 through GD-30, GD-32, and
    GD-34.

  Player Data persists item and loadout records, while Items, Equipment &
  Loadouts remains authoritative for what those records mean and which mutations
  are valid. Items authorizes and records consumable use; Combat, Player Survival
  & Recovery, or Abilities & Cooperative Actions resolves the requested effect.
- **Consequences:** The first release avoids inventing trading, capacity
  management, or a separate inventory lifecycle. A future feature that adds
  independent inventory rules may justify a later split, but current item
  ownership, equipment, cosmetics, loadout selection, and consumables form one
  coherent domain. Economy and Commerce can request validated mutations without
  taking ownership of item semantics.
- **Detailed-spec impact:** `ITEMS_AND_EQUIPMENT.md` must settle item identity,
  definitions versus owned records, slot rules, loadout locking, consumable
  lifecycle, modifier validation, cosmetic separation, persistence semantics,
  mutation requests, and safe extension points. Schemas and storage remain for
  technical architecture.
- **Deferred:** Final item names, catalogs, stat ranges, tiers, traits, charges,
  power budgets, advanced slots, sockets, sets, and later inventory complexity
  belong to the naming pass, content catalogs, balance framework, future system
  specs, and playtesting.

### SM-11: Builds & Specialization boundary

- **Owner answer:** The proposed separate system and detailed specification are
  approved: “Yes that's fine.”
- **Map decision:** **Builds & Specialization** is a separate **First-Release
  Runtime System**.
  - **Purpose:** Turn unlocked specialization options into a valid behavior-
    changing build without changing rhythm fairness, instrument freedom, or the
    combat control set.
  - **Owns:** The universal offense, defense, support, and Hype/utility option
    categories; one major behavior-changing slot and three supporting-rule slots;
    cross-category mixing for every instrument; beginner role presets and the
    advanced-editor access gate; three saved build presets; free respec outside
    active combat and build locking during encounters; option compatibility,
    stacking order, shared power budgets, and synergy caps; build-specific
    modifier definitions; production of a validated resolved modifier set; and
    enforcement that builds cannot alter charts, judgments, timing windows,
    controls, core movement timing, accessibility rights, or reward eligibility.
  - **Does not own:** Gear or item stats; base Signature Special, Band Call, or
    Crescendo definitions; progression awards; persistent storage; presentation;
    or direct mutation of Combat, Survival, Positioning, or Ability state.
  - **Depends on:** Player/Campaign/Boss Mastery for unlocked options; Player
    Data for durable presets; Items, Equipment & Loadouts for the complete player
    configuration; Abilities & Cooperative Actions for permitted ability hooks;
    Combat, Player Survival & Recovery, and Tactical Positioning & Movement for
    permitted domain hooks; Difficulty & Scaling and fairness invariants; and UI
    for preset/editor interaction.
  - **Used by:** Items, Equipment & Loadouts; Combat; Player Survival & Recovery;
    Tactical Positioning & Movement; Abilities & Cooperative Actions; Solo
    Support where explicitly permitted; Multiplayer staging validation; Results
    & Feedback; UI Presentation; and Analytics & Playtest Evidence.
  - **First-release role:** First-Release Runtime System.
  - **Needs detailed spec?:** Yes—`BUILDS_AND_SPECIALIZATION.md`; final priority
    is assigned in SM-17.
  - **Major unresolved decisions:** Final player-facing terminology; option and
    preset catalogs; exact category/slot counts if playtesting changes the
    baseline; hook contracts; incompatibility and stacking rules; synergy caps;
    power budgets; preset versioning; and behavior when an option changes or is
    retired.
  - **Source references:** GD-17, GD-18, GD-24 through GD-29, GD-32, and GD-34.

  Progression owns which options are unlocked. Builds owns configuration
  validity and resolved modifier semantics. Player Data persists presets. Items
  owns gear. Abilities owns base ability behavior. Each consuming gameplay
  system applies only explicitly permitted resolved modifiers within its own
  domain contract.
- **Consequences:** Gear-based power and behavior-changing specialization remain
  distinct even when presented together in a loadout flow. The build system can
  grow through new options and interactions without acquiring new combat buttons
  or direct ownership of other systems' state. Cross-system modifier hooks must
  be explicitly allowed and validated rather than inferred from item-like data.
- **Detailed-spec impact:** `BUILDS_AND_SPECIALIZATION.md` is required because
  combinatorial validation, effect hooks, stacking, caps, presets, and fairness
  rules affect several systems and would be unsafe to bury inside
  `ITEMS_AND_EQUIPMENT.md`. Its final dependency priority is assigned in SM-17.
- **Deferred:** Working terms may be used internally but cannot ship. Final names
  and option language require the dedicated naming-and-tone pass. Exact effect
  catalog, unlock pace, values, caps, and balance require content design and
  testing. Persistent schema and migration belong to technical architecture.

### SM-12: Rewards, Loot & Economy and Commerce boundaries

- **Owner answer:** The proposed separation and shared specification are
  approved: “Sure.”
- **Map decision:** **Rewards, Loot & Economy** and **Commerce** are separate
  **First-Release Runtime Systems** documented in one shared detailed design
  specification.

  **Rewards, Loot & Economy**
  - **Purpose:** Turn encounter outcomes and player participation into fair,
    durable rewards and deterministic resource transactions.
  - **Owns:** Reward eligibility and calculation from outcome, meaningful
    participation, difficulty, performance, and positional risk; banked and
    unbanked Risk Bonus state and its bounded final effect; Cohesion Bonus reward
    effects; general-resource and boss-material balances; boss loot pools and
    deterministic crafting progress; earned random drops of complete fixed-stat
    items; guaranteed, first-clear, and signature-material rules; duplicate
    salvage; deterministic crafting and upgrade transactions; consumable costs
    and approved resource sinks; idempotent reward-transaction orchestration;
    and economy prohibitions including no repair/death taxes, paid luck, respec
    fees, energy limits, or daily earning caps.
  - **Does not own:** Boss outcome; raw musical/combat/participation facts; item
    semantics or loadout validity; progression-track semantics; paid offers;
    presentation; or persistence implementation. Deep crafting is not part of
    the first-release system.
  - **Depends on:** Boss Encounters for outcome; Rhythm Gameplay and Combat for
    performance/contribution facts; Tactical Positioning & Movement for risk-
    tier events; Multiplayer for active/absent/inactive participation facts;
    Difficulty & Scaling; Items, Equipment & Loadouts; Player/Campaign/Boss
    Mastery; Player Data; content catalogs; and Results & Feedback.
  - **Used by:** Items, Equipment & Loadouts; Player/Campaign/Boss Mastery;
    Commerce equivalence validation; Results & Feedback; Order Hub & Navigation;
    UI Presentation; Player Data; and Analytics & Playtest Evidence.

  **Commerce**
  - **Purpose:** Handle optional Robux purchases without bypassing progression
    fairness or exploiting the target audience.
  - **Owns:** The paid product catalog; store-unlock eligibility; purchase
    initiation, confirmation, receipt handling, and duplicate-purchase
    protection; mapping every stat-bearing paid item to an exact earnable
    equivalent; tier-ceiling and stat-budget validation; granting purchased
    equipment only at the player's currently unlocked tier; enforcing permitted
    and prohibited product categories; and preventing paid prompts during
    battle, recovery, defeat, results, and retry.
  - **Does not own:** The granted item's semantics; earned reward calculation;
    campaign progression; Roblox Marketplace behavior; store presentation; or
    persistent storage implementation.
  - **Depends on:** Roblox Marketplace as an external platform capability;
    Items, Equipment & Loadouts; Rewards, Loot & Economy for equivalence and
    balance comparisons; Player/Campaign/Boss Mastery for current tier and store
    unlock prerequisites; Onboarding state; Player Data; UI Presentation; and
    Communication & Safety/age-appropriate policy.
  - **Used by:** Items, Equipment & Loadouts; Player Data; Order Hub & Navigation;
    UI Presentation; Results/flow safeguards; and Analytics & Playtest Evidence.

  Rewards, Loot & Economy owns a once-only transaction plan. Items and
  Progression validate and apply their domain mutations. Player Data durably
  commits the resulting transaction. Commerce owns why and whether a paid grant
  is valid; Items owns the granted item afterward.
- **Consequences:** Paid and earned transactions cannot blur together or take
  ownership of item/progression semantics. Bounded deterministic first-release
  crafting remains available without implying the deferred deep-crafting system.
  Randomness may exist in earned complete-item drops but never in paid products.
  Transaction orchestration has one domain owner while atomic durability remains
  a technical Player Data concern.
- **Detailed-spec impact:** Both systems require detailed design and will be
  separate major sections of `REWARDS_AND_ECONOMY.md`, covering encounter
  rewards, resources, loot, deterministic paths, crafting/upgrades/salvage,
  transaction guarantees, and Commerce safeguards. Final priority is assigned
  in SM-17. Technical receipt validation, security, atomic persistence, retry,
  and recovery belong to technical architecture.
- **Deferred:** Exact reward quantities, drop chances, deterministic-path length,
  resource costs, item/loot catalogs, upgrade tables, risk/cohesion caps, store
  catalog, Robux prices, and balance tests belong to content catalogs,
  `BALANCE_FRAMEWORK.md`, Commerce review, and playtesting. Deep crafting and all
  unapproved monetization categories remain deferred or prohibited as stated by
  the GDD.

### SM-13: Unified Progression boundary

- **Owner answer:** The proposed unified system is approved as presented:
  “Yes.”
- **Map decision:** **Progression** is one **First-Release Runtime System** with
  three coordinated tracks: general player progression, campaign progression,
  and boss mastery/personal bests.
  - **Purpose:** Preserve meaningful long-term advancement and determine what
    content, options, and world states the player has earned.
  - **Owns:** General player progression and system unlocks; campaign
    destinations, first-clear state, and recovered Shattered Song fragments;
    per-boss difficulty availability and the Normal-victory requirement for
    Hard; boss mastery ranks and milestones; personal-best records by boss,
    instrument, and difficulty; unlock eligibility for specialization options,
    recipes, equipment choices, cosmetics, lore, titles, and saved-build
    capacity; current campaign tier and recommended-power information; old-item
    uplift eligibility; progression effects of meaningful victory and failure;
    campaign-driven hub-restoration state; and the rules preventing daily
    streaks, expiring progress, energy, or exclusive rotating rewards.
  - **Does not own:** Raw rhythm, combat, outcome, or roster facts; the canonical
    meaningful-participation result; item ownership; reward/economy
    orchestration; milestone content itself; persistence implementation; hub
    presentation; or results presentation.
  - **Depends on:** Boss Encounters for shared outcome and first-clear target;
    Multiplayer, Rhythm Gameplay, and Combat for participation evidence; Rewards,
    Loot & Economy for the canonical meaningful-participation result and once-
    only transaction; Items, Equipment & Loadouts and Builds & Specialization
    for unlock consumers; Difficulty & Scaling; Player Data; progression and
    milestone content catalogs; and Results & Feedback.
  - **Used by:** Items, Equipment & Loadouts; Builds & Specialization; Rewards,
    Loot & Economy; Commerce; Multiplayer difficulty selection; Order Hub &
    Navigation; Onboarding; Results & Feedback; UI Presentation; Player Data;
    and Analytics & Playtest Evidence.
  - **First-release role:** First-Release Runtime System.
  - **Needs detailed spec?:** Yes—`PROGRESSION.md`.
  - **Major unresolved decisions:** Track state transitions; first-clear and
    replay idempotency; participation-to-progress rules; difficulty-unlock
    records; mastery milestones; personal-best metrics and update rules; unlock
    catalog; recommended-power semantics; current-tier and old-item uplift rules;
    and the exact hub-state output contract.
  - **Source references:** GD-01, GD-08, GD-13, GD-21, GD-24 through GD-32, and
    GD-34.

  Rewards, Loot & Economy owns the canonical meaningful-participation result and
  orchestrates the once-only overall transaction. Progression calculates and
  validates its own player-progress, campaign, mastery, difficulty-unlock, and
  personal-best mutations. Player Data persists the committed state.
- **Consequences:** The three tracks share outcome, transaction, and unlock
  semantics without becoming independent systems prematurely. Narrative,
  milestone, reward, and hub content may vary without taking ownership of
  progression state. Future campaign or mastery complexity may justify a split,
  but only through an explicit systems-map amendment.
- **Detailed-spec impact:** `PROGRESSION.md` is required to define the three
  tracks, their state transitions, inputs, mutation rules, unlock contracts,
  first-clear behavior, personal bests, and hub outputs. Exact amounts, ranks,
  thresholds, rates, and recommended-power curves remain in
  `BALANCE_FRAMEWORK.md` and content catalogs.
- **Deferred:** Final progression terminology, rank counts, milestone catalog,
  unlock pace, reward values, personal-best metrics, and balance are downstream
  design/content/test work. Persistent schema, atomic commit, rollback, and
  migration belong to technical architecture.

### SM-14: Hub, Onboarding, and Results classifications

- **Owner answer:** The proposed classifications and boundaries are approved:
  “Sure.”
- **Map decision:** **Order Hub & Navigation** and **Results & Feedback** are
  **Orchestrated Experiences/Surfaces**. **Onboarding & Practice** is a
  **First-Release Runtime System**.

  **Order Hub & Navigation**
  - **Purpose:** Give players a readable physical home and fast routes into the
    game's encounter, preparation, progression, practice, social, and voluntary-
    store flows.
  - **Owns:** Spatial and navigational composition of the phasing-shard field;
    stable shard interaction points; practice, workshop, story/mastery, social,
    and store anchors; fast access to unlocked shard tiers; application of
    Progression's restoration state to the visible hub; routes into encounter
    selection, loadouts, upgrades, practice, and Commerce; and stable landmarks/
    optional activities that never become compulsory errands.
  - **Does not own:** Campaign or unlock state; matchmaking; item/loadout state;
    economy transactions; store eligibility or purchases; social-platform
    behavior; or the domain rules behind any destination it exposes.
  - **Depends on:** Progression; Multiplayer; Items, Equipment & Loadouts;
    Rewards, Loot & Economy; Commerce; Onboarding & Practice; UI Presentation;
    Audio Presentation; and authored hub/world content.
  - **Used by:** Onboarding & Practice; Multiplayer entry; Items/Builds/Economy/
    Commerce surfaces; Results-return routing; and Analytics & Playtest Evidence.

  **Onboarding & Practice**
  - **Purpose:** Teach the minimum playable vocabulary safely, remember what was
    taught or skipped, and expose replayable practice and references.
  - **Owns:** Tutorial sequence and checkpoints; completion, explicit skipping,
    and replay; safe practice-module state; contextual teaching triggers in the
    first boss; prompt eligibility and history; the completed-or-skipped fact
    required for public matchmaking; the onboarding portion of store-unlock
    eligibility; offering calibration and control/settings access at appropriate
    points; and the rule that contextual teaching never pauses or rewinds an
    active encounter.
  - **Does not own:** Calibration mathematics or storage; rhythm judgments;
    combat rules; authored practice charts; public-matchmaking state; Commerce;
    or presentation components.
  - **Depends on:** Rhythm Gameplay; Combat; Boss Encounters; Tactical
    Positioning & Movement; Player Survival & Recovery; Abilities & Cooperative
    Actions; Input/Settings/Accessibility; Song, Chart & Encounter Authoring;
    Progression; Commerce; UI Presentation; Audio Presentation; and Player Data.
  - **Used by:** Multiplayer public-queue eligibility; Commerce store eligibility;
    Order Hub & Navigation; UI Presentation; Player Data; and Analytics &
    Playtest Evidence.

  **Results & Feedback**
  - **Purpose:** Explain what happened, what the player earned, how they
    performed, and what they can do next without delaying retry or blaming
    players.
  - **Owns:** Combining settled domain facts into the immediate summary;
    separating shared outcome from personal performance rating; exact reason
    display; presentation of already-granted rewards and unlocks; Performance,
    Combat, Band, and Progress detail views; up to two private evidence-based
    suggestions; adaptive primary-next-action selection; routes for Retry, Stay
    with Band, Loadout/Upgrades, Continue Story, and Return to Hub; and the no-
    claim, no-public-ranking, no-blame, and no-paid-prompt presentation rules.
  - **Does not own:** Encounter outcome; reward calculation or grant; progression
    mutation; persistent personal-best state; rematch membership; item/loadout
    state; or Commerce offers.
  - **Depends on:** Boss Encounters; Rhythm Gameplay; Combat; Player Survival &
    Recovery; Tactical Positioning & Movement; Abilities & Cooperative Actions;
    Solo Support; Multiplayer; Rewards, Loot & Economy; Progression; Items,
    Equipment & Loadouts; UI Presentation; Audio Presentation; and Analytics-
    derived private comparison inputs where approved.
  - **Used by:** Multiplayer rematch/leave routing; Order Hub & Navigation;
    Items/Builds/Progression follow-up routes; and Analytics & Playtest Evidence.

  Calibration is deliberately excluded from Onboarding ownership beyond offering
  the entry point; SM-15 assigns its calculation, settings, storage, and runtime
  application boundaries.
- **Consequences:** Hub and result composition can evolve without becoming
  duplicate owners of progression, matchmaking, rewards, or items. Onboarding
  retains the small amount of durable domain state and gating logic that cannot
  be reduced to presentation alone. Results may derive presentation-level
  ratings and suggestions but never modifies the authoritative outcomes it
  explains.
- **Detailed-spec impact:** All three require substantial sections in
  `UI_UX.md`, including complete flows, states, error/recovery behavior, input-
  device variants, accessibility, and phone-first layouts. Separate
  `ORDER_HUB.md`, `ONBOARDING.md`, or `RESULTS.md` system specifications are not
  initially required. The hub may later receive an authored content/world brief.
- **Deferred:** Exact hub layout, travel affordances, screen/component layouts,
  tutorial copy, prompt timing, performance-rating formula, suggestion logic,
  and adaptive-action presentation belong to `UI_UX.md`, content/world briefs,
  localization, and playtesting. Persistent onboarding/prompt schema and routing
  implementation belong to technical architecture.

### SM-15: UI, Input/Settings/Calibration, Accessibility, and Audio boundaries

- **Owner answer:** The proposed division and specification grouping are
  approved: “Sure.”
- **Map decision:** **UI Presentation**, **Input, Settings & Calibration**, and
  **Audio Presentation** are separate **First-Release Runtime Systems**.
  **Accessibility** is a **Cross-Cutting Requirement** and mandatory design
  scaffold rather than an optional subsystem.

  **UI Presentation**
  - **Purpose:** Present every domain's semantic state and player action in a
    coherent, phone-first, device-appropriate visual interface.
  - **Owns:** HUD and screen composition; persistent versus contextual
    information hierarchy; menus, dialogs, navigation, focus, loading, disabled,
    confirmation, and error states; responsive layouts, safe areas, UI scaling,
    and touch-target presentation; rendering semantic cues supplied by gameplay
    systems; device-appropriate labels and control references; captions,
    subtitles, source labels, and visual equivalents for audio cues; and the
    component/flow presentation for Hub, Onboarding, Results, loadouts,
    progression, matchmaking, and Commerce.
  - **Does not own:** Combat, encounter, rhythm, progression, reward, item,
    matchmaking, or Commerce state; physical-input interpretation; accessibility
    policy; audio mixing; or persistent settings storage.
  - **Depends on:** Semantic state and actions from every runtime system; Input,
    Settings & Calibration; Accessibility; Audio Presentation caption/cue
    metadata; Communication & Safety; Art Direction; and Player Data for loaded
    preferences.
  - **Used by:** Every player-facing runtime system and orchestrated experience.

  **Input, Settings & Calibration**
  - **Purpose:** Convert supported physical controls into stable semantic actions
    and provide player-controlled profiles for timing, comfort, accessibility,
    and presentation.
  - **Owns:** Touch, keyboard/mouse, and gamepad mapping into common semantic
    actions; hub, encounter, menu, and contextual input modes; active-device
    detection and binding profiles; supported remapping; touch handedness, pad
    size, spacing, position, and opacity; settings definitions and current
    profile values; guided calibration, outlier rejection, suggested offset,
    test, and manual adjustment; and exposing saved calibration/accessibility
    preferences to consuming systems.
  - **Does not own:** The gameplay consequence of an action; UI rendering; rhythm
    judgment; application of audio/visual settings by their consuming systems;
    or persistence implementation.
  - **Depends on:** Roblox device/input capabilities; Accessibility policy; UI
    Presentation; Audio Presentation; Player Data; and semantic action contracts
    from Hub, Rhythm, Combat, Positioning, Abilities, Multiplayer, and other
    player-facing systems.
  - **Used by:** Rhythm Gameplay; Combat; Tactical Positioning & Movement;
    Abilities & Cooperative Actions; Multiplayer; Onboarding & Practice; UI
    Presentation; Audio Presentation; Player Data; and Analytics & Playtest
    Evidence.

  **Accessibility**
  - **Purpose:** Make critical gameplay, navigation, communication, and feedback
    perceivable and operable without changing difficulty, rewards, privacy, or
    player dignity.
  - **Owns:** Cross-system requirements that critical information is not conveyed
    through color or sound alone; shape, label, placement, motion, captions, and
    source identification reinforce cues; UI/staff/notes/touch controls can
    scale; motion, flashing, bloom, particles, camera effects, haptics, and audio
    layers can be reduced independently; approved assists such as Hold Assist are
    difficulty-independent; accessibility never reduces rewards, mastery,
    campaign credit, matchmaking, or privacy; and no public accessibility label
    or performance shaming is permitted.
  - **Does not own:** Another system's semantic state, settings values,
    presentation implementation, difficulty, rewards, or persistent profile.
  - **Depends on:** Every domain exposing sufficient semantic state; UI
    Presentation; Input, Settings & Calibration; Audio Presentation;
    Communication & Safety; Content Authoring validation; and playtest evidence.
  - **Used by:** Every runtime system, orchestrated experience, content package,
    validator, and detailed specification.

  **Audio Presentation**
  - **Purpose:** Preserve musical clarity and communicate performance, danger,
    cooperation, and world response through a controllable, accessible mix.
  - **Owns:** Stable song/stem playback presentation; local-instrument emphasis
    for judgments, movement, downing, and recovery; boss, combat, group-action,
    crowd, and ambience mixing; critical-cue priority and effect ducking; audio
    buses, dynamic-range presets, mono compatibility, captions metadata, and
    source identification; aggregate band-performance audio; and restrained
    haptic/impact-feedback requests.
  - **Does not own:** The musical clock; rhythm judgments; combat/encounter/group
    semantics; source chart or encounter authoring; settings values; or caption
    rendering.
  - **Depends on:** Rhythm Gameplay's clock and judgments; semantic events from
    Combat, Survival, Boss Encounters, Positioning, Abilities, Solo Support, and
    Multiplayer; Song, Chart & Encounter Authoring; Input/Settings profiles;
    Accessibility; UI Presentation; and authored audio assets/metadata.
  - **Used by:** Every player-facing runtime system, Hub, Onboarding, Results,
    Communication & Safety, and Analytics & Playtest Evidence.

  The `ux-decisions` framework informed the classification of Accessibility as
  a reusable scaffold: domain systems own semantic meaning, presentation systems
  render it consistently, and accessibility requirements are applied during
  every interface/content decision rather than checked only at the end.
- **Consequences:** Semantic state stays with the gameplay owner while visual,
  input, and audio systems remain independently testable. Calibration has one
  procedural owner; Rhythm only applies its output. Accessibility cannot be
  deprioritized as presentation polish or implemented inconsistently per boss.
  Responsive audio remains a core gameplay-presentation capability substantial
  enough to specify independently from screen UI.
- **Detailed-spec impact:** `UI_UX.md` is required and covers UI Presentation;
  Input, Settings & Calibration; Accessibility; Order Hub; Onboarding & Practice;
  Results & Feedback; and every major navigation flow. `AUDIO_PRESENTATION.md` is
  required separately for responsive music, mix priorities, cue contracts,
  buses, accessibility, haptics requests, and content requirements. No standalone
  `INPUT.md` or `ACCESSIBILITY.md` is initially required. Final document priority
  is assigned in SM-17.
- **Deferred:** Exact responsive layouts, component library, focus behavior,
  touch measurements, binding UI, calibration algorithm details, settings
  catalog, audio buses/levels, cue catalog, mix targets, and validation/test
  matrices belong to the detailed specs and playtesting. Roblox APIs, profile
  storage schema, rendering/mixing implementation, and performance budgets belong
  to technical architecture.

### SM-16: Player Data and Analytics & Playtest Evidence boundaries

- **Owner answer:** The proposed runtime-platform and supporting-production
  boundaries are approved: “Yes.” Communication & Safety remains resolved by
  SM-08.
- **Map decision:** **Player Data** is a **First-Release Runtime Platform
  System**. **Analytics & Playtest Evidence** is a
  **First-Release-Supporting Production System** that includes bounded runtime
  instrumentation but does not own gameplay behavior.

  **Player Data**
  - **Purpose:** Durably preserve player-owned facts and cross-domain
    transactions without taking ownership of what those facts mean.
  - **Owns:** Loading, saving, and recovery guarantees; durable transaction
    commits across affected domains; record versioning and migration; default-
    profile creation; concurrent-session and stale-write protection; retry,
    rollback, and failure-recovery policy; durable storage for progression,
    inventory, currencies, purchases, builds, loadouts, mastery, personal bests,
    settings, calibration, onboarding, prompt history, and unlocks; and player-
    visible behavior when data is unavailable or cannot be safely saved.
  - **Does not own:** The semantic meaning or mutation rules of a domain record;
    reward, item, progression, Commerce, build, or settings decisions; ephemeral
    encounter/reconnect state; UI presentation; or the Roblox storage service
    itself.
  - **Depends on:** Mutation/validation contracts from every durable domain;
    Rewards, Loot & Economy transaction orchestration; Commerce receipt/grant
    decisions; UI failure/recovery presentation; Communication & Safety/privacy
    requirements; and external Roblox persistence, account, privacy, and
    temporary-storage capabilities.
  - **Used by:** Progression; Items, Equipment & Loadouts; Builds &
    Specialization; Rewards, Loot & Economy; Commerce; Multiplayer; Onboarding &
    Practice; Input, Settings & Calibration; Results & Feedback; UI Presentation;
    and Analytics & Playtest Evidence where approved.

  **Analytics & Playtest Evidence**
  - **Purpose:** Produce trustworthy, privacy-conscious evidence that the game is
    understandable, fair, accessible, and ready.
  - **Owns:** The analytics event and metric catalog; shared collection and
    segmentation rules; device, difficulty, instrument, boss, solo/co-op,
    roster-size, and accessibility segmentation; joining observation, player
    explanation, telemetry, and voluntary behavior into readiness evidence;
    GD-34 readiness-gate calculations and reports; data-quality checks so
    averages cannot hide failed instruments, devices, or player groups;
    research consent, safeguarding, retention, and access boundaries; and the
    rule that analytics never changes difficulty, rewards, matchmaking, or
    public player labels automatically.
  - **Does not own:** The semantic gameplay facts it records; live gameplay
    decisions; progression/reward state; public scoring; automatic difficulty;
    persistent player-domain records; or external analytics transport.
  - **Depends on:** Semantic event contracts from every system; Player Data only
    where explicitly approved; Accessibility and Communication & Safety policy;
    UI/Onboarding/Results instrumentation; research operations; and external
    Roblox telemetry, privacy, account, and consent capabilities.
  - **Used by:** Design review; content validation; Difficulty & Scaling;
    `BALANCE_FRAMEWORK.md`; every detailed spec's validation plan; playtest and
    release-readiness decisions; and Results' private historical suggestions only
    where the approved privacy boundary permits it.

  Gameplay systems own the semantic facts they emit. Analytics owns how those
  facts are collected, segmented, combined, quality-checked, and reported.
  Communication & Safety remains the SM-08 cross-cutting requirement and is not
  duplicated here.
- **Consequences:** Persistence mechanics cannot acquire authority to invent
  domain changes, while cross-domain grants have one durable commit boundary.
  Ephemeral reconnection remains a session concern. Analytics serves evidence
  and validation without becoming a hidden gameplay-control or public-ranking
  system. Roblox capabilities remain explicit dependencies rather than game-
  owned responsibilities.
- **Detailed-spec impact:** `PLAYER_DATA.md` is required for the durable-record
  inventory, ownership matrix, transaction requirements, save/recovery
  guarantees, version/migration policy, and player-visible failure behavior.
  `PLAYTEST_AND_ANALYTICS.md` is required for event meanings, segmentation,
  evidence collection, readiness reports, research safeguards, and validation
  procedures. Exact data/event schemas, datastore/telemetry architecture,
  encryption/security, retention implementation, and transport remain technical.
- **Deferred:** Save cadence, retry limits, migration mechanics, record schemas,
  platform budgets, deletion/export handling, event payloads, identifiers,
  retention periods, consent procedure, study protocol, sample segmentation,
  statistical confidence, and operational sign-off require the detailed specs,
  technical architecture, privacy/safety review, and research planning.

### SM-17: Detailed-spec backlog and dependency order

- **Owner direction:** The initially proposed position of
  `CONTENT_AUTHORING.md` at number 12 is rejected: “#12 needs to be done before
  we start doing gameplay stuff because we need the song data as a starting
  point.”
- **Working map decision:** `CONTENT_AUTHORING.md` moves to the start of the
  detailed-spec sequence. Its first pass must establish the offline song-data
  foundation before gameplay design or implementation depends on it:
  - maintained source inputs and source-of-truth rules;
  - the existing platform-neutral `tools/chart-pipeline/` bundle as the starting
    contract;
  - required song, stem, beat-grid, instrument-chart, difficulty, timing,
    structural, Activity Map, validation, and version metadata;
  - deterministic build/validation behavior and human approval gates; and
  - the exported package boundary consumed by Roblox.

  Because some event and validator requirements are refined by the downstream
  Rhythm Gameplay, Combat, Boss Encounters, Abilities, Multiplayer, and
  Accessibility specifications, Content Authoring receives a required
  reconciliation pass after those behavioral contracts are written. That pass
  extends or corrects the offline contract; it does not move authoring into
  Roblox or allow runtime systems to invent private song-data formats.
- **Consequences:** Gameplay work begins from real canonical song data rather
  than parallel assumptions. `RHYTHM_GAMEPLAY.md` consumes the first approved
  contract and identifies any required additions through an explicit authoring-
  contract revision. The same rule applies to later encounter/group validators.
  The existing chart pipeline is extended deliberately rather than replaced by
  speculative Roblox-side data generation.
- **Owner approval:** The owner approved the revised complete order: “Yes.”
- **Detailed-spec impact:** The finite, approved specification sequence is:
  1. `CONTENT_AUTHORING.md` — establish the offline source, song-data, validation,
     and exported runtime-bundle contract.
  2. `RHYTHM_GAMEPLAY.md`.
  3. `COMBAT.md`, including Player Survival & Recovery.
  4. `BOSS_ENCOUNTERS.md`, including Tactical Positioning & Movement.
  5. `PROGRESSION.md`.
  6. `ITEMS_AND_EQUIPMENT.md`, including Inventory and consumables.
  7. `ABILITIES_AND_COOPERATIVE_ACTIONS.md`, including Solo Support.
  8. `MULTIPLAYER.md`, including Communication & Safety.
  9. `REWARDS_AND_ECONOMY.md`, including Commerce.
  10. `BUILDS_AND_SPECIALIZATION.md`.
  11. `UI_UX.md`, including Input/Settings/Calibration, Accessibility, Hub,
      Onboarding, and Results.
  12. `AUDIO_PRESENTATION.md`.
  13. `PLAYER_DATA.md`.

  After specifications 2 through 12 identify their complete authored-data and
  validation needs, `CONTENT_AUTHORING.md` receives a mandatory reconciliation
  pass. Only then, and after the other architecture-critical specifications are
  sufficiently settled, may `TECHNICAL_ARCHITECTURE.md` become canonical.
  `BALANCE_FRAMEWORK.md` and `PLAYTEST_AND_ANALYTICS.md` are supporting documents
  developed alongside later specifications rather than additional entries in
  the thirteen-spec sequence.

  No standalone specification is initially required for Player Survival &
  Recovery, Tactical Positioning & Movement, Solo Support, Difficulty & Scaling,
  Communication & Safety, Commerce, Inventory, consumables, Order Hub,
  Onboarding, Results, Input, or Accessibility because each has an explicit
  approved parent document above.
- **Deferred:** No specification is removed by this amendment. Exact initial
  contract fields belong to the first Content Authoring interview/spec, while
  file formats and implementation architecture remain technical.

### SM-18: Final ownership and dependency reconciliation

- **Owner answer:** The owner approved the final audit and authorized the
  canonical result. The response `6es` is understood as “Yes” in the direct
  context of the approval question.
- **Map decision:** The final audit is accepted:
  - all 25 retained entries have an approved classification and primary
    responsibility;
  - every major state, outcome, transaction, flow, and presentation
    responsibility has one primary owner;
  - all GD-01 through GD-34 responsibilities are covered;
  - apparent cycles are explicit semantic handshakes rather than duplicated
    state ownership;
  - the thirteen-spec order and Content Authoring-first reconciliation gate are
    coherent;
  - first-release non-goals and uncommitted genre features are explicit; and
  - remaining design, content, naming, balance, research, and technical questions
    have named destinations.
- **Consequences:** The interview is complete.
  [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md) becomes the canonical bridge from the GDD to
  detailed system specifications. Later documents may refine internal behavior
  and interfaces but must explicitly amend the map to merge, split, add, remove,
  reclassify, or transfer ownership between systems.
- **Detailed-spec impact:** Work begins with `CONTENT_AUTHORING.md` and the
  offline song-data contract, followed by the remaining SM-17 sequence.
- **Deferred:** Only the routed questions already listed under each system
  remain. No unresolved item blocks approval of the systems map itself.

## 5. Emerging system inventory

The complete candidate inventory in SM-01 is approved. Its five domains and
entries are recorded in the SM-01 decision above. SM-02 establishes the labels
used below; classifications are added as each boundary question is resolved.

Approved classifications so far:

- **Rhythm Gameplay:** First-Release Runtime System.
- **Combat:** First-Release Runtime System.
- **Player Survival & Recovery:** First-Release Runtime System.
- **Boss Encounters:** First-Release Runtime System.
- **Tactical Positioning & Movement:** First-Release Runtime System; explicit
  SM-01 inventory refinement.
- **Abilities & Cooperative Actions:** First-Release Runtime System.
- **Solo Support:** First-Release Runtime System.
- **Difficulty & Scaling:** Cross-Cutting Requirement.
- **Multiplayer Sessions, Parties & Matchmaking:** First-Release Runtime System.
- **Communication & Safety:** Cross-Cutting Requirement.
- **Song, Chart & Encounter Authoring:** First-Release-Supporting Production
  System; offline and platform-neutral, outside Roblox runtime.
- **Items, Equipment & Loadouts:** First-Release Runtime System; Inventory and
  consumables are internal responsibilities.
- **Builds & Specialization:** First-Release Runtime System.
- **Rewards, Loot & Economy:** First-Release Runtime System.
- **Commerce:** First-Release Runtime System.
- **Progression:** First-Release Runtime System with player, campaign, and boss-
  mastery tracks.
- **Order Hub & Navigation:** Orchestrated Experience/Surface.
- **Onboarding & Practice:** First-Release Runtime System.
- **Results & Feedback:** Orchestrated Experience/Surface.
- **UI Presentation:** First-Release Runtime System.
- **Input, Settings & Calibration:** First-Release Runtime System.
- **Accessibility:** Cross-Cutting Requirement.
- **Audio Presentation:** First-Release Runtime System.
- **Player Data:** First-Release Runtime Platform System.
- **Analytics & Playtest Evidence:** First-Release-Supporting Production System.

## 6. Emerging detailed-spec backlog

SM-17 approved thirteen detailed specifications in this order:

1. `CONTENT_AUTHORING.md`
2. `RHYTHM_GAMEPLAY.md`
3. `COMBAT.md`
4. `BOSS_ENCOUNTERS.md`
5. `PROGRESSION.md`
6. `ITEMS_AND_EQUIPMENT.md`
7. `ABILITIES_AND_COOPERATIVE_ACTIONS.md`
8. `MULTIPLAYER.md`
9. `REWARDS_AND_ECONOMY.md`
10. `BUILDS_AND_SPECIALIZATION.md`
11. `UI_UX.md`
12. `AUDIO_PRESENTATION.md`
13. `PLAYER_DATA.md`

The first Content Authoring pass establishes the offline song-data contract. It
receives a mandatory reconciliation after specifications 2 through 12 and before
technical architecture is finalized. `BALANCE_FRAMEWORK.md` and
`PLAYTEST_AND_ANALYTICS.md` are supporting documents.

## 7. Final ownership and dependency audit

The approved SM-18 audit found no unassigned first-release responsibility.

| Responsibility | Primary owner |
|---|---|
| Offline song/chart/encounter data and approval | Song, Chart & Encounter Authoring |
| Current song time, note judgment, and normalized musical contribution | Rhythm Gameplay |
| Intent routing and calculated combat effects | Combat |
| Ward, downed, revival, and re-entry state | Player Survival & Recovery |
| Active boss attempt, Resolve/Momentum, attacks, finishing, and outcome | Boss Encounters |
| Arena graph, location, travel, movement charge, cover, and risk tier | Tactical Positioning & Movement |
| Signature, Band Call, Crescendo, Join In, and effect lifecycle | Abilities & Cooperative Actions |
| Acolyte state and fixed solo support | Solo Support |
| Difficulty/player-count invariants and allowed scaling dimensions | Difficulty & Scaling |
| Party, queue, roster, reconnect, AFK, rematch, and ping delivery | Multiplayer |
| Safe communication and anti-coercion invariants | Communication & Safety |
| Canonical meaningful-participation eligibility | Rewards, Loot & Economy |
| Item ownership semantics, equipment, loadout, cosmetics, and consumables | Items, Equipment & Loadouts |
| Specialization configuration and resolved build modifiers | Builds & Specialization |
| Earned rewards, resources, crafting/upgrades, and transaction orchestration | Rewards, Loot & Economy |
| Paid catalog and purchase eligibility | Commerce |
| Player, campaign, difficulty-unlock, mastery, and personal-best state | Progression |
| Hub spatial/navigation composition | Order Hub & Navigation |
| Tutorial, practice, completion/skip, and prompt state | Onboarding & Practice |
| Result explanation, derived feedback, and next-action routing | Results & Feedback |
| Visual presentation and component/navigation states | UI Presentation |
| Physical-to-semantic input, settings, and calibration profiles | Input, Settings & Calibration |
| Cross-system accessibility invariants | Accessibility |
| Runtime mix, responsive instrument audio, and critical audio cues | Audio Presentation |
| Durable storage, migration, and cross-domain commit | Player Data |
| Evidence collection, segmentation, and readiness reporting | Analytics & Playtest Evidence |

The following apparent cycles are intentional design handshakes, not duplicated
ownership:

- Multiplayer establishes the roster and starts an attempt; Boss Encounters owns
  the active attempt and returns its outcome.
- Boss Encounters configures/starts Rhythm Gameplay; Rhythm owns song time and
  returns musical boundaries consumed by the encounter.
- Rewards orchestrates a once-only transaction; Items and Progression validate
  their domain mutations; Player Data durably commits the combined result.
- Gameplay systems expose semantic state to UI and Audio; Input returns semantic
  player actions to the owning systems.
- Content Authoring consumes validation requirements from gameplay specs and
  exports approved packages back to runtime consumers through the required
  reconciliation process.

GDD coverage is complete: GD-01 through GD-10 map to Hub/Multiplayer/Results,
Boss/Rhythm/UI/Audio/Authoring; GD-11 through GD-23 map to Combat, Survival,
Boss, Positioning, Abilities, Solo Support, Scaling, Multiplayer, and Safety;
GD-24 through GD-28 map to Items, Builds, Economy, Progression, and Commerce;
and GD-29 through GD-34 map to Onboarding, Hub, Authoring/content, Results,
Accessibility/Input/Audio/Safety, and Analytics.

The exclusion register remains explicit: PvP, user-authored songs, free-roaming
worlds, deep crafting, player trading, paid recovery, paid randomness, and
multi-song raids are not first-release systems. Quests, achievements, daily or
retention mechanics, live operations, and guilds are not assumed for the first
release. Content catalogs, narrative, boss/song briefs, world layouts, asset
production, balance values, technical authority/networking/security, and file/
data schemas are routed to their named downstream documents rather than treated
as orphan systems.

## 8. Working change log

- **2026-08-18:** Initialized the progressive record alongside the 18-question
  interview plan. No owner decisions have yet been recorded.
- **2026-08-18:** Recorded the owner's unmodified approval of the five-domain
  candidate inventory in SM-01. Progress is 1 of 18 questions resolved.
- **2026-08-18:** Recorded SM-02's approved classification rules and replaced
  “launch” with “first-release” throughout the systems-map documents. Progress
  is 2 of 18 questions resolved.
- **2026-08-18:** Recorded SM-03's approved Rhythm Gameplay boundary and its role
  as the first gameplay specification after Content Authoring. Progress is 3 of
  18 questions resolved.
- **2026-08-18:** Recorded SM-04's separation of Combat from Player Survival &
  Recovery and their shared second gameplay specification, `COMBAT.md`.
  Progress is 4 of 18 questions resolved.
- **2026-08-18:** Recorded SM-05's separation of Boss Encounters from Tactical
  Positioning & Movement and their shared third gameplay specification,
  `BOSS_ENCOUNTERS.md`. Progress is 5 of 18 questions resolved.
- **2026-08-18:** Recorded SM-06's separate Abilities & Cooperative Actions and
  Solo Support systems and their shared later specification. Progress is 6 of
  18 questions resolved.
- **2026-08-18:** Recorded SM-07's delegated decision: Difficulty & Scaling is a
  cross-cutting requirement, with system-specific sections and a later shared
  `BALANCE_FRAMEWORK.md`. Progress is 7 of 18 questions resolved.
- **2026-08-18:** Recorded SM-08's Multiplayer runtime boundary and Communication
  & Safety cross-cutting policy, to be detailed together in `MULTIPLAYER.md`.
  Progress is 8 of 18 questions resolved.
- **2026-08-18:** Recorded SM-09's authoring boundary and the owner's explicit
  requirement that it extend the existing offline `tools/chart-pipeline/`
  toolchain rather than live in Roblox. Progress is 9 of 18 questions resolved.
- **2026-08-18:** Recorded SM-10's unified Items, Equipment & Loadouts boundary,
  with Inventory and consumables retained as internal responsibilities. Progress
  is 10 of 18 questions resolved.
- **2026-08-18:** Recorded SM-11's separate Builds & Specialization boundary and
  later `BUILDS_AND_SPECIALIZATION.md`. Progress is 11 of 18 questions resolved.
- **2026-08-18:** Recorded SM-12's separate Rewards, Loot & Economy and Commerce
  boundaries and shared `REWARDS_AND_ECONOMY.md`. Progress is 12 of 18 questions
  resolved.
- **2026-08-18:** Recorded SM-13's unified Progression boundary with player,
  campaign, and boss-mastery tracks. Progress is 13 of 18 questions resolved.
- **2026-08-18:** Recorded SM-14's Hub and Results surfaces and Onboarding &
  Practice runtime boundary, all to be detailed in `UI_UX.md`. Progress is 14 of
  18 questions resolved.
- **2026-08-18:** Recorded SM-15's UI, Input/Settings/Calibration, Accessibility,
  and Audio boundaries, with `UI_UX.md` and `AUDIO_PRESENTATION.md` required.
  Progress is 15 of 18 questions resolved.
- **2026-08-18:** Recorded SM-16's Player Data and Analytics & Playtest Evidence
  boundaries, with `PLAYER_DATA.md` and `PLAYTEST_AND_ANALYTICS.md` required.
  Progress is 16 of 18 questions resolved.
- **2026-08-18:** Recorded the SM-17 owner constraint that
  `CONTENT_AUTHORING.md` and its initial song-data contract precede gameplay
  specifications. SM-17 remained open until the revised order was approved.
- **2026-08-18:** Recorded approval of SM-17's revised thirteen-spec order and
  mandatory Content Authoring reconciliation gate. Progress is 17 of 18
  questions resolved.
- **2026-08-18:** Recorded approval of SM-18's final audit, completed the
  interview, archived this working record, and produced canonical
  `SYSTEMS_MAP.md`. Progress is 18 of 18 questions resolved.
