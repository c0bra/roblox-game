# Bands Battle Systems Map

- **Status:** Approved systems-map baseline
- **Approved:** 2026-08-18
- **Parent design:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Higher authorities:** [`GAME_VISION.md`](GAME_VISION.md) and
  [`ART_DIRECTION.md`](ART_DIRECTION.md)
- **Decision source:** [`SYSTEMS_MAP_WORKING.md`](SYSTEMS_MAP_WORKING.md)
- **Interview plan:** [`SYSTEMS_MAP_QUESTIONS.md`](SYSTEMS_MAP_QUESTIONS.md)

## 1. Role and authority

This document is the canonical responsibility and dependency map for Bands
Battle. It translates the approved player-facing design into stable system
boundaries, identifies the source of truth for every major rule and state, and
defines which detailed design documents must exist before technical architecture
or implementation is allowed to fill in the remaining details.

`GAME_DESIGN.md` remains authoritative for player behavior. `GAME_VISION.md`
remains authoritative for product purpose, audience, tone, and scope.
`ART_DIRECTION.md` remains authoritative for visual language. This map assigns
responsibility for those approved decisions; it does not silently change them.

A system boundary is a design ownership boundary, not an instruction to create
one Roblox service, module, class, remote, or datastore. Concrete client/server
authority, networking, persistence, security, schemas, APIs, and file layout
belong to `TECHNICAL_ARCHITECTURE.md` after the required system specifications
are sufficiently settled.

In this document, **first-release** means the first publicly playable product
defined by `GAME_DESIGN.md`: the Order hub and onboarding, three replayable
bosses, solo and three-to-six-player co-op, rhythm combat, basic equipment,
rewards, upgrades, progression, accessibility, safety, and the supporting
content pipeline.

## 2. Entry classifications

- **First-Release Runtime System:** operates in the shipped player experience.
- **First-Release Runtime Platform System:** provides a game-owned runtime
  platform guarantee, such as durable player data, without taking ownership of
  domain semantics.
- **First-Release-Supporting Production System:** creates, validates, or
  evaluates first-release content outside the player-facing runtime.
- **Cross-Cutting Requirement:** defines shared invariants that several systems
  must enforce rather than owning their mutable state.
- **Orchestrated Experience/Surface:** composes domain systems into a player flow
  without taking ownership of the underlying domain facts.
- **Deferred Future System / Explicit Non-Goal:** appears only in the boundary
  register unless later approval establishes stable responsibilities.

Roblox-provided storage, transport, matchmaking, Marketplace, filtering,
blocking, reporting, privacy, age-control, account, and analytics capabilities
are external dependencies. Bands Battle systems own only the game-specific
behavior and policy layered on those capabilities.

## 3. Inventory overview

| Domain | Entry | Classification | Detailed specification |
|---|---|---|---|
| Core Battle | Rhythm Gameplay | First-Release Runtime System | `RHYTHM_GAMEPLAY.md` |
| Core Battle | Combat | First-Release Runtime System | `COMBAT.md` |
| Core Battle | Player Survival & Recovery | First-Release Runtime System | `COMBAT.md` |
| Core Battle | Boss Encounters | First-Release Runtime System | `BOSS_ENCOUNTERS.md` |
| Core Battle | Tactical Positioning & Movement | First-Release Runtime System | `BOSS_ENCOUNTERS.md` |
| Core Battle | Abilities & Cooperative Actions | First-Release Runtime System | `ABILITIES_AND_COOPERATIVE_ACTIONS.md` |
| Core Battle | Solo Support | First-Release Runtime System | `ABILITIES_AND_COOPERATIVE_ACTIONS.md` |
| Core Battle | Difficulty & Scaling | Cross-Cutting Requirement | System sections + `BALANCE_FRAMEWORK.md` |
| Multiplayer | Multiplayer Sessions, Parties & Matchmaking | First-Release Runtime System | `MULTIPLAYER.md` |
| Multiplayer | Communication & Safety | Cross-Cutting Requirement | `MULTIPLAYER.md` |
| Progression and Meta | Items, Equipment & Loadouts | First-Release Runtime System | `ITEMS_AND_EQUIPMENT.md` |
| Progression and Meta | Builds & Specialization | First-Release Runtime System | `BUILDS_AND_SPECIALIZATION.md` |
| Progression and Meta | Rewards, Loot & Economy | First-Release Runtime System | `REWARDS_AND_ECONOMY.md` |
| Progression and Meta | Commerce | First-Release Runtime System | `REWARDS_AND_ECONOMY.md` |
| Progression and Meta | Progression | First-Release Runtime System | `PROGRESSION.md` |
| Experience Shell | Order Hub & Navigation | Orchestrated Experience/Surface | `UI_UX.md` |
| Experience Shell | Onboarding & Practice | First-Release Runtime System | `UI_UX.md` |
| Experience Shell | Results & Feedback | Orchestrated Experience/Surface | `UI_UX.md` |
| Experience Shell | UI Presentation | First-Release Runtime System | `UI_UX.md` |
| Experience Shell | Input, Settings & Calibration | First-Release Runtime System | `UI_UX.md` |
| Experience Shell | Accessibility | Cross-Cutting Requirement | `UI_UX.md` + every affected spec |
| Experience Shell | Audio Presentation | First-Release Runtime System | `AUDIO_PRESENTATION.md` |
| Content and Platform | Song, Chart & Encounter Authoring | First-Release-Supporting Production System | `CONTENT_AUTHORING.md` |
| Content and Platform | Player Data | First-Release Runtime Platform System | `PLAYER_DATA.md` |
| Content and Platform | Analytics & Playtest Evidence | First-Release-Supporting Production System | `PLAYTEST_AND_ANALYTICS.md` |

## 4. Core Battle

### 4.1 Rhythm Gameplay

- **Purpose:** Turn authored instrument charts and player inputs into musically
  aligned judgments and normalized performance contribution while keeping the
  full song as the encounter clock.
- **Owns:** Current song time and musical boundaries at the design level;
  runtime chart playback; tap, hold, repeat, alternate, and rest interpretation;
  input-to-note matching; Perfect/Great/Good/Miss judgments; early/late and hold
  feedback; scoring groups; pre-combat normalized contribution; chart
  suspension/re-entry; application of calibration and Hold Assist; and solo
  pause/resume timing.
- **Does not own:** Authoring, device bindings, saved settings, combat
  conversion, boss state, audio mixing, results presentation, or analytics
  collection.
- **Depends on:** Approved authoring packages, encounter configuration, selected
  instrument/difficulty, Input/Settings/Calibration, Accessibility, and
  Difficulty & Scaling.
- **Used by:** Combat, Boss Encounters, Survival & Recovery, Abilities &
  Cooperative Actions, Solo Support, Multiplayer, Audio Presentation, Results,
  and Analytics.
- **Needs detailed spec?:** Yes—`RHYTHM_GAMEPLAY.md`, the first gameplay spec
  after the initial Content Authoring contract.
- **Major unresolved decisions:** Exact input matching, rapid/overlapping input
  behavior, aggregation and normalization, suspension/re-entry, pause/resume,
  calibration application, and semantic output contracts.
- **GDD sources:** GD-02, GD-04 through GD-10, GD-12, GD-14 through GD-19,
  GD-29, GD-33, and GD-34.

### 4.2 Combat

- **Purpose:** Convert rhythm performance, intent, position, equipment, and build
  effects into combat consequences.
- **Owns:** Attack/Defend/Special intent and queuing; routing normalized
  performance; calculation of Resolve pressure, mitigation, Ward reinforcement,
  restoration, support, and ability contribution; permitted post-score
  modifiers; combat attribution; and the rule that a miss does not directly
  damage Ward.
- **Does not own:** Rhythm judgment, boss/Resolve state, Ward state, ability
  definitions, rewards, or persistent items.
- **Depends on:** Rhythm Gameplay, Boss Encounters, Positioning & Movement,
  Items & Equipment, Builds & Specialization, Abilities & Cooperative Actions,
  and Difficulty & Scaling.
- **Used by:** Boss Encounters, Survival & Recovery, Abilities, Solo Support,
  Rewards, Results, Audio, and Analytics.
- **Needs detailed spec?:** Yes—`COMBAT.md`.
- **Major unresolved decisions:** Formulae, modifier ordering, caps, attribution,
  semantic effect contracts, and balance values.
- **GDD sources:** GD-11 through GD-19, GD-21, GD-24, GD-25, GD-32, and GD-34.

### 4.3 Player Survival & Recovery

- **Purpose:** Manage whether each player remains active, becomes downed, and
  returns to play.
- **Owns:** Ward and its thresholds; application of post-mitigation damage,
  restoration, and reinforcement; downed/target-ineligible state; cooperative
  revival state; the one-use solo emergency recovery opportunity; revived Ward;
  and re-entry protection.
- **Does not own:** Boss attack timing, Defend performance, recovery-chart
  judgments, encounter defeat, consumable definitions, or presentation.
- **Depends on:** Combat effects, Rhythm Gameplay, Boss Encounters and Activity
  Maps, Multiplayer roster state, Abilities, consumable effects, and Difficulty
  & Scaling.
- **Used by:** Boss Encounters, Combat, Multiplayer, Solo Support, Abilities,
  Results, Audio, UI, and Analytics.
- **Needs detailed spec?:** Yes—as a separate major section of `COMBAT.md`.
- **Major unresolved decisions:** Ward/damage/restoration values, revival
  contribution rules, recovery-window semantics, and re-entry edge cases.
- **GDD sources:** GD-11, GD-12, GD-14 through GD-17, GD-20 through GD-23,
  GD-29, GD-32, and GD-34.

### 4.4 Boss Encounters

- **Purpose:** Orchestrate a complete song-shaped boss attempt and determine its
  shared outcome.
- **Owns:** Active-attempt lifecycle; five flexible song functions; Resolve
  layers and openings; Momentum; finishing-cadence evaluation; application of
  boss-directed effects; Telegraph/Commit/Impact/Recovery attacks; legal event
  selection; targeting, hazards, recovery/group opportunities; and exact
  victory or defeat reason.
- **Does not own:** Rhythm judgment, combat formulae, Ward state, session
  membership, rewards, or creation of boss packages.
- **Depends on:** Authoring packages, Rhythm, Combat, Survival, Positioning,
  Abilities, Solo Support, Multiplayer roster state, and Difficulty & Scaling.
- **Used by:** Rhythm configuration, Combat, Survival, Positioning, Abilities,
  Solo Support, Multiplayer, Rewards, Results, Audio, UI, and Analytics.
- **Needs detailed spec?:** Yes—`BOSS_ENCOUNTERS.md`.
- **Major unresolved decisions:** State transitions, conflict arbitration,
  attack/target selection, Resolve/Momentum details, content contracts, and
  failure/recovery edge cases.
- **GDD sources:** GD-01 through GD-03, GD-10, GD-13 through GD-23, GD-29,
  GD-31, GD-32, and GD-34.

### 4.5 Tactical Positioning & Movement

- **Purpose:** Maintain each player's legal arena location and movement state
  while exposing tactical risk, cover, and attack geometry.
- **Owns:** Arena graph; current location/travel; legal destinations; movement
  charges and beat-based recovery; settling; multi-edge travel; displacement;
  shared-location/no-body-blocking rules; cover/hazard occupancy; risk tier; and
  application of authored graph changes.
- **Does not own:** Input bindings, boss attack selection, combat modifier
  calculation, Ward damage, reward calculation, or arena authoring.
- **Depends on:** Boss Encounters, authored arena data, Rhythm boundaries,
  semantic movement input, Survival state, Solo Support, and Difficulty &
  Scaling.
- **Used by:** Boss Encounters, Combat, Survival, Solo Support, Abilities,
  Rewards, Results, UI, Audio, and Analytics.
- **Needs detailed spec?:** Yes—as a separate major section of
  `BOSS_ENCOUNTERS.md`.
- **Major unresolved decisions:** Graph changes, movement request/edge semantics,
  cover/hazard rules, and numeric movement/risk values.
- **GDD sources:** GD-03, GD-06, GD-11, GD-14 through GD-16, GD-20, GD-21,
  GD-24, GD-31, GD-32, and GD-34.

### 4.6 Abilities & Cooperative Actions

- **Purpose:** Manage music-aligned personal powers and coordinated band
  performances after Rhythm and Combat determine contribution.
- **Owns:** Signature Special, Band Call, and Crescendo definitions; Hype;
  readiness/arming/consumption; Call initiation and lockout; invitations,
  Join In, eligibility, cancellation, musical-boundary scheduling, contribution
  combination, group tiers, and effect resolution through owning domains.
- **Does not own:** Special intent, authored opportunity selection, revival
  state, loadout ownership, build rules, rhythm judgments, Resolve, Ward, or
  presentation.
- **Depends on:** Authoring/Activity Maps, Rhythm, Combat, Boss Encounters,
  Survival, Multiplayer, Items/Loadouts, Builds, Input, and Difficulty & Scaling.
- **Used by:** Combat, Boss Encounters, Survival, Solo Support, Multiplayer,
  Results, Audio, UI, and Analytics.
- **Needs detailed spec?:** Yes—`ABILITIES_AND_COOPERATIVE_ACTIONS.md`.
- **Major unresolved decisions:** Ability catalog, scheduling/cancellation edge
  cases, Hype/Call rates, contribution/tier rules, permitted build hooks, and
  effect contracts.
- **GDD sources:** GD-12, GD-16 through GD-21, GD-24 through GD-26, GD-29,
  GD-31, GD-32, and GD-34.

### 4.7 Solo Support

- **Purpose:** Make solo encounters complete through visible, predictable
  assistance without fabricated rhythm performance.
- **Owns:** Vanguard/Warden/Herald runtime state and fixed functions; authored
  cadences; suppression/recovery; formation requests; capped fixed group
  contributions; and prohibitions on fake charts, judgments, performance credit,
  or independent Resolve breaks.
- **Does not own:** Solo emergency recovery, general scaling, tactical-location
  state, judgments, or applied combat effects.
- **Depends on:** Rhythm, Combat, Boss Encounters, Positioning, Survival,
  Abilities, Activity Maps, and Difficulty & Scaling.
- **Used by:** Combat, Boss Encounters, Abilities, Results, UI, Audio, and
  Analytics.
- **Needs detailed spec?:** Yes—as a major section of
  `ABILITIES_AND_COOPERATIVE_ACTIONS.md`.
- **Major unresolved decisions:** Cadences, suppression rules, fixed
  contributions, values, and presentation contracts.
- **GDD sources:** GD-16, GD-18 through GD-21, GD-31, GD-32, and GD-34.

### 4.8 Difficulty & Scaling

- **Purpose:** Change challenge and population pressure without changing song
  speed, encounter identity, musical fairness, accessibility rights, or maximum
  available contribution.
- **Owns:** Easy/Normal/Hard profiles; normalized-contribution invariants;
  one-to-six-human relationships; duplicate neutrality; allowed scaling
  dimensions; solo/co-op equivalence goals; Cohesion principles; accessibility
  reward neutrality; and positional risk/reward constraints.
- **Does not own:** Mutable runtime state, difficulty unlocks, current roster,
  positions, or domain-specific application.
- **Depends on:** GDD invariants and segmented Analytics/Playtest evidence.
- **Used by:** Rhythm, Combat, Survival, Boss Encounters, Positioning, Abilities,
  Solo Support, Multiplayer, Authoring, Rewards, Progression, Results, and
  Analytics.
- **Needs detailed spec?:** No standalone system spec. Every affected spec
  contains a scaling section; `BALANCE_FRAMEWORK.md` holds shared values.
- **Major unresolved decisions:** Exact matrices, curves, thresholds, caps,
  reward modifiers, and acceptable solo/co-op variance.
- **GDD sources:** GD-07, GD-08, GD-13 through GD-21, GD-24 through GD-28,
  GD-31, GD-33, and GD-34.

## 5. Multiplayer

### 5.1 Multiplayer Sessions, Parties & Matchmaking

- **Purpose:** Move consenting players from shard selection into a stable roster
  and then into individually chosen follow-up actions.
- **Owns:** Party membership/leadership/consent; public matching; queues and the
  two-player choice; ready/staging and lock timing; no join-in-progress;
  encounter/active roster; disconnect, grace, rejoin, AFK, inactivity/resume;
  rematch/refill/leave; leader transfer; and ping delivery/rate/muting.
- **Does not own:** Loadout contents, boss outcome, gameplay-domain state,
  reward/progression calculation, or durable progression.
- **Depends on:** Hub entry, valid loadouts, Boss Encounters, Rhythm, Survival,
  Abilities, Difficulty & Scaling, Communication & Safety, Results, Player Data
  where durable facts are needed, and external Roblox services.
- **Used by:** Boss Encounters, Survival, Abilities, Difficulty consumers,
  Rewards, Progression, Results, UI, Audio, and Analytics.
- **Needs detailed spec?:** Yes—`MULTIPLAYER.md`.
- **Major unresolved decisions:** Queue/ready/grace/AFK/refill values, region and
  skill inputs, failures, rejoin transport, roster-change details, and
  participation thresholds.
- **GDD sources:** GD-01, GD-06, GD-18 through GD-23, GD-29, GD-30, GD-32
  through GD-34.

### 5.2 Communication & Safety

- **Purpose:** Make cooperation understandable and safe without requiring voice,
  unrestricted text, punitive policing, or coercion.
- **Owns:** Preset-ping policy; protection of automatic critical cues; safe
  defaults; anti-coercion; and no friendly fire, vote-kick, body blocking,
  negative contribution, forced follow-up, or spending others' resources.
- **Does not own:** Session state, platform moderation records, Roblox filtering
  or account controls, or presentation of domain-authored critical cues.
- **Depends on:** External Roblox safety capabilities and approved Accessibility
  policy.
- **Used by:** Multiplayer, Combat, Boss Encounters, Positioning, Abilities,
  Commerce, Results, UI, Audio, Player Data policy, and Analytics.
- **Needs detailed spec?:** No standalone spec; a major `MULTIPLAYER.md` section
  plus requirements in every affected spec.
- **Major unresolved decisions:** Ping localization/rates, moderation
  integration, privacy review, and operational safety testing.
- **GDD sources:** GD-18, GD-19, GD-21 through GD-23, GD-28 through GD-34.

## 6. Progression and Meta

### 6.1 Items, Equipment & Loadouts

- **Purpose:** Represent what a player owns and brings into an encounter,
  producing one validated loadout.
- **Owns:** Item/consumable/cosmetic definitions and owned collections; fixed
  stats, traits, tier/rank; power/action-reference/consumable/appearance slots;
  equip/validation; staging and encounter locks; combat inventory restriction;
  quantities and consumption authorization; resolved equipment modifiers; and
  prohibited-modifier enforcement.
- **Does not own:** Earning, drops, crafting/salvage/upgrade transactions,
  currencies, purchases, ability behavior, specialization, combat calculation,
  persistence implementation, or UI.
- **Depends on:** Rewards/Economy, Commerce, Progression, Abilities, Builds,
  Player Data, catalogs, and Multiplayer lock state.
- **Used by:** Rhythm instrument selection, Combat, Survival, Abilities,
  Multiplayer staging, Builds, Rewards, Commerce, Results, UI, Audio, and
  Analytics.
- **Needs detailed spec?:** Yes—`ITEMS_AND_EQUIPMENT.md`.
- **Major unresolved decisions:** Definition-versus-instance identity, mutation
  contracts, slot validation, consumable lifecycle, modifier allowlist/order,
  cosmetics, and extension points.
- **GDD sources:** GD-01, GD-17, GD-18, GD-24 through GD-30, GD-32, and GD-34.

### 6.2 Builds & Specialization

- **Purpose:** Turn unlocked options into behavior-changing builds without
  changing rhythm fairness, instrument freedom, or the control set.
- **Owns:** Universal functional categories; major/supporting slots; mixing;
  beginner presets and advanced-editor gate; saved build presets; free
  out-of-combat respec and encounter lock; compatibility, stacking, power
  budgets, synergy caps; build modifiers; resolved modifier output; and fairness
  prohibitions.
- **Does not own:** Gear, base abilities, progression awards, persistence,
  presentation, or direct mutation of consumer state.
- **Depends on:** Progression unlocks, Items, Abilities and permitted hooks,
  Combat/Survival/Positioning hooks, Difficulty & Scaling, Player Data, and UI.
- **Used by:** Items/loadout resolution, Combat, Survival, Positioning,
  Abilities, Solo Support where permitted, Multiplayer staging, Results, UI, and
  Analytics.
- **Needs detailed spec?:** Yes—`BUILDS_AND_SPECIALIZATION.md`.
- **Major unresolved decisions:** Final terminology, option catalog, hook
  contracts, stacking/incompatibility, caps, budgets, preset versioning, and
  retired-option behavior.
- **GDD sources:** GD-17, GD-18, GD-24 through GD-29, GD-32, and GD-34.

### 6.3 Rewards, Loot & Economy

- **Purpose:** Turn outcomes and participation into fair, durable earned rewards
  and deterministic resource transactions.
- **Owns:** Canonical meaningful-participation eligibility; reward
  calculation; banked/unbanked Risk Bonus; Cohesion reward effect; currencies
  and boss materials; loot pools and deterministic paths; earned fixed-item
  drops; guaranteed/first-clear/signature rules; salvage; bounded first-release
  crafting/upgrades; consumable costs; transaction orchestration; and economy
  prohibitions.
- **Does not own:** Boss outcome, raw evidence, item/progression semantics, paid
  offers, presentation, or persistence implementation.
- **Depends on:** Boss outcome, Rhythm/Combat contribution, Positioning risk
  facts, Multiplayer participation facts, Difficulty & Scaling, Items,
  Progression, Player Data, and catalogs.
- **Used by:** Items, Progression, Commerce equivalence, Results, Hub economy
  surfaces, UI, Player Data, and Analytics.
- **Needs detailed spec?:** Yes—`REWARDS_AND_ECONOMY.md`.
- **Major unresolved decisions:** Quantities, chances, costs, deterministic path,
  catalogs, transaction edge cases, and Risk/Cohesion caps.
- **GDD sources:** GD-11, GD-13, GD-15, GD-21, GD-24, GD-26 through GD-28,
  GD-31, GD-32, and GD-34.

### 6.4 Commerce

- **Purpose:** Handle optional Robux purchases without bypassing progression
  fairness or exploiting the target audience.
- **Owns:** Paid catalog; store eligibility; purchase/confirmation/receipt
  lifecycle; duplicate protection; earnable equivalents; tier/stat validation;
  current-tier grants; allowed/prohibited products; and prohibited prompt
  surfaces.
- **Does not own:** Granted-item semantics, earned rewards, progression,
  Marketplace behavior, store UI, or persistence implementation.
- **Depends on:** External Roblox Marketplace, Items, Rewards/Economy,
  Progression tier, Onboarding, Player Data, UI, and Communication & Safety.
- **Used by:** Items, Player Data, Hub/store surfaces, UI, and Analytics.
- **Needs detailed spec?:** Yes—as a separate major section of
  `REWARDS_AND_ECONOMY.md`.
- **Major unresolved decisions:** Catalog, prices, equivalence validation,
  receipt/recovery behavior, review procedure, and compliance testing.
- **GDD sources:** GD-26, GD-28 through GD-30, GD-32 through GD-34.

### 6.5 Progression

- **Purpose:** Preserve advancement and determine which content, options, and
  world states have been earned.
- **Owns:** General progression and system unlocks; campaign destinations,
  first clears, fragments, and tiers; per-boss difficulty unlocks; boss mastery;
  personal bests; unlock eligibility; recommended power; old-item uplift
  eligibility; victory/failure progress; hub restoration; and non-expiring
  progression policy.
- **Does not own:** Raw evidence, canonical participation eligibility, item
  ownership, reward orchestration, milestone content, persistence, or
  presentation.
- **Depends on:** Boss outcome, Multiplayer/Rhythm/Combat evidence,
  Rewards/Economy eligibility and transaction, Difficulty & Scaling, Player
  Data, and milestone catalogs.
- **Used by:** Items, Builds, Rewards, Commerce, Multiplayer difficulty choice,
  Hub, Onboarding, Results, UI, Player Data, and Analytics.
- **Needs detailed spec?:** Yes—`PROGRESSION.md`.
- **Major unresolved decisions:** Track transitions, first-clear idempotency,
  progression amounts, mastery/personal-best rules, unlock catalog,
  recommended-power semantics, uplift, and hub output.
- **GDD sources:** GD-01, GD-08, GD-13, GD-21, GD-24 through GD-32, and GD-34.

## 7. Experience Shell

### 7.1 Order Hub & Navigation

- **Purpose:** Provide a readable physical home and fast routes into encounter,
  preparation, progression, practice, social, and voluntary-store flows.
- **Owns:** Spatial/navigation composition; shard and functional-anchor
  interactions; fast access; application of visible restoration state; and
  optional activity/landmark rules.
- **Does not own:** Campaign, matchmaking, items, economy, Commerce eligibility,
  or platform social behavior.
- **Depends on:** Progression, Multiplayer, Items, Builds, Rewards, Commerce,
  Onboarding, UI, Audio, and authored world content.
- **Used by:** Onboarding, encounter entry, preparation/economy/store surfaces,
  Results return routing, and Analytics.
- **Needs detailed spec?:** No standalone system spec; a major `UI_UX.md`
  section plus a later hub content/world brief.
- **Major unresolved decisions:** Exact layout, travel affordances, activation
  treatment, authored restoration stages, and world content.
- **GDD sources:** GD-01, GD-29, GD-30, GD-32, and GD-33.

### 7.2 Onboarding & Practice

- **Purpose:** Teach the minimum playable vocabulary safely and remember what
  was completed, skipped, or should be prompted again.
- **Owns:** Sequence/checkpoints; completion, skip, replay; safe practice state;
  contextual triggers; prompt eligibility/history; public-matchmaking gate; the
  onboarding part of store eligibility; calibration/settings entry; and
  non-pausing contextual instruction.
- **Does not own:** Calibration math/storage, rhythm/combat rules, authored
  practice charts, matchmaking, Commerce, or presentation.
- **Depends on:** Authoring, Rhythm, Combat, Boss Encounters, Positioning,
  Survival, Abilities, Input/Settings, Progression, Commerce, UI, Audio, and
  Player Data.
- **Used by:** Multiplayer eligibility, Commerce eligibility, Hub, UI, Player
  Data, and Analytics.
- **Needs detailed spec?:** Yes—as a major section of `UI_UX.md`.
- **Major unresolved decisions:** Tutorial copy, prompt timing, checkpoints,
  practice content, skip/reference presentation, and failure/recovery behavior.
- **GDD sources:** GD-01, GD-06, GD-07, GD-29, GD-30, GD-33, and GD-34.

### 7.3 Results & Feedback

- **Purpose:** Explain what happened, what was earned, how the player performed,
  and what can happen next without delaying retry or blaming players.
- **Owns:** Derived summary; outcome/performance separation; exact reason
  display; presentation of already-granted rewards/unlocks; detail views;
  private suggestions; adaptive next action; follow-up routes; and no-claim,
  no-ranking, no-blame, no-paid-prompt rules.
- **Does not own:** Outcome, rewards/grants, progression mutation, personal-best
  state, rematch membership, loadout state, or Commerce.
- **Depends on:** Boss, Rhythm, Combat, Survival, Positioning, Abilities, Solo
  Support, Multiplayer, Rewards, Progression, Items, UI, Audio, and approved
  Analytics-derived inputs.
- **Used by:** Multiplayer follow-up, Hub return, preparation/progression routes,
  UI, and Analytics.
- **Needs detailed spec?:** Yes—as a major section of `UI_UX.md`.
- **Major unresolved decisions:** Rating and suggestion formulas, adaptive-action
  rules, layouts, animation timing, and accessibility validation.
- **GDD sources:** GD-01, GD-07, GD-21 through GD-23, GD-26 through GD-28,
  GD-32 through GD-34.

### 7.4 UI Presentation

- **Purpose:** Present every domain's semantic state and player action in a
  coherent, phone-first, device-appropriate interface.
- **Owns:** HUD/screen composition; hierarchy; navigation/focus and component
  states; responsive layout/safe areas/scaling; rendering semantic cues; device
  labels; captions/subtitles/source labels; and presentation for every major
  flow.
- **Does not own:** Domain state, physical-input interpretation, accessibility
  policy, audio mixing, or persistent settings.
- **Depends on:** Semantic contracts from all player-facing systems, Input &
  Settings, Accessibility, Audio metadata, Communication & Safety, Art
  Direction, and loaded preferences.
- **Used by:** Every player-facing runtime system and orchestrated experience.
- **Needs detailed spec?:** Yes—`UI_UX.md`.
- **Major unresolved decisions:** Responsive layouts, component/state catalog,
  focus/navigation, measurements, loading/error/recovery behavior, and complete
  device/accessibility matrices.
- **GDD sources:** GD-01, GD-03, GD-05 through GD-09, GD-11 through GD-19,
  GD-22 through GD-24, and GD-28 through GD-34.

### 7.5 Input, Settings & Calibration

- **Purpose:** Convert supported physical controls into stable semantic actions
  and provide player-controlled timing, comfort, accessibility, and presentation
  profiles.
- **Owns:** Touch/keyboard/gamepad mapping; context modes; active-device
  detection; profiles and remapping; touch configuration; settings definitions
  and values; guided/manual calibration; and profile outputs.
- **Does not own:** Gameplay consequences, UI rendering, rhythm judgment,
  application of settings by consumers, or persistence implementation.
- **Depends on:** External Roblox input/device capabilities, Accessibility, UI,
  Player Data, and semantic action contracts.
- **Used by:** Rhythm, Combat, Positioning, Abilities, Multiplayer, Onboarding,
  UI, Audio, Player Data, and Analytics.
- **Needs detailed spec?:** Yes—as a major section of `UI_UX.md`.
- **Major unresolved decisions:** Complete binding map, context precedence,
  calibration details, device transitions, settings catalog, profile behavior,
  and input failure cases.
- **GDD sources:** GD-03, GD-05 through GD-08, GD-15, GD-18, GD-19, GD-22,
  GD-29, GD-30, GD-33, and GD-34.

### 7.6 Accessibility

- **Purpose:** Make critical play, navigation, communication, and feedback
  perceivable and operable without changing difficulty, rewards, privacy, or
  dignity.
- **Owns:** Multimodal/non-color requirements; shape/label/placement/motion
  reinforcement; scalable UI/staff/notes/touch controls; independently reducible
  motion/flashing/effects/haptics/audio; difficulty-independent assists; reward
  and access neutrality; and no public accessibility label or shaming.
- **Does not own:** Domain semantic state, setting values, implementation,
  difficulty, rewards, or persistent profiles.
- **Depends on:** Semantic state from every domain, UI, Input/Settings, Audio,
  Communication & Safety, Authoring validation, and playtest evidence.
- **Used by:** Every runtime system, surface, content package, validator, and
  detailed specification.
- **Needs detailed spec?:** No standalone spec; a major `UI_UX.md` section and
  mandatory acceptance criteria in every affected spec.
- **Major unresolved decisions:** Exact option behavior, acceptance matrices,
  device combinations, content validation, and target-age usability evidence.
- **GDD sources:** GD-03, GD-05 through GD-09, GD-11, GD-13 through GD-16,
  GD-18, GD-19, GD-23, GD-28 through GD-34.

### 7.7 Audio Presentation

- **Purpose:** Preserve musical clarity and communicate performance, danger,
  cooperation, and world response through a controllable, accessible mix.
- **Owns:** Stable song/stem presentation; local-instrument response; gameplay,
  crowd, and ambience mixing; critical cue priority/ducking; buses, dynamic
  range, mono compatibility, and caption/source metadata; aggregate band audio;
  and restrained haptic/impact requests.
- **Does not own:** Musical clock, judgments, gameplay semantics, source
  authoring, settings values, or caption rendering.
- **Depends on:** Rhythm clock/judgments; semantic events from Combat, Survival,
  Boss, Positioning, Abilities, Solo Support, and Multiplayer; Authoring; Input/
  Settings; Accessibility; UI; and audio assets.
- **Used by:** Every player-facing runtime system, Hub, Onboarding, Results,
  Communication & Safety, and Analytics.
- **Needs detailed spec?:** Yes—`AUDIO_PRESENTATION.md`.
- **Major unresolved decisions:** Bus/cue catalog, mix targets, responsive stem
  behavior, caption metadata, haptic requests, device profiles, and performance
  budgets.
- **GDD sources:** GD-03, GD-07, GD-09, GD-13 through GD-20, GD-23, GD-29
  through GD-34.

## 8. Content and Platform

### 8.1 Song, Chart & Encounter Authoring

- **Purpose:** Turn approved music and encounter concepts into reviewed,
  validated, versioned runtime packages.
- **Owns:** Offline source intake and provenance; analysis suggestions; human
  chart/difficulty authoring; Activity Maps and ensemble coverage; encounter
  timelines; validation aggregation; approval/versioning; and runtime export.
  Automation and AI may suggest but never approve or publish.
- **Does not own:** Roblox runtime execution, gameplay-domain semantics, or an
  in-game authoring interface.
- **Depends on:** Approved source music/briefs, domain validation requirements,
  Accessibility, human review, and external processing tools.
- **Used by:** Rhythm, Boss Encounters, Positioning, Survival, Abilities, Solo
  Support, Multiplayer validation, Audio, and Analytics.
- **Needs detailed spec?:** Yes—`CONTENT_AUTHORING.md`, specification 1.
- **Major unresolved decisions:** Expanded bundle fields, Activity Map and event
  contracts, approval/rework/version lifecycle, validators/evidence, and Roblox
  export adaptation.
- **GDD sources:** GD-02, GD-04 through GD-10, GD-14 through GD-19, GD-21,
  GD-29, GD-31, GD-33, and GD-34.

This system is an offline, platform-neutral extension of the maintained
[`tools/chart-pipeline/`](tools/chart-pipeline/README.md), not a Roblox runtime
feature. Its first pass establishes canonical song data before gameplay specs.
It receives a mandatory reconciliation after specifications 2 through 12 and
before technical architecture is finalized.

### 8.2 Player Data

- **Purpose:** Durably preserve player-owned facts and cross-domain transactions
  without taking ownership of domain meaning.
- **Owns:** Loading/saving/recovery guarantees; cross-domain durable commit;
  versioning/migration; defaults; concurrency/stale-write protection;
  retry/rollback/failure policy; approved durable records; and player-visible
  unsafe/unavailable-save behavior.
- **Does not own:** Domain mutation semantics, gameplay decisions, ephemeral
  reconnect state, UI, or external storage services.
- **Depends on:** Durable-domain contracts, Rewards transaction orchestration,
  Commerce decisions, UI failure presentation, safety/privacy policy, and
  external Roblox capabilities.
- **Used by:** Progression, Items, Builds, Rewards, Commerce, Multiplayer where
  durable facts apply, Onboarding, Input/Settings, Results, UI, and approved
  Analytics.
- **Needs detailed spec?:** Yes—`PLAYER_DATA.md`.
- **Major unresolved decisions:** Record inventory, transaction matrix, save
  cadence, version/migration, failure/recovery, deletion/export, and technical
  budgets.
- **GDD sources:** GD-07, GD-08, GD-16 through GD-18, GD-22 through GD-30,
  GD-32 through GD-34.

### 8.3 Analytics & Playtest Evidence

- **Purpose:** Produce privacy-conscious evidence that the game is
  understandable, fair, accessible, and ready.
- **Owns:** Event/metric catalog; collection/segmentation; evidence synthesis;
  GD-34 reports; data-quality checks; research consent/safeguarding/retention/
  access boundaries; and prohibition on automatic gameplay or public-label
  changes.
- **Does not own:** Semantic source facts, live gameplay, rewards/progression,
  public scoring, durable domain records, or external transport.
- **Depends on:** Semantic events from every system, approved Player Data use,
  Accessibility, Communication & Safety, research operations, and external
  Roblox telemetry/privacy capabilities.
- **Used by:** Design review, Authoring validation, Difficulty & Scaling,
  `BALANCE_FRAMEWORK.md`, spec validation plans, Results where privately
  approved, and release-readiness decisions.
- **Needs detailed spec?:** Yes—`PLAYTEST_AND_ANALYTICS.md` as a supporting
  document.
- **Major unresolved decisions:** Event meanings/payloads, identifiers,
  retention/consent, study protocol, segmentation, confidence, and operational
  sign-off.
- **GDD sources:** GD-07 through GD-10, GD-21 through GD-23, GD-28, GD-29,
  GD-32 through GD-34.

## 9. Dependency and outcome views

### Core encounter

```text
Offline approved song/encounter package
                |
                v
       Rhythm Gameplay clock
                |
                v
        normalized performance
                |
                v
             Combat
          /     |      \
         v      v       v
   Boss Resolve Ward   Abilities/group effects
         |       |       |
         +-------+-------+
                 |
                 v
       Boss Encounters outcome
```

Boss Encounters and Tactical Positioning exchange authored attack geometry,
location, cover, hazard, and movement facts. Multiplayer supplies roster state;
Solo Support supplies fixed assistance only in solo. Difficulty & Scaling
constrains every applicable step.

### Session and post-battle flow

```text
Order Hub / shard
       |
       v
Multiplayer staging and roster
       |
       v
Boss Encounter
       |
       v
Results & Feedback
   |       |        |
 Retry   Stay     Hub / preparation / story
```

### Earned transaction

```text
Outcome + performance + roster + risk facts
                    |
                    v
          Rewards, Loot & Economy
             /               \
            v                 v
   Items mutation       Progression mutation
             \               /
              v             v
             Player Data commit
```

Commerce creates a separate paid transaction path through external Roblox
Marketplace capabilities, validates equivalence and tier policy, and hands the
resulting item mutation to Items before Player Data commits it.

### Player-experience contract

```text
Physical controls
      |
      v
Input, Settings & Calibration
      |
      v
semantic actions -> owning gameplay systems -> semantic state/events
                                             /                 \
                                            v                   v
                                     UI Presentation     Audio Presentation
```

Accessibility and Communication & Safety constrain the entire loop. Analytics
observes approved semantic events but never controls gameplay automatically.

## 10. Detailed specification sequence

The approved order is:

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

`CONTENT_AUTHORING.md` first establishes the offline song-data and exported
bundle contract. After specifications 2 through 12 identify their complete
authored-data and validation needs, it receives a mandatory reconciliation pass.
Only after that reconciliation and the other architecture-critical
specifications are sufficiently settled may `TECHNICAL_ARCHITECTURE.md` become
canonical.

`BALANCE_FRAMEWORK.md` and `PLAYTEST_AND_ANALYTICS.md` are supporting documents
developed alongside later specifications. The required naming-and-tone pass is
a separate product task.

No standalone specification is initially required for Player Survival &
Recovery, Tactical Positioning & Movement, Solo Support, Difficulty & Scaling,
Communication & Safety, Commerce, Inventory, consumables, Order Hub, Onboarding,
Results, Input, or Accessibility because each has an explicit parent document.

## 11. Exclusions and routed work

### Explicit first-release non-goals

- PvP
- user-authored songs
- free-roaming worlds
- deep crafting beyond the bounded deterministic first-release capability
- player trading
- paid recovery
- paid randomness
- multi-song raids

Quests, achievements, daily or retention mechanics, live operations, and guilds
are not assumed for the first release. Adding one requires an explicit design
change rather than genre convention.

### Not systems

Individual songs, bosses, arenas, items, rewards, screens, hub locations,
narrative beats, and asset packages are content or surfaces unless a later
approved reusable ruleset gives them an independent lifecycle.

Content catalogs, narrative and boss/song briefs, world layouts, asset
production, final names, balance values, research procedure, technical
authority/networking/security, and file/data schemas are routed to their named
content, naming, balance, playtest, system-specification, or technical-
architecture documents.

## 12. Approval and change control

The bounded owner interview resolved SM-01 through SM-18 on 2026-08-18. Every
first-release responsibility has one primary owner or an explicit cross-cutting
classification. Dependency cycles documented here are semantic handshakes, not
permission for duplicated mutable state.

A later specification may refine a system's internal rules, values, content
contract, or interfaces without changing this map. A material merge, split,
reclassification, ownership transfer, new first-release system, or change to the
Content Authoring-first dependency must amend this document explicitly and cite
the superseded decision.
