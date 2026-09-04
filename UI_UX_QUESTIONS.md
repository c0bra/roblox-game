# Bands Battle UI/UX Specification Questions

- **Status:** Interview complete; 12 of 12 questions resolved and reconciled
- **Started:** 2026-08-30
- **Parent systems:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#7-experience-shell)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Rhythm dependency:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)
- **Encounter dependency:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Multiplayer dependency:** [`MULTIPLAYER.md`](MULTIPLAYER.md)
- **Items/preset dependency:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)
- **Working record:** [`UI_UX_WORKING.md`](UI_UX_WORKING.md)
- **Canonical result:** [`UI_UX.md`](UI_UX.md)

## 1. Interview method

This interview uses four checkpoints of three questions. It follows a task-first
UX process: establish player jobs and information architecture, define the
encounter interaction surface, resolve learning and post-battle flows, then
complete accessibility and every non-ideal interface state.

Answers are persisted after each checkpoint. When all twelve questions are
resolved, they are reconciled into canonical `UI_UX.md`. Exact measurements,
visual styling, shipping names, and implementation technology remain downstream
unless a value is necessary to protect usability or gameplay fairness.

## 2. Fixed inherited decisions

- The experience is phone-first while supporting touch, keyboard/mouse, and
  gamepad without changing gameplay opportunity or rewards.
- The Order hub is a physical home with a tiered phasing-shard landmark, stable
  activation points, readable locks, visible higher campaign tiers, evolving
  restoration, and distinct supporting anchors.
- Returning players need fast routes to playable shards and essential menus.
  Retry Same Shard bypasses the hub, and public matchmaking may continue during
  compatible hub movement and menu use.
- Shards open an encounter card rather than starting through proximity. Players
  then choose solo, public matchmaking, or current party before difficulty,
  preparation, Ready, and final loadout lock.
- The normal battle camera frames the performer below and the boss above. The
  boss, arena telegraphs, and tactical positions remain legible at phone size.
- Persistent encounter information includes survival state, selected intent,
  personal Signature/Hype state, boss resistance and phase, and three fixed
  rhythm pads. Phrase, movement, attack, cooperative, consumable, revival, and
  recovery surfaces are contextual.
- Rhythm controls never move or gain unrelated meanings. Current-device labels
  or glyphs appear on prompts, and occasional actions may queue to musical
  boundaries.
- The complete baseline semantic action map from GD-06 is binding. Keyboard and
  gamepad remapping applies where Roblox permits; touch supports handedness and
  adjustable secondary-control layout.
- First-time practice is four to six minutes, checkpointed, replayable, safely
  repeatable, and skippable after control confirmation. It teaches one concept
  at a time, saves after each module, and never requires Perfect.
- Advanced systems are taught contextually during the first boss without
  pausing or rewinding cooperative play. Contextual teaching may be disabled and
  no important instruction becomes permanently missable.
- Results begin with a phone-first summary of outcome, exact reason, personal
  rating, important already-granted rewards/unlocks, and one prominent adaptive
  next action. Details are optional, private, non-ranking, and non-blaming.
- Store access is voluntary, unavailable until onboarding and one encounter are
  complete, and absent from combat, downing, recovery, defeat, results, and
  immediate retry.
- Critical meaning never relies on color, sound, motion, or haptics alone.
  Accessibility assists remain private, independent of difficulty, and neutral
  to rewards, mastery, campaign credit, and matchmaking.
- Accessibility settings are available before onboarding and from every safe
  menu. Solo pause freezes song and encounter with a beat-counted resume;
  cooperative play cannot pause the shared song and explains this before entry.
- UI presents semantic facts from owning systems. It never recalculates or
  silently repairs gameplay, rewards, progression, loadouts, or saved data.

## 3. Question plan

### Checkpoint A - Experience architecture, hub, and navigation

#### UX-01 - Player jobs, experience modes, and navigation hierarchy

- **Status:** Resolved 2026-08-30.
- **Decision needed:** What are the primary jobs players must accomplish, and
  how should the experience shell keep their current mode and next action clear?
- **Must resolve:** First-time, returning, queued, staging, encounter, results,
  and recovery modes; primary versus secondary jobs; global destinations;
  current-location and back behavior; no dead ends; state preservation; and the
  maximum visible navigation depth.

#### UX-02 - Physical hub wayfinding and fast-access routes

- **Status:** Resolved 2026-08-30.
- **Decision needed:** How do world navigation and menus cooperate so the hub is
  memorable without making repeat play or preparation slow?
- **Must resolve:** Spawn/return placement, shard/anchor readability, locked and
  newly unlocked states, interaction range and confirmation, map/wayfinding,
  fast play, shortcuts, queue persistence, restoration changes, accessibility
  routes, and optional social/world content.

#### UX-03 - Responsive shell, safe areas, and device navigation

- **Status:** Resolved 2026-08-30.
- **Decision needed:** Which navigation and layout model should adapt across
  phone, tablet, desktop, and gamepad while preserving the same task structure?
- **Must resolve:** Primary destinations, compact menu, tabs/drawers/panels,
  controller focus, keyboard navigation, touch targets, safe areas, orientation,
  UI scaling, text expansion, return focus, deep-link/context return, and modal
  limits.

### Checkpoint B - Encounter HUD, controls, and readable action

#### UX-04 - Persistent encounter hierarchy and contextual surfaces

- **Status:** Resolved 2026-08-31.
- **Decision needed:** What stays visible during battle, what appears only at the
  point of need, and how is attention divided between staff, boss, and team?
- **Must resolve:** Persistent HUD regions, phrase staff/pads, survival/intent/
  Hype, Resolve/phase, contextual priority, cooperative invitations, movement,
  consumables, revival/recovery, multiple simultaneous states, occlusion, and
  glance/read-time targets.

#### UX-05 - Semantic controls, remapping, and active-device transitions

- **Status:** Resolved 2026-08-31.
- **Decision needed:** How do physical inputs map to semantic actions and remain
  safe when focus, device, context, or bindings change?
- **Must resolve:** GD-06 mappings, fixed rhythm controls, context precedence,
  simultaneous inputs, queued actions, remapping conflicts, reserved/platform
  controls, active-device detection, rapid device switching, touch edit mode,
  focus capture, disconnect, and control reference.

#### UX-06 - Telegraphs, camera, position, and cue arbitration

- **Status:** Resolved 2026-08-31.
- **Decision needed:** How do arena and interface cues reinforce one another
  without hiding the boss or creating contradictory instructions?
- **Must resolve:** Attack geometry, targets/safe areas, position graph and
  travel/recovery, directed camera moments, reduced motion, off-screen/source
  indicators, overlapping attacks, cue priority, committed versus advisory
  states, interruption limits, and impossible-combination handling.

### Checkpoint C - Learning, preparation, and post-battle flow

#### UX-07 - Onboarding, practice, and contextual teaching

- **Status:** Resolved 2026-08-31.
- **Decision needed:** How does the approved practice sequence deliver first
  value quickly and then hand off to nonintrusive first-boss teaching?
- **Must resolve:** Entry and accessibility setup, starter instrument, six
  modules, success/retry/checkpoint behavior, skip confirmation, public gate,
  replay/reference, prompt trigger/history/suppression, first-boss script,
  failure recovery, and experienced/returning players.

#### UX-08 - Staging, loadout, inventory, builds, upgrades, and store surfaces

- **Status:** Resolved 2026-08-31.
- **Decision needed:** How should dense preparation and collection tasks remain
  understandable on phones without hiding exact consequences?
- **Must resolve:** Task-based information architecture, current versus saved
  full spec presets, comparison, filters/sort/search, progressive disclosure,
  validation and atomic Apply, Ready invalidation, queue-safe editing, locked/
  empty/full states, purchase safeguards, undo/confirmation, and return path.

#### UX-09 - Results summary, evidence, and next actions

- **Status:** Resolved 2026-08-31.
- **Decision needed:** How does Results explain the attempt within seconds while
  making deeper evidence and the right follow-up easy to reach?
- **Must resolve:** Brief presentation/skip, immediate hierarchy, exact-reason
  priority, adaptive primary action, performance/combat/band/progress detail,
  personal-best comparison, private suggestions, reward/progress animation,
  rematch/refill states, loadout route, no claim/store/blame, and exit timing.

### Checkpoint D - Accessibility, system states, and implementation contract

#### UX-10 - Settings, calibration, accessibility, and saved profiles

- **Status:** Resolved 2026-08-31.
- **Decision needed:** Which settings exist, where they are available, and how
  do players preview, save, reset, and understand their effects?
- **Must resolve:** Calibration, remapping, touch layout, Hold Assist, scale/
  scroll speed, contrast/color vision, motion/flash/effects/haptics, captions/
  subtitles, audio handoff, language, solo pause, cooperative limitation,
  per-device/global profile scope, defaults, preview, reset, and privacy.

#### UX-11 - Feedback, focus, loading, failure, and recovery states

- **Status:** Resolved 2026-08-31.
- **Decision needed:** How does every surface remain truthful and recoverable
  during waiting, partial failure, network change, invalid state, or save risk?
- **Must resolve:** Default/focus/active/disabled/loading/success/error states;
  empty/min/max content; skeleton/progress behavior; cancel/retry; preserved
  input; stale data; reconnect; queue and party changes; save unavailable/unsafe;
  transactional pending/failure; duplicate action prevention; and nonblocking
  notifications.

#### UX-12 - Design system, semantic outputs, localization, and acceptance

- **Status:** Resolved 2026-08-31.
- **Decision needed:** Which component, content, semantic-event, and test
  contracts make the UI implementable without an agent inventing behavior?
- **Must resolve:** Tokens and component catalog, complete state matrix, domain
  fact-to-presentation mapping, priority and deduplication, focus/announcement,
  privacy, age-appropriate copy, internal-name gate, localization/RTL/text
  expansion, device/accessibility matrix, performance budgets, analytics handoff,
  Content Authoring register, and observable usability gates.

## 4. Completion criteria

`UI_UX.md` is complete only when:

- UX-01 through UX-12 are resolved;
- every major player job has a clear start, current location, completion, back,
  cancellation, and recovery route without excessive hierarchy;
- hub spectacle and physical progression coexist with fast repeat play and
  accessible stable wayfinding;
- encounter information and controls remain readable, fixed where timing
  demands it, and fair across touch, keyboard/mouse, and gamepad;
- onboarding reaches real play quickly, remains replayable/skippable, and never
  permanently hides required knowledge;
- preparation and results use progressive disclosure without hiding exact
  consequences, already-granted rewards, failures, or next actions;
- settings and accessibility preserve every critical cue and remain independent
  of difficulty, reward, public identity, and monetization;
- all ideal, empty, minimum, maximum, loading, disabled, error, disconnect,
  stale, and recovery states are specified; and
- semantic inputs/outputs, component states, localization, privacy, validation,
  and usability evidence leave no implementation-agent design choice.

## 5. Change log

- **2026-08-30:** Created the concise 12-question UI/UX interview from the
  approved GDD, Systems Map, canonical dependencies, and task-first UX review.
- **2026-08-30:** Approved UX-01 through UX-03, completing experience
  architecture, physical/fast hub navigation, and the responsive device shell.
  Progress is 3 of 12 questions.
- **2026-08-31:** Approved UX-04 through UX-06, completing the encounter HUD,
  semantic controls/device transitions, and telegraph/camera/cue checkpoint.
  Progress is 6 of 12 questions.
- **2026-08-31:** Approved UX-07 through UX-09, completing onboarding/practice,
  preparation/collection, and Results/post-battle checkpoint C. Progress is 9
  of 12 questions.
- **2026-08-31:** Approved UX-10 through UX-12, completing settings/accessibility,
  system-state/recovery, and implementation-contract checkpoint D. All twelve
  questions were reconciled into canonical `UI_UX.md`.
