# Bands Battle UI/UX Working Record

- **Status:** Interview in progress; 0 of 12 questions resolved
- **Started:** 2026-08-30
- **Question plan:** [`UI_UX_QUESTIONS.md`](UI_UX_QUESTIONS.md)
- **Parent systems:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#7-experience-shell)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Planned canonical result:** `UI_UX.md`

## 1. Role of this record

This file preserves approved answers, refinements, inherited constraints, and
cross-system handoffs while the UI/UX interview is active. It is evidence for
the canonical specification, not the final authority.

## 2. Inherited boundary

UI/UX owns experience composition, navigation, focus, responsive presentation,
semantic action mapping, settings/calibration definitions, onboarding
orchestration, results presentation, and accessibility acceptance. It presents
and routes semantic facts from owning systems; it does not recalculate gameplay,
resolve transactions, change rewards, repair loadouts, invent progression, or
implement persistence.

The complete inherited decision set is recorded in
[`UI_UX_QUESTIONS.md`](UI_UX_QUESTIONS.md#2-fixed-inherited-decisions).

## 3. Decision record

### Checkpoint A - Experience architecture, hub, and navigation

#### UX-01 - Player jobs, experience modes, and navigation hierarchy

- **Status:** Open.

#### UX-02 - Physical hub wayfinding and fast-access routes

- **Status:** Open.

#### UX-03 - Responsive shell, safe areas, and device navigation

- **Status:** Open.

### Checkpoint B - Encounter HUD, controls, and readable action

#### UX-04 - Persistent encounter hierarchy and contextual surfaces

- **Status:** Open.

#### UX-05 - Semantic controls, remapping, and active-device transitions

- **Status:** Open.

#### UX-06 - Telegraphs, camera, position, and cue arbitration

- **Status:** Open.

### Checkpoint C - Learning, preparation, and post-battle flow

#### UX-07 - Onboarding, practice, and contextual teaching

- **Status:** Open.

#### UX-08 - Staging, loadout, inventory, builds, upgrades, and store surfaces

- **Status:** Open.

#### UX-09 - Results summary, evidence, and next actions

- **Status:** Open.

### Checkpoint D - Accessibility, system states, and implementation contract

#### UX-10 - Settings, calibration, accessibility, and saved profiles

- **Status:** Open.

#### UX-11 - Feedback, focus, loading, failure, and recovery states

- **Status:** Open.

#### UX-12 - Design system, semantic outputs, localization, and acceptance

- **Status:** Open.

## 4. Content/configuration reconciliation register

- No new authoring requirements have been approved yet.
- Final reconciliation must distinguish runtime UI configuration, localized
  content, authored encounter cue metadata, and facts owned by other systems.
- `CONTENT_AUTHORING.md` must gain only song/encounter-owned cue and validation
  requirements, not general UI layout, input mapping, settings, or hub behavior.

## 5. Open handoffs

- Gameplay systems own semantic state, timings, legality, and results; UI owns
  their composition, labels, focus, announcements, and device presentation.
- Multiplayer owns queue/party/rematch membership and communication safety; UI
  owns visible state, consent surfaces, focus, mute/block/report access, and
  recovery routing.
- Items, Builds, Progression, Rewards, and Commerce own domain validation and
  mutation; UI owns task flow, exact disclosure, confirmation, and failure
  presentation.
- Audio Presentation owns mix and audible cue output; UI/UX owns caption and
  subtitle rendering plus visual/haptic reinforcement requirements.
- Player Data owns durable profile/configuration guarantees; UI/UX owns setting
  definitions and player-visible save-unavailable or unsafe-save treatment.

## 6. Change log

- **2026-08-30:** Created the working record. Progress is 0 of 12 questions.
