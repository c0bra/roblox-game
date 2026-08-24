# Bands Battle System Specifications Tracker

- **Status:** In progress; 6 of 13 canonical specifications complete
- **Started:** 2026-08-19
- **Authority:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md)
- **Current specification:** `ABILITIES_AND_COOPERATIVE_ACTIONS.md`

## Purpose

This document tracks the owner interview and approval state for the thirteen
detailed design specifications required by `SYSTEMS_MAP.md`. It does not replace
those specifications or change their approved order.

Each specification proceeds through four persisted states:

1. **Question plan:** a finite `<SPEC>_QUESTIONS.md` defines the decisions needed.
2. **Working record:** `<SPEC>_WORKING.md` records owner answers and consequences
   at meaningful checkpoints.
3. **Canonical draft:** the resolved decisions are reconciled into `<SPEC>.md`.
4. **Approved:** the canonical document passes its completion audit and becomes
   the authority for its defined system boundaries.

## Progress

| # | Canonical specification | State | Interview progress | Notes |
|---:|---|---|---|---|
| 1 | [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md) | Approved baseline | 24 of 24 | Mandatory reconciliation after specs 2–12 |
| 2 | [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md) | Approved | 12 of 12 | Canonical specification published |
| 3 | [`COMBAT.md`](COMBAT.md) | Approved | 12 of 12 | Includes Player Survival & Recovery |
| 4 | [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md) | Approved | 12 of 12 | Includes Tactical Positioning & Movement |
| 5 | [`PROGRESSION.md`](PROGRESSION.md) | Approved | 12 of 12 | Three tracks plus full spec presets |
| 6 | [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md) | Approved | 12 of 12 | Includes inventory, presets, consumables, cosmetics |
| 7 | `ABILITIES_AND_COOPERATIVE_ACTIONS.md` | Interview in progress | 0 of 12 | Includes Solo Support |
| 8 | `MULTIPLAYER.md` | Not started | — | Includes Communication & Safety |
| 9 | `REWARDS_AND_ECONOMY.md` | Not started | — | Includes Commerce |
| 10 | `BUILDS_AND_SPECIALIZATION.md` | Not started | — | Working names require later naming pass |
| 11 | `UI_UX.md` | Not started | — | Includes hub, onboarding, results, input, and accessibility |
| 12 | `AUDIO_PRESENTATION.md` | Not started | — | Runtime musical and semantic audio presentation |
| 13 | `PLAYER_DATA.md` | Not started | — | Final architecture-critical platform contract |

## Required reconciliation

`CONTENT_AUTHORING.md` receives a mandatory reconciliation after specifications
2 through 12 identify their complete authored-data and validation needs. The
thirteen-document program is not complete until that reconciliation resolves all
new authoring requirements and the cross-spec audit finds no orphaned or
contradictory contract.

`BALANCE_FRAMEWORK.md` and `PLAYTEST_AND_ANALYTICS.md` remain required supporting
documents and will be developed alongside the specifications that supply their
inputs. They are tracked separately from the thirteen canonical system specs.

## Change log

- **2026-08-19:** Created the tracker and began the Content Authoring interview.
- **2026-08-19:** Completed Content Authoring checkpoint A. Overall progress is
  0 of 13 specifications and 3 of 24 Content Authoring questions.
- **2026-08-19:** Completed Content Authoring checkpoint B. Overall progress is
  0 of 13 specifications and 6 of 24 Content Authoring questions.
- **2026-08-19:** Approved CA-08 and CA-09 and opened a density-aware CA-07
  refinement. Overall progress is 0 of 13 specifications and 8 of 24 Content
  Authoring questions.
- **2026-08-19:** Approved the density-aware CA-07 rule and completed Content
  Authoring checkpoint C. Overall progress is 0 of 13 specifications and 9 of
  24 Content Authoring questions.
- **2026-08-20:** Clarified dynamic revival-window behavior and recorded the
  initial local web-app direction for CA-14. Resolved progress remains 9 of 24.
- **2026-08-20:** Approved CA-10 through CA-12 and the CA-14 internal web-app
  surface. Overall progress is 0 of 13 specifications and 13 of 24 Content
  Authoring questions.
- **2026-08-20:** Approved CA-13 and CA-15, completing checkpoint E. Overall
  progress is 0 of 13 specifications and 15 of 24 Content Authoring questions.
- **2026-08-20:** Approved CA-16 through CA-18, completing checkpoint F.
  Overall progress is 0 of 13 specifications and 18 of 24 Content Authoring
  questions.
- **2026-08-21:** Approved CA-19 through CA-21, completing checkpoint G.
  Overall progress is 0 of 13 specifications and 21 of 24 Content Authoring
  questions.
- **2026-08-21:** Simplified owner-created-audio traceability, approved CA-22
  through CA-24, and published the canonical `CONTENT_AUTHORING.md` baseline.
  Overall progress is 1 of 13 specifications complete.
- **2026-08-21:** Approved RG-01 through RG-03, completing Rhythm Gameplay
  checkpoint A. Overall progress remains 1 of 13 specifications complete and
  3 of 12 Rhythm Gameplay questions.
- **2026-08-21:** Approved RG-04 through RG-06, completing Rhythm Gameplay
  checkpoint B. Overall progress remains 1 of 13 specifications complete and
  6 of 12 Rhythm Gameplay questions.
- **2026-08-21:** Approved RG-07 through RG-09, completing Rhythm Gameplay
  checkpoint C. Overall progress remains 1 of 13 specifications complete and
  9 of 12 Rhythm Gameplay questions.
- **2026-08-21:** Approved RG-10 through RG-12 and published canonical
  `RHYTHM_GAMEPLAY.md`. Overall progress is 2 of 13 specifications complete;
  Combat is next.
- **2026-08-21:** Created the 12-question Combat/Survival interview and working
  record. Combat progress is 0 of 12.
- **2026-08-21:** Approved CM-01 through CM-03, completing Combat checkpoint A.
  Overall progress remains 2 of 13 specifications complete and 3 of 12 Combat
  questions.
- **2026-08-21:** Approved CM-04 through CM-06, completing Combat checkpoint B.
  Overall progress remains 2 of 13 specifications complete and 6 of 12 Combat
  questions.
- **2026-08-21:** Approved CM-07 through CM-09, completing Combat checkpoint C.
  Overall progress remains 2 of 13 specifications complete and 9 of 12 Combat
  questions.
- **2026-08-21:** Approved CM-10 through CM-12 and published canonical
  `COMBAT.md` with Player Survival & Recovery. Overall progress is 3 of 13
  specifications complete; Boss Encounters is next.
- **2026-08-21:** Created the 12-question Boss Encounter/Positioning interview
  and working record. Boss Encounter progress is 0 of 12.
- **2026-08-22:** Approved BE-01 through BE-03, completing Boss Encounter
  checkpoint A. Overall progress remains 3 of 13 specifications complete and
  3 of 12 Boss Encounter questions.
- **2026-08-22:** Approved BE-04 through BE-06, completing Boss Encounter
  checkpoint B. Overall progress remains 3 of 13 specifications complete and
  6 of 12 Boss Encounter questions.
- **2026-08-22:** Approved BE-07 through BE-09, completing Boss Encounter
  checkpoint C. Overall progress remains 3 of 13 specifications complete and
  9 of 12 Boss Encounter questions.
- **2026-08-22:** Approved BE-10 through BE-12 and published canonical
  `BOSS_ENCOUNTERS.md` with Tactical Positioning & Movement. Overall progress is
  4 of 13 specifications complete; Progression is next.
- **2026-08-22:** Created the 12-question Progression interview and working
  record. Progression progress is 0 of 12.
- **2026-08-22:** Approved PG-01 through PG-03, completing Progression checkpoint
  A. Overall progress remains 4 of 13 specifications complete and 3 of 12
  Progression questions.
- **2026-08-24:** Approved PG-04 through PG-06, completing Progression checkpoint
  B and defining three full quick-switch combat-configuration presets. Overall
  progress remains 4 of 13 specifications complete and 6 of 12 Progression
  questions.
- **2026-08-24:** Approved PG-07 through PG-09, completing Progression checkpoint
  C. Overall progress remains 4 of 13 specifications complete and 9 of 12
  Progression questions.
- **2026-08-24:** Approved PG-10 through PG-12 and published canonical
  `PROGRESSION.md`. Overall progress is 5 of 13 specifications complete; Items &
  Equipment is next.
- **2026-08-24:** Created the 12-question Items & Equipment interview and working
  record. Items & Equipment progress is 0 of 12.
- **2026-08-24:** Approved IE-01 through IE-03, completing Items & Equipment
  checkpoint A. Overall progress remains 5 of 13 specifications complete and
  3 of 12 Items & Equipment questions.
- **2026-08-24:** Approved IE-04 through IE-06, completing Items & Equipment
  checkpoint B. Overall progress remains 5 of 13 specifications complete and
  6 of 12 Items & Equipment questions.
- **2026-08-24:** Approved IE-07 through IE-09, completing Items & Equipment
  checkpoint C. Overall progress remains 5 of 13 specifications complete and
  9 of 12 Items & Equipment questions.
- **2026-08-24:** Approved IE-10 through IE-12 and published canonical
  `ITEMS_AND_EQUIPMENT.md`. Overall progress is 6 of 13 specifications complete;
  Abilities & Cooperative Actions is next.
- **2026-08-24:** Created the 12-question Abilities/Cooperative Actions/Solo
  Support interview and working record. Progress is 0 of 12.
