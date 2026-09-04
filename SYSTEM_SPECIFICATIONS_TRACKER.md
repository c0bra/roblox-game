# Bands Battle System Specifications Tracker

- **Status:** Complete; 13 of 13 canonical specifications approved and final
  reconciliation complete
- **Started:** 2026-08-19
- **Completed:** 2026-09-02
- **Authority:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md)
- **Current specification:** Complete; supporting documents remain separate

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
| 1 | [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md) | Approved and reconciled | 24 of 24 | Mandatory reconciliation completed 2026-09-02 |
| 2 | [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md) | Approved | 12 of 12 | Canonical specification published |
| 3 | [`COMBAT.md`](COMBAT.md) | Approved | 12 of 12 | Includes Player Survival & Recovery |
| 4 | [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md) | Approved | 12 of 12 | Includes Tactical Positioning & Movement |
| 5 | [`PROGRESSION.md`](PROGRESSION.md) | Approved | 12 of 12 | Three tracks plus full spec presets |
| 6 | [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md) | Approved | 12 of 12 | Includes inventory, presets, consumables, cosmetics |
| 7 | [`ABILITIES_AND_COOPERATIVE_ACTIONS.md`](ABILITIES_AND_COOPERATIVE_ACTIONS.md) | Approved | 12 of 12 | Includes Solo Support |
| 8 | [`MULTIPLAYER.md`](MULTIPLAYER.md) | Approved | 12 of 12 | Includes Communication & Safety |
| 9 | [`REWARDS_AND_ECONOMY.md`](REWARDS_AND_ECONOMY.md) | Approved | 12 of 12 | Includes Commerce |
| 10 | [`BUILDS_AND_SPECIALIZATION.md`](BUILDS_AND_SPECIALIZATION.md) | Approved | 12 of 12 | Working names require later naming pass |
| 11 | [`UI_UX.md`](UI_UX.md) | Approved | 12 of 12 | Includes hub, onboarding, results, input, and accessibility |
| 12 | [`AUDIO_PRESENTATION.md`](AUDIO_PRESENTATION.md) | Approved | 12 of 12 | Runtime musical and semantic audio presentation |
| 13 | [`PLAYER_DATA.md`](PLAYER_DATA.md) | Approved | 12 of 12 | Final architecture-critical platform contract |

## Completed reconciliation

The mandatory `CONTENT_AUTHORING.md` reconciliation was completed on 2026-09-02
after specifications 2 through 12 identified their complete authored-data and
validation needs. The consolidated contract assigns each runtime content field
an owner, consumer, compatibility impact, fallback, and validation/review path.

The final cross-specification audit found no orphaned ownership, competing
durable/content source of truth, or contradictory player-facing contract among
the thirteen canonical specifications. Deferred numbers, catalogs, assets,
platform policy details, and implementation mechanisms remain explicitly routed
to balance, content, naming, playtest, policy, or technical architecture.

`BALANCE_FRAMEWORK.md` and `PLAYTEST_AND_ANALYTICS.md` remain required supporting
documents and will be developed alongside the specifications that supply their
inputs. They are tracked separately from the thirteen canonical system specs.

## Final audit coverage

The 2026-09-02 audit verified:

- all thirteen canonical files exist, are approved, and link to their system-map
  authority and decision records;
- the song-specific extensible-role model, Normal difficulty terminology,
  dynamic recovery candidates, no join-in-progress, and locked active snapshots
  remain consistent across consumers;
- Rhythm, Combat, Encounter, Ability, Multiplayer, Economy, configuration, UI,
  Audio, and Player Data facts each retain one semantic owner;
- content/package revisions, system catalogs, frozen attempt snapshots, and
  durable player revisions remain distinct and explicitly referenced;
- rewards, purchases, consumables, Progression, Items, presets/builds, and
  corrections share atomic/idempotent persistence behavior without duplicating
  domain meaning;
- client, other-player, support, Analytics, authoring, and operational exposure
  follows the same minimum-data/privacy boundary; and
- all registered Content Authoring needs are consolidated in its section 14 and
  traced back to the owning canonical specification.

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
- **2026-08-24:** Approved AC-01 through AC-03, completing Abilities checkpoint
  A. Overall progress remains 6 of 13 specifications complete and 3 of 12
  Abilities questions.
- **2026-08-24:** Approved AC-04 through AC-06, completing Abilities checkpoint
  B. Overall progress remains 6 of 13 specifications complete and 6 of 12
  Abilities questions.
- **2026-08-24:** Approved AC-07 through AC-09, completing Abilities checkpoint
  C. Overall progress remains 6 of 13 specifications complete and 9 of 12
  Abilities questions.
- **2026-08-25:** Approved AC-10 through AC-12 and published canonical
  `ABILITIES_AND_COOPERATIVE_ACTIONS.md` with Solo Support. Overall progress is
  7 of 13 specifications complete; Multiplayer is next.
- **2026-08-25:** Created the 12-question Multiplayer/Communication & Safety
  interview and working record. Multiplayer progress is 0 of 12.
- **2026-08-25:** Approved MP-01 through MP-03, completing Multiplayer checkpoint
  A. Overall progress remains 7 of 13 specifications complete and 3 of 12
  Multiplayer questions.
- **2026-08-25:** Approved MP-04 through MP-06, completing Multiplayer checkpoint
  B. Overall progress remains 7 of 13 specifications complete and 6 of 12
  Multiplayer questions.
- **2026-08-25:** Approved MP-07 through MP-09, completing Multiplayer checkpoint
  C. Overall progress remains 7 of 13 specifications complete and 9 of 12
  Multiplayer questions.
- **2026-08-25:** Approved MP-10 through MP-12 and published canonical
  `MULTIPLAYER.md` with Communication & Safety. Overall progress is 8 of 13
  specifications complete; Rewards & Economy is next.
- **2026-08-25:** Created the 12-question Rewards/Economy/Commerce interview and
  working record. Rewards & Economy progress is 0 of 12.
- **2026-08-26:** Approved RE-01 through RE-03, completing Rewards & Economy
  checkpoint A. Overall progress remains 8 of 13 specifications complete and
  3 of 12 Rewards & Economy questions.
- **2026-08-26:** Approved RE-04 through RE-06, completing Rewards & Economy
  checkpoint B. Overall progress remains 8 of 13 specifications complete and
  6 of 12 Rewards & Economy questions.
- **2026-08-26:** Approved RE-07 through RE-09, completing Rewards & Economy
  checkpoint C. Overall progress remains 8 of 13 specifications complete and
  9 of 12 Rewards & Economy questions.
- **2026-08-26:** Approved RE-10 through RE-12 and published canonical
  `REWARDS_AND_ECONOMY.md` with Commerce. Overall progress is 9 of 13
  specifications complete; Builds & Specialization is next.
- **2026-08-26:** Created the 12-question Builds & Specialization interview and
  working record. Builds & Specialization progress is 0 of 12.
- **2026-08-26:** Approved BS-01 through BS-03, completing Builds & Specialization
  checkpoint A. Overall progress remains 9 of 13 specifications complete and
  3 of 12 Builds & Specialization questions.
- **2026-08-27:** Approved BS-04 through BS-06, completing Builds & Specialization
  checkpoint B. Overall progress remains 9 of 13 specifications complete and
  6 of 12 Builds & Specialization questions.
- **2026-08-30:** Approved BS-07 through BS-09, completing Builds & Specialization
  checkpoint C. Overall progress remains 9 of 13 specifications complete and
  9 of 12 Builds & Specialization questions.
- **2026-08-30:** Approved BS-10 through BS-12 and published canonical
  `BUILDS_AND_SPECIALIZATION.md`. Overall progress is 10 of 13 specifications
  complete; UI/UX is next.
- **2026-08-30:** Created the 12-question UI/UX interview and working record.
  UI/UX progress is 0 of 12.
- **2026-08-30:** Approved UX-01 through UX-03, completing UI/UX checkpoint A.
  Overall progress remains 10 of 13 specifications complete and 3 of 12 UI/UX
  questions.
- **2026-08-31:** Approved UX-04 through UX-06, completing UI/UX checkpoint B.
  Overall progress remains 10 of 13 specifications complete and 6 of 12 UI/UX
  questions.
- **2026-08-31:** Approved UX-07 through UX-09, completing UI/UX checkpoint C.
  Overall progress remains 10 of 13 specifications complete and 9 of 12 UI/UX
  questions.
- **2026-08-31:** Approved UX-10 through UX-12 and published canonical
  `UI_UX.md`. Overall progress is 11 of 13 specifications complete; Audio
  Presentation is next.
- **2026-08-31:** Created the 12-question Audio Presentation interview and
  working record. Audio Presentation progress is 0 of 12.
- **2026-09-01:** Approved AP-01 through AP-03, completing Audio Presentation
  checkpoint A. Overall progress remains 11 of 13 specifications complete and
  3 of 12 Audio Presentation questions.
- **2026-09-01:** Approved AP-04 through AP-06, completing Audio Presentation
  checkpoint B. Overall progress remains 11 of 13 specifications complete and
  6 of 12 Audio Presentation questions.
- **2026-09-01:** Approved AP-07 through AP-09, completing Audio Presentation
  checkpoint C. Overall progress remains 11 of 13 specifications complete and
  9 of 12 Audio Presentation questions.
- **2026-09-01:** Approved AP-10 through AP-12 and published canonical
  `AUDIO_PRESENTATION.md`. Overall progress is 12 of 13 specifications complete;
  Player Data is next.
- **2026-09-01:** Created the 12-question Player Data interview and working
  record. Player Data progress is 0 of 12.
- **2026-09-02:** Resolved Player Data PD-01 through PD-03: durable boundaries,
  the logical source-fact record, deterministic first creation, and safe
  authoritative loading. Player Data progress is 3 of 12.
- **2026-09-02:** Resolved Player Data PD-04 through PD-06: semantic mutation
  plans, atomic cross-domain transactions, frozen outcomes, idempotent replay,
  exclusive session authority, and stale-write rejection. Player Data progress
  is 6 of 12.
- **2026-09-02:** Resolved Player Data PD-07 through PD-09: critical save timing,
  lifecycle flushing, Save Unsafe outage behavior, migration evidence,
  corruption quarantine, restoration, and forward repair. Player Data progress
  is 9 of 12.
- **2026-09-02:** Resolved Player Data PD-10 through PD-12: verified receipt
  fulfillment/restoration, privacy and account lifecycle, semantic outputs,
  operational evidence, runbooks, and disaster testing. Player Data interview
  reached 12 of 12 and canonical drafting began.
- **2026-09-02:** Published and audited canonical `PLAYER_DATA.md`, completing
  all 13 specifications. Reconciled `CONTENT_AUTHORING.md` against specs 2
  through 12, closed their reconciliation registers, aligned receipt status
  vocabulary, validated local design links and document structure, and completed
  the final cross-specification ownership/consistency audit.
