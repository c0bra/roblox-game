# Bands Battle Builds and Specialization Specification Questions

- **Status:** Completed; 12 of 12 questions resolved
- **Started:** 2026-08-26
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#62-builds--specialization)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Progression dependency:** [`PROGRESSION.md`](PROGRESSION.md)
- **Items/preset dependency:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)
- **Abilities dependency:** [`ABILITIES_AND_COOPERATIVE_ACTIONS.md`](ABILITIES_AND_COOPERATIVE_ACTIONS.md)
- **Working record:** [`BUILDS_AND_SPECIALIZATION_WORKING.md`](BUILDS_AND_SPECIALIZATION_WORKING.md)
- **Planned canonical result:** `BUILDS_AND_SPECIALIZATION.md`

## 1. Interview method

This interview uses four checkpoints of three questions. It inherits settled
one-major/three-supporting structure, four universal functional categories,
cross-instrument mixing, full spec presets, free out-of-combat respec, typed
post-score modifiers, and all Rhythm/movement/reward fairness prohibitions.

Answers are persisted after each checkpoint. When all twelve questions are
resolved, they are reconciled into canonical `BUILDS_AND_SPECIALIZATION.md`.
All player-facing names remain explicitly unapproved pending the later naming
and tone pass.

## 2. Fixed inherited decisions

- Every instrument draws from the same four functional categories: offense/
  Momentum/risk; Ward/Defend/revival; teammate/group/Cohesion; and Hype/
  Signature/movement-triggered/hybrid utility.
- A specialization configuration equips one major behavior-changing rule and
  three smaller supporting rules. Categories may mix freely; an instrument is
  not a class and no pure role/composition is required.
- New players use clear beginner role presets before General Progression opens
  the advanced editor. General Progression and Boss Mastery unlock options.
- The three approved full spec presets each store the complete specialization
  configuration alongside role, gear, abilities, and consumables. Builds must
  not create a competing second preset system.
- Respec is free in the hub or unlocked staging. Active encounter snapshots are
  immutable and use exact definition/balance revisions.
- Gear carries most direct power. Specialization emphasizes conditional behavior,
  strengths/tradeoffs, group interactions, and hybrids under shared budgets/caps.
- Rules may alter consequences after normalized scoring or respond to identified
  gameplay events. They cannot change charts, notes, judgments, timing windows,
  calibration, Hold Assist, controls, song speed, or playable material.
- Rules cannot alter movement travel/recovery/dash charges, telegraph fairness,
  position risk/reward ratios, recovery/revival attempt counts, rewards/
  eligibility, matchmaking, item slots, ability slots, consumable slots, or paid
  value.
- Rules cannot fabricate performance, copy one contribution into multiple full
  routes, create negative teammate output, recurse, or let acolytes earn/receive
  build effects in the first release.
- Every effect uses typed owner-domain hooks, deterministic order, power-budget
  cost, caps/stacking, target/fallback, exact attribution, and idempotency.
- Supporting rules do not add required combat buttons. Long-term depth expands
  the option/interaction catalog rather than the control set.
- `Discipline`, `Build Core`, and `Technique` are internal placeholders only and
  must not ship as player-facing names.

## 3. Question plan

### Checkpoint A — Structure, definitions, and access

#### BS-01 — Configuration shape and functional categories

- **Status:** Resolved 2026-08-26.

- **Decision needed:** What exactly constitutes one valid specialization while
  preserving universal mixing and avoiding a second preset system?
- **Must resolve:** One major/three supporting, category identities, cross-
  category/instrument availability, duplicate selections, empty slots, beginner
  templates, relationship to full spec presets, no class/composition, no new
  controls, and internal naming.

#### BS-02 — Option definition and typed hook contract

- **Status:** Resolved 2026-08-26.

- **Decision needed:** Which fields make each major/supporting rule deterministic
  and safe for every consuming system?
- **Must resolve:** Stable identity/revision, slot/category/tags, unlock,
  trigger/condition/source, hook stage/domain, effect/target/fallback, budget,
  cap/stacking/duration, incompatibility, cues/description, prohibited domains,
  attribution, and idempotency.

#### BS-03 — Beginner access, editor unlock, respec, and option unlocks

- **Status:** Resolved 2026-08-26.

- **Decision needed:** How do players move from readable templates to free
  advanced experimentation without irreversible or invalid choices?
- **Must resolve:** Starter templates, default selection, editor gate, General/
  Mastery unlock sources, preview of locked options, free hub/staging respec,
  ready reset, active lock, three spec presets, option ownership, retirement,
  and no purchase gate.

### Checkpoint B — Major/supporting behavior and shared budgets

#### BS-04 — Major-rule scope, tradeoffs, and reliability

- **Status:** Resolved 2026-08-27.

- **Decision needed:** How powerful may the one major behavior change be without
  becoming a direct class/required option or breaking reliable base effects?
- **Must resolve:** Legal transformation examples, base preservation, condition/
  tradeoff, one primary route, trigger frequency, guaranteed/optional effects,
  targets, difficulty/roster, failure/fallback, forbidden authority, and cues.

#### BS-05 — Supporting-rule triggers, stacking, and recursion

- **Status:** Resolved 2026-08-27.

- **Decision needed:** How do three smaller rules combine around events without
  copying contribution or producing infinite trigger chains?
- **Must resolve:** Legal trigger sources, same-option duplicates, independent
  conditions, once-per-source processing, derived-event eligibility, internal
  order, additive/multiplicative limits, cooldown/duration, suppressed/invalid
  state, zero input, and attribution.

#### BS-06 — Power budget, category caps, synergies, and incompatibility

- **Status:** Resolved 2026-08-27.

- **Decision needed:** How does a configuration prove total value remains within
  budget across gear, abilities, and specialization interactions?
- **Must resolve:** Major/support costs, configuration budget, hook-category and
  global caps, synergy cost/cap, gear/ability combination, over-budget behavior,
  explicit incompatibility, duplicate tags, deterministic resolution, balance
  revision, validator, and no mandatory build.

### Checkpoint C — Domain hooks and runtime resolution

#### BS-07 — Combat and Survival hook allowlist

- **Status:** Resolved 2026-08-30.

- **Decision needed:** Which post-score Attack, Defend, Ward, support, downing,
  revival, and recovery interactions are legal?
- **Must resolve:** Intent/contribution stages, Resolve/Momentum, mitigation/
  reinforcement/restoration, teammate protection, downed/revival facts, bounded
  recovery received, target/distribution, one route, monotonicity, zero input,
  protected base, and prohibited attempts/invulnerability.

#### BS-08 — Positioning, Hype, Signature, and group-action hooks

- **Status:** Resolved 2026-08-30.

- **Decision needed:** Which reactions to position/movement completion and
  ability/group state are legal without altering core timing/resources?
- **Must resolve:** Settled position/risk tags, movement-complete trigger, no
  travel/recovery/risk-ratio change, Hype gain/use, Signature base/bonus, Call
  readiness/potency/lockout/use, Crescendo shares/tiers, group targets, no extra
  routes/charges, and acolyte exclusion.

#### BS-09 — Multi-source runtime order, roster/difficulty, and disconnect

- **Status:** Resolved 2026-08-30.

- **Decision needed:** How does the resolved build behave predictably with gear,
  abilities, multiple players, downing, and connection changes?
- **Must resolve:** Immutable snapshot, modifier stages/order, cap application,
  source/target identities, roster snapshots, duplicate instruments, difficulty,
  down/disconnect/return, committed effects, no retroactive mutation, solo,
  attribution, and semantic evidence.

### Checkpoint D — Validation, lifecycle, outputs, and completeness

#### BS-10 — Configuration validation, preset application, and migration

- **Status:** Resolved 2026-08-30.

- **Decision needed:** How does an edited or saved configuration become a valid
  immutable loadout and survive catalog changes safely?
- **Must resolve:** Unlock/slot/category, budget/cap/incompatibility, definition
  availability, spec-preset validation/application, staging compatibility, Ready
  reset, snapshot lock, missing/disabled/retired options, migration, grandfather/
  compensation, no silent substitution, and idempotency.

#### BS-11 — Player disclosure, naming, accessibility, and privacy

- **Status:** Resolved 2026-08-30.

- **Decision needed:** What must players understand about a rule before selecting
  it and when it triggers, while keeping placeholder names out of shipping UI?
- **Must resolve:** Exact behavior/tradeoff/caps, comparison, trigger feedback,
  inactive/fallback explanation, beginner language, advanced details, multimodal
  cues, naming-pass handoff, preset names, party-visible summary, private build/
  recommendations, and no misleading percentages.

#### BS-12 — Semantic outputs, catalog completeness, and test matrix

- **Status:** Resolved 2026-08-30.

- **Decision needed:** Which facts/catalog fields/validators/tests make the
  system implementation-ready without inventing build behavior?
- **Must resolve:** Edit/validate/apply/lock/migrate and runtime trigger/effect/
  cap facts, identities/revisions/times, consumer outputs, persistence/Analytics,
  option/category/unlock catalogs, prohibited-hook validator, combinations/
  roster/difficulty/device/accessibility tests, Content Authoring register,
  deferred values/names, and completion audit.

## 4. Completion criteria

`BUILDS_AND_SPECIALIZATION.md` is complete only when:

- BS-01 through BS-12 are resolved;
- one major/three supporting options mix across universal categories without a
  second preset system or required composition;
- every option and synergy uses typed bounded post-score/event hooks;
- major/supporting rules preserve base reliability, one-route, monotonicity,
  zero-input, movement, recovery-count, reward, and control invariants;
- budget/cap/incompatibility validation covers gear/ability combinations and
  prevents recursive or mandatory builds;
- beginner templates, advanced-editor unlock, free respec, locks, and retirement
  behavior are deterministic and non-coercive;
- placeholder names cannot ship and every behavior is disclosed accessibly; and
- catalog/output/test requirements leave no implementation-agent design choice.

## 5. Change log

- **2026-08-26:** Created the concise 12-question Builds & Specialization
  interview from the approved GDD and canonical dependencies.
- **2026-08-26:** Approved BS-01 through BS-03, completing structure,
  definitions, and access checkpoint A. Progress is 3 of 12 questions.
- **2026-08-27:** Approved BS-04 through BS-06, completing behavior and power-
  budget checkpoint B. Progress is 6 of 12 questions.
- **2026-08-30:** Approved BS-07 through BS-09, completing domain hooks and
  runtime resolution checkpoint C. Progress is 9 of 12 questions.
- **2026-08-30:** Approved BS-10 through BS-12, completing validation, lifecycle,
  disclosure, and outputs checkpoint D. All 12 questions are resolved and the
  canonical specification was published.
