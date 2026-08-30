# Bands Battle Builds and Specialization Working Record

- **Status:** Completed; reconciled into canonical specification
- **Started:** 2026-08-26
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#62-builds--specialization)
- **Interview plan:** [`BUILDS_AND_SPECIALIZATION_QUESTIONS.md`](BUILDS_AND_SPECIALIZATION_QUESTIONS.md)
- **Progression dependency:** [`PROGRESSION.md`](PROGRESSION.md)
- **Items/preset dependency:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)
- **Abilities dependency:** [`ABILITIES_AND_COOPERATIVE_ACTIONS.md`](ABILITIES_AND_COOPERATIVE_ACTIONS.md)
- **Planned canonical result:** `BUILDS_AND_SPECIALIZATION.md`

## 1. Role of this record

This document persists owner decisions while the Builds & Specialization
interview is in progress. It is not canonical until reconciled into
`BUILDS_AND_SPECIALIZATION.md`.

## 2. Inherited boundary

Builds owns option/category catalogs; one-major/three-supporting configuration;
beginner templates and advanced-editor behavior; unlock consumption; free edit/
respec; typed hook resolution; budgets/caps/stacking/incompatibility/synergy;
resolved build modifiers; validation; and option migration/retirement.

It does not own gear, base abilities, Progression awards, full spec-preset
storage/application, gameplay-domain state, item/economy transactions, raw
Rhythm judgment, persistence implementation, presentation, or final naming.

## 3. Approved inputs

- Progression editor/option unlocks and exact player progression snapshot.
- Items full spec-preset configuration, loadout validation, staging/lock state,
  exact gear modifiers, and immutable encounter snapshot.
- Combat/Survival/Positioning/Abilities typed hook schemas, event facts, caps,
  prohibited domains, and balance revision.
- Boss Encounter difficulty, roster, position/risk, musical boundary, and active
  attempt identity.
- Player Data current configurations/presets and catalog migration state.

## 4. Decision record

### Checkpoint A — Structure, definitions, and access

#### BS-01 — Configuration shape and functional categories

- **Status:** Resolved 2026-08-26.
- A valid first-release specialization configuration contains exactly one major
  rule and three supporting rules. All slots are required and the same exact
  option definition cannot occupy more than one slot.
- Every option belongs to one primary internal functional category and may carry
  cross-category tags. The four first-release categories cover offense/Momentum/
  dangerous-position play; Ward/Defend/revival support; teammate/group/Cohesion;
  and Hype/Signature/movement-triggered/hybrid utility.
- Any configuration may mix categories freely. Options are available across
  instruments unless a required typed gameplay hook is genuinely unavailable;
  validation explains that incompatibility without turning instruments into
  classes or producing a required composition.
- No option adds a required battle control or changes the one-major/three-support
  shape. Beginner templates are complete selectable configurations, not extra
  durable build-preset slots.
- The active configuration is part of the current loadout and of each of Items'
  three full spec presets. Builds does not create a competing save/apply system.
- Every system/category/slot/template/option label, including Discipline, Build
  Core, Technique, offense/protection/balanced, and Specialization itself, is an
  internal working term pending the dedicated naming/tone pass.

#### BS-02 — Option definition and typed hook contract

- **Status:** Resolved 2026-08-26.
- Each immutable option revision declares stable identity, major/supporting slot,
  primary category/tags, Progression unlock reference, trigger event/source and
  conditions, eligible owner/target/runtime states, and owning domain.
- It also declares exact hook stage, typed effect request, target/distribution/
  invalid-target fallback, duration/cooldown/once-per-source rules, power-budget
  cost/category, per-effect and global caps, stacking/synergy/incompatibility
  tags, numeric balance revision, and deterministic internal order.
- Presentation-neutral data includes a plain-language behavior and tradeoff,
  relevant exact values, trigger/inactive/fallback reason keys, semantic cue
  keys, and accessible alternatives. Player-facing names remain separate data.
- Runtime commands/facts carry attempt, snapshot, player, option revision,
  triggering source event, target, logical time, and idempotency identity.
- An option never mutates Resolve, Ward, movement, survival, Hype, group state,
  reward, or other domain data directly. It issues a typed request; the owner
  revalidates current state/caps and applies or rejects it.
- Invalid or unavailable conditions produce no effect and an identified reason.
  The system never silently substitutes another option, trigger, target, value,
  or behavior.

#### BS-03 — Beginner access, editor unlock, respec, and option unlocks

- **Status:** Resolved 2026-08-26.
- Onboarding initially offers three complete working templates: offense-focused,
  protection/support-focused, and balanced hybrid. Their names are unapproved;
  Balanced is the safe complete default when selection is skipped.
- Before the advanced editor unlocks, the player may switch among unlocked
  beginner templates freely. Templates use universal options and never bind the
  player to the current instrument or an irreversible path.
- General Progression opens the advanced editor and Items' three full spec
  presets together. General Progression and Boss Mastery unlock additional
  options. Locked options remain visible with exact earnable source and never a
  Robux-only path.
- After unlock, templates remain optional one-click starting configurations. An
  explicit preview/confirmation is required before replacing a customized
  configuration; other spec presets remain unchanged.
- Edit/template application/respec costs no resource and is allowed only in the
  hub or unlocked staging. A snapshot-affecting change clears Multiplayer Ready.
  Final lock freezes exact revisions for the attempt; later edits affect only
  future snapshots.
- Unlock state is account-wide and the same option may be referenced by multiple
  spec presets. Retirement/disablement never auto-selects a replacement; its
  detailed saved-preset/migration behavior is resolved in BS-10.

### Checkpoint B — Major/supporting behavior and shared budgets

#### BS-04 — Major-rule scope, tradeoffs, and reliability

- **Status:** Resolved 2026-08-27.
- The one major rule is a substantial bounded transformation of an existing
  decision, not an unconditional flat-stat upgrade. Each major declares a
  meaningful condition, tradeoff, redistribution, or changed use pattern.
- Legal patterns include exchanging some personal Defend value for shared
  protection, emphasizing Attack earned through dangerous play, changing a Band
  Call's approved distribution/balance, or changing an approved use of Hype/
  Signature bonus while preserving their core contracts.
- A major may split or redirect one effect only inside one declared total budget.
  It cannot copy the primary contribution route, turn one source into multiple
  full effects, or re-enter the normalized contribution pipeline.
- Committed Signature and Band Call bases, Crescendo minimum result, revival/
  recovery reliability, and other owner-domain guaranteed effects remain intact.
  A major may adjust only explicitly allowlisted bonus/distribution hooks.
- Each major resolves at most once per causal source under its frequency/
  cooldown contract. Difficulty/roster may affect legal targets/caps but cannot
  silently change its definition or create composition dependence.
- When condition/target becomes invalid, the deterministic definition fallback
  preserves ordinary baseline behavior instead of erasing earned output or
  selecting a different build rule.
- Trigger, tradeoff, transformed distribution, fallback, cap, and resulting
  effect receive clear multimodal semantic feedback. No major adds a required
  control.

#### BS-05 — Supporting-rule triggers, stacking, and recursion

- **Status:** Resolved 2026-08-27.
- Supporting rules are smaller conditional effects triggered only by approved
  normalized or semantic sources, such as contribution resolution, intent
  change, committed attack/impact, Ward threshold, completed movement/settling,
  position state, group commit/result, Hype state, or revival state.
- Each option processes an identified root event at most once. Duplicate,
  delayed, or out-of-order delivery returns the established eligibility/result.
- Build-generated effect events are non-triggering by default and cannot cause a
  new build evaluation. An explicit synergy observes the same original root
  event and snapshot rather than another option's derived output.
- Options resolve at declared pipeline stages; inside one stage, category/order
  then stable option identity determine order. Same additive-category values add
  and an independently budgeted multiplicative category applies once.
- Ordinary zero performance stays zero. A non-performance utility is legal only
  as a separately budgeted effect from a real authoritative event such as
  movement completion; it is not represented as instrument contribution.
- Invalid, suppressed, downed, disconnected, unavailable, or cooling-down rules
  produce no effect and an identified reason. Missed triggers are never banked,
  replayed, or emitted as catch-up after recovery/reconnection.
- Durations/cooldowns use declared musical boundaries where gameplay is song-
  aligned and authoritative server time only where no musical clock exists.
  Source/target attribution remains exact.

#### BS-06 — Power budget, category caps, synergies, and incompatibility

- **Status:** Resolved 2026-08-27.
- First-release build balance uses a 100-unit abstract configuration envelope:
  the major may consume at most 40 units and each supporting slot at most 20.
  Lower-cost effects leave unused capacity and cannot transfer it automatically.
- Option values and conditional maximums must fit their slot envelope in catalog
  validation. Any declared multi-option synergy is charged within the
  participating option envelopes and is initially capped around 15% of total
  build value; it cannot create a separate unbudgeted layer.
- Gear retains a separate direct-power budget. Gear, build, and ability modifiers
  in the same hook category combine together under the owning domain's shared
  category/global cap. Same-category sources add before one cap, so source order
  cannot privilege gear or builds; separately budgeted categories multiply only
  at fixed stages.
- The editor/loadout validator shows exact resolved values, contributing sources,
  unused budget, synergy cost, and any legal cap saturation before Ready. A cap
  is an explicit formula result, not a hidden runtime repair.
- Logically contradictory transformations or mutually exclusive unique tags use
  declared symmetric incompatibility. The editor blocks the whole invalid
  configuration and explains every conflict; it never drops, replaces, reorders,
  or weakens a selected rule silently.
- Any cross-category mixture that satisfies slots, unlocks, hook availability,
  and explicit incompatibility remains legal. Balance and option-usage testing
  must reject a catalog where one option/configuration becomes mandatory.

### Checkpoint C — Domain hooks and runtime resolution

#### BS-07 — Combat and Survival hook allowlist

- **Status:** Resolved 2026-08-30.
- Builds may conditionally modify post-normalized Attack conversion and legal
  Momentum handling; Defend mitigation, Ward reinforcement/restoration, and
  deterministic teammate protection; authentic revival contribution; returned
  Ward; and re-entry protection within owner-domain typed hooks and caps.
- A major may make an explicitly snapshotted, budgeted maximum-Ward or other
  substantial tradeoff. A supporting rule cannot grant unconditional maximum-
  Ward/direct-stat power; its value remains conditional and secondary to gear.
- Revival effects still require authentic exclusively redirected participant
  performance. A rule may add bounded positive progress or returned-state value,
  but cannot auto-revive, use zero/fabricated contribution, remove the opportunity
  cost, or bypass target and Activity Map constraints.
- Solo recovery rules cannot add/preserve/refund attempts, bypass or shorten the
  challenge, fabricate inputs, prevent failure/downing, or create invulnerability.
  Only explicitly allowlisted returned Ward/protection values may change inside
  global caps.
- One contribution retains one primary route. Stronger normalized performance
  cannot produce a weaker result under identical state, and ordinary zero input
  remains zero except separately budgeted event utility.
- Builds cannot affect unopened Resolve, layer sequencing, Finishing Cadence,
  committed attack target/geometry/time, all-humans-down/outcome, owner-domain
  guaranteed bases, friendly-fire prohibitions, or core Survival state authority.
- Multi-target protection declares fixed-total division or roster-aware capped
  per-recipient value with deterministic eligibility/fallback; there is no new
  manual teammate-target control.

#### BS-08 — Positioning, Hype, Signature, and group-action hooks

- **Status:** Resolved 2026-08-30.
- A build may observe settled location/risk tags, authentic dangerous-position
  performance, and completed movement/settling events to modify a later legal
  effect or create a separately budgeted visible utility request.
- It cannot change route/travel duration, movement/dash charges, recovery or
  settling, location graph/legality/occupancy, cover/hazard/telegraph geometry,
  or the universal position Attack/incoming-danger/reward ratios. Passive
  occupancy alone creates no performance or reward.
- Hype hooks may adjust allowlisted slow/fast gain values or how a Signature's
  positive bonus is allocated. They cannot add a charge, retain overflow, auto-
  fire, change state/consumption/refund timing, bypass Special routing, alter
  between-attempt reset, or weaken the guaranteed Signature base/resolution.
- Band Call hooks may affect allowlisted personal readiness rate, personal share,
  potency, or typed target/distribution within shared budgets. They cannot add a
  use, bypass readiness/pending reservation, reduce the shared lockout floor,
  invent a candidate, force participation, or remove the guaranteed base.
- Crescendo hooks may modify only the player's bounded authentic share or a
  separately budgeted legal target effect. They cannot add/select/reschedule an
  activation, change candidate/preview/eligibility, lower tier thresholds, remove
  Echo, or let one player represent an inactive roster.
- Group/revival contributions remain independently nonnegative and one-route.
  Builds never alter reward calculations/eligibility or acolyte triggers,
  recipients, functions, fixed shares, suppression, performance, or attribution.

#### BS-09 — Multi-source runtime order, roster/difficulty, and disconnect

- **Status:** Resolved 2026-08-30.
- Items' final loadout lock stores exact build configuration/options, definition/
  balance revisions, 100-unit costs, synergy and incompatibility resolution,
  gear/ability modifiers, category/global caps, and resulting typed modifier set.
- At each identified root event, the owning runtime validates the immutable
  snapshot and current source/target state, computes base conversion/effect,
  applies the major transformation at its declared stage, combines same-stage
  gear/build categories, applies supporting additions and legal position/
  encounter/target/difficulty stages, then clamps once at the owning effect cap.
- Same-category sources add before the shared cap; independently budgeted
  categories multiply once at fixed stages. Major precedes supporting additions
  at the same stage, and stable option identity is the final deterministic tie-
  breaker. No network arrival order changes the musical result.
- Committed build-derived effects retain their guarantee through downing,
  disconnect, target change, or phase change using their declared fallback.
  Uncommitted future triggers stop while the source is unavailable.
- Player-owned durations/cooldowns continue on the authoritative musical clock
  during absence and do not reset/extend on downing/reconnect. Rejoin restores the
  exact locked build state and remaining durations under Multiplayer.
- Duplicate instruments do not change build legality or value. Roster and
  difficulty influence only declared target/distribution/cap inputs at the
  definition's snapshot boundary and never rewrite a prior committed effect.
- Solo uses the same human build contracts; acolytes remain excluded. Every
  eligible/omitted trigger, input/output stage, source/target, pre/post value,
  cap/saturation, fallback, duration, and final effect retains private semantic
  attribution to its exact option and root event.

### Checkpoint D — Validation, lifecycle, outputs, and completeness

#### BS-10 — Configuration validation, preset application, and migration

- **Status:** Resolved 2026-08-30.
- The editor may retain an incomplete draft, but Apply/Ready requires exactly one
  unlocked active major, three distinct unlocked active supports, all required
  owner-domain hooks, budget/category/global-cap compliance, and no declared
  incompatibility or prohibited effect.
- Applying a template/configuration to the current loadout or saving/applying it
  inside one full spec preset is atomic. Failure changes nothing and reports all
  unlock, slot, definition, hook, budget/cap, incompatibility, and scenario issues.
- Editing a nonactive saved preset does not clear staging Ready. Any successful
  edit/application that changes the current staging loadout clears that player's
  Ready. Final lock freezes exact build and resolved modifier revisions.
- Compatible ordinary balance revisions apply only to future snapshots and
  disclose material value changes. A behavior-identical identity/revision mapping
  may migrate idempotently without choice when hooks, budget, and semantics match.
- Structural retirement/disablement/replacement makes affected current/saved
  future configurations Incomplete until the player explicitly selects a legal
  option. Retired unlock value maps to an equal-or-broader replacement unlock
  without replay/cost, but selection is never silently substituted.
- Active attempts retain their locked revision. Only an exceptional critical
  integrity/security fault may invalidate the attempt through the owner-domain
  No Contest path; it never mutates one player's build mid-song.
- Template revisions affect only later explicit template applications. Customized
  and saved configurations never auto-adopt a changed template.

#### BS-11 — Player disclosure, naming, accessibility, and privacy

- **Status:** Resolved 2026-08-30.
- Before selection, each option shows its trigger/condition, tradeoff, exact
  current resolved values, budget/caps, cooldown/duration, targets/distribution,
  fallback, synergies/incompatibilities, affected systems, and comparison with
  the currently selected option.
- Beginner presentation uses a concise role/behavior summary and expandable exact
  detail. Advanced presentation exposes full calculation/source/cap evidence.
  Neither mode hides drawbacks, clipped value, or conditional reliability.
- Runtime uses compact contextual state/trigger/cap/cooldown/fallback cues without
  moving rhythm controls, adding required buttons, or creating persistent timing
  clutter. Repeated inactive-condition messages are summarized rather than spammed.
- Critical meaning uses text/icon/shape and optional audio/haptics; color/sound
  alone is insufficient. Motion/flash/color/comfort settings may transform or
  suppress presentation without changing build state or output.
- Discipline, Build Core, Technique, Specialization, and all current category/
  template/option labels are internal keys only. The dedicated naming/tone pass
  must supply approved player-facing/localized strings before publication.
- Other players see safe role/readiness, appearance, and necessary shared-effect
  cues only. Exact options/configuration, calculated power, private recommendations,
  unlock state, migration issues, and performance/build evidence remain private.

#### BS-12 — Semantic outputs, catalog completeness, and test matrix

- **Status:** Resolved 2026-08-30.
- Builds emits authoritative facts for draft/edit/template selection, validation
  findings, save/apply/Ready invalidation, snapshot lock, unlock/availability,
  deprecation/disablement/retirement/migration, and replacement requirements.
- Runtime facts cover option eligibility/trigger/omission, major transformation,
  supporting/synergy evaluation, modifier/source combination, budget/cap
  saturation, target/fallback, duration/cooldown/expiry, and typed effect request/
  owner result.
- Facts carry exact player, current/full spec-preset/snapshot, option/catalog/
  balance revision, root event, attempt, musical/server time, source/target,
  pre/post values, reason, and idempotency identity.
- UI/Audio receive presentation-neutral facts; Items receives configuration/
  resolved modifiers; gameplay domains receive typed requests; Player Data stores
  drafts/configurations/revisions/migrations; Results receives private evidence;
  Analytics receives privacy-reviewed option/validation/trigger/cap semantics.
- Catalog verification covers every option/template/instrument hook, every legal
  one-major/three-supporting combination, representative gear/ability cap edges,
  all difficulties and rosters one through six, zero/max performance, target
  invalidation, down/disconnect/rejoin, duration/cooldown, and acolyte exclusion.
- Build definitions are global catalog data. Song packages only declare their
  normal supported roles/hooks and may not contain hidden build behavior,
  overrides, multipliers, option bans, or player-specific substitutions.

## 5. Content/configuration reconciliation register

- Catalogs require stable four-category identities, exact one-major/three-
  supporting slot shape, cross-instrument availability/hook requirements,
  beginner-template definitions, and explicit internal-versus-player-name fields.
- Option definitions require every identity/unlock/trigger/hook/effect/target/
  fallback/budget/cap/stacking/incompatibility/order/cue/description/idempotency
  field in BS-02. Validation rejects direct domain mutation or silent fallback.
- Progression/template configuration requires the three complete beginner
  templates, editor/full-spec-preset gate, visible earnable unlock sources, free
  hub/staging edit, Ready invalidation, confirmation before overwrite, and no
  paid-only functional option.
- Major/supporting catalogs require declared tradeoff/transformation, baseline/
  guaranteed-effect preservation, one-source/one-route behavior, root-event and
  derived-event policy, deterministic stages/order, cooldown/duration/fallback,
  and multimodal feedback.
- Balance data requires the 100-unit/40-major/20-supporting envelopes, option and
  synergy costs, initial 15% synergy cap, category/global hook caps shared with
  gear/abilities, explicit incompatibilities, exact cap preview, and mandatory-
  build/combination test evidence.
- Consumer hook schemas require every legal Combat/Survival/Positioning/Hype/
  Signature/Call/Crescendo input, stage, typed output, base/zero/monotonicity/
  target/cap/fallback rule, and the complete prohibited-authority list in BS-07/
  BS-08.
- Runtime verification requires immutable multi-source snapshot/order, root-event
  processing, same-category/global caps, committed/uncommitted and duration/
  disconnect behavior, roster/difficulty snapshots, acolyte exclusion, and exact
  semantic attribution from BS-09.
- Lifecycle/migration configuration requires draft versus applicable state,
  atomic full-spec integration, active-loadout Ready invalidation, balance/
  behavior-identical/structural change classes, equal-or-broader unlock mapping,
  no silent option/template substitution, and active-snapshot preservation.
- Publication requires complete beginner/advanced disclosure, exact value/cap/
  tradeoff evidence, multimodal cue keys, strict internal-versus-player-facing
  naming separation, privacy allowlists, semantic facts, and the exhaustive/
  representative combination matrix in BS-12.

## 6. Open handoffs

- `PROGRESSION.md` owns editor/option unlocks and general/mastery award sources.
- `ITEMS_AND_EQUIPMENT.md` owns three full spec presets, validation/application,
  immutable loadout snapshots, and gear modifiers.
- `COMBAT.md`, `BOSS_ENCOUNTERS.md`, and
  `ABILITIES_AND_COOPERATIVE_ACTIONS.md` own legal runtime state/effect domains.
- UI/Audio present semantic build facts; Player Data stores configurations/
  revisions/migration; Analytics receives privacy-reviewed evidence.
- The later naming/tone pass owns every player-facing system/category/slot/
  option/template name.

## 7. Change log

- **2026-08-26:** Created the working record. Progress is 0 of 12 questions.
- **2026-08-26:** Approved BS-01 through BS-03. Configuration/preset shape,
  universal categories, complete typed option definitions, beginner templates,
  editor unlock, and free respec are resolved; progress is 3 of 12 questions.
- **2026-08-27:** Approved BS-04 through BS-06. Major transformations,
  nonrecursive supporting triggers, deterministic order, 100-unit slot envelopes,
  budgeted synergies, shared caps, and incompatibility are resolved; progress is
  6 of 12 questions.
- **2026-08-30:** Approved BS-07 through BS-09. Combat/Survival/Positioning/
  ability/group hook allowlists and prohibitions, immutable cross-source order,
  roster/difficulty/disconnect behavior, and attribution are resolved; progress
  is 9 of 12 questions.
- **2026-08-30:** Approved BS-10 through BS-12. Atomic validation/full-spec
  application, option/template lifecycle and migration, exact accessible/private
  disclosure, naming gate, semantic outputs, and completeness tests are resolved.
  All 12 questions were reconciled into the canonical specification.
