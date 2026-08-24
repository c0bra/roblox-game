# Bands Battle Progression Specification Questions

- **Status:** Interview complete; 12 of 12 questions resolved and reconciled
- **Started:** 2026-08-22
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#65-progression)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Encounter outcome authority:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Working record:** [`PROGRESSION_WORKING.md`](PROGRESSION_WORKING.md)
- **Canonical result:** [`PROGRESSION.md`](PROGRESSION.md)

## 1. Interview method

This interview uses four checkpoints of three questions. It keeps Campaign,
General Progression, and Boss Mastery separate and does not decide reward
quantities, item transactions, persistence implementation, or UI layout.

Answers are persisted after each checkpoint. When all twelve questions are
resolved, they are reconciled into canonical `PROGRESSION.md`.

## 2. Fixed inherited decisions

- Long-term advancement has three distinct tracks: Campaign, General Player
  Progression, and per-boss Mastery. Final player-facing names remain open.
- A player's first valid victory against a boss on any difficulty restores its
  fragment and advances the campaign when canonical meaningful-participation
  eligibility is met. Being downed at the end does not by itself remove credit.
- Easy and Normal start available for every unlocked boss. A valid Normal
  victory unlocks Hard separately for that boss.
- General progression comes from meaningful victories and failed attempts,
  unlocks systems/options, and provides only limited direct statistical power.
- Respec is free outside combat and no progression choice is irreversible.
- Each boss starts with approximately ten visible mastery ranks shared across
  instruments. Victory/failure both grant participation-scaled mastery, with
  victory substantially more efficient.
- Personal bests remain separate by instrument and difficulty even though boss
  mastery is shared.
- Boss mastery may make lore, cosmetics, recipes, specialization options,
  titles, and deterministic milestones eligible. Campaign fragments, boss
  materials, signature combat items, and first-clear advancement require
  victory through their owning systems.
- Current-tier bosses remain the best source of current-tier power. Older items
  may later be raised using mostly current-tier resources plus original-boss
  material.
- Recommended power is advisory, never an entry gate.
- No required daily streak, energy system, expiring progression, exclusive
  rotating reward, or endless mastery power ladder is permitted.
- Boss Encounters may produce Invalid / No Contest. That outcome is neither a
  campaign victory nor a gameplay defeat; Rewards owns any compensation.

## 3. Question plan

### Checkpoint A — Campaign access and difficulty

#### PG-01 — Campaign credit and first-clear idempotency [Resolved]

- **Decision needed:** What exact evidence grants a fragment/first-clear and how
  is duplicate or late processing made harmless?
- **Must resolve:** Valid outcome, meaningful participation, downed/disconnected
  players, any-difficulty credit, first-clear identity, repeat victory,
  duplicate/out-of-order events, Defeat, and No Contest.
- **Owner decision:** A valid Victory plus canonical meaningful-participation
  eligibility grants the player's per-boss first clear on any difficulty. A
  downed or disconnected end state does not remove credit when eligibility is
  retained. The player/boss first-clear key is idempotent, so duplicate, late,
  or out-of-order processing cannot duplicate a fragment or unlock. Repeat
  victories remain eligible for their normal mastery/reward paths. Defeat and
  Invalid / No Contest never create campaign credit.

#### PG-02 — Campaign graph, tiers, and hub restoration [Resolved]

- **Decision needed:** How does one first clear unlock destinations and durable
  world restoration without hard-coding all future campaigns?
- **Must resolve:** Node/prerequisite model, initial state, next destinations,
  shard tiers, fragment/hub state, atomic transitions, monotonicity, replay,
  catalog revisions, and presentation handoff.
- **Owner decision:** Campaign access is a versioned stable-identity prerequisite
  graph, even when launch content forms a simple linear path. A first clear
  atomically records its fragment, completes the boss node, unlocks every newly
  satisfied destination/tier, and advances a monotonic hub-restoration state.
  Replay cannot repeat those mutations. Hub/UI only present the authoritative
  progression snapshot. Catalog changes use explicit migration and cannot remove
  or downgrade earned nodes, fragments, access, or restoration.

#### PG-03 — Difficulty unlocks and recommendations [Resolved]

- **Decision needed:** Which difficulties may each player enter and how can the
  game advise without coercing or bypassing access?
- **Must resolve:** Easy/Normal availability, per-boss Hard unlock, Normal credit,
  party/matchmaking checks, campaign relationship, recommendation evidence,
  privacy, no auto-change, and Hard-only reward limits.
- **Owner decision:** Easy and Normal are available for each campaign-unlocked
  boss. A meaningful Normal Victory unlocks Hard for that player/boss only; Easy
  campaign credit does not. Every party member must independently satisfy access
  and cannot be carried into locked content. Hard is never required for campaign
  advancement or essential combat power, though it may support bounded mastery
  cosmetics/titles/badges and improved ordinary rewards. Recommendations are
  private evidence-based suggestions that never auto-select, lock, or gate
  matchmaking.

### Checkpoint B — General player progression

#### PG-04 — General progress earning [Resolved]

- **Decision needed:** Which completed attempts earn broad progress and how do
  victory, failure, participation, disconnect, practice, and No Contest differ?
- **Must resolve:** Canonical eligibility input, source event, relative efficiency,
  caps, anti-idle rules, retry farming, practice/onboarding, invalid attempts,
  rounding, and attribution.
- **Owner decision:** Every completed attempt with canonical meaningful
  participation may grant broad progress. Victory has a substantially larger
  base award; Defeat grants a capped participation-scaled practice award.
  Legitimate repeated failures remain useful, but idle/noneligible play earns
  nothing and cannot substitute for victory-owned gear, fragments, or boss
  materials. Practice/onboarding grants its own completion state, not farmable
  account progress. Invalid / No Contest contributes only through an explicit
  compensation event, initially no worse than the normal failure award for
  confirmed participation.

#### PG-05 — System and option unlock catalog [Resolved]

- **Decision needed:** What belongs in the general unlock sequence and what must
  remain available from the start?
- **Must resolve:** Tutorial gates, public matchmaking, systems, saved builds,
  specialization/editor, equipment breadth, consumables, accessibility/settings,
  unlock prerequisites, retroactive catalog changes, and player communication.
- **Owner decision:** Accessibility, settings, calibration, practice, core
  controls, starter equipment, and the first boss's Easy/Normal access begin
  available. Completing or explicitly skipping practice unlocks public
  matchmaking. General progression reveals advanced build editing and broader
  specialization, equipment, consumable, and other complexity rather than
  withholding accessibility or core play. Stable versioned prerequisites
  evaluate retroactively, notify once, and never relock an earned unlock after
  catalog change.

#### PG-06 — Power limits, choice, and respec [Resolved]

- **Decision needed:** How much direct power may general progression grant and
  how are reversible option choices handled?
- **Must resolve:** Direct-stat ceiling, gear versus account power, selectable
  options, free out-of-combat respec, preset-slot unlocks, encounter lock,
  refunds/retired options, and no loss-farming replacement for victory.
- **Owner decision:** General progression's direct permanent stats are small,
  front-loaded milestones with an initial ceiling around 10% of a complete
  first-tier loadout's power budget. Gear and successful boss progress remain
  primary power. Unlock currency/points are never spent or lost; configuration
  may change freely outside active combat. When advanced builds open, all three
  baseline player-defined spec presets open together. Each preset references a
  complete combat configuration (role/instrument, power gear, Signature Special,
  Band Call, prepared consumable types, and specialization choices) for one-step
  switching in the hub or unlocked pre-battle staging before final loadout lock.
  It never duplicates items/consumables or bypasses ownership/access. Retired
  options migrate to equivalent unlocked choices.

### Checkpoint C — Boss mastery and personal records

#### PG-07 — Mastery earning and rank completion [Resolved]

- **Decision needed:** How is participation-scaled mastery applied across
  victory/failure, instruments, difficulties, and post-completion replay?
- **Must resolve:** Per-boss identity, approximately ten ranks, XP curve,
  victory efficiency, difficulty factor, meaningful participation, shared
  instruments, cap/completion, overflow, and no endless power.
- **Owner decision:** Every boss has a finite visible track, initially around ten
  ranks, shared across all instruments. Eligible Victory and Defeat grants use
  canonical participation evidence; Victory is substantially more efficient and
  higher difficulties may add only a bounded bonus. One grant may cross multiple
  ranks and carries overflow between them until the finite cap. Completion
  discards further mastery XP rather than creating hidden/endless levels; replay
  continues through personal records and ordinary rewards. Invalid / No Contest
  grants mastery only through an explicit compensation event.

#### PG-08 — Mastery milestone eligibility [Resolved]

- **Decision needed:** What does progression unlock at mastery milestones, and
  where does reward/item transaction ownership begin?
- **Must resolve:** Milestone catalog, deterministic claims/grants, lore,
  cosmetics, recipes, specialization, titles, combat options, victory-only
  exclusions, retroactive changes, duplicates, and completion rewards.
- **Owner decision:** Rank crossing automatically and idempotently satisfies
  stable cataloged milestone eligibility for lore, cosmetics, titles, recipes,
  bounded specialization sidegrades, or deterministic rewards. Progression owns
  the eligibility fact; Rewards/Items owns any resulting grant transaction. A
  failed grant stays pending and retryable without duplicate delivery. Catalog
  additions grant retroactively to already-qualified players and never revoke
  prior unlocks. Fragments, boss materials, signature combat items, and
  first-clear progress remain on their victory-owned paths; mastery never
  becomes an essential-power ladder.

#### PG-09 — Personal-best records [Resolved]

- **Decision needed:** Which attempt facts can become a personal best and how are
  incompatible contexts segmented?
- **Must resolve:** Boss/instrument/difficulty keys, solo/co-op, roster size,
  score categories, valid outcomes, failures, disconnect coverage, tie-breaking,
  calibration/accessibility privacy, version changes, and self-comparison only.
- **Owner decision:** Comparable records are segmented by boss, instrument,
  difficulty, solo/co-op context, and roster grouping for categories materially
  changed by roster. Rhythm, combat, survival, positional, and other record
  categories compare independently using category-specific ordered metrics.
  Valid Victories and Defeats may update when meaningful participation and
  coverage meet the category requirement. No Contest updates only explicitly
  preserved trustworthy categories. Exact ties preserve the earlier record.
  Calibration/accessibility choices are private and create no penalty or public
  label. Incompatible chart/scoring revisions archive prior records and begin a
  new comparable set rather than overwriting history. Records are self-comparison
  only.

### Checkpoint D — Power tiers, ordering, and outputs

#### PG-10 — Recommended power [Resolved]

- **Decision needed:** How is advisory power calculated/presented without
  becoming a gear-score gate or social exclusion tool?
- **Must resolve:** Inputs, boss/difficulty recommendation, confidence/bands,
  under-equipped access, party privacy, stale catalogs, no matchmaking gate,
  and improvement routes.
- **Owner decision:** Recommended power is a private readiness band such as
  Prepared, Challenging, or Far Below, never a mandatory score. It evaluates the
  selected complete spec preset against versioned boss/difficulty expectations
  across offense, Ward/defense, and relevant utility rather than privileging
  damage. Accessibility and private performance never lower the band. Other
  players do not see it and Multiplayer cannot gate on it. Improvement routes
  point to earned bosses/upgrades/recipes/build changes without store prompts.

#### PG-11 — Old-item uplift eligibility [Resolved]

- **Decision needed:** When may an older item be raised to a later campaign tier
  while current bosses remain the primary power source?
- **Must resolve:** Tier access, eligible items, recipe unlock, current/original
  resource split, resulting cap, repeat uplift, paid-equivalent parity,
  sidegrade intent, and ownership handoffs.
- **Owner decision:** Reaching a later campaign tier may unlock one-tier-at-a-
  time uplift recipes for eligible owned combat items. Uplift preserves item
  identity, trait, and appearance; moves it only to the player's current tier;
  resets it to that tier's base upgrade rank; and never exceeds its normal cap.
  It consumes mostly current-tier resources plus original-boss material, so
  current bosses remain necessary. No item auto-scales or jumps tiers. Paid
  equivalents use identical recipes/costs. Progression owns eligibility; Economy
  and Items own the transaction/mutation.

#### PG-12 — Transaction order, semantic outputs, and permanence [Resolved]

- **Decision needed:** How are concurrent/duplicate progression events applied
  atomically and exposed to Player Data, Results, Hub, UI, and Analytics?
- **Must resolve:** Stable event/idempotency keys, prerequisite order, atomic
  unlock bundle, conflict/retry, exact revisions, derived versus stored state,
  no expiration/regression, migration, event catalog, and completion audit.
- **Owner decision:** Each stable progression event processes once: validate
  source/revisions, apply totals, determine crossings, atomically commit all
  campaign/difficulty/system/mastery/milestone/record changes, then issue
  idempotent downstream grant requests. Concurrent/retried delivery cannot
  duplicate or partially expose a bundle. Durable source facts and grandfathered
  unlocks are stored; current catalog-derived eligibility is recomputed from
  them. Earned progress never expires/regresses and migrations preserve
  equivalence. Every change emits attributed semantic facts for Player Data,
  Results, Hub, UI, and Analytics.

## 4. Completion criteria

`PROGRESSION.md` is complete only when:

- PG-01 through PG-12 are resolved;
- Campaign, General Progression, Boss Mastery, and personal records cannot grant
  or overwrite one another's authority;
- first-clear and unlock transitions are atomic, monotonic, and idempotent;
- failure remains useful without replacing victory-owned advancement;
- instrument changes do not fragment boss mastery;
- recommended power remains advisory;
- old-content uplift cannot bypass current-tier bosses;
- Invalid / No Contest never becomes campaign credit by inference; and
- outputs are complete for persistence and player-facing consumers.

## 5. Change log

- **2026-08-22:** Created the concise 12-question plan from the approved GDD,
  Systems Map, Boss Encounter outcome contract, and existing progression rules.
- **2026-08-22:** Resolved PG-01 through PG-03, establishing first-clear credit,
  campaign/hub transitions, and per-player difficulty access.
- **2026-08-24:** Resolved PG-04 through PG-06, establishing broad progress,
  unlock sequencing, bounded account power, free respec, and three full combat-
  configuration presets.
- **2026-08-24:** Resolved PG-07 through PG-09, establishing finite shared boss
  mastery, durable milestone eligibility, and compatible personal-best records.
- **2026-08-24:** Resolved PG-10 through PG-12 and reconciled all twelve answers
  into canonical `PROGRESSION.md`.
