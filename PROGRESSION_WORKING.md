# Bands Battle Progression Working Record

- **Status:** Complete decision record; 12 of 12 questions reconciled
- **Started:** 2026-08-22
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#65-progression)
- **Interview plan:** [`PROGRESSION_QUESTIONS.md`](PROGRESSION_QUESTIONS.md)
- **Encounter dependency:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Canonical result:** [`PROGRESSION.md`](PROGRESSION.md)

## 1. Role of this record

This document persists owner decisions while the Progression interview is in
progress. It is not canonical until reconciled into `PROGRESSION.md`.

## 2. Inherited boundary

Progression owns campaign nodes/destinations, first clears, fragments, campaign
tiers and hub-restoration state; general progression and system/option unlocks;
per-boss Hard access; boss mastery; personal-best semantics; unlock eligibility;
recommended power; old-item uplift eligibility; and the rule that progression
does not expire or regress.

It does not own raw gameplay evidence, canonical meaningful-participation rules,
reward/economy calculations or transactions, item ownership/mutation,
persistence implementation, catalog content, or presentation.

## 3. Approved inputs

- Immutable Boss Encounter result/outcome identity and exact content/balance
  revisions.
- Rewards/Multiplayer canonical meaningful-participation eligibility and
  transaction results where required.
- Rhythm/Combat/Positioning/Abilities personal result facts for record
  evaluation.
- Versioned campaign, general-unlock, mastery-milestone, difficulty, item-tier,
  and recommendation catalogs.
- Current durable player progression snapshot from Player Data.

## 4. Decision record

### Checkpoint A — Campaign access and difficulty

#### PG-01 — Campaign credit and first-clear idempotency

- **Status:** Approved.
- Campaign credit requires an immutable Boss Encounter **Victory** plus the
  canonical meaningful-participation eligibility supplied by Rewards/Multiplayer.
- A player who is downed or disconnected at encounter end still receives credit
  when prior participation preserves that eligibility. Current survival or
  connection state does not overwrite the canonical decision.
- Easy, Normal, or Hard may grant the per-boss first clear. The stable mutation
  key combines player and boss campaign-node identity, with the causal encounter
  result retained for audit.
- The first accepted event records completion once. Duplicate, retried, late,
  or out-of-order copies return the already-established state and cannot grant a
  second fragment, repeat unlock animation as a new fact, or overwrite the
  original completion evidence.
- Repeat victories do not recreate first-clear mutations but remain ordinary
  inputs for mastery, records, rewards, and replay.
- Defeat and Invalid / No Contest never create campaign completion, a fragment,
  destination access, or a Normal-victory Hard unlock by inference.

#### PG-02 — Campaign graph, tiers, and hub restoration

- **Status:** Approved.
- A versioned campaign catalog contains stable node/destination/tier identities,
  boss association, prerequisite expression, fragment identity, hub-restoration
  consequence, and player-facing content references.
- Launch may use a simple three-boss linear order, but runtime semantics are a
  prerequisite graph so later branches or parallel destinations require catalog
  data rather than a progression rewrite.
- A first clear commits one atomic monotonic bundle:
  1. mark the boss node complete;
  2. record its restored fragment;
  3. evaluate and unlock every newly satisfied destination and shard tier; and
  4. advance the highest newly satisfied hub-restoration state.
- Failure partway through persistence cannot expose a fragment without its node
  or a destination without its prerequisites; Player Data later defines the
  atomic storage/retry mechanism.
- Replaying a completed node never repeats its campaign/hub mutations. Hub and UI
  consume the snapshot and may animate a newly committed transition once, but
  presentation is not authority.
- Campaign progress is monotonic. Catalog revision/migration may map, merge, or
  replace retired nodes while preserving earned equivalence; it cannot remove a
  fragment, relock accessible content, lower the hub state, or require replay to
  recover previously earned access.

#### PG-03 — Difficulty unlocks and recommendations

- **Status:** Approved.
- Easy and Normal are accessible for every boss whose campaign node/destination
  is unlocked. Difficulty does not create a separate campaign story route.
- Hard eligibility is a permanent per-player/per-boss fact granted only by a
  valid meaningful-participation Normal Victory. Easy first clear advances the
  campaign but does not satisfy this prerequisite.
- Multiplayer receives individual boss/difficulty eligibility for every member.
  A leader, invitation, party, or matchmaking result cannot bypass another
  player's locked campaign node or Hard requirement.
- Hard never gates the next campaign destination, Shattered Song fragment,
  essential combat item, required build option, or baseline story. It may
  improve normal reward efficiency and grant bounded mastery cosmetics, titles,
  or badges.
- Private recommendations may use recent personal records, timing trends,
  survival, and encounter success. They suggest moving up or down but never
  alter selected difficulty, hide an available choice, identify the suggestion
  publicly, or gate matchmaking/rewards.

### Checkpoint B — General player progression

#### PG-04 — General progress earning

- **Status:** Approved.
- Progression consumes an immutable completed-attempt/result identity plus the
  canonical meaningful-participation decision; it does not recalculate gameplay
  eligibility from raw scores.
- A valid Victory receives a substantially larger base general-progress award.
  A Defeat receives a capped practice award scaled inside the eligible
  participation range. Exact curves, boss-tier factors, and difficulty bonuses
  remain balance data.
- Every attempt applies at most one general-progress event. Repeating a boss or
  legitimately failing multiple times remains useful without a daily cap or
  hidden diminishing-return timer, but each attempt's cap and meaningful-
  participation requirement prevent idle farming.
- General failure progress unlocks breadth and limited baseline milestones. It
  cannot grant campaign fragments, boss-specific materials, signature items,
  first-clear choices, or other victory-owned advancement.
- Practice/onboarding modules have checkpoint/completion/skip facts but do not
  produce repeatable account progress.
- Invalid / No Contest is neither Victory nor Defeat and does not enter either
  curve automatically. Rewards may issue one explicitly identified compensation
  grant for confirmed participation, initially targeted at no worse than the
  ordinary failure-progress value; duplicates remain idempotent.
- Calculation uses versioned balance data and deterministic final rounding while
  retaining pre-rounded evidence.

#### PG-05 — System and option unlock catalog

- **Status:** Approved.
- Available from first arrival are accessibility and comfort settings,
  calibration, replayable practice, core rhythm/combat/movement controls, one
  usable starter configuration, and the first campaign boss on Easy/Normal.
- Completing or explicitly skipping the checkpointed practice unlocks public
  matchmaking. Practice completion is an onboarding gate, not a general-level
  grind or minimum-grade test.
- General progression may reveal advanced specialization/build editing, the
  complete preset surface, broader option catalogs, and wider equipment and
  consumable choices. Campaign owns destinations, per-boss Normal victory owns
  Hard, and Boss Mastery owns its milestone eligibility.
- Core play, accessibility, safety, settings, calibration, manual controls,
  campaign-critical actions, and the ability to equip a valid starter loadout
  are never hidden behind account progress.
- Each unlock definition has stable identity, catalog revision, prerequisites,
  resulting eligibility/system state, and one-time notification metadata.
- When a revised catalog introduces an unlock whose prerequisites the player
  already satisfies, it grants retroactively and notifies once. A revision may
  not relock earned access; retired unlocks migrate to equivalent or broader
  eligibility.

#### PG-06 — Power limits, choice, respec, and spec presets

- **Status:** Approved.
- Direct permanent statistical power from general progression is limited to
  small, front-loaded, fixed milestones. The initial design ceiling is roughly
  10% of a complete first-tier loadout's power budget, subject to balance tests
  but never an endless account-stat ladder.
- Gear and current-tier successful boss progression remain the main direct-power
  sources. Repeated loss/general progress cannot reproduce victory-owned items,
  materials, upgrades, or current-tier access.
- Progress totals and unlocked options are never spent. A player freely changes
  specialization choices outside active combat; no respec fee, refund flow, or
  irreversible branch exists.
- When the advanced build/spec surface unlocks, all three baseline saved preset
  slots unlock together. The player does not grind each quality-of-life slot.
- A preset is a player-named complete combat-configuration reference containing:
  - selected playable role/instrument;
  - Instrument, Ward Core, and Resonator equipment references;
  - Signature Special and Band Call references;
  - two prepared consumable-type references; and
  - major/supporting specialization choices.
- Applying a preset performs one validated switch in the hub or pre-battle
  staging before final loadout lock. It never swaps during an active attempt or
  after multiplayer staging locks the loadout.
- Presets reference owned/unlocked definitions and existing item instances; they
  never duplicate gear, create consumable quantity, bypass progression, or spend
  unavailable resources. A missing/retired/incompatible reference makes the
  preset visibly incomplete until the player repairs it or an approved migration
  supplies an equivalent.
- Exact equip/quantity validation belongs to Items & Equipment; specialization
  compatibility and terminology belong to Builds & Specialization; Player Data
  owns atomic persistence/application.
- Retired progression options preserve earned equivalence through explicit
  migration and cannot erase a player's ability to form a previously legal
  functional build.

### Checkpoint C — Boss mastery and personal records

#### PG-07 — Mastery earning and rank completion

- **Status:** Approved.
- Each boss has one stable mastery-track identity shared across every playable
  instrument/role. Switching musical identity never forks or restarts mastery.
- The first release targets approximately ten visible finite ranks per boss.
  Each rank has a versioned cumulative requirement; the complete track and
  upcoming milestones are visible rather than hidden behind an endless level.
- A completed eligible attempt produces at most one mastery grant for its causal
  player/boss result. The grant combines outcome base, canonical meaningful-
  participation factor, and any bounded declared difficulty factor.
- Victory is substantially more mastery-efficient than Defeat. Failure remains
  useful practice progress but cannot become the fastest route through
  intentional loss or AFK behavior.
- Higher difficulty may provide a modest bounded efficiency bonus; Easy and
  Normal remain legitimate completion routes, and no rank requires Hard.
- One grant may cross several ranks. XP overflow carries through subsequent rank
  thresholds atomically until the finite track cap.
- At completion, further ordinary mastery XP is discarded rather than stored as
  invisible prestige or converted into uncapped power. Replay continues through
  personal records, normal boss rewards, deterministic economy paths, and
  enjoyment.
- Invalid / No Contest is not a Victory or Defeat mastery source. An explicit
  compensation event may grant the configured value once when the later Rewards
  policy authorizes it.

#### PG-08 — Mastery milestone eligibility

- **Status:** Approved.
- A versioned milestone catalog maps stable boss/rank identities to one or more
  eligibility results such as lore/archive entries, cosmetics, titles, recipes,
  specialization sidegrades, or deterministic reward grants.
- Crossing a rank automatically evaluates every newly satisfied milestone. It
  never requires the player to discover and press a separate claim button merely
  to preserve an earned unlock.
- Progression records the milestone-eligibility fact idempotently. Pure access
  facts become available immediately; any item/currency/cosmetic transaction is
  an identified request to Rewards/Items with the milestone as its causal key.
- A failed downstream grant remains Pending and retries safely. It never marks
  Delivered early, loses the reward, or produces duplicates on retry.
- New catalog milestones whose requirements are already satisfied grant
  retroactively and notify once. A revision cannot revoke a delivered milestone
  or reduce earned rank.
- Combat-relevant specialization rewards are bounded options/sidegrades, not
  mandatory current-tier power. Signature combat items, boss-specific materials,
  campaign fragments, and first-clear progression remain victory-owned even when
  mastery also advances through failure.
- Mastery completion may have a major deterministic cosmetic/title/lore
  milestone but does not unlock endless statistical ranks.

#### PG-09 — Personal-best records

- **Status:** Approved.
- Every record uses stable player, boss, instrument/role, difficulty, mode, and
  record-compatibility identity. Solo and co-op are separate. A category whose
  value materially changes with roster size also includes the configured roster
  size/band grouping.
- Categories update independently. Initial families include rhythm execution,
  participation/hold performance, personal combat contribution, Ward/survival,
  position/risk, cooperative help, and overall personal performance; the later
  Results specification defines the final concise catalog.
- Each category declares its ordered comparison tuple, minimum meaningful-
  participation/coverage requirement, valid attempt/outcome states, and
  supporting evidence. A run may improve one category without replacing any
  other record.
- Both Victory and Defeat can establish a record when the category evidence is
  valid. Disconnect absence lowers coverage normally and may disqualify only the
  categories whose minimum was not met; prior accepted play is not erased.
- Invalid / No Contest may update only categories explicitly marked trustworthy
  by the encounter result. A global clock/scoring invalidation marks affected
  categories ineligible rather than guessing.
- An exact tie under the category's full ordered tuple preserves the earlier
  achievement as the record. The tied run may remain in attempt history without
  replacing it.
- Calibration, Hold Assist, and other accessibility settings do not create a
  lower-value class, public tag, or separate leaderboard. Private technical
  evidence remains outside the record key.
- Content revisions declare a record-compatibility identity. A compatible
  revision may continue the set; an incompatible chart/scoring change archives
  the old record and starts a new current set without deleting history.
- Records are for comparison with the player's own history. This system exposes
  no global, party, friend, or public damage ranking.

### Checkpoint D — Power tiers, ordering, and outputs

#### PG-10 — Recommended power

- **Status:** Approved.
- Recommended power is private advisory context for one selected complete spec
  preset against one boss/difficulty revision. It is never a universal social
  score or required entry threshold.
- The versioned recommendation profile evaluates multiple relevant dimensions,
  initially Attack/offense capacity, maximum/current-tier Ward and defensive
  conversion, and useful ability/support/utility capacity. It recognizes valid
  specializations rather than ranking every build by damage.
- The player sees a concise readiness band such as **Prepared**,
  **Challenging**, or **Far Below**, the major limiting dimension when helpful,
  and uncertainty when a catalog/profile is stale. Exact internal estimates do
  not need to become a large public number.
- A valid under-equipped player may enter regardless of band. Multiplayer,
  parties, matchmaking, public profiles, and other players cannot see or gate on
  the recommendation.
- Accessibility settings, calibration, Hold Assist, difficulty suggestions,
  private performance history, age, and spending do not reduce readiness.
- Improvement routes identify earned options such as an available boss,
  deterministic recipe, upgrade, compatible owned item, or build adjustment.
  They do not open the store, compare paid equipment, or imply purchase is
  required.

#### PG-11 — Old-item uplift eligibility

- **Status:** Approved.
- Uplift eligibility requires the player to have reached a later campaign/item
  tier, own an eligible non-consumable combat item, and unlock the item's stable
  uplift recipe through its declared campaign/mastery/recipe source.
- One transaction advances exactly one tier and cannot exceed the player's
  current unlocked tier or that item's normal versioned tier cap. Multiple-tier
  catch-up requires the corresponding sequential recipes/transactions.
- The uplift preserves stable item lineage, fixed trait/behavior, and unlocked
  appearance while using the target tier's ordinary stat budget. It starts at
  that tier's base upgrade rank; subsequent ranks use normal upgrade rules.
- Cost uses mostly current-tier resources plus material associated with the
  item's original boss. Thus replay supplies identity-specific material while
  current bosses remain necessary for current power.
- Uplift never occurs automatically and cannot turn an old item above current-
  tier standards. It supports favorite identities and sidegrades rather than a
  cheaper replacement for progression against new bosses.
- Paid/stat-bearing equivalents obey identical eligibility, material cost,
  target tier, base rank, and cap. Purchase never supplies automatic future
  scaling.
- Progression emits eligibility. Rewards/Economy validates/spends resources, and
  Items & Equipment performs the atomic item-instance mutation.

#### PG-12 — Transaction order, semantic outputs, and permanence

- **Status:** Approved.
- Every mutation request has stable player, source-event, progression-event,
  catalog revision, and causal result identity. One causal event can commit once.
- Semantic processing order is:
  1. validate source outcome/eligibility and exact relevant revisions;
  2. apply general/mastery/other totals using deterministic balance data;
  3. determine every newly crossed rank, level, prerequisite, and record change;
  4. atomically commit the complete campaign, fragment, destination/tier, hub,
     difficulty, system, option, mastery, milestone-eligibility, preset-access,
     and personal-record bundle; and
  5. issue idempotent downstream Reward/Item grant or mutation requests.
- Concurrent/retried/out-of-order events either commit against the latest valid
  snapshot or retry. They cannot duplicate totals, expose half an unlock bundle,
  regress a record, or lose a pending downstream milestone grant.
- Durable source facts include first clears, fragments, completed nodes,
  general/mastery totals, delivered/grandfathered unlocks, milestone states,
  difficulty access, practice/public access, record history, and migration
  evidence. Current catalog-derived availability may be recomputed from those
  facts while grandfathered access remains explicit.
- Progress never expires or decreases because of inactivity, season, catalog
  rotation, failed attempt, lower difficulty, item change, or respec. There are
  no daily streaks, energy, expiring ranks, or exclusive rotating progression.
- A migration maps retired identities to equal-or-broader earned equivalence and
  records its source/target revision. It never demands replay to recover access.
- Progression emits attributed facts for first clear/duplicate, fragment,
  destination/tier/hub state, difficulty, general totals/levels, unlocks,
  preset-access, mastery/ranks, milestone eligibility/pending/delivered,
  personal records/archive, recommendation, uplift eligibility, migration, and
  rejected/duplicate processing.
- Player Data persists these semantics. Results, Hub, Onboarding, Items, Builds,
  Rewards, Commerce, Multiplayer, UI, and Analytics consume them without taking
  ownership.

## 5. Open handoffs

- `REWARDS_AND_ECONOMY.md` owns meaningful-participation eligibility, quantities,
  granted resources/items, deterministic reward paths, and compensation.
- `ITEMS_AND_EQUIPMENT.md` owns durable item instances and uplift mutation.
- `BUILDS_AND_SPECIALIZATION.md` owns use of unlocked build options/presets and
  respec behavior inside the progression constraints. Together with Items, it
  must implement the approved three full combat-configuration presets.
- `MULTIPLAYER.md` owns encounter/difficulty consent and entry validation.
- `PLAYER_DATA.md` owns persistence, concurrency, migration, and recovery of the
  durable facts specified here.
- Hub, Onboarding, UI, Results, and Analytics consume semantic unlock/progress
  facts without owning them.

## 6. Content Authoring reconciliation register

- Approved content revisions need a stable record-compatibility identity and an
  explicit compatible/incompatible relationship to the preceding revision for
  each personal-best category affected by chart/scoring changes.
- Campaign, mastery, unlock, and recommendation catalogs reference stable boss
  and encounter identities but remain Progression/system data rather than song-
  authored timeline data.

## 7. Change log

- **2026-08-22:** Created the working record. Progress is 0 of 12 questions.
- **2026-08-22:** Approved PG-01 through PG-03. Progress is 3 of 12 questions.
  Established campaign credit, prerequisite/hub state, and difficulty access.
- **2026-08-24:** Approved PG-04 through PG-06. Progress is 6 of 12 questions.
  Established general earning/unlocks, bounded power, free respec, and three
  complete quick-switch spec presets.
- **2026-08-24:** Approved PG-07 through PG-09. Progress is 9 of 12 questions.
  Established finite boss mastery, milestone grants, and personal-best records.
- **2026-08-24:** Approved PG-10 through PG-12 and reconciled all twelve
  decisions into canonical `PROGRESSION.md`.
