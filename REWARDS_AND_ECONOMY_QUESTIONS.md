# Bands Battle Rewards and Economy Specification Questions

- **Status:** Completed; 12 of 12 questions resolved
- **Started:** 2026-08-25
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#63-rewards-loot--economy)
- **Included system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#64-commerce)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Progression dependency:** [`PROGRESSION.md`](PROGRESSION.md)
- **Items dependency:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)
- **Multiplayer dependency:** [`MULTIPLAYER.md`](MULTIPLAYER.md)
- **Working record:** [`REWARDS_AND_ECONOMY_WORKING.md`](REWARDS_AND_ECONOMY_WORKING.md)
- **Planned canonical result:** `REWARDS_AND_ECONOMY.md`

## 1. Interview method

This interview uses four checkpoints of three questions. It inherits settled
outcomes, performance/contribution evidence, banked Risk, group contribution,
progression awards, item ownership/mutations, no-public-ranking, and voluntary
store boundaries. It focuses on canonical participation eligibility, reward
assembly, deterministic acquisition, economy transactions, and safe Commerce.

Answers are persisted after each checkpoint. When all twelve questions are
resolved, they are reconciled into canonical `REWARDS_AND_ECONOMY.md` including
the Commerce section.

## 2. Fixed inherited decisions

- Rewards are personal and never divided from a shared pool. Party size does not
  change the core drop table or make solo/full-band the required farm strategy.
- Victory grants general resource, boss material, boss mastery, and a chance at
  a complete fixed item/cosmetic; first victory separately triggers campaign/
  fragment progression.
- Meaningfully participated failure grants modest general resource and mastery,
  but no boss material, signature boss item, or first-clear campaign reward.
- Easy, Normal, and Hard share the boss combat-item pool. Higher difficulties may
  improve quantities, complete-drop chance, or starting rank, not exclusive
  essential functionality.
- Personal performance, banked positional Risk, and Cohesion may add bounded
  positive bonuses. Weak play never turns the result into all or nothing.
- Every victory advances a visible deterministic boss-item path initially aimed
  at roughly four to six clears for one chosen standard combat item.
- Items have fixed stats/traits and about three guaranteed upgrade ranks per
  tier. Craft/upgrade previews are exact and never fail, destroy, downgrade, or
  impose timers after confirmation.
- Duplicate items can be voluntarily salvaged into useful boss material while
  their appearance entitlement remains. Cosmetics do not duplicate into power.
- Only general resource and boss-specific material families exist in the first
  release. There is no trading, repair/death fee, paid luck, punitive respec,
  energy, daily cap, random affix/reroll, or failure-is-best farming loop.
- Paid access appears only in a voluntarily opened post-onboarding hub store,
  never combat, downing/recovery, defeat, Results, or immediate Retry.
- First-release paid products are direct deterministic cosmetics and permanent
  equipment only. Every paid power item has an exact earnable same-tier
  functional equivalent and no paid-only trait/stat advantage.
- No loot boxes/random bundles, resources/materials, consumables, boosts,
  recovery, luck/drop modifiers, content access, subscriptions/passes,
  convenience, or progression skips are sold.

## 3. Question plan

### Checkpoint A — Eligibility and reward assembly

#### RE-01 — Meaningful-participation eligibility

- **Status:** Resolved 2026-08-26.

- **Decision needed:** Which evidence makes one player eligible for encounter
  rewards without punishing late network failure or rewarding idling?
- **Must resolve:** Outcome-independent evidence, minimum activity/coverage,
  genuine attempts, downed/help/support credit, connection absence, voluntary
  leave, Departed, Inactive/repeat AFK, late failure, solo, accessibility,
  eligibility tiers, privacy, and deterministic reason.

#### RE-02 — Victory, failure, and No Contest base grants

- **Status:** Resolved 2026-08-26.

- **Decision needed:** What base reward families does each immutable outcome
  authorize and which progression facts remain owned elsewhere?
- **Must resolve:** Personal grants/no split, Victory general/boss/drop/mastery,
  first clear, Failure general/mastery prohibitions, early departure, Defeat
  reasons, No Contest compensation/refunds, consumables, progression handoff,
  result freeze, and no reroll.

#### RE-03 — Difficulty, performance, Risk, and Cohesion bonuses

- **Status:** Resolved 2026-08-26.

- **Decision needed:** How do positive modifiers improve rewards without creating
  punitive cliffs, public competition, or unsafe farming incentives?
- **Must resolve:** Allowed dimensions, additive/multiplicative order, per-source/
  total caps, banked versus unbanked Risk, performance normalization, Cohesion,
  difficulty, active roster, solo parity, accessibility neutrality, disconnect
  coverage, floor/ceiling, and attribution.

### Checkpoint B — Resources, loot, and deterministic acquisition

#### RE-04 — Resource definitions, sources, sinks, and tier relevance

- **Status:** Resolved 2026-08-26.

- **Decision needed:** How do the two earned resource families remain readable,
  useful, and non-exploitable across campaign tiers?
- **Must resolve:** General/boss-specific identity, grants, sources, sinks,
  quantities/caps, old/current-tier use, uplift relationship, conversion,
  expiration, negative balances, trading/gifting, faucets/sinks, and naming.

#### RE-05 — Boss pools, complete drops, guarantees, and milestones

- **Status:** Resolved 2026-08-26.

- **Decision needed:** How are random complete rewards selected from fixed pools
  while preserving progression guarantees and duplicate safety?
- **Must resolve:** Pool revision, eligible item/cosmetic types, shared difficulty
  pool, chance/rank modifiers, fixed identity, first-clear starter choice,
  signature rules, mastery/deterministic milestones, owned cosmetic handling,
  duplicate power items, disclosure, selection seed, and pity versus path.

#### RE-06 — Deterministic path, chosen item, and duplicate salvage

- **Status:** Resolved 2026-08-26.

- **Decision needed:** How does every victory advance a visible chosen-item path
  and how do duplicate items retain value?
- **Must resolve:** Boss material progress, four-to-six-clear target, chosen item,
  changing choice, existing material, standard versus signature items, craft
  readiness, random drop interaction, duplicate identity, appearance retention,
  salvage output, equipped/locked protection, confirmation, and no forced
  deletion.

### Checkpoint C — Earned economy transactions

#### RE-07 — Crafting contracts and exact-result confirmation

- **Status:** Resolved 2026-08-26.

- **Decision needed:** Which first-release recipes exist and how does crafting
  atomically create a chosen fixed result?
- **Must resolve:** Recipe identity/revision/unlock, ingredients, boss/tier,
  ownership checks, exact preview, confirmation, atomic spend/grant, duplicate
  result, failure/retry, no timers/chance/reroll, batch behavior, and UI handoff.

#### RE-08 — Upgrades, uplift, salvage, consumables, and ordinary shop

- **Status:** Resolved 2026-08-26.

- **Decision needed:** How do remaining earned-resource sinks mutate ownership
  without downgrade, coercion, or hidden loss?
- **Must resolve:** Three-rank baseline, costs, exact stat preview, max rank,
  current-tier uplift recipe, rank reset, salvage selection/output, appearance,
  equipped/preset references, consumable pricing/stack cap, normal shop, refunds,
  and no repair/respec/death fees.

#### RE-09 — Transaction orchestration, concurrency, and economy integrity

- **Status:** Resolved 2026-08-26.

- **Decision needed:** What makes every grant/spend/mutation safe through retry,
  stale data, duplicate delivery, and service failure?
- **Must resolve:** Transaction identity, preconditions/snapshot, ledger order,
  atomicity, idempotency, concurrent devices/servers, insufficient balance,
  partial failure, reservation, compensation, No Contest, negative/overflow,
  audit, exploit response, and Player Data handoff.

### Checkpoint D — Commerce and outputs

#### RE-10 — Store eligibility, voluntary surfaces, and prompt limits

- **Status:** Resolved 2026-08-26.

- **Decision needed:** When may the Robux store appear and how does declining
  keep it from becoming pressure?
- **Must resolve:** Onboarding/encounter gate, hub/menu entry, voluntary open,
  one-time new-stock notice, close/decline suppression, prohibited gameplay/
  results/retry surfaces, age/account/platform eligibility, unavailable state,
  truthful seasonal dates, no urgency tricks, and UI separation.

#### RE-11 — Paid catalog, earnable equivalence, and power review

- **Status:** Resolved 2026-08-26.

- **Decision needed:** Which direct products are legal and how is exact same-tier
  functional equivalence proven?
- **Must resolve:** Cosmetic/equipment only, deterministic SKU, current unlocked
  tier, no auto-scaling, exact stat/trait/budget, earnable route, exclusive
  appearance, prohibited categories, duplicate protection, balance tests,
  automated validator, human review, future category review, and publication.

#### RE-12 — Purchase receipt, restoration, semantic outputs, and audit

- **Status:** Resolved 2026-08-26.

- **Decision needed:** How does a Marketplace purchase become one durable grant
  without double-charge/grant, loss, or ambiguity?
- **Must resolve:** Quote/confirmation, exact price/product/result, receipt
  authority, pending/retry/recovery, idempotent grant, already-owned result,
  cancellation/failure, Robux refund boundary, history/restoration, safe data,
  Reward/Economy/Commerce events, consumers, compliance/operational testing,
  and completion audit.

## 4. Completion criteria

`REWARDS_AND_ECONOMY.md` is complete only when:

- RE-01 through RE-12 are resolved;
- canonical participation eligibility distinguishes meaningful play, absence,
  voluntary departure, and repeated inactivity without using accuracy as blame;
- every outcome authorizes an exact personal base grant and No Contest path;
- bonuses are bounded, private, positive, and accessibility/roster safe;
- random complete drops cannot replace the visible deterministic path;
- every craft/upgrade/uplift/salvage/consumable transaction is exact, atomic,
  idempotent, and loss-safe;
- the two-resource economy has no prohibited coercive/exploitative system;
- every paid power product proves its same-tier earnable equivalent and all
  prohibited product/prompt categories are rejected; and
- receipt recovery and all semantic outputs are complete, privacy-safe, and
  operationally auditable.

## 5. Change log

- **2026-08-25:** Created the concise 12-question Rewards/Economy/Commerce
  interview from the approved GDD and canonical dependencies.
- **2026-08-26:** Approved RE-01 through RE-03, completing eligibility and reward
  assembly checkpoint A. Progress is 3 of 12 questions.
- **2026-08-26:** Approved RE-04 through RE-06, completing resources, loot, and
  deterministic acquisition checkpoint B. Progress is 6 of 12 questions.
- **2026-08-26:** Approved RE-07 through RE-09, completing earned-economy
  transactions checkpoint C. Progress is 9 of 12 questions.
- **2026-08-26:** Approved RE-10 through RE-12, completing Commerce and outputs
  checkpoint D. All 12 questions are resolved and the canonical specification
  was published.
