# Bands Battle Rewards and Economy Working Record

- **Status:** Completed; reconciled into canonical specification
- **Started:** 2026-08-25
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#63-rewards-loot--economy)
- **Included system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#64-commerce)
- **Interview plan:** [`REWARDS_AND_ECONOMY_QUESTIONS.md`](REWARDS_AND_ECONOMY_QUESTIONS.md)
- **Progression dependency:** [`PROGRESSION.md`](PROGRESSION.md)
- **Items dependency:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)
- **Multiplayer dependency:** [`MULTIPLAYER.md`](MULTIPLAYER.md)
- **Planned canonical result:** `REWARDS_AND_ECONOMY.md`

## 1. Role of this record

This document persists owner decisions while the Rewards, Economy, and Commerce
interview is in progress. It is not canonical until reconciled into
`REWARDS_AND_ECONOMY.md`.

## 2. Inherited boundary

Rewards/Economy owns meaningful-participation eligibility, per-player reward
calculation, currencies/materials, deterministic/random loot paths, crafting,
upgrade/uplift/salvage/ordinary-shop/consumable transactions, transaction
orchestration, and economy prohibitions.

Commerce owns legal paid catalog/surfaces, store eligibility, exact-equivalence
validation, quote/confirmation/receipt/grant recovery, duplicate protection, and
paid-category/prompt prohibitions.

They do not own encounter outcome/raw evidence, item/progression semantics,
Marketplace billing, durable persistence implementation, UI layout, or Results
presentation.

## 3. Approved inputs

- Immutable Boss Encounter outcome/reason/configuration and No Contest facts.
- Identified Rhythm/Combat/Positioning/Abilities contribution, coverage, banked
  Risk, Cohesion, and performance evidence.
- Multiplayer deployment/active-roster, connection absence, departure,
  inactivity/resume, and participation facts.
- Progression first-clear/tier/mastery/unlock/uplift eligibility and mutations.
- Items catalogs, ownership/mutation preconditions, entitlements, stacks,
  snapshots, and duplicate results.
- Player Data balances/ledger/history and Roblox Marketplace eligibility,
  product, price, receipt, and platform-result facts.

## 4. Decision record

### Checkpoint A — Eligibility and reward assembly

#### RE-01 — Meaningful-participation eligibility

- **Status:** Resolved 2026-08-26.
- Rewards computes one private canonical participation decision from identified
  attempt, role/chart, Rhythm coverage, genuine input attempts, cooperative/
  revival performance, connection, survival, departure, and AFK evidence. It
  does not use public comparison or accuracy as an eligibility test.
- Full eligibility initially requires genuine attempts in at least 20% of the
  role's total playable scoring groups and at least three groups when that many
  exist. It also requires attempts in at least half of the eligible groups that
  occurred while the player was connected, Active, unsuspended, and able to
  perform. Threshold math is deterministic and revisioned.
- Authentic early/late/Miss input, Attack/Defend/Special material, revival, Band
  Call, and Crescendo chart performance counts as an attempt. Rests, chart
  dropouts, downed time, connection absence, Rhythm suspension, and unavailable
  group material do not become ignored opportunities.
- A downed player or involuntarily network-Departed player may retain full
  eligibility from completed evidence. A voluntary mid-attempt Leave or permanent
  Inactive state after the one Resume forfeits ordinary outcome rewards even if
  some earlier evidence was meaningful.
- Solo and multiplayer use role/chart-normalized thresholds. Approved
  accessibility, calibration, Hold Assist, device, difficulty, and instrument
  choice never reduce eligibility or identify a lower-value category.
- The player receives `Eligible`, `Insufficient Participation`, or `Forfeited`
  plus a private evidence/reason summary. Other players receive no eligibility
  label or underlying participation evidence.

#### RE-02 — Victory, failure, and No Contest base grants

- **Status:** Resolved 2026-08-26.
- Each eligible player's frozen immutable Victory independently authorizes a
  general-resource grant, boss-material grant, one complete boss-pool drop roll,
  visible deterministic-path progress, and identified eligibility facts for
  Progression's campaign/general/mastery processing. Nothing is divided from or
  reduced by a shared party pool.
- Each eligible Defeat authorizes modest general resource and failure/general/
  mastery Progression facts. It grants no boss material, complete-item/cosmetic
  roll, signature boss reward, first-clear/fragment/destination, or Hard unlock.
- A downed or involuntarily disconnected/Departed player at the result receives
  the same outcome base when canonical participation remains Eligible. An
  Insufficient or Forfeited player receives no ordinary outcome grant.
- Invalid / No Contest grants no outcome loot, boss material, drop roll,
  deterministic victory progress, campaign credit, or ordinary first-clear. All
  causal committed consumables refund and unused reservations release under
  Items. Confirmed meaningful participants receive one failure-equivalent
  general-resource and Progression compensation event.
- A player without meaningful participation in No Contest receives only causal
  system-fault refunds. No compensation path may grant more than the configured
  failure-equivalent ceiling or imply gameplay Victory/Defeat.
- The result, eligibility, reward/balance/catalog revisions, and random selection
  seed freeze before calculation. Processing and reconnection/retry are
  idempotent; they return the established transaction/result instead of rerolling.

#### RE-03 — Difficulty, performance, Risk, and Cohesion bonuses

- **Status:** Resolved 2026-08-26.
- Reward assembly applies the outcome base, then the selected difficulty profile,
  then independently calculated nonnegative performance, Risk, and Cohesion or
  solo-equivalent breadth packets. It clamps the combined packet and rounds once
  using the published revision.
- Initial hypotheses cap personal normalized performance near +15%, individually
  banked positional Risk near +25%, Cohesion near +15%, and all non-difficulty
  bonuses together near +40%. Difficulty values and exact caps remain tuning,
  but no source may bypass its own or the combined cap.
- A catalog explicitly states which modifiers may affect general/boss-material
  quantity, complete-drop chance, or starting upgrade rank. They never change
  the boss pool, deterministic reward identity, campaign/Hard eligibility,
  guaranteed first-clear choice, essential-function availability, or number of
  independent roll opportunities.
- Performance is normalized for instrument, difficulty, authentic material, and
  trustworthy coverage. Base eligibility prevents poor accuracy from erasing the
  base. Network absence supplies no performance bonus and cannot improve its
  factor.
- Only Positioning's already banked personal Risk contributes; unbanked value
  lost by movement/downing supplies none. Risk never comes from passive
  occupancy, acolytes, another player, items, or paid products.
- Multiplayer Cohesion measures broad successful human participation using the
  difficulty profile and active/eligible roster evidence. Weak/inactive players
  add little or no positive share but never subtract value or create blame.
  Solo uses an equivalent human-only breadth factor under the same cap; acolytes
  supply none.
- Accessibility, calibration, Hold Assist, input device, role duplication, and
  party size do not lower reward floors/caps or create separate reward tables.
  Results privately attribute each applied bonus without publishing rankings.

### Checkpoint B — Resources, loot, and deterministic acquisition

#### RE-04 — Resource definitions, sources, sinks, and tier relevance

- **Status:** Resolved 2026-08-26.
- First release has exactly two broad earned resource families: one permanent
  account-wide general-resource balance and one permanent material balance per
  boss. Player-facing names remain open for the naming/tone pass.
- General resource pays for ordinary item upgrades, basic crafting, consumables,
  and the ordinary earned shop. Boss material pays for that boss's combat items,
  traits, cosmetics, upgrades, and the original-boss portion of an approved later
  item-uplift recipe.
- Resource definitions and ledger entries carry stable identity, boss where
  applicable, tier/source revision, exact integer quantity, cause/transaction,
  and pre/post balance. No transaction may create a negative balance.
- Resources do not expire, decay, earn interest, convert into Robux, convert
  between bosses, or exchange between the two families. First release has no
  trading, gifting, marketplace, mail transfer, or player-facing balance cap.
- Internal numeric storage has a safe maximum; a grant that cannot fit fails
  atomically and alerts rather than discarding value. There is no daily earning
  cap, energy limit, hidden diminishing return, debt, or forced sink.
- Older bosses retain unique item/cosmetic/mastery/recipe uses, but their fixed
  encounter-tier reward profiles do not scale upward with the player's current
  tier. Current-tier encounters remain the efficient source of current-tier
  power, and uplift requires mostly current-tier inputs under Progression.

#### RE-05 — Boss pools, complete drops, guarantees, and milestones

- **Status:** Resolved 2026-08-26.
- Each boss has a versioned, visible reward pool containing fixed-definition
  power items and cosmetics plus eligibility, weight/chance, starting-rank,
  duplicate, first-clear, signature, and milestone metadata.
- Easy, Normal, and Hard share the same functional combat-item pool. Difficulty
  may change the disclosed overall complete-drop chance or starting rank but
  cannot add an essential exclusive entry, change item function, or add an
  independent roll opportunity.
- Each eligible Victory performs exactly one complete-drop roll using the frozen
  encounter/player/pool revision and deterministic selection seed. The pool,
  overall chance, and per-entry chances/weights are disclosed. Retry/reconnect
  returns the same result.
- A successful power-item selection creates one normal fixed equal-function
  instance even if already owned. A selected already-owned cosmetic returns an
  Items Duplicate result and automatically converts through a disclosed
  boss-material transaction instead of creating another entitlement.
- Authored first-clear choices and mastery guarantees are separate deterministic
  awards in the catalog and do not replace or consume the ordinary roll. They
  grant once through stable milestone/first-clear keys.
- A Signature combat item cannot remain random-only. Every Signature declares a
  visible victory-based recipe or finite mastery/clear milestone. It may be more
  demanding than the standard-item path but cannot require Hard or spending.
- There is no hidden pity counter or undisclosed dynamic chance. The permanent
  boss-material path provides explicit bad-luck protection, and random success
  never resets or consumes it.

#### RE-06 — Deterministic path, chosen item, and duplicate salvage

- **Status:** Resolved 2026-08-26.
- Boss material is the authoritative deterministic-path balance; there is no
  separate hidden pity/progress currency. Each Victory's grant advances every
  affordable recipe from that boss equally until the player chooses a spend.
- In the hub, the player may pin one standard boss item as a private goal. The
  surface shows exact recipe/material cost, current balance, progress, and an
  estimate based on the selected difficulty's disclosed base reward.
- Initial economy tuning targets roughly four to six baseline eligible victories
  to afford one chosen standard combat item. Changing or clearing the pin is free
  and never reserves, spends, or resets material.
- Receiving the pinned item randomly does not spend/reset material or auto-craft.
  The player keeps both and may freely pin another goal. Standard recipe identity
  and Signature-specific deterministic paths remain distinct and visible.
- A duplicate power item remains a separate equal-function instance until the
  owner voluntarily salvages it. Salvage preview initially targets roughly
  25–35% of equivalent crafting cost, varies only by disclosed tier/rank rules,
  and is always lower than recreating the same item.
- An active locked-snapshot instance cannot be salvaged. A hub-equipped or
  preset-referenced instance requires explicit confirmation listing affected
  configurations; Items then leaves those references Missing/Incomplete without
  substitution.
- Salvage never removes the permanently unlocked appearance entitlement. There
  is no forced deletion, automatic power-item salvage, inventory-pressure sink,
  or hidden material loss.

### Checkpoint C — Earned economy transactions

#### RE-07 — Crafting contracts and exact-result confirmation

- **Status:** Resolved 2026-08-26.
- First-release crafting recipes may create fixed standard boss combat items,
  explicitly authored deterministic Signature items, and selected boss cosmetics
  or traits. There is no procedural item, affix, quality roll, or reroll recipe.
- Each immutable recipe revision declares stable identity, unlock/prerequisite,
  boss and campaign tier, one exact item/entitlement definition and starting
  rank, exact general/boss-material inputs, allowed quantity, and availability/
  replacement relationship.
- The hub workshop shows exact output identity, tier/rank, stat/trait/appearance,
  owned count or entitlement state, pinned-goal relationship, ingredient balances,
  and final cost before an explicit confirmation.
- A power-item duplicate is allowed because it creates a distinct normal
  instance, but the preview warns when the player already owns an equal-function
  item. An already-owned cosmetic recipe is blocked as Owned and spends nothing.
- One confirmation creates one output. The ingredient spend and Items grant
  commit atomically under one transaction identity. Any validation/service
  failure creates neither spend nor grant and can retry the same identity.
- Crafting has no chance, completion timer, queue, hidden fee, batch ambiguity,
  failure roll, destruction, downgrade, or output variation. A stale recipe/
  balance refreshes the quote instead of accepting changed terms.

#### RE-08 — Upgrades, uplift, salvage, consumables, and ordinary shop

- **Status:** Resolved 2026-08-26.
- A first-release item initially supports roughly three guaranteed ranks inside
  its current tier. Each rank definition declares exact resulting stat/trait
  state and general-plus-relevant-boss-material cost. Max rank is explicit.
- Upgrade preview shows the selected unique instance, current/next rank and
  values, trait continuity, exact cost, affected equipped/preset references, and
  resulting state. Commit preserves the instance identity and references and
  can never fail randomly, lower power, or destroy the item.
- Progression-authorized uplift moves one eligible old item up exactly one tier
  using mostly current-tier inputs plus original-boss material. It preserves
  identity, trait, appearance entitlement, and references while visibly resetting
  rank to the new tier's base as already defined by Items.
- An active locked-snapshot item cannot upgrade, uplift, or salvage. Other
  referenced-item mutations follow Items' exact warning, automatic reference
  preservation for same-instance mutation, or Missing/Incomplete behavior for
  salvage.
- The ordinary earned shop uses a stable non-random catalog of low-cost
  consumables and selected basic items/cosmetics priced in general resource.
  Consumable bundles/stack caps and exact post-purchase quantity are disclosed;
  over-cap purchase rejects without spending.
- Earned purchases and resources never expire. There are no repair, death,
  respec, cancellation, storage, convenience, rush, or completion-timer fees.
  A valid confirmed earned transaction is final; failure commits nothing, and
  later retirement/migration uses explicit compensation rather than hidden loss.

#### RE-09 — Transaction orchestration, concurrency, and economy integrity

- **Status:** Resolved 2026-08-26.
- Every reward, craft, upgrade, uplift, salvage, earned-shop purchase, duplicate
  conversion, refund, and compensation receives stable player/action/source
  transaction identity, exact catalog/balance revision, expected state version,
  and preconditions.
- The authority serializes relevant mutations per player and validates current
  balance, ownership, lock/reference state, eligibility, definition/recipe,
  storage bound, and prior causal transaction before commit.
- One atomic commit records the immutable ledger entry, resource pre/post
  balances, and exact Items/Progression mutation or causal handoff. Retrying the
  same identity returns the committed or Pending result; it never repeats a roll,
  spend, grant, removal, mutation, refund, or compensation.
- Concurrent or stale requests fail closed with a current-state refresh.
  Insufficient balance, overflow, invalid ownership, changed recipe, or failed
  precondition creates no partial transaction and never clamps silently or makes
  a balance negative.
- If a cross-service persistence result is uncertain, the durable journal remains
  `Pending` and recovers to the exact intended grant or exact rollback. Results
  communicates Pending clearly and gameplay Retry need not wait. A client cannot
  choose a different result or key to reroll.
- Committed history is append-only. A correction uses a linked, reasoned,
  idempotent compensation transaction instead of editing/deleting the original.
  No Contest refunds reference exact attempt/use identities and cannot exceed
  causal committed quantities.
- Player Data persists balances, ledger/journal, Items/Progression results, and
  recovery state. Audit/Analytics receive privacy-reviewed transaction semantics;
  invalid/exploit attempts cannot mint, duplicate, transfer, or destroy value.

### Checkpoint D — Commerce and outputs

#### RE-10 — Store eligibility, voluntary surfaces, and prompt limits

- **Status:** Resolved 2026-08-26.
- The paid store becomes eligible only after onboarding and one completed
  encounter. Completion does not require Victory, a grade, public matchmaking,
  or a spending-linked achievement.
- Eligible access exists only through a clearly labeled physical hub shop or a
  paid-store menu the player voluntarily opens. Paid and general-resource shop
  surfaces remain visually and semantically distinct.
- A hub surface may show one dismissible notice for genuinely new stock. Dismissal
  persists until a later real catalog revision adds qualifying stock; closing or
  declining cannot trigger repeat prompts for the same revision.
- Purchase prompts/Robux buttons/store comparisons are prohibited during
  onboarding, staging, deployment, combat, downing, recovery/revival, Defeat,
  Results/reward reveals, immediate Retry, and earned-resource shortage.
- Roblox account, age, region, commerce, or platform restrictions produce an
  honest Unavailable state and never a workaround, pressure message, lost
  gameplay function, or lower reward.
- Genuine seasonal cosmetics may show truthful dates. Fake/restarting scarcity,
  discounts, countdowns, urgency, ambiguous rarity, rescue framing, personalized
  pressure, or obscured paid-versus-earned labeling is prohibited.

#### RE-11 — Paid catalog, earnable equivalence, and power review

- **Status:** Resolved 2026-08-26.
- First-release paid SKUs are individually identified deterministic permanent
  cosmetics or equipment only. The exact product is shown before platform
  confirmation; there is no paid randomization or ambiguous result.
- Each stat-bearing paid definition references one exact same-campaign-tier
  earnable equivalent. Functional equivalence requires identical slot, stat
  budget, primary stat, trait behavior, modifier hooks/caps, and upgrade table.
  Only appearance and paid-source lineage may differ.
- The page discloses exact Robux price, product and grant tier, full stat/trait/
  appearance/upgrade state, owned/Pending state, and an Earn Through Play route
  naming the boss, recipe, mastery milestone, or progression source.
- The grant uses the player's currently unlocked tier frozen in the quote. It
  never jumps ahead, auto-scales later, or receives cheaper/non-earned upgrades.
  Paid and earned instances follow identical normal upgrade/uplift costs/rules.
- Owned or Pending SKU state blocks another checkout. Owning only the functional
  earned equivalent does not block an explicitly desired exclusive appearance,
  but the page clearly identifies the already-owned function.
- Automated equivalence/power-budget/prohibited-hook validation, human design and
  economy review, and the same combat-balance tests must all pass before
  publication. Any mismatch blocks the SKU and its dependent store revision.
- Prohibited products include loot boxes/random bundles/gacha/prize wheels;
  resources/materials/consumables/boosts/luck/drop modifiers; revival/recovery/
  Ward/Hype/Call charges; bosses/songs/campaign access; subscriptions/passes;
  convenience/progression skips; and any paid-only gameplay function. A future
  category requires a new explicit owner/compliance review.

#### RE-12 — Purchase receipt, restoration, semantic outputs, and audit

- **Status:** Resolved 2026-08-26.
- Checkout creates an authoritative quote with player, SKU/product/catalog
  revision, platform price, exact definition/tier/grant, earnable equivalent,
  owned/Pending state, and expiry before invoking Roblox confirmation.
- Commerce never trusts client-reported purchase success. Only a verified
  server/platform receipt may authorize the quoted grant. Product identity and
  mapping are immutable and cannot later point an old receipt at a new item.
- Each platform receipt has one immutable idempotency identity and progresses
  through `Pending` to `Granted`, `Already Processed`, `Canceled`, or `Recovery
  Required`. Item/entitlement grant and purchase history commit durably before
  Commerce acknowledges successful processing.
- Repeated, delayed, simultaneous, or out-of-order delivery returns the same
  result. Storage uncertainty stays Pending and retries the same grant without
  requiring/recommending another purchase. Canceled/unverified checkout grants
  nothing.
- Restoration replays verified ownership/history into the exact original
  entitlement or item/tier, not a current-tier or newly mapped substitute. It
  cannot duplicate an already restored grant.
- If a distinct duplicate purchase bypasses Owned/Pending protection, it enters
  visible `Recovery Required`. Only a Roblox-authorized refund/support procedure
  may settle it; the game does not promise/fabricate Robux refunds or substitute
  resources, materials, power, or random goods.
- Commerce stores no payment credentials. The owner sees quote/purchase/history/
  recovery state; Items and Player Data receive exact grant/receipt references;
  privacy-reviewed operations/Analytics receive minimal product/status/failure
  facts. Other players receive nothing.
- Receipt tests cover price/catalog changes, ownership races, duplicate/delayed/
  out-of-order receipt, storage loss/recovery, retirement, account restriction,
  restoration, platform failure, and idempotent grant/history publication.

## 5. Content/configuration reconciliation register

- Participation configuration requires revisioned total-playable and currently-
  eligible group rules, 20%/three-group/half-engagement thresholds, voluntary-
  leave/permanent-inactive exclusions, accessibility neutrality, and private
  reason keys.
- Reward tables require exact Victory/Defeat/No Contest base families, permitted
  difficulty effects, performance/Risk/Cohesion curves and per-source/combined
  caps, deterministic rounding, pool revision, and result/selection seed.
- Validation rejects shared-pool division, negative/cliff bonuses, accessibility
  penalties, pool/first-clear mutation by bonuses, repeated rolls, and No Contest
  grants above its approved compensation boundary.
- Resource/economy catalogs require stable general and per-boss material
  identities, legal source/sink/tier relationships, safe storage bounds, no
  conversion/trade/expiry, and current-tier-efficiency validation.
- Boss pools require exact revisions, entries/chances/weights, starting-rank and
  duplicate rules, deterministic seed contract, authored first-clear/milestone/
  Signature paths, cosmetic-duplicate conversion, and full disclosure.
- Standard boss recipes and pinned-goal data require exact costs, four-to-six-
  victory tuning evidence, no reservation/reset, and salvage tables below
  recreation cost with lock/reference/appearance safeguards.
- Recipe/upgrade/uplift/earned-shop catalogs require exact outputs, costs,
  unlocks, tiers/ranks, duplicate/owned behavior, stack/storage bounds,
  replacements, and preview keys. Validation rejects chance, timers, random
  variation, hidden fees, downgrade/destruction, and locked-snapshot mutation.
- Transaction configuration requires stable identities, expected-state and
  catalog revisions, serialized/atomic ledger plus Items mutation, durable
  Pending recovery, idempotent causal compensation, safe numeric bounds, and a
  complete concurrent/stale/partial-failure test matrix.
- Commerce catalogs require eligibility/prompt policy, immutable SKU/product-
  grant mapping, exact earnable-equivalence links, current-tier quote behavior,
  Owned/Pending protection, prohibited-category checks, truthful seasonal data,
  and automated/human/balance approval evidence.
- Receipt operations require authoritative quotes, platform verification,
  idempotent Pending/grant/history state, exact restoration, duplicate recovery,
  platform refund boundary, privacy allowlists, and the full compliance/failure
  test matrix in RE-12.

## 6. Open handoffs

- `BOSS_ENCOUNTERS.md` owns immutable outcome, difficulty/roster snapshot,
  post-break value, banked Risk facts, and No Contest.
- `RHYTHM_GAMEPLAY.md`, `COMBAT.md`, `ABILITIES_AND_COOPERATIVE_ACTIONS.md`, and
  `MULTIPLAYER.md` own raw normalized contribution/coverage/group/absence facts.
- `PROGRESSION.md` owns campaign/general/mastery eligibility and mutations.
- `ITEMS_AND_EQUIPMENT.md` owns exact item/stack/entitlement mutations and
  duplicate results.
- Player Data owns durable balances/ledger/history; Results/UI present committed
  facts; Analytics receives approved semantics.
- Roblox Marketplace owns Robux billing/receipt authority; Commerce validates and
  converts recognized receipts into Items grants.

## 7. Change log

- **2026-08-25:** Created the working record. Progress is 0 of 12 questions.
- **2026-08-26:** Approved RE-01 through RE-03. Canonical meaningful
  participation, outcome base grants/No Contest compensation, and bounded
  difficulty/performance/Risk/Cohesion assembly are resolved; progress is 3 of
  12 questions.
- **2026-08-26:** Approved RE-04 through RE-06. Two permanent resource families,
  transparent deterministic boss pools/guarantees, visible chosen-item progress,
  duplicate behavior, and voluntary salvage are resolved; progress is 6 of 12.
- **2026-08-26:** Approved RE-07 through RE-09. Fixed exact crafting, upgrades/
  uplift/shop/salvage sinks, and atomic idempotent loss-safe transaction
  orchestration are resolved; progress is 9 of 12 questions.
- **2026-08-26:** Approved RE-10 through RE-12. Voluntary non-coercive store
  access, exact paid/earned equivalence and prohibited products, and verified
  idempotent receipt/restoration/recovery are resolved. All 12 questions were
  reconciled into the canonical specification.
