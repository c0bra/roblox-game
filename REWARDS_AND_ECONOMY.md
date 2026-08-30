# Bands Battle Rewards, Economy, and Commerce

- **Status:** Approved
- **Approved:** 2026-08-26
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#63-rewards-loot--economy)
- **Included system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#64-commerce)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Progression dependency:** [`PROGRESSION.md`](PROGRESSION.md)
- **Items dependency:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)
- **Multiplayer dependency:** [`MULTIPLAYER.md`](MULTIPLAYER.md)
- **Decision source:** [`REWARDS_AND_ECONOMY_WORKING.md`](REWARDS_AND_ECONOMY_WORKING.md)
- **Interview plan:** [`REWARDS_AND_ECONOMY_QUESTIONS.md`](REWARDS_AND_ECONOMY_QUESTIONS.md)

## 1. Role and authority

Rewards/Economy owns canonical meaningful-participation eligibility; personal
reward calculation; general resource and boss materials; complete-drop pools and
deterministic acquisition; fixed crafting, upgrade, uplift, salvage, earned-shop,
and consumable transactions; transaction orchestration; and economy prohibitions.

Commerce owns legal paid catalogs/surfaces, store eligibility, exact earnable-
equivalence validation, authoritative quote/confirmation/receipt/grant recovery,
duplicate-purchase protection, and paid-product/prompt prohibitions.

This document does not own encounter outcome/raw evidence, item/progression
semantics, Marketplace billing/refund authority, durable persistence
implementation, UI layout, or Results presentation. It consumes exact facts and
emits identified grant/mutation decisions to their owners.

## 2. Governing invariants

1. **Rewards are personal:** no shared pool is divided by party size or another
   player's performance.
2. **Eligibility measures participation, not skill:** authentic attempts count;
   accuracy, accessibility, instrument, and public comparison do not gate.
3. **Victory owns boss power progression:** failure/No Contest cannot grant boss
   material, complete rolls, Signature rewards, or campaign first clear.
4. **Every eligible result has a floor:** performance/Risk/Cohesion add bounded
   positives and never erase the base.
5. **One roll means one frozen roll:** reconnect/retry cannot reroll or add
   opportunities.
6. **Random drops do not control builds:** boss material always advances a
   visible chosen-item path; Signature items have deterministic victory paths.
7. **Owned value is fixed:** no random affixes, quality ranges, failure upgrades,
   rerolls, item destruction, or hidden downgrade.
8. **Resources are permanent and simple:** two broad earned families, no expiry,
   conversion, trading, daily cap, energy, debt, or coercive fee.
9. **Transactions are atomic and idempotent:** uncertain state is Pending, never
   silent loss, duplication, negative balance, or changed result.
10. **The paid store is voluntary:** no rescue/results/retry/shortage prompt,
    pressure trick, or hidden paid/earned boundary.
11. **Paid power is exactly earnable:** same-tier function, budget, hooks, and
    upgrade rules; only appearance/source lineage may differ.
12. **Billing authority remains external:** only verified platform receipts
    grant, while Roblox owns Robux billing/refunds/account restrictions.

## 3. Inputs, identities, and frozen result

Rewards consumes immutable attempt outcome/reason/configuration; content,
balance, pool, and difficulty revisions; role/chart and normalized Rhythm/Combat/
Ability evidence; banked Risk and Cohesion facts; Multiplayer connection,
departure, inactivity, and participation coverage; Progression eligibility; and
Items ownership/mutation results.

Commerce additionally consumes current campaign tier, onboarding/completed-
encounter/store eligibility, exact Items equivalence, Player Data history, and
Roblox product/price/receipt/account availability.

Every result, selection seed, eligibility decision, calculation, quote, and
transaction names exact player, source attempt/action, catalogs/revisions,
preconditions, and idempotency identity. Result/pool/seed freeze before reward
calculation; no late packet or retry can change them.

## 4. Canonical meaningful participation

Rewards privately classifies each deployment member as `Eligible`, `Insufficient
Participation`, or `Forfeited` with a deterministic reason/evidence summary.

Initial full eligibility requires:

- genuine attempts in at least 20% of that role's total playable scoring groups;
- at least three attempted groups when the chart contains three or more; and
- attempts in at least half of eligible groups occurring while connected,
  Active, unsuspended, and able to perform.

Authentic early/late/Miss input and Attack/Defend/Special, revival, Band Call, or
Crescendo chart material counts. Rests, dropouts, downed time, connection absence,
Rhythm suspension, and unavailable material are not ignored opportunities.

Downed or involuntarily network-Departed players may qualify from retained
evidence. Voluntary mid-attempt Leave or permanent Inactive after the one Resume
is Forfeited for ordinary outcome rewards.

Solo and multiplayer normalize by role/chart. Accessibility, calibration, Hold
Assist, device, difficulty, and duplicate role never lower value or create a
public/lower reward class. Only the owner and authorized consumers receive the
decision/reason.

## 5. Base grants by outcome

Each eligible **Victory** independently authorizes:

- general resource;
- boss-specific material;
- exactly one complete boss-pool roll;
- visible deterministic-path progress through the material grant; and
- canonical facts for Progression's campaign, General Progression, Boss Mastery,
  first-clear, Hard-unlock, record, and milestone evaluation.

Each eligible **Defeat** authorizes modest general resource plus failure/general/
mastery facts. It grants no boss material, complete roll, Signature reward,
first-clear fragment/destination, or Hard unlock.

Downed or involuntarily Departed players receive the same base when Eligible.
Insufficient/Forfeited players receive no ordinary grant.

**Invalid / No Contest** authorizes no outcome loot, material, roll,
deterministic victory path, campaign credit, or first clear. Items refunds all
causal committed consumables and releases unused reservations. Confirmed
meaningful participants receive one failure-equivalent general-resource and
Progression compensation event. Others receive only causal system-fault refunds.

Compensation never exceeds its configured failure-equivalent ceiling or implies
Victory/Defeat. Each frozen result/eligibility processes once.

## 6. Difficulty and positive bonuses

Assembly order is:

1. outcome base;
2. selected difficulty profile;
3. independent nonnegative personal performance, banked Risk, and Cohesion or
   solo-equivalent breadth packets;
4. per-source and combined clamps; then
5. deterministic final rounding with pre-rounded evidence retained.

Initial tuning hypotheses are about +15% performance, +25% banked Risk, +15%
Cohesion, and +40% combined non-difficulty bonus. Exact values remain revisioned
balance data.

Catalogs explicitly allow each modifier to affect resource quantity, complete-
drop chance, or starting item rank. They cannot change pool identity, add rolls,
alter guaranteed/first-clear items, grant campaign/Hard access, or create
essential difficulty-exclusive function.

Performance normalizes instrument/difficulty/material/trustworthy coverage. Only
Positioning's banked personal Risk counts. Cohesion uses broad successful human
participation under difficulty thresholds; weak/inactive players add little or
none, never a negative or public blame fact. Solo uses equivalent human-only
breadth and acolytes add nothing.

Accessibility/settings/device/role duplication/party size never lower reward
floors/caps or create separate tables. Results privately attributes bonuses.

## 7. Resource model and tier relevance

First release has:

- one permanent account-wide general-resource balance; and
- one permanent boss-material balance per boss.

General resource funds ordinary upgrades, basic crafting, consumables, and the
earned shop. Boss material funds that boss's combat items, traits, cosmetics,
upgrades, and original-boss portion of eligible uplift.

Entries carry stable resource/boss identity, source tier/revision, integer amount,
cause/transaction, and pre/post balance. Balances never become negative.

Resources do not expire, decay, accrue interest, convert to Robux, exchange
between bosses/families, trade, gift, mail-transfer, or face a player-visible cap.
Internal overflow fails atomically and alerts instead of discarding value. There
is no daily cap, energy, diminishing return, debt, or forced sink.

Older encounters retain unique uses but fixed reward profiles do not scale to
the player's tier. Current-tier encounters remain efficient for current-tier
power; uplift consumes mostly current-tier inputs.

## 8. Boss pools, rolls, and guarantees

Each boss pool revision declares fixed-definition power/cosmetic entries,
eligibility, weight/chance, starting rank, duplicate handling, first-clear,
Signature, and milestone metadata.

Easy/Normal/Hard share the functional combat pool. Difficulty may improve the
disclosed complete chance or starting rank, not entries, essential function, or
roll count.

Each eligible Victory performs one deterministic-seeded roll using frozen
attempt/player/pool identity. Pool, overall chance, and entry chances/weights are
visible. Retry/reconnect returns the established result.

A power-item duplicate creates another normal equal-function instance. An owned
cosmetic result converts through an exact disclosed boss-material transaction.
First-clear choices and mastery guarantees are additional stable-key deterministic
awards and do not replace/consume the roll.

Every Signature combat item has a visible victory-based recipe or finite mastery/
clear milestone, may be more demanding than a standard item, and cannot require
Hard or Robux. There is no hidden pity/dynamic chance because material supplies
the explicit path. A random success never resets/consumes it.

## 9. Chosen-item path and salvage

Boss material itself is deterministic-path progress. The player may privately
pin one standard boss item in the hub and see exact cost, balance, progress, and
a selected-difficulty base-clear estimate.

Initial tuning targets roughly four to six baseline eligible Victories for one
chosen standard combat item. Changing/clearing the pin is free and never
reserves/spends/resets material. Receiving the item randomly does not spend or
auto-craft; the owner keeps value and may choose a new goal.

Duplicate power items remain separate until voluntarily salvaged. Preview shows
the exact result, initially around 25–35% of equivalent crafting cost and always
below recreation. Active locked instances cannot salvage. Referenced instances
require explicit warning and leave configurations Missing/Incomplete without
substitution. Appearance entitlements remain permanent. No forced/automatic
salvage, deletion, inventory pressure, or hidden loss exists.

## 10. Exact crafting

First-release recipes may create fixed standard boss combat items, explicit
Signature paths, and selected boss cosmetics/traits. Each revision declares
identity, unlock/prerequisite, boss/tier, exact output/start rank, exact general/
boss-material inputs, quantity, and availability/replacement.

The hub shows output function/appearance, tier/rank, owned status/count, pinned
goal, balances, and final cost before explicit confirmation. Power duplicates
are allowed with warning; owned cosmetic crafting is blocked without spending.

One confirmation creates one output. Ingredient spend and Items grant commit
atomically. Failure creates neither and retries the same identity. There is no
chance, timer, queue, hidden fee, batch ambiguity, failure roll, destruction,
downgrade, or output variation. Stale terms refresh before confirmation.

## 11. Upgrades, uplift, and earned shop

An item initially supports roughly three guaranteed ranks within tier. Each
declares exact result and general-plus-boss-material cost. Preview shows instance,
current/next values, trait continuity, affected references, and cost. Commit
preserves identity/references and never randomly fails, lowers, or destroys.

Progression-authorized uplift moves one eligible old item one tier using mostly
current-tier inputs plus original-boss material. It preserves identity, trait,
appearance, and references while visibly resetting to the new tier's base rank.

Locked-snapshot items cannot upgrade/uplift/salvage. Other mutations follow Items'
reference preservation/warning/Incomplete rules.

The stable non-random earned shop sells low-cost consumables and selected basic
items/cosmetics for general resource. Bundles/stack caps and post-quantity are
exact; over-cap attempts spend nothing. Nothing expires. No repair, death,
respec, cancellation, storage, convenience, rush, or timer fee exists.

## 12. Earned transaction orchestration

Every grant, craft, upgrade, uplift, salvage, earned purchase, duplicate
conversion, refund, and compensation has stable transaction identity, exact
catalog/balance revision, expected state version, and preconditions.

Authority serializes relevant player mutations and validates balance, ownership,
lock/reference state, eligibility, definition/recipe, storage bound, and causal
history. One atomic commit records append-only ledger, pre/post balances, and
Items/Progression mutation/handoff.

Same-identity retry returns Committed or Pending without repeating any random
roll, spend, grant, removal, mutation, refund, or compensation. Concurrent/stale,
insufficient, overflow, invalid, or changed requests fail closed and refresh,
never partially commit/clamp/produce negative balances.

Uncertain cross-service persistence remains visibly Pending and recovers to the
exact grant or rollback. Gameplay Retry need not wait. Corrections are linked,
reasoned, idempotent compensation entries, never silent history edits. No Contest
refunds cannot exceed causal committed quantities.

## 13. Commerce eligibility and voluntary surfaces

The paid store unlocks after onboarding and one completed encounter without a
Victory/grade/public-play requirement. It exists only as a labeled physical hub
shop or paid menu voluntarily opened, visually/semantically separate from the
earned shop.

One dismissible notice may mark genuinely new stock. Dismissal persists until a
later qualifying catalog revision. No Robux prompt/button/comparison appears in
onboarding, staging/deployment, combat, downing/recovery/revival, Defeat, Results/
rewards, immediate Retry, or earned-resource shortage.

Roblox account/age/region/commerce/platform restriction produces honest
Unavailable without workaround, pressure, or gameplay loss. Seasonal dates are
truthful. Fake/restarting scarcity, discount, countdown, urgency, rarity,
rescue framing, personalized pressure, and obscured paid/earned labels are
prohibited.

## 14. Paid catalog and exact equivalence

First-release SKUs are individual deterministic permanent cosmetics/equipment.
Every paid power definition references one exact same-tier earnable equivalent
with identical slot, stat budget, primary stat, trait, modifier hooks/caps, and
upgrade table. Only appearance/source lineage differs.

The page discloses exact Robux price, grant tier/function/appearance/upgrade
state, owned/Pending state, and Earn Through Play route. Grant tier freezes at
the player's currently unlocked quote tier; it never jumps/auto-scales. Paid and
earned items use identical earned upgrade/uplift costs.

Owned/Pending SKU blocks checkout. Owning the earnable function alone does not
block a desired exclusive appearance, but the page says that function is owned.

Automated equivalence/power/prohibited-hook checks, human design/economy review,
and common combat tests must pass. Prohibited are randomization/bundles/gacha;
resources/materials/consumables/boosts/luck/drop changes; revival/recovery/Ward/
Hype/Call charges; content/campaign; subscriptions/passes; convenience/skips;
and paid-only function. Future categories need new explicit review.

## 15. Quote, receipt, grant, and restoration

Checkout creates an expiring authoritative quote naming player, immutable SKU/
product/catalog revision, platform price, exact definition/tier/grant, earnable
equivalent, and owned/Pending state before Roblox confirmation.

Only a verified server/platform receipt authorizes grant. Client success is not
authority. Product mapping cannot be repointed later.

Each receipt identity moves from Pending to Granted, Already Processed, Canceled,
or Recovery Required. Items grant and purchase history commit durably before
successful acknowledgment. Duplicate/delayed/concurrent/out-of-order delivery
returns the same result. Storage uncertainty stays Pending and retries without
another purchase; unverified/canceled checkout grants nothing.

Restoration recreates the exact original entitlement/item/tier and cannot
duplicate it. A distinct duplicate purchase bypassing protection enters visible
Recovery Required. Only Roblox-authorized refund/support may settle it; the game
does not promise/fabricate Robux refund or substitute resource/material/power/
random goods.

## 16. Semantic, persistence, and privacy contract

Rewards/Economy emits identified facts for eligibility/reason; base/difficulty/
performance/Risk/Cohesion calculation; resource/roll/drop/guarantee/path/pin;
craft/upgrade/uplift/salvage/shop; ledger/Pending/commit/rollback/compensation;
and balance/mutation result.

Commerce emits store eligibility/notice/dismissal; catalog/equivalence/review;
quote/confirm; receipt Pending/grant/processed/cancel/recovery; restoration; and
platform/compliance failure.

Player Data stores balances, pinned goals, immutable ledgers/journals, roll/
guarantee history, transaction outcomes, purchase history, restoration, and
recovery state. Items/Progression receive exact mutation facts. UI/Results show
committed/Pending facts without delaying Retry. Analytics/operations get minimal
privacy-reviewed semantics.

Commerce stores no payment credential. Only the owner sees eligibility,
balances, inventory goals, quotes, purchases, and recovery. No player sees
another's spending, eligibility, bonuses, participation reason, or ledger.

## 17. Catalog, balance, and verification requirements

Required revisioned data/testing includes:

- participation group/evidence rules, outcome grants, bonus curves/caps/order,
  difficulty profiles, rounding, and compensation;
- resource identities/sources/sinks/bounds/tier relationships;
- boss pool entries/chances/weights/seeds/duplicates/guarantees/Signature paths;
- recipes, goals, salvage, ranks/uplift, earned-shop/stack rules;
- atomic transaction/journal/recovery/compensation and concurrency tests;
- Commerce eligibility/prompt policy, immutable SKU grants, equivalence links,
  prohibited products, seasonal truth, and approval evidence; and
- quote/receipt/duplicate/restoration/platform failure/privacy/compliance tests.

Publication rejects negative/cliff/shared rewards, repeated rolls, random-only
Signature power, hidden pity, conversion/trade/expiry, recreation-value salvage,
random/destructive/timed crafting, partial transactions, coercive prompts,
paid-power mismatch, prohibited products, mutable mappings, or non-idempotent
receipts.

## 18. Deferred tuning and technical work

Behavior is complete; these remain versioned economy/product/architecture work:

- resource/item/trait/cosmetic names and complete catalogs;
- quantities/chances/weights, difficulty/bonus caps, four-to-six-clear target,
  salvage percentages, rank/craft/uplift/consumable costs;
- first-clear/Signature/mastery guarantee catalogs;
- durable transaction/receipt/state recovery implementation and operations;
- Robux prices, store presentation, platform compliance, seasonal assets, and
  support/refund procedures; and
- economy simulation, exploit testing, retention/privacy, and alert thresholds.

Tuning may not create failure-optimal farming, random-only build access, random
affixes/rerolls, daily/energy/expiry/trading/debt/fees, partial loss, paid rescue/
resource/luck/content/convenience, superior paid function, pressure prompts,
mutable receipt grants, or client-authorized purchase success.

## 19. Approval and change control

The owner interview resolved RE-01 through RE-12 on 2026-08-26. This document is
the canonical Rewards, Economy, and Commerce design specification.

A material change to participation eligibility, outcome grants, bonus order/
caps, resource families, pool/roll/deterministic path, transaction guarantees,
store surfaces, paid-equivalence/prohibitions, or receipt/restoration requires an
explicit amendment citing the superseded rule. Numeric/catalog tuning inside
these boundaries creates a new revision and never changes a frozen result,
transaction, quote, or receipt mapping.
