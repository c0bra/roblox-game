# Bands Battle Progression

- **Status:** Approved
- **Approved:** 2026-08-24
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#65-progression)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Encounter outcome authority:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Decision source:** [`PROGRESSION_WORKING.md`](PROGRESSION_WORKING.md)
- **Interview plan:** [`PROGRESSION_QUESTIONS.md`](PROGRESSION_QUESTIONS.md)

## 1. Role and authority

This document defines durable advancement and eligibility across three separate
tracks:

1. **Campaign:** boss first clears, fragments, destinations, tiers, hub
   restoration, and per-boss difficulty access.
2. **General Player Progression:** broad progress, system/option access, bounded
   baseline power, advanced configuration, and spec-preset access.
3. **Boss Mastery:** finite per-boss ranks, milestone eligibility, and replay
   goals shared across instruments.

It also owns personal-best semantics, private recommended-power bands, old-item
uplift eligibility, monotonic unlock rules, and progression event ordering.

Progression does not own raw gameplay evidence; meaningful-participation
calculation; reward quantities/transactions; item ownership/mutation; build or
ability behavior; persistence implementation; catalog content production; or
UI presentation. It consumes authoritative results and emits stable eligibility
and advancement facts to those systems.

Final player-facing names for the three tracks remain open.

## 2. Governing invariants

1. **Tracks remain distinct:** Campaign, General Progression, and Boss Mastery
   cannot silently grant or overwrite one another's authority.
2. **Victory owns campaign advancement:** Defeat and No Contest cannot create a
   fragment or first-clear destination.
3. **Practice is useful:** meaningful failure may advance General Progression and
   Boss Mastery without replacing victory-owned rewards/power.
4. **First clear is idempotent:** duplicate, late, retried, or concurrent events
   cannot repeat a fragment or unlock.
5. **Mastery follows the boss:** changing instrument never restarts its track.
6. **Power breadth exceeds account stats:** gear/current-tier success remains the
   main direct-power source.
7. **Choices are reversible:** no progression choice spends earned totals or
   requires paid/out-of-combat respec.
8. **Three complete presets:** the baseline spec-preset feature switches a whole
   owned/unlocked combat configuration, not only a specialization subtree.
9. **Recommendations are private advice:** they never become entry, social, or
   matchmaking gates.
10. **Current content remains necessary:** old-item uplift requires current-tier
    progress/resources and cannot auto-scale.
11. **Progress is permanent:** no daily streak, energy, expiration, seasonal
    regression, or exclusive rotating advancement exists.
12. **Atomic bundles:** a player never sees half of a campaign or rank transition.

## 3. Inputs and catalogs

Progression consumes:

- immutable Boss Encounter outcome and exact content/balance revisions;
- Rewards/Multiplayer canonical meaningful-participation eligibility;
- Rhythm, Combat, Positioning, Ability, and Results facts approved for record
  evaluation;
- downstream Reward/Item transaction status for milestone delivery; and
- the current durable player progression snapshot from Player Data.

Versioned catalogs provide stable identities and prerequisites for:

- campaign nodes, destinations, fragments, tiers, and hub restoration;
- general levels/milestones and system/option access;
- per-boss difficulty unlocks;
- boss mastery ranks and milestones;
- personal-record categories/compatibility;
- recommendation profiles; and
- item-tier uplift eligibility/recipes.

A catalog revision never implicitly removes earned equivalence. Migrations are
explicit and auditable.

## 4. Campaign credit and first clear

Campaign credit requires one immutable **Victory** plus canonical meaningful-
participation eligibility. A player downed or disconnected at encounter end
still qualifies when their retained participation is eligible. Easy, Normal, or
Hard may grant first clear.

The stable key combines player and boss campaign-node identity and retains the
causal encounter result. First accepted processing records completion once.
Duplicate, late, out-of-order, or retried delivery returns the established state
without another fragment, mutation, or newly earned notification.

Repeat victories remain eligible for ordinary mastery, records, rewards, and
replay. Defeat and Invalid / No Contest never grant a first clear, fragment,
destination, campaign tier, or Normal-victory Hard unlock.

## 5. Campaign graph and hub restoration

Campaign access is a versioned stable-identity prerequisite graph. Launch may
use a linear three-boss path, while later branches or parallel destinations need
only catalog definitions.

A first clear atomically:

1. completes the boss node;
2. records its restored Shattered Song fragment;
3. unlocks every newly satisfied destination and shard tier; and
4. advances the highest newly satisfied hub-restoration state.

No consumer may see a fragment without its completed node or access without its
prerequisites. Replay never repeats campaign/hub mutations.

Hub/UI renders this snapshot and may animate a newly committed transition once;
it cannot invent access. Campaign state is monotonic. Migration may map/merge/
replace retired nodes while preserving earned equivalence, but cannot remove a
fragment, relock content, lower hub restoration, or demand replay.

## 6. Difficulty access and recommendations

Easy and Normal are available for each campaign-unlocked boss. Hard is a
per-player/per-boss permanent fact granted only by a valid meaningful Normal
Victory. Easy victory may advance Campaign but never satisfies the Hard
prerequisite.

Multiplayer receives individual boss/difficulty eligibility for every member.
Leaders, invitations, parties, and matchmaking cannot carry someone into locked
campaign content or Hard.

Hard never gates campaign destinations, fragments, essential combat items,
required build options, or baseline story. It may improve normal reward
efficiency or supply bounded mastery cosmetics, titles, and badges.

Difficulty suggestions are private and evidence-based. They may recommend up or
down but never alter a selection, hide choices, identify the suggestion publicly,
or gate rewards/matchmaking.

## 7. General progress earning

Progression accepts one identified general-progress event per completed eligible
attempt. It uses canonical meaningful participation rather than recomputing
eligibility from raw gameplay.

- **Victory:** substantially larger base progress.
- **Defeat:** capped practice progress scaled inside the eligible participation
  range.
- **Invalid / No Contest:** no automatic outcome award; Rewards may issue one
  explicit compensation grant for confirmed participation, initially no worse
  than ordinary failure progress.
- **Practice/onboarding:** separate checkpoint/completion facts, never repeatable
  account-progress farming.

Legitimate repeat failures remain useful without hidden daily diminishing
returns. Per-attempt caps and participation exclude idling. Failure progress
cannot grant fragments, boss materials, signature items, first-clear choices, or
other victory-owned advancement.

Calculations use versioned balance data, deterministic final rounding, and
retained pre-rounded evidence.

## 8. General unlock catalog

Available from first arrival:

- accessibility, comfort, settings, and calibration;
- replayable practice and core controls;
- one usable starter configuration; and
- the first campaign boss on Easy/Normal.

Completing or explicitly skipping practice unlocks public matchmaking without a
grade test. General Progression later reveals advanced build/spec editing, the
complete preset surface, broader option catalogs, and wider equipment/
consumable choices.

Core play, safety/accessibility, settings, manual controls, campaign-critical
actions, and ability to equip a valid starter loadout are never account-level
gates.

Each unlock has stable identity, catalog revision, prerequisites, result, and
one-time notification metadata. Newly added unlocks grant retroactively when
prerequisites already hold. Revisions do not relock earned access; retired
unlocks migrate to equal-or-broader eligibility.

## 9. Bounded power, respec, and spec presets

General Progression's direct permanent stats are small, front-loaded fixed
milestones. The initial ceiling is roughly 10% of a complete first-tier
loadout's power budget, subject to balance testing but never endless.

Gear and successful current-tier boss progress remain primary direct-power
sources. Progress totals/options are never spent. Specialization can be changed
freely outside active combat with no fee, refund flow, or irreversible branch.

When advanced builds open, all three baseline player-named spec-preset slots open
together. A preset references:

- playable role/instrument;
- Instrument, Ward Core, and Resonator item instances;
- Signature Special and Band Call;
- two prepared consumable types; and
- major/supporting specialization choices.

Applying one performs a single validated switch in the hub or pre-battle staging
before final loadout lock. It never swaps during an active attempt, creates gear,
duplicates consumables, bypasses ownership/access, or spends missing quantity.

A missing, retired, or incompatible reference makes the preset visibly
incomplete until repaired or migrated. Items owns equip/quantity validation,
Builds owns specialization compatibility/names, and Player Data owns atomic
persistence/application.

## 10. Boss Mastery

Each boss has one stable finite mastery track shared across instruments. First
release targets approximately ten visible ranks with versioned cumulative
requirements and visible upcoming milestones.

An eligible result grants mastery once using outcome base, meaningful-
participation factor, and any bounded difficulty factor. Victory is substantially
more efficient than Defeat; higher difficulty may add a modest bonus, but no
rank requires Hard.

One grant may cross multiple ranks, carrying overflow through their thresholds
atomically. At finite completion, further ordinary mastery XP is discarded
rather than becoming hidden prestige or endless power. Replay continues through
personal records, ordinary rewards, economy paths, and enjoyment.

No Contest is not an outcome source; only an explicitly authorized compensation
event may grant its configured value once.

## 11. Mastery milestones

A versioned milestone catalog maps boss/rank to lore/archive access, cosmetics,
titles, recipes, bounded specialization sidegrades, or deterministic reward
requests.

Rank crossing evaluates all newly satisfied milestones automatically. Pure
access becomes available immediately. Item/currency/cosmetic delivery becomes
an identified downstream transaction keyed by the milestone. A failed delivery
remains Pending and retries idempotently rather than losing or duplicating it.

New milestones grant retroactively to qualified players and notify once. A
revision never revokes a delivered milestone or earned rank.

Mastery combat options remain bounded sidegrades, not essential current-tier
power. Signature combat items, boss materials, campaign fragments, and
first-clear progress remain victory-owned even though failure may advance
mastery. Completion may award a major deterministic cosmetic/title/lore reward,
not endless stats.

## 12. Personal-best records

Records key by player, boss, instrument/role, difficulty, solo/co-op mode, record
compatibility, and roster grouping for categories materially changed by roster.

Categories update independently. Initial families include rhythm execution,
participation/holds, personal combat, Ward/survival, position/risk, cooperative
help, and overall personal performance. The later Results/UI contract selects a
concise final visible catalog.

Each category declares an ordered comparison tuple, minimum participation/
coverage, valid attempt/outcome states, and supporting evidence. Victory and
Defeat may qualify. Disconnect disqualifies only categories whose coverage fails.
No Contest may update only categories explicitly preserved as trustworthy.

An exact full-tuple tie preserves the earlier record. Calibration, Hold Assist,
and accessibility are private and do not create lower-value/public categories.

Content revisions declare record compatibility. Compatible revisions continue a
set; incompatible chart/scoring changes archive the old set and start a new
current one without deleting history. Records compare the player only with their
own history; no public damage leaderboard exists.

## 13. Recommended power

Recommended power is private advice for one selected complete spec preset
against one boss/difficulty revision. It is not a universal social number.

A versioned profile evaluates multiple dimensions, initially offense, Ward/
defensive conversion, and relevant ability/support utility. It recognizes valid
specializations rather than ranking only damage. The player sees a concise band
such as **Prepared**, **Challenging**, or **Far Below**, a useful limiting
dimension, and uncertainty when data is stale.

Any valid player may enter regardless. Other players, parties, matchmaking, and
public profiles cannot see or gate on it. Accessibility, calibration, Hold
Assist, private performance, age, and spending do not reduce readiness.

Improvement routes point only to earned bosses, deterministic recipes, upgrades,
compatible owned items, or build adjustments. They never open the store or
imply payment is required.

## 14. Old-item uplift eligibility

Uplift requires a later campaign/item tier, an eligible owned non-consumable
combat item, and its stable unlocked recipe.

Each transaction advances exactly one tier, cannot exceed the player's current
tier or item's normal cap, preserves item lineage/trait/appearance, and starts at
the target tier's base upgrade rank. Multi-tier catch-up requires sequential
recipes.

Costs use mostly current-tier resources plus original-boss material. This keeps
current bosses necessary while preserving older identity and sidegrades. Uplift
never auto-runs or exceeds current-tier standards. Paid equivalents use identical
eligibility, cost, target tier, rank, and cap.

Progression emits eligibility; Rewards/Economy spends resources and Items mutates
the item atomically.

## 15. Atomic processing and permanence

Each mutation has stable player, source, progression-event, catalog revision,
and causal-result identity. Processing order is:

1. validate outcome/eligibility and exact revisions;
2. apply totals with deterministic balance data;
3. determine all new rank/level/prerequisite/record changes;
4. atomically commit the complete campaign, difficulty, general, mastery,
   milestone, preset, and record bundle; and
5. issue idempotent downstream Reward/Item requests.

Concurrent/retried/out-of-order events commit against the current valid snapshot
or retry. They cannot duplicate totals, expose a partial bundle, regress a
record, or lose Pending delivery.

Durable source facts include first clears, fragments, completed nodes,
general/mastery totals, delivered/grandfathered unlocks, milestone states,
difficulty/practice/public access, records/archive, and migration evidence.
Catalog-derived availability may be recomputed while grandfathered access
remains explicit.

Progress never expires or decreases through inactivity, season, rotation,
failure, easier difficulty, item changes, or respec. Migrations map retired
identities to equal-or-broader equivalence and never require replay.

## 16. Semantic outputs

Progression emits attributed facts for:

- first clear/duplicate, fragment, campaign node, destination/tier, and hub state;
- difficulty access and recommendation;
- general totals/levels and unlocks;
- preset-surface access;
- mastery grant/rank/cap/completion;
- milestone eligibility/Pending/Delivered;
- personal record/current/archive/tie;
- recommended-power band/evidence revision;
- uplift eligibility;
- migration; and
- rejected/duplicate/concurrent processing.

Every fact carries player, stable source/cause, relevant catalog/content/balance
revisions, pre/post state, and idempotency identity. Player Data persists;
Results, Hub, Onboarding, Items, Builds, Rewards, Commerce, Multiplayer, UI, and
Analytics consume without owning the semantics.

## 17. Content Authoring reconciliation register

The following requirements were reconciled into
[`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md#14-cross-specification-handoffs-and-reconciliation)
on 2026-09-02:

- Content revisions need stable record-compatibility identity and explicit
  compatible/incompatible relationship to the preceding revision for every
  personal-best category affected by chart/scoring changes.
- Campaign, mastery, unlock, and recommendation catalogs reference stable boss/
  encounter identities but remain system data rather than song-timeline data.

## 18. Deferred tuning and technical work

Behavior is complete; these remain versioned balance/architecture choices:

- general and mastery award curves, failure/victory ratios, caps, and rank XP;
- exact general direct-power ceiling after testing;
- system/option and milestone catalogs;
- final personal-record categories/comparison tuples/coverage;
- recommendation profiles, dimensions, thresholds, and confidence;
- uplift catalogs and resource costs;
- atomic persistence/concurrency/recovery implementation; and
- final track, rank, milestone, and preset terminology.

Tuning may not turn failure into campaign credit, create endless power, fragment
mastery by instrument, gate access through recommendations, weaken full preset
validation, auto-scale old/paid items, or expire earned advancement.

## 19. Approval and change control

The owner interview resolved PG-01 through PG-12 on 2026-08-24. This document is
the canonical Progression design specification.

A material change to track ownership, first-clear/difficulty rules, campaign
monotonicity, general-power limits, three full presets, mastery finiteness,
record compatibility, recommendation privacy, uplift constraints, or atomic
permanence requires an explicit amendment citing the superseded rule.
