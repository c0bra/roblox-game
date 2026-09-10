# Bands Battle Progression

- **Status:** Approved
- **Approved:** 2026-08-24
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#65-progression)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Encounter outcome authority:** [`BOSS_ENCOUNTERS.md`](BOSS_ENCOUNTERS.md)
- **Player-language contract:** [`NAMING_AND_TONE.md`](NAMING_AND_TONE.md)
- **Decision source:** [`PROGRESSION_WORKING.md`](PROGRESSION_WORKING.md)
- **Interview plan:** [`PROGRESSION_QUESTIONS.md`](PROGRESSION_QUESTIONS.md)

## 1. Role and authority

This document defines durable advancement and eligibility across three separate
tracks:

1. **Song Restoration:** boss first clears, Song Fragments, destinations, Tiers, hub
   restoration, and per-boss difficulty access.
2. **Order Rank:** broad progress, system/option access, bounded
   baseline power, advanced configuration, and Loadout access.
3. **Boss Mastery:** finite per-boss ranks, milestone eligibility, and replay
   goals shared across instruments.

It also owns personal-best semantics, private recommended-power bands, old-item
Reforge eligibility, monotonic unlock rules, and progression event ordering.

Progression does not own raw gameplay evidence; meaningful-participation
calculation; reward quantities/transactions; item ownership/mutation; Battle Style or
ability behavior; persistence implementation; catalog content production; or
UI presentation. It consumes authoritative results and emits stable eligibility
and advancement facts to those systems.

`Song Restoration`, `Order Rank`, and `Boss Mastery` are the approved
player-facing track names under `NAMING_AND_TONE.md`.

## 2. Governing invariants

1. **Tracks remain distinct:** Song Restoration, Order Rank, and Boss Mastery
   cannot silently grant or overwrite one another's authority.
2. **Victory owns Song Restoration:** Defeat and No Contest cannot create a Song
   Fragment or first-clear destination.
3. **Practice is useful:** meaningful failure may advance Order Rank and
   Boss Mastery without replacing victory-owned rewards/power.
4. **First clear is idempotent:** duplicate, late, retried, or concurrent events
   cannot repeat a Song Fragment or unlock.
5. **Boss Mastery follows the boss:** changing instrument never restarts its track.
6. **Power breadth exceeds account stats:** gear/current-tier success remains the
   main direct-power source.
7. **Choices are reversible:** no progression choice spends earned totals or
   requires paid/out-of-combat respec.
8. **Three complete Loadouts:** the baseline Loadout feature switches a whole
   owned/unlocked combat configuration, not only a specialization subtree.
9. **Recommendations are private advice:** they never become entry, social, or
   matchmaking gates.
10. **Current content remains necessary:** old-item Reforge requires current-tier
    progress/resources and cannot auto-scale.
11. **Progress is permanent:** no daily streak, energy, expiration, seasonal
    regression, or exclusive rotating advancement exists.
12. **Atomic bundles:** a player never sees half of a Song Restoration or Order
    Rank transition.

## 3. Inputs and catalogs

Progression consumes:

- immutable Boss Encounter outcome and exact content/balance revisions;
- Rewards/Multiplayer canonical meaningful-participation eligibility;
- Rhythm, Combat, Positioning, Ability, and Results facts approved for record
  evaluation;
- downstream Reward/Item transaction status for milestone delivery; and
- the current durable player progression snapshot from Player Data.

Versioned catalogs provide stable identities and prerequisites for:

- Song Restoration nodes, destinations, Song Fragments, Tiers, and hub restoration;
- Order Rank levels/milestones and system/option access;
- per-boss difficulty unlocks;
- Boss Mastery Ranks and milestones;
- personal-record categories/compatibility;
- recommendation profiles; and
- item-tier Reforge eligibility/recipes.

A catalog revision never implicitly removes earned equivalence. Migrations are
explicit and auditable.

## 4. Song Restoration credit and first clear

Song Restoration credit requires one immutable **Victory** plus canonical meaningful-
participation eligibility. A player downed or disconnected at encounter end
still qualifies when their retained participation is eligible. Easy, Normal, or
Hard may grant first clear.

The stable key combines player and boss Song Restoration node identity and retains the
causal encounter result. First accepted processing records completion once.
Duplicate, late, out-of-order, or retried delivery returns the established state
without another Song Fragment, mutation, or newly earned notification.

Repeat victories remain eligible for ordinary Boss Mastery, records, rewards, and
replay. Defeat and Invalid / No Contest never grant a first clear, Song Fragment,
destination, Song Restoration Tier, or Normal-victory Hard unlock.

## 5. Song Restoration graph and hub restoration

Song Restoration access is a versioned stable-identity prerequisite graph. The
first public release may
use a linear three-boss path, while later branches or parallel destinations need
only catalog definitions.

A first clear atomically:

1. completes the boss node;
2. records its restored Song Fragment;
3. unlocks every newly satisfied destination and shard tier; and
4. advances the highest newly satisfied hub-restoration state.

No consumer may see a Song Fragment without its completed node or access without
its prerequisites. Replay never repeats Song Restoration/hub mutations.

Hub/UI renders this snapshot and may animate a newly committed transition once;
it cannot invent access. Song Restoration state is monotonic. Migration may map/merge/
replace retired nodes while preserving earned equivalence, but cannot remove a
Song Fragment, relock content, lower hub restoration, or demand replay.

## 6. Difficulty access and recommendations

Easy and Normal are available for each Song-Restoration-unlocked boss. Hard is a
per-player/per-boss permanent fact granted only by a valid meaningful Normal
Victory. Easy victory may advance Song Restoration but never satisfies the Hard
prerequisite.

Multiplayer receives individual boss/difficulty eligibility for every member.
Leaders, invitations, parties, and matchmaking cannot carry someone into locked
Song Restoration content or Hard.

Hard never gates Song Restoration destinations, Song Fragments, essential combat items,
required Battle Style options, or baseline story. It may improve normal reward
efficiency or supply bounded Boss Mastery Cosmetics, titles, and badges.

Difficulty suggestions are private and evidence-based. They may recommend up or
down but never alter a selection, hide choices, identify the suggestion publicly,
or gate rewards/matchmaking.

## 7. Order Rank progress earning

Progression accepts one identified Order Rank progress event per completed eligible
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
cannot grant Song Fragments, Boss Echoes, signature items, first-clear choices, or
other victory-owned advancement.

Calculations use versioned balance data, deterministic final rounding, and
retained pre-rounded evidence.

## 8. Order Rank unlock catalog

Available from first arrival:

- accessibility, comfort, settings, and calibration;
- replayable practice and core controls;
- one usable starter configuration; and
- the first story boss on Easy/Normal.

Completing or explicitly skipping practice unlocks public matchmaking without a
grade test. Order Rank later reveals advanced Battle Style editing, the
complete Loadout surface, broader option catalogs, and wider equipment/Battle
Supply choices.

Core play, safety/accessibility, settings, manual controls, story-critical
actions, and ability to equip a valid starter Loadout are never account-level
gates.

Each unlock has stable identity, catalog revision, prerequisites, result, and
one-time notification metadata. Newly added unlocks grant retroactively when
prerequisites already hold. Revisions do not relock earned access; retired
unlocks migrate to equal-or-broader eligibility.

## 9. Bounded power, respec, and Loadouts

Order Rank's direct permanent stats are small, front-loaded fixed
milestones. The initial ceiling is roughly 10% of a complete first-tier
Loadout's power budget, subject to balance testing but never endless.

Gear and successful current-tier boss progress remain primary direct-power
sources. Progress totals/options are never spent. Specialization can be changed
freely outside active combat with no fee, refund flow, or irreversible branch.

When advanced Battle Style editing opens, all three player-named Loadout slots open
together. A Loadout references:

- playable role/instrument;
- Instrument, Ward Core, and Resonator item instances;
- Signature and Band Call;
- two prepared Battle Supply types; and
- one Anthem and three Riff choices within the Battle Style.

Applying one performs a single validated switch in the hub or pre-battle staging
before final Loadout lock. It never swaps during an active attempt, creates gear,
duplicates Battle Supplies, bypasses ownership/access, or spends missing quantity.

A missing, retired, or incompatible reference makes the Loadout visibly
incomplete until repaired or migrated. Items owns equip/quantity validation,
Builds owns Battle Style compatibility; Naming & Tone owns player strings; and
Player Data owns atomic
persistence/application.

## 10. Boss Mastery

Each boss has one stable finite Boss Mastery track shared across instruments. First
release targets approximately ten visible ranks with versioned cumulative
requirements and visible upcoming milestones.

An eligible result grants Boss Mastery once using outcome base, meaningful-
participation factor, and any bounded difficulty factor. Victory is substantially
more efficient than Defeat; higher difficulty may add a modest bonus, but no
rank requires Hard.

One grant may cross multiple ranks, carrying overflow through their thresholds
atomically. At finite completion, further ordinary Boss Mastery XP is discarded
rather than becoming hidden prestige or endless power. Replay continues through
personal records, ordinary rewards, economy paths, and enjoyment.

No Contest is not an outcome source; only an explicitly authorized compensation
event may grant its configured value once.

## 11. Boss Mastery milestones

A versioned milestone catalog maps boss/rank to lore/archive access, cosmetics,
titles, recipes, bounded specialization sidegrades, or deterministic reward
requests.

Rank crossing evaluates all newly satisfied milestones automatically. Pure
access becomes available immediately. Item/currency/cosmetic delivery becomes
an identified downstream transaction keyed by the milestone. A failed delivery
remains Pending and retries idempotently rather than losing or duplicating it.

New milestones grant retroactively to qualified players and notify once. A
revision never revokes a delivered milestone or earned rank.

Boss Mastery combat options remain bounded sidegrades, not essential current-tier
power. Signature combat items, Boss Echoes, Song Fragments, and
first-clear progress remain victory-owned even though failure may advance Boss
Mastery. Completion may award a major deterministic cosmetic/title/lore reward,
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

Recommended power is private advice for one selected complete Loadout
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
compatible owned items, or Battle Style adjustments. They never open the store or
imply payment is required.

## 14. Old-item Reforge eligibility

Reforge requires a later Song Restoration/item Tier, an eligible owned combat
item that is not a Battle Supply, and its stable unlocked Recipe.

Each transaction advances exactly one tier, cannot exceed the player's current
tier or item's normal cap, preserves item lineage/trait/appearance, and starts at
the target tier's base upgrade rank. Multi-tier catch-up requires sequential
recipes.

Costs use mostly current-tier Resonance plus Boss Echoes from the item's original
boss. This keeps current bosses necessary while preserving older identity and sidegrades. Reforge
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
4. atomically commit the complete Song Restoration, difficulty, Order Rank, Boss
   Mastery, milestone, Loadout, and record bundle; and
5. issue idempotent downstream Reward/Item requests.

Concurrent/retried/out-of-order events commit against the current valid snapshot
or retry. They cannot duplicate totals, expose a partial bundle, regress a
record, or lose Pending delivery.

Durable source facts include first clears, Song Fragments, completed nodes,
Order Rank/Boss Mastery totals, delivered/grandfathered unlocks, milestone states,
difficulty/practice/public access, records/archive, and migration evidence.
Catalog-derived availability may be recomputed while grandfathered access
remains explicit.

Progress never expires or decreases through inactivity, season, rotation,
failure, easier difficulty, item changes, or respec. Migrations map retired
identities to equal-or-broader equivalence and never require replay.

## 16. Semantic outputs

Progression emits attributed facts for:

- first clear/duplicate, Song Fragment, Song Restoration node, destination/Tier,
  and hub state;
- difficulty access and recommendation;
- Order Rank totals/levels and unlocks;
- Loadout-surface access;
- Boss Mastery grant/rank/cap/completion;
- milestone eligibility/Pending/Delivered;
- personal record/current/archive/tie;
- recommended-power band/evidence revision;
- Reforge eligibility;
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
- Song Restoration, Boss Mastery, unlock, and recommendation catalogs reference stable boss/
  encounter identities but remain system data rather than song-timeline data.

## 18. Deferred tuning and technical work

Behavior is complete; these remain versioned balance/architecture choices:

- Order Rank and Boss Mastery award curves, failure/victory ratios, caps, and Rank XP;
- exact Order Rank direct-power ceiling after testing;
- system/option and milestone catalogs;
- final personal-record categories/comparison tuples/coverage;
- recommendation profiles, dimensions, thresholds, and confidence;
- Reforge catalogs and resource costs;
- atomic persistence/concurrency/recovery implementation; and
- future milestone and catalog-entry names under `NAMING_AND_TONE.md`.

Tuning may not turn failure into Song Restoration credit, create endless power,
fragment Boss Mastery by instrument, gate access through recommendations, weaken full Loadout
validation, auto-scale old/paid items, or expire earned advancement.

## 19. Approval and change control

The owner interview resolved PG-01 through PG-12 on 2026-08-24. This document is
the canonical Progression design specification.

A material change to track ownership, first-clear/difficulty rules, Song
Restoration monotonicity, Order Rank power limits, three saved Loadouts, Boss
Mastery finiteness, record compatibility, recommendation privacy, Reforge
constraints, or atomic
permanence requires an explicit amendment citing the superseded rule.
