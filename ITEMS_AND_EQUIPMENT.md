# Bands Battle Items and Equipment

- **Status:** Approved
- **Approved:** 2026-08-24
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#61-items-equipment--loadouts)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Progression dependency:** [`PROGRESSION.md`](PROGRESSION.md)
- **Combat contract:** [`COMBAT.md`](COMBAT.md)
- **Authoring dependency:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)
- **Decision source:** [`ITEMS_AND_EQUIPMENT_WORKING.md`](ITEMS_AND_EQUIPMENT_WORKING.md)
- **Interview plan:** [`ITEMS_AND_EQUIPMENT_QUESTIONS.md`](ITEMS_AND_EQUIPMENT_QUESTIONS.md)

## 1. Role and authority

This document defines immutable item/consumable/cosmetic catalogs, durable owned
instances/stacks/entitlements, quantities, fixed tier/rank power, loadout slots,
three complete spec presets, equip/validation, immutable encounter snapshots,
prepared consumable authorization, cosmetic compatibility, legal item modifiers,
and application of authoritative inventory mutations.

It does not determine drops, prices, crafting/salvage/upgrade/uplift costs,
Commerce receipts, progression eligibility, ability/build behavior, Combat
formulae, persistence implementation, or UI layout. Owning systems provide
identified decisions; Items applies and exposes the resulting state.

## 2. Governing invariants

1. **Definitions are fixed:** no random affixes, stat ranges, quality rolls, or
   reroll seeds.
2. **Ownership shapes remain distinct:** power instances, consumable stacks, and
   cosmetic entitlements cannot be confused or duplicated.
3. **One readable item:** first-release power gear has one primary stat and one
   trait.
4. **Three power slots:** Instrument, Ward Core, and Resonator form the required
   gear surface.
5. **Complete presets switch atomically:** a spec applies in full or not at all.
6. **The arrangement decides role availability:** owned instruments cannot create
   absent song material.
7. **Post-score effects only:** items never change Rhythm, chart, movement,
   telegraph, recovery-count, or position-ratio fairness.
8. **Locked means immutable:** an active encounter uses one exact snapshot
   regardless of later inventory/catalog changes.
9. **Prepared consumables only:** two reserved types are the whole combat
   inventory surface.
10. **Spend only on commitment:** invalid/canceled requests do not consume.
11. **Cosmetics have no power:** appearance cannot alter gameplay recognition or
    critical cues.
12. **Evolution is explicit:** migration/retirement never silently deletes,
    substitutes, or weakens owned identity.

## 3. Catalog and ownership model

### Definitions

An immutable versioned definition declares stable identity, slot/type, supported
tier/rank tables, fixed primary stat, fixed trait/effects, modifier allowlist,
role/cosmetic references, compatibility tags, assets, availability, and migration
relationships.

### Power-item instances

Each owned Instrument, Ward Core, or Resonator has unique player-inventory
identity and references a definition plus current tier, upgrade rank,
acquisition/transaction lineage, and mutation/migration state. Final stats and
traits derive from exact definition/tier/rank revision; instances do not store
secret rolls.

Duplicate drops create distinct equal-function instances. Rewards may later
salvage one; duplicate identity never implies higher quality.

### Consumable stacks

Each player has at most one quantity stack per consumable definition. It stores
owned, reserved, committed/consumed, and refund/mutation evidence. Individual
uses are not permanent item instances.

### Cosmetic entitlements

A permanent entitlement records access to one cosmetic definition. Duplicate
awards return a Duplicate result to Rewards/Economy rather than creating another
or stronger entitlement.

First release has no hard power-item inventory cap, paid storage pressure,
expiring mailbox, or forced deletion. Filtering/sorting and voluntary salvage
handle duplicates.

## 4. Loadout surface

A complete combat configuration contains:

- exactly one owned Instrument instance;
- exactly one owned Ward Core instance;
- exactly one owned Resonator instance;
- exactly one unlocked Signature Special reference;
- exactly one unlocked Band Call reference;
- zero, one, or two distinct prepared consumable-type references; and
- one valid major/supporting specialization configuration.

Appearance is separate and cannot satisfy a combat slot. References do not copy
their targets; multiple presets may point to one owned instance/unlock/stack.

## 5. Full spec presets

When Progression opens advanced builds, three baseline player-named preset slots
become available together. Each stores the complete configuration above plus its
selected role/instrument.

Applying a preset in the hub validates global ownership/unlocks/compatibility.
Applying in selected-encounter staging additionally validates song role,
difficulty/boss, ability/build, and consumable readiness.

Application is atomic. If any required reference is missing, locked, retired,
disabled, incompatible, or invalid, the existing loadout remains unchanged and
all repair issues display together. The saved preset remains **Incomplete**; it
never drops slots, chooses alternatives, changes role, or spends resources.

A zero-quantity optional consumable reference remains remembered as **Empty** and
provides no charge without invalidating required gear.

Switching is allowed only in the hub or unlocked staging. Multiplayer's final
lock freezes it for deployment/combat/rejoin.

## 6. Instrument roles and song compatibility

A global extensible role catalog defines stable playable musical identities.
Drums, vocals, guitar, and bass are initial examples, not a closed roster. Piano,
synthesizer, percussion, strings, and later roles are valid where authentic.

Each Instrument definition references exactly one role plus its visual/audio and
cosmetic family. Item variants within that role may emphasize offense, defense,
support, or hybrids.

An approved song package declares roles with authentic playable chart material
and controllable audio/equivalent mapping. Instrumental songs may omit vocals;
sparse/atmospheric/absent parts may be unavailable.

Staging checks the exact content revision. Unsupported roles are encounter-
incompatible: no chart fabrication, silent substitution, or automatic preset
edit. The owned Instrument/preset remains globally valid for other content.

Duplicate humans may use the same role/variant. A role never fixes combat class.
Explicit thematic ability restrictions are legal only when alternatives remain
and no required composition results.

## 7. Tier, rank, primary stat, and trait

Every definition contains exact values for supported tiers and a base plus
roughly three upgrade ranks per tier. The instance stores tier/rank; consumers
derive primary value, trait parameters, comparison facts, and modifiers.

First-release gear visibly exposes one primary stat and one fixed readable
trait. Internal trait parameters are not random secondary affixes.

Upgrade preview shows exact pre/post tier/rank/value/trait/cost/compatibility.
An authorized upgrade never fails, lowers value, destroys the instance, or
chooses a random result.

Uplift preserves instance/definition lineage, trait, and appearance entitlement,
moves one tier, and resets to the new tier's base rank under Progression.

Starter, earned, crafted, mastery, paid, and earnable-equivalent items use the
same derivation. Functional equivalence means exact stat budget and trait effect.

## 8. Modifier allowlist

Each functional definition emits typed records compatible with `COMBAT.md`:
source item/slot, effect tags, authoritative condition, pipeline stage/category,
power-budget cost, cap/stacking, duration, and attribution.

Slot hooks are constrained:

- **Instrument:** role plus permitted Attack, Defend, support, readiness, or
  hybrid emphasis/trait hooks.
- **Ward Core:** maximum Ward, Defend conversion, mitigation, reinforcement,
  restoration, or bounded recovery received.
- **Resonator:** Attack conversion, Hype, Signature, support/group, or Band Call
  readiness/potency.

No item may alter charts, timing windows, judgments, calibration, Hold Assist,
movement, telegraphs, recovery attempt count, autoplay, invulnerability,
positional baseline ratios, reward eligibility, or required controls.

Contribution traits obey Combat monotonicity/zero rules, cannot copy full value,
and cannot recurse. Static maximum Ward or event-driven utility needs a separate
allowlisted hook and authoritative event.

Catalog validation rejects unknown tags, illegal stages, over-budget effects,
recursion, incompatible caps, and prohibited domains. Runtime never repairs an
illegal definition through silent clamping.

## 9. Final loadout validation and snapshot

Final staging validates:

- exact item/role/ability/build/consumable catalog and balance revisions;
- instance ownership, tier/rank, enabled/mutation state, and slot;
- no salvaged/replaced/duplicate unique-instance slot use;
- Progression unlock and tier access;
- selected song's role/chart/audio support;
- Signature/Band Call/specialization compatibility;
- consumable reference, quantity/reservation, and caps; and
- all modifiers against allowlists/budgets/caps.

It reports all issues together. Invalid required gear/role/modifiers cannot mark
Ready or deploy.

Final lock emits one immutable snapshot with exact instances/definitions/tier/
rank, role, action/consumable/build/cosmetic references, typed modifiers, and
reservations. Catalog/balance/inventory updates apply only to future snapshots.
Disconnect/rejoin restores the same snapshot and remaining charges.

The full inventory cannot upgrade, salvage, swap, or equip during an attempt.

## 10. Prepared consumables

Two optional slots must reference different consumable definitions. At final
lock, each atomically reserves:

`min(owned unreserved quantity, definition encounter cap)`

Reservation makes units unavailable elsewhere but does not consume. Lower
quantity produces fewer visible charges; zero shows Empty. Presets remember
types, not copied quantity, and reserve fresh each attempt.

Used units become Consumed under section 11. Unused reservations release after
Victory, Defeat, canceled deployment, safe teardown, or completed compensation.

Gear, builds, difficulty, paid items, and other loadout elements cannot add
slots, duplicate the same type, raise charge caps, or access unprepared stacks.
Combat UI exposes only prepared types/remaining charges.

## 11. Consumable authorization, spending, and refund

Each use has stable request identity and validates snapshot, slot/type, reserved
charge, player/target/effect eligibility, cooldown/lockout, and revisions.

A valid request may wait for its definition's boundary. It becomes **Committed**
only when the owning effect system guarantees execution. Commit atomically
converts one reserved unit to durably Consumed and publishes the effect request.
The same request cannot decrement/apply twice.

Invalid state/target, no charge, duplicate input, canceled queue, or pre-Commit
interruption spends nothing. After Commit, ordinary downing, disconnect,
movement, intent, Victory, or Defeat does not refund.

If a system fault commits quantity but prevents guaranteed effect resolution,
one request-keyed refund restores it. Invalid / No Contest creates one
idempotent compensation transaction refunding all consumables committed by that
player in that attempt, without exceeding causal committed quantities.

Rejoin restores snapshot charges and committed identities, never hub quantity.

## 12. Cosmetics and appearance

Initial stat-free categories include stagewear, Instrument finish/skin,
aura/performance effect, and title. Entitlements equip only in compatible
appearance slots and contain no stat, trait, modifier, hitbox, or gameplay tag.

Combat presets do not change appearance. A future separate appearance-preset
feature may exist. If appearance becomes incompatible, presentation uses the
owned/default safe compatible form while preserving entitlement/preference.

Local flash/particle/bloom/motion/color/comfort settings may reduce/hide effects
without changing ownership. Replication follows safe platform/performance policy.

Cosmetics must pass moderation, target-age, phone-performance, and critical-cue
occlusion checks. They cannot alter targeting silhouette, obscure charts/
telegraphs, change animation/camera timing, or imitate gameplay/reward states.

## 13. Authoritative inventory mutations

Items applies only identified Reward/Economy/Commerce outcomes:

- grant/craft creates one exact unique power instance;
- consumable grant/spend/refund changes one stack quantity;
- cosmetic grant creates entitlement or returns Duplicate;
- upgrade/uplift mutates the same instance tier/rank; and
- salvage removes one selected instance.

Each result has player/source/revision/idempotency/precondition identity and
applies once. Duplicate delivery returns the committed result. Atomic failure
commits nothing and retries with the same key.

Locked-snapshot instances cannot mutate. Salvaging a hub-equipped or preset-
referenced instance requires explicit confirmation listing affected configs;
those references become Missing/Incomplete, never substitutes. Upgrade/uplift
preserves references but previews any new compatibility issue.

Paid and earned grants use identical mutation/equivalence behavior.

## 14. Catalog versioning, retirement, and extension

Stable definition identities have immutable revisions recording semantics,
compatibility, effects, assets, predecessor/migration, and availability.

A compatible update reaches existing instances through explicit idempotent
migration before a future snapshot, never active combat. Migration preserves
instance identity, tier/rank where compatible, lineage, appearance, and preset
references, and reports material changes.

Safe retired items may remain usable. An exploit/moderation/dependency/hook
failure may require disablement only with clear status plus functional-equivalent
migration or fair compensation. Nothing is silently deleted/replaced.

Future traits, sockets, sets, sidegrades, families, or slots must use typed hooks
and shared budgets, add no mandatory buttons/rhythm authority/random rerolls/paid
storage pressure, and preserve early items through explicit paths. Old gear never
receives free current-tier scaling.

## 15. Semantic output and persistence contract

Items emits identified facts for:

- definition/revision availability/migration;
- instance grant/create/duplicate/equip/upgrade/uplift/salvage/disable/replace;
- stack grant/reserve/release/commit/refund/quantity;
- entitlement grant/duplicate/equip/fallback;
- loadout/preset create/edit/validate/apply/incomplete/repair;
- role/song incompatibility and findings;
- immutable snapshot lock/release; and
- transaction success/reject/duplicate/retry/rollback.

Events carry exact player, definition/instance/stack/entitlement/preset/snapshot,
catalog/content/balance revisions, source transaction/encounter/use, pre/post
state, modifier/compatibility evidence, and idempotency identity.

Player Data stores instances/lineage, stacks/reservations, entitlements/
appearance, current loadout, three named presets and validation state, migration/
transaction history, and locked references.

The owner/UI sees full facts. Multiplayer gets readiness/equipped role/permitted
summary; Combat/Abilities/Audio get locked function; Rewards/Commerce get
transaction results; Analytics gets non-sensitive semantics. Others may see role
and safe appearance, never full inventory, quantities, purchase history, private
recommendations, incomplete presets, or migration findings.

## 16. Content Authoring reconciliation register

- Song packages must reference stable global role identities and explicitly
  declare playable/unavailable roles with compatible chart/audio mappings.
- Validation rejects unknown/retired role identities and never infers support
  from item names or a conventional four-instrument list.

## 17. Deferred tuning and technical work

Behavior is complete; these remain catalog/balance/architecture work:

- item/consumable/cosmetic catalogs, exact stats/traits, tiers/ranks/caps;
- modifier budgets and final comparison presentation;
- consumable effects, quantities, cooldowns, prices, and encounter caps;
- cosmetic slots/assets/performance budgets;
- transaction/persistence/concurrency implementation;
- migration/retirement/compensation procedures; and
- later extension catalog/slot schedule.

Tuning may not add random affixes, storage pressure, silent preset substitution,
unsupported-role fabrication, Rhythm authority, charge inflation, cosmetic
power, active-snapshot mutation, or silent item deletion.

## 18. Approval and change control

The owner interview resolved IE-01 through IE-12 on 2026-08-24. This document is
the canonical Items & Equipment design specification.

A material change to ownership shapes, fixed stats, three power slots, full
preset atomicity, role compatibility, modifier allowlists, snapshot lock,
consumable reservation/spending/refund, cosmetic separation, or explicit
migration requires an amendment citing the superseded rule.
