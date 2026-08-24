# Bands Battle Items and Equipment Specification Questions

- **Status:** Interview complete; 12 of 12 questions resolved and reconciled
- **Started:** 2026-08-24
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#61-items-equipment--loadouts)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Progression dependency:** [`PROGRESSION.md`](PROGRESSION.md)
- **Combat modifier contract:** [`COMBAT.md`](COMBAT.md)
- **Working record:** [`ITEMS_AND_EQUIPMENT_WORKING.md`](ITEMS_AND_EQUIPMENT_WORKING.md)
- **Canonical result:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)

## 1. Interview method

This interview uses four checkpoints of three questions. It defines item,
inventory, loadout, consumable, cosmetic, and mutation semantics without deciding
drop chances, resource prices, ability behavior, build-option behavior,
persistence architecture, or UI layout.

Answers are persisted after each checkpoint. When all twelve questions are
resolved, they are reconciled into canonical `ITEMS_AND_EQUIPMENT.md`.

## 2. Fixed inherited decisions

- First-release power gear has exactly three player-facing slots: Instrument,
  Ward Core, and Resonator.
- An Instrument establishes the performed instrument/role and supplies a modest
  primary emphasis plus one distinctive trait. Every instrument category must
  support offensive, defensive, and support-oriented variants rather than a
  class lock.
- Ward Cores affect maximum Ward, Defend conversion, reinforcement, restoration,
  or bounded recovery received.
- Resonators affect Attack conversion, Hype, Signature potency, support, or Band
  Call readiness.
- Separate action references contain one Signature Special, one Band Call, and
  two prepared consumable types with limited encounter quantity/charges.
- Appearance slots/unlocks are separate and carry no combat stats.
- A first-release item exposes one primary stat and one readable trait, uses
  fixed values, and has no random affix ranges or reroll loop.
- Equipment modifies only post-Rhythm combat consequences under `COMBAT.md`.
  It never changes charts, judgment windows, calibration, telegraphs, movement,
  recovery counts, autoplay, position ratios, reward eligibility, or core
  controls.
- The full inventory is inaccessible during combat. The validated locked loadout
  is the entire encounter equipment surface.
- Three full player-named spec presets reference instrument/role, all power gear,
  Signature, Band Call, two consumable types, and specialization choices; they
  switch atomically only in the hub or unlocked staging before final lock.
- Roles are extensible and song-specific. Drums/vocals/guitar/bass are only
  starting tool categories; piano, synthesizer, percussion, strings, or later
  roles may be valid, while absent/sparse parts may be unavailable.
- Rewards/Economy owns acquisition, drops, salvage, crafting, upgrading, uplift
  resource spending, and transactions. Items owns resulting inventory mutation.

## 3. Question plan

### Checkpoint A — Identity, ownership, and complete loadouts

#### IE-01 — Definition, instance, stack, and entitlement identity [Resolved]

- **Decision needed:** Which durable shapes represent fixed gear, duplicates,
  consumables, and cosmetic ownership?
- **Must resolve:** Catalog definition versus player instance, stable identity,
  derived fixed stats, tier/rank, acquisition lineage, duplicate instances,
  consumable stacks, cosmetic entitlements, quantities, inventory capacity, and
  no random affixes.
- **Owner decision:** Immutable versioned catalog definitions describe fixed
  gear semantics. Each owned power item is a unique instance referencing its
  definition plus tier, upgrade rank, and acquisition lineage; stats/traits are
  derived and never random affixes. Duplicate drops are separate instances.
  Consumables are quantity stacks by definition, and cosmetics are permanent
  entitlements rather than power instances. First release has no hard inventory
  capacity or paid storage pressure.

#### IE-02 — Loadout slots and atomic spec-preset application [Resolved]

- **Decision needed:** What constitutes one valid complete combat configuration
  and how does quick switching succeed or fail?
- **Must resolve:** Required/optional slots, exact instance/reference ownership,
  action references, consumable quantities, specialization handoff, atomic
  validation, incomplete preset, hub/staging availability, final lock, and no
  partial/silent substitution.
- **Owner decision:** A valid combat loadout requires one owned Instrument,
  Ward Core, and Resonator instance plus one unlocked Signature Special and Band
  Call; the two consumable-type references may be empty. A full spec preset also
  references its role and specialization choices. Application validates every
  ownership, unlock, compatibility, and consumable reference atomically. Either
  the whole configuration applies or the prior loadout remains, with all errors
  shown. The saved preset remains visibly incomplete for repair. Switching is
  legal only in the hub or unlocked staging before final loadout lock; there is
  no silent substitution or active-combat swap.

#### IE-03 — Instrument/role and song compatibility [Resolved]

- **Decision needed:** How does an owned Instrument map to an extensible playable
  role and become valid for a particular song/encounter?
- **Must resolve:** Role catalog identity, instrument variants, song-supported
  roles, sparse/absent material, duplicate roles, unsupported presets, audio/
  visual references, thematic ability restrictions, and no class composition.
- **Owner decision:** Each Instrument definition references one stable extensible
  role identity and its visual/audio family. The selected song package declares
  authentic supported playable roles with approved chart/audio mappings.
  Encounter staging accepts the Instrument only when its role is supported;
  unavailable/sparse material is never fabricated and no substitute is chosen
  automatically. Such a preset remains globally valid but encounter-incompatible.
  Duplicate players may use the same role. Instrument variants can support any
  combat emphasis, and ability restrictions must be explicit thematic exceptions
  that never create a required composition or class lock.

### Checkpoint B — Stats, traits, and combat resolution

#### IE-04 — Tier, upgrade rank, primary stat, and trait [Resolved]

- **Decision needed:** Which values belong to definition, tier variant, and
  instance, and how do rank/up-lift changes derive final readable stats?
- **Must resolve:** Fixed stat source, one-primary/one-trait rule, tier/rank cap,
  upgrade curve, uplift reset, preview, equality, starter items, comparison, and
  no randomized ranges.
- **Owner decision:** Each immutable definition contains exact values for every
  supported tier and roughly three upgrade ranks. An instance stores tier/rank;
  final values derive from the definition. First-release gear exposes one visible
  primary stat and one fixed trait. Upgrades preview exact results, never fail,
  and never roll values. Uplift preserves identity/trait and begins at the next
  tier's base rank. Starter, earned, and paid-equivalent items follow the same
  derivation/equality rules.

#### IE-05 — Modifier allowlist and resolution [Resolved]

- **Decision needed:** How do equipped items emit legal typed modifiers through
  Combat without reimplementing formulas or forbidden gameplay changes?
- **Must resolve:** Allowed effect tags by slot, definition validation,
  conditional traits, Combat stage/category/budget, caps, source attribution,
  zero-performance behavior, illegal definition rejection, and extension.
- **Owner decision:** Definitions emit typed modifiers using Combat's declared
  category, authoritative condition, pipeline stage, power budget, cap, and
  attribution contract. Instrument hooks may cover allowed offense/defense/
  support effects; Ward Cores are limited to Ward/Defend/restoration/recovery;
  Resonators are limited to Attack/Hype/Signature/support/Band Call. A
  contribution-derived trait cannot create value from zero, recurse, or affect
  Rhythm. Static/event-driven effects require explicit allowlisted hooks.
  Catalog validation rejects illegal definitions rather than runtime clamping.

#### IE-06 — Loadout validation and encounter snapshot [Resolved]

- **Decision needed:** Which checks produce one immutable resolved loadout for
  staging/encounter use?
- **Must resolve:** Ownership/unlock/tier/slot/role/ability/build/consumable
  validation, duplicate reference rules, definition versions, resolved modifier
  output, lock timing, queue changes, disconnect/rejoin, and snapshot identity.
- **Owner decision:** Final staging validation checks exact definition/catalog
  versions, owned and mutable instance state, slot type, role/song support, tier
  access, ability/build compatibility, and consumable readiness. It resolves one
  immutable identified loadout snapshot containing every legal reference and
  modifier at final lock. An invalid configuration cannot deploy and reports all
  issues. Catalog/balance/inventory changes after lock cannot change the attempt;
  disconnect/rejoin restores the same snapshot.

### Checkpoint C — Consumables and cosmetics

#### IE-07 — Prepared consumable slots and charges [Resolved]

- **Decision needed:** How do two prepared types map inventory quantity into
  encounter-available uses?
- **Must resolve:** Stack identity, same-type duplicate slots, per-attempt cap,
  quantity reservation, displayed charges, eligibility, difficulty/build limits,
  empty slots, refill, and no combat inventory browsing.
- **Owner decision:** The two optional prepared slots must reference different
  consumable definitions. At final lock, each reserves up to its versioned per-
  encounter cap from the owned stack without consuming it. Lower quantity yields
  fewer visible charges but remains valid; zero yields Empty. Presets remember
  types and reserve again for later attempts. Unused reservations return after
  the attempt. Gear, builds, and difficulty cannot add slots or charges. Only
  prepared reserved uses exist in combat; inventory browsing is unavailable.

#### IE-08 — Consumption authorization and recovery [Resolved]

- **Decision needed:** At what point is a consumable spent and how do cancellation,
  duplicate input, disconnect, server failure, and No Contest resolve safely?
- **Must resolve:** Request/commit/effect boundaries, idempotency, invalid use,
  cooldown/target, interruption, quantity decrement, receipt/retry, refund,
  encounter result independence, and attribution.
- **Owner decision:** A use validates the prepared slot, reserved charge,
  player/target/effect state, cooldown, and request identity. Reserved quantity
  becomes durably consumed only at the authoritative Commit boundary where the
  effect is guaranteed to execute. Invalid, duplicate, or pre-Commit-canceled
  requests spend nothing. After Commit, ordinary downing, disconnect, Victory,
  or Defeat does not refund. A system failure between spend/effect refunds that
  use. Invalid / No Contest refunds all attempt-committed consumables through one
  idempotent compensation transaction. Every transition retains source lineage.

#### IE-09 — Appearance ownership and equipment [Resolved]

- **Decision needed:** How are stagewear, skins/finishes, auras/effects, titles,
  and other cosmetics owned/equipped without gameplay authority?
- **Must resolve:** Entitlement versus instance, permanent unlock, duplicate
  handling, slot/category compatibility, instrument appearance, effects/safety
  settings, multiplayer replication, moderation, stat prohibition, and preset
  relationship.
- **Owner decision:** Stagewear, instrument finishes/skins, auras/performance
  effects, titles, and later appearance categories are permanent stat-free
  entitlements equipped in compatible appearance slots. Duplicate awards become
  Economy-owned compensation. Combat spec presets do not change appearance.
  An incompatible appearance temporarily uses the owned/default compatible
  presentation without removing the entitlement. Local safety/accessibility may
  reduce/hide effects. Cosmetics must pass catalog/platform moderation and never
  affect hitboxes, targeting, visibility, critical cues, animation timing,
  gameplay recognition, or any modifier.

### Checkpoint D — Mutations, evolution, and outputs

#### IE-10 — Acquisition and item mutation handoff [Resolved]

- **Decision needed:** How do reward, purchase, salvage, craft, upgrade, and
  uplift transactions change inventory exactly once?
- **Must resolve:** Transaction/result identity, grant instance creation,
  duplicate drop, salvage removal, upgrade/uplift mutation, atomicity, equipped/
  preset references, paid equivalence, retry/rollback, and audit lineage.
- **Owner decision:** Each Reward/Economy/Commerce result has a stable
  idempotency key and creates or mutates exactly one declared instance, stack, or
  entitlement outcome. Crafting creates; upgrade/uplift preserves the same
  instance identity; salvage removes one unlocked instance. Salvaging an equipped
  or preset-referenced instance requires explicit confirmation listing every
  affected configuration, which becomes visibly incomplete without substitution.
  Locked encounter snapshots cannot mutate. Failure commits nothing or retries
  safely, and all lineage remains auditable.

#### IE-11 — Catalog versioning, retirement, and extension [Resolved]

- **Decision needed:** How do catalog changes preserve owned items/presets while
  leaving room for later slots, traits, sockets, sets, and techniques?
- **Must resolve:** Definition immutability, balance revision, migrations,
  grandfathering, retired/disabled items, preset repair, compensation,
  extension-point constraints, no mandatory buttons, and early-item viability.
- **Owner decision:** Definition revisions are immutable under a stable identity.
  Existing instances adopt a compatible new revision only through explicit
  migration affecting future loadout snapshots, never an active attempt, while
  preserving tier/rank/ownership lineage and preset references. Unsafe retired
  items may be disabled only with clear equivalent migration or compensation,
  never silent deletion. Future traits/sockets/sets/slots must remain inside
  typed modifier budgets, add no mandatory battle button, and preserve early
  items through explicit compatible evolution.

#### IE-12 — Semantic outputs and persistence contract [Resolved]

- **Decision needed:** Which item/loadout/consumable/cosmetic facts must be
  exposed to consumers and durably stored?
- **Must resolve:** Event catalog, exact revisions/identities, source/transaction
  attribution, inventory snapshot, loadout/preset validation, modifier output,
  quantities/reservations, privacy, UI/Audio/Analytics, Player Data boundary,
  and completion audit.
- **Owner decision:** Items emits attributed facts for grants/instances/stacks/
  entitlements, equip/preset validation/application, encounter snapshots,
  reservations/use/refunds, cosmetics, upgrades/uplifts/salvage, migration, and
  rejected/duplicate transactions. Player Data persists exact identities,
  revisions, quantities, references, states, and lineage. Consumers see only
  their required semantic subset. Other players may see equipped role and safe
  appearance where appropriate, never full inventory, recommendations,
  incomplete presets, or acquisition history.

## 4. Completion criteria

`ITEMS_AND_EQUIPMENT.md` is complete only when:

- IE-01 through IE-12 are resolved;
- fixed definitions, owned instances, stacks, and entitlements cannot be
  confused or duplicated;
- full spec presets apply atomically and never silently substitute;
- song-specific role compatibility supports an extensible catalog;
- item modifiers can only use the approved post-score effect surface;
- consumable quantities cannot be double-spent or lost on invalid use;
- cosmetics have no combat authority;
- mutations preserve equipped/preset references or expose explicit repair; and
- all durable facts and semantic outputs are complete for Player Data.

## 5. Change log

- **2026-08-24:** Created the concise 12-question plan from the approved GDD,
  Systems Map, Combat contract, Progression preset contract, and Content
  Authoring role model.
- **2026-08-24:** Resolved IE-01 through IE-03, establishing inventory identity,
  atomic complete presets, and extensible song-role compatibility.
- **2026-08-24:** Resolved IE-04 through IE-06, establishing fixed item power,
  modifier allowlists, final validation, and immutable loadout snapshots.
- **2026-08-24:** Resolved IE-07 through IE-09, establishing prepared reservation,
  atomic consumption/refunds, and separate stat-free cosmetics.
- **2026-08-24:** Resolved IE-10 through IE-12 and reconciled all twelve answers
  into canonical `ITEMS_AND_EQUIPMENT.md`.
