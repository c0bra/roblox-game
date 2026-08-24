# Bands Battle Items and Equipment Working Record

- **Status:** Complete decision record; 12 of 12 questions reconciled
- **Started:** 2026-08-24
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#61-items-equipment--loadouts)
- **Interview plan:** [`ITEMS_AND_EQUIPMENT_QUESTIONS.md`](ITEMS_AND_EQUIPMENT_QUESTIONS.md)
- **Progression dependency:** [`PROGRESSION.md`](PROGRESSION.md)
- **Combat dependency:** [`COMBAT.md`](COMBAT.md)
- **Canonical result:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)

## 1. Role of this record

This document persists owner decisions while the Items & Equipment interview is
in progress. It is not canonical until reconciled into
`ITEMS_AND_EQUIPMENT.md`.

## 2. Inherited boundary

Items & Equipment owns immutable item/consumable/cosmetic definitions, owned
item instances/stacks/entitlements, quantities, fixed stats/traits/tier/rank,
power/action/consumable/appearance slots, equip and complete-preset validation,
staging/encounter locks, combat inventory restrictions, consumption
authorization, resolved legal equipment modifiers, and mutation application.

It does not own earning/drop/crafting/salvage/upgrade/uplift prices or
transactions, paid receipt validation, ability/build behavior, Combat formulae,
progression eligibility, persistence implementation, or presentation.

## 3. Approved inputs

- Versioned role, item, consumable, cosmetic, ability, build, and effect catalogs.
- Durable player inventory and three complete spec-preset references.
- Progression/campaign/tier/unlock eligibility.
- Song/encounter supported-role and exact content revision.
- Multiplayer staging/final-lock state.
- Reward/Economy/Commerce identified transaction results.

## 4. Decision record

### Checkpoint A — Identity, ownership, and complete loadouts

#### IE-01 — Definition, instance, stack, and entitlement identity

- **Status:** Approved.
- A catalog **definition** has stable identity and immutable versioned semantics:
  slot/type, tier variants, fixed primary stat, fixed trait, allowed modifiers,
  role/cosmetic references, compatibility tags, and presentation references.
- An owned power **instance** has a globally unique player-inventory identity and
  references one definition plus current tier, upgrade rank, acquisition source/
  transaction lineage, and migration state. Final stats/traits derive from the
  immutable definition/tier/rank and are not arbitrary stored rolls.
- Every dropped/crafted/purchased power item creates a distinct instance even
  when another instance references the same definition. A duplicate has no
  hidden stronger roll and may later be salvaged through Rewards/Economy.
- A consumable **stack** is one per player/consumable definition and stores
  quantity plus reservation/mutation evidence. Individual uses are not permanent
  item instances.
- A cosmetic **entitlement** permanently records access to a definition. It does
  not need duplicate owned instances and cannot carry power stats.
- First release imposes no hard gear-instance inventory cap, expiring mailbox,
  deletion pressure, or purchasable storage capacity. Filtering/sorting and
  voluntary salvage handle duplicates without coercion.
- Random affixes, stat ranges, quality rolls, and reroll seeds do not exist. The
  same definition/tier/rank has the same functional values for every player.

#### IE-02 — Loadout slots and atomic spec-preset application

- **Status:** Approved.
- A complete loadout contains exactly one owned instance in each required power
  slot: Instrument, Ward Core, and Resonator.
- It references exactly one unlocked Signature Special and one unlocked Band
  Call. Two prepared consumable-type slots are optional and may be empty. The
  complete spec preset also references role/instrument and major/supporting
  specialization configuration.
- References do not copy their targets. The same owned instance, unlock, or
  consumable stack remains authoritative when several presets point to it.
- Applying a preset in the hub validates the complete global configuration.
  Applying it for a selected encounter in staging additionally validates role/
  song, boss/difficulty, ability/build, and consumable eligibility.
- Preset application is atomic. If any required instance is missing, unequippable,
  locked, retired/disabled without migration, incompatible, or otherwise invalid,
  no slot changes; the current loadout remains intact and every repair issue is
  shown together.
- An invalid preset stays saved and explicitly Incomplete rather than silently
  dropping slots, selecting replacements, changing role, or consuming anything.
- A consumable reference may remain saved with zero owned quantity, but it is
  visibly Empty and supplies no encounter charges; IE-07 will finalize slot and
  reservation behavior.
- Switching is allowed in the hub and in pre-battle staging while individual
  loadouts remain unlocked. The validated snapshot freezes at Multiplayer's
  final lock and cannot change during deployment, combat, disconnect, or rejoin.

#### IE-03 — Instrument/role and song compatibility

- **Status:** Approved.
- A global extensible role catalog assigns stable identity and capabilities to
  playable musical roles. Initial drums, vocals, guitar, and bass categories are
  examples rather than a closed or mandatory list; piano, synthesizer,
  percussion, strings, and later authentic roles are valid.
- Each Instrument definition references exactly one playable role identity plus
  its instrument visual/audio family and allowed cosmetic family. Multiple fixed
  item variants for one role may emphasize offense, Ward, support, or hybrids.
- An approved song package declares only roles supported by authentic playable
  chart material and controllable audio/equivalent mapping. Instrumental songs
  may omit vocals; sparse/atmospheric/absent parts may be unavailable.
- Staging validates the equipped Instrument role against the exact encounter
  content revision. Unsupported means encounter-incompatible: no fabricated
  chart, universal replacement part, silent substitute, or automatic preset
  edit occurs.
- The preset and owned Instrument remain globally valid for other songs. UI
  identifies the incompatible role and offers explicit player-controlled repair.
- Multiple humans may equip the same role and Instrument definition/variant.
  Duplicate roles never conflict or change normalization.
- Instrument role never fixes Attack/Defend/Special or party class. Every role
  can have offensive, defensive, support, and hybrid variants. A Signature or
  other option may have an explicit thematic restriction only when alternative
  valid compositions remain and no encounter requires that pairing.
- Items exposes role/visual/audio references; Content Authoring owns chart/audio
  availability, Audio Presentation owns mix behavior, and Abilities/Builds own
  their definitions.

### Checkpoint B — Stats, traits, and combat resolution

#### IE-04 — Tier, upgrade rank, primary stat, and trait

- **Status:** Approved.
- Each immutable item definition contains the exact functional table for every
  supported campaign/item tier and upgrade rank. First release targets a base
  rank plus roughly three guaranteed upgrade ranks per tier.
- The owned instance stores only definition identity, current tier, current rank,
  lineage, and mutation state. Primary-stat value, trait parameters, comparison
  facts, and final modifiers derive from the exact definition/catalog revision.
- Each first-release power item presents one primary stat and one readable fixed
  trait. Internal effect parameters needed to implement that trait do not become
  random secondary affixes or hidden quality rolls.
- Upgrade preview shows exact pre/post tier, rank, primary value, trait behavior,
  cost handoff, and resulting compatibility before confirmation. An approved
  upgrade transaction never fails, lowers value, destroys the instance, or
  selects a random result.
- Uplift preserves stable instance lineage, item identity, fixed trait, and
  appearance entitlement while moving one tier and resetting to that tier's base
  rank under `PROGRESSION.md`.
- Starter, drop, craft, mastery-granted, earned-store-equivalent, and paid-
  equivalent instances use the same definition/tier/rank derivation. Equal
  function means exact stat budget and trait effect, not merely a similar label.
- Item comparison uses these exact facts and selected loadout context. UI may
  summarize tradeoffs but cannot invent an overall quality roll.

#### IE-05 — Modifier allowlist and resolution

- **Status:** Approved.
- Every functional definition declares typed modifier/effect records compatible
  with `COMBAT.md`: source item/slot, affected effect tags, authoritative
  condition, legal pipeline stage/category, power-budget cost, cap/stacking,
  duration, and source attribution.
- Slot allowlists are:
  - **Instrument:** selected role plus permitted Attack, Defend, support,
    readiness, or hybrid emphasis/trait hooks.
  - **Ward Core:** maximum Ward, Defend conversion, threat mitigation,
    reinforcement, restoration, or bounded recovery-received hooks.
  - **Resonator:** Attack conversion, Hype generation, Signature potency,
    support/group effects, or Band Call readiness/potency hooks.
- A definition cannot modify charts, judgments, timing, calibration, Hold Assist,
  movement, telegraphs, recovery count, invulnerability, autoplay, positional
  baseline ratios, reward eligibility, or another prohibited domain.
- Contribution-derived item traits obey Combat monotonicity and zero-performance
  rules. They cannot turn zero normalized contribution into an ordinary effect,
  copy the full value, or recursively re-enter the pipeline.
- Static effects such as maximum Ward or an explicitly event-triggered utility
  require a separately budgeted allowlisted hook and authoritative event. The
  general trait field is not permission for arbitrary scripting.
- Catalog/build-time validation rejects unknown tags, illegal stages, over-budget
  definitions, recursive graphs, incompatible caps, and prohibited effects.
  Runtime does not make an illegal item safe by silently clamping or ignoring
  part of it.
- Later item categories, traits, sockets, or sets must emit through the same
  typed contract unless an explicit design amendment grants a new hook.

#### IE-06 — Loadout validation and encounter snapshot

- **Status:** Approved.
- Final validation checks:
  - immutable item/role/ability/build/consumable catalog and balance revisions;
  - exact instance ownership, current tier/rank, enabled/mutation state, and slot;
  - no destroyed/salvaged/replaced or duplicate slot use of one unique instance;
  - campaign/tier/system/ability/build unlock eligibility;
  - selected song's role/chart/audio support;
  - Signature, Band Call, and specialization compatibility;
  - prepared consumable reference, quantity/reservation, and encounter limits;
    and
  - every emitted modifier against its allowlist, category budget, and caps.
- Validation reports all actionable issues together. A player is not marked
  Ready and cannot be deployed with a missing/invalid required slot, unsupported
  role, illegal modifier, or incompatible locked configuration.
- Before final lock, hub/staging changes produce a newly validated candidate
  snapshot. They do not mutate a snapshot already handed to an attempt.
- At Multiplayer's final lock, Items emits one immutable identified loadout
  snapshot containing exact instance/definition/tier/rank, role, action and
  consumable references, build revision, resolved typed modifiers, cosmetic
  references, and reservation facts.
- Catalog/balance/inventory updates after lock apply only to future snapshots.
  The active attempt uses the bound versions and cannot hot-swap, salvage,
  upgrade, equip, or consume from the full inventory outside its prepared slots.
- Disconnect/rejoin restores the same attempt snapshot and remaining committed
  encounter consumable charges; it never reloads the player's newer hub loadout.

### Checkpoint C — Consumables and cosmetics

#### IE-07 — Prepared consumable slots and charges

- **Status:** Approved.
- A loadout has two optional prepared consumable-type slots. Both may be empty;
  when filled they must reference two different consumable definitions so one
  type cannot double its per-encounter cap by occupying both slots.
- Each consumable definition has one player stack, stable identity, versioned
  eligibility, maximum owned quantity if any, and per-encounter charge cap.
- At final loadout lock, each filled slot atomically reserves `min(owned
  unreserved quantity, per-encounter cap)`. Reservation makes those units
  unavailable to another transaction but does not consume them.
- A quantity below cap enters with that smaller visible charge count. Zero
  quantity leaves the remembered slot visibly Empty and supplies no use; because
  consumable slots are optional, this does not invalidate otherwise complete
  required gear.
- A full spec preset remembers consumable definition references, not charges or
  copied quantity. Every new staging lock performs a fresh reservation from the
  current owned stacks.
- Used reserved units transition to Consumed under IE-08. Every unused reservation
  releases after Victory, Defeat, canceled deployment, safe attempt teardown, or
  resolved compensation.
- Gear, builds, difficulty, paid items, and other loadout elements cannot add a
  third slot, equip the same type twice, increase the definition's encounter
  charge cap, or access unprepared stacks during combat. They may affect potency
  only through an explicitly legal post-score/effect hook.
- Combat UI exposes only the two prepared types and remaining committed charges.
  The full inventory cannot open or substitute items during an attempt.

#### IE-08 — Consumption authorization and recovery

- **Status:** Approved.
- Every use has stable request identity and validates encounter/loadout snapshot,
  prepared slot/type, remaining reserved charge, current player/target/effect
  eligibility, cooldown/lockout, and definition/balance revision.
- A valid request may enter a short queued/pending state if its definition needs
  a musical/effect boundary. It becomes **Committed** only when the owning effect
  system guarantees execution.
- Commit atomically converts one exact reserved unit to durably Consumed and
  publishes the identified effect request. The same request cannot decrement or
  apply twice.
- Invalid target/state, unavailable effect, no charge, duplicate input, canceled
  queue, or interruption before Commit produces no consumption. Feedback
  explains the failure without auto-queuing another use.
- After Commit, ordinary player downing, disconnect, movement, intent change,
  Victory, or Defeat does not refund the unit. The committed effect resolves or
  follows its definition's post-Commit behavior.
- If an authoritative system fault commits quantity but prevents the effect from
  entering guaranteed resolution, one refund keyed to the original request
  restores exactly that unit.
- An Invalid / No Contest attempt creates one idempotent compensation transaction
  that refunds every consumable unit committed by that player in the attempt.
  It cannot refund an unspent reservation twice or create quantity beyond the
  attempt's committed lineage.
- Disconnect/rejoin restores remaining snapshot charges and all prior committed
  use identities. It never reloads quantity from the hub stack.
- Items owns reservation/consumption/refund state; the consumable's owning effect
  system owns targeting/behavior; Rewards/Economy/Player Data later defines the
  durable transaction implementation and operational retry.

#### IE-09 — Appearance ownership and equipment

- **Status:** Approved.
- Appearance uses permanent entitlements rather than duplicate power instances.
  Initial compatible categories include stagewear, Instrument finish/skin,
  aura/performance effect, and title, with extensible later cosmetic categories.
- An entitlement may be equipped only in its declared appearance category and
  compatibility family. It contains no stat, trait, modifier, hidden power,
  reward multiplier, hitbox, or gameplay tag.
- A duplicate entitlement award is reported to Rewards/Economy for its declared
  compensation/salvage-equivalent result; it never creates a second cosmetic
  instance or a stronger cosmetic.
- Combat spec presets contain gameplay configuration only and do not change
  appearance. A later separate appearance-preset feature may be added without
  changing combat presets.
- If the equipped finish/effect is incompatible with a newly selected Instrument
  or becomes unavailable, presentation temporarily uses the owned/default safe
  compatible appearance. The entitlement and explicit equipped preference remain
  so it can reappear when compatible; no substitute purchase occurs.
- Local flash, particle, bloom, motion, color-vision, and comfort settings may
  reduce or hide cosmetic effects without unequipping them for ownership. Other
  players receive only safe replicated appearance within platform/performance
  policy.
- Cosmetics must pass catalog/platform moderation, target-age standards, phone-
  scale performance budgets, and critical-cue occlusion tests. They cannot alter
  silhouette recognition needed for targeting, obscure notes/boss telegraphs,
  change animation timing, affect camera, or imitate gameplay/reward state.

### Checkpoint D — Mutations, evolution, and outputs

#### IE-10 — Acquisition and item mutation handoff

- **Status:** Approved.
- Items applies only an authoritative identified outcome from Rewards/Economy or
  Commerce. It does not decide drop, price, cost, receipt, or eligibility.
- Every transaction/result has stable idempotency identity, player, source,
  catalog/balance revisions, exact requested mutation, and expected precondition.
- Application semantics are:
  - a power-item grant/craft creates one new unique instance referencing the
    exact definition/tier/base rank and source lineage;
  - a consumable grant/spend/refund changes one stack by the exact authorized
    quantity;
  - a cosmetic grant creates one permanent entitlement or returns Duplicate for
    Economy compensation;
  - an upgrade/uplift mutates tier/rank of the same instance and preserves
    identity/lineage/references; and
  - salvage removes exactly one selected instance and emits its authoritative
    removed identity to Economy.
- The same result applies once. A duplicate returns the committed result without
  creating another instance, quantity, entitlement, rank, or salvage value.
- An instance in an active locked encounter snapshot cannot be upgraded,
  uplifted, salvaged, disabled by ordinary catalog change, or otherwise mutated
  until the snapshot releases.
- Salvaging a hub-equipped or preset-referenced instance requires explicit
  confirmation listing every affected loadout/preset. On commit, those exact
  references remain as visible Missing/Incomplete repair evidence rather than
  silently selecting another owned item.
- Upgrade/uplift preserves valid equipped/preset references because instance
  identity remains. If the new revision creates a compatibility issue, the
  transaction preview must disclose it before commit and resulting validation
  marks the affected configuration explicitly.
- Paid and earned grants enter the same instance/mutation model and functional
  equivalence checks. Purchase lineage does not grant different upgrade,
  salvage, preset, or tier behavior.
- Atomic application either commits the complete mutation/audit record or
  commits nothing. Retried recovery uses the same idempotency key and cannot
  compensate twice.

#### IE-11 — Catalog versioning, retirement, and extension

- **Status:** Approved.
- A stable definition identity may have immutable revisions. Every revision
  records exact semantics, compatibility, balance/effect references, assets,
  predecessor/migration, and availability state.
- Existing instances do not silently change in an active encounter. A compatible
  catalog/balance update applies through one explicit idempotent instance-
  definition migration before a future loadout snapshot.
- Migration preserves owned instance identity, tier, rank where compatible,
  acquisition lineage, appearance entitlement, equipped/preset references, and
  functional-equivalence obligations. It reports material player-facing changes.
- A retired definition may remain owned/usable when safe. If an exploit,
  moderation issue, missing dependency, or removed hook requires disabling it,
  the player receives clear status and an explicit functional-equivalent
  migration or fair compensation. Items are never silently deleted or replaced
  with an unrelated object.
- Presets keep the stable reference through compatible migration. An incompatible
  retirement leaves a visible repair issue until the approved equivalent is
  applied or the player edits it.
- Future item families, advanced traits, sockets, set interactions, sidegrades,
  or additional slots must use typed Combat/Ability/Build hooks and shared power
  budgets. They cannot add mandatory combat buttons, rhythm authority, paid
  storage pressure, random-affix rerolls, or automatic obsolescence.
- Early items remain valid within their original tier and may gain only explicit
  compatible uplift/extension paths. New content can broaden choices without
  erasing old identity or granting old items free current-tier scaling.

#### IE-12 — Semantic outputs and persistence contract

- **Status:** Approved.
- Items & Equipment emits causally linked facts for:
  - definition/revision availability and migration;
  - item-instance grant/create/duplicate/equip/unequip/upgrade/uplift/salvage/
    disable/replace;
  - consumable-stack grant/reserve/release/commit/refund/quantity;
  - cosmetic-entitlement grant/duplicate/equip/fallback;
  - full loadout and spec-preset create/edit/validate/apply/incomplete/repair;
  - role/song incompatibility and all validation findings;
  - final immutable encounter snapshot and release; and
  - authoritative transaction success/reject/duplicate/retry/rollback evidence.
- Events carry player, definition/instance/stack/entitlement/preset/snapshot,
  exact catalog/content/balance revisions, source transaction/result, causal
  encounter/use, pre/post state, modifier/compatibility evidence, and stable
  idempotency identity as applicable.
- Player Data durably stores item instances and lineage, stack quantities and
  reservations, entitlements/equipped appearance, current loadout, three preset
  references/names/validation state, mutation/migration state, transaction
  application history, and locked snapshot references.
- UI and the owning player may inspect complete inventory/loadout/preset facts.
  Multiplayer receives readiness, equipped role, and permitted loadout summary;
  Combat/Abilities/Audio consume the locked functional snapshot; Commerce/
  Rewards consume transaction results; Analytics observes non-sensitive semantic
  events.
- Other players may see the equipped role/instrument and safe replicated
  appearance where gameplay/social presentation needs them. They never receive
  full inventory, currencies/quantities, acquisition/purchase history, private
  recommendation, incomplete preset details, or internal migration findings.

## 5. Content Authoring reconciliation register

- Every approved song package must reference stable global role identities and
  provide explicit playable/unavailable status plus compatible chart/audio
  mappings for its offered roles.
- Runtime/package validation must reject an unknown/retired role identity and
  never infer compatibility from an item name or conventional instrument list.

## 6. Open handoffs

- `REWARDS_AND_ECONOMY.md` owns grants, drops, crafting, salvage, upgrades,
  uplift resource spending, and Commerce receipts/compensation.
- `PROGRESSION.md` owns option/tier/recipe eligibility and the three complete
  spec-preset access rule.
- `BUILDS_AND_SPECIALIZATION.md` owns specialization configuration and its
  compatibility inside a full preset.
- `ABILITIES_AND_COOPERATIVE_ACTIONS.md` owns Signature/Band Call definitions and
  effect behavior referenced by loadouts.
- `MULTIPLAYER.md` owns staging/queue/final lock and rejoin session state.
- `PLAYER_DATA.md` owns durable inventory/preset persistence, atomic concurrency,
  migration, and recovery.
- UI, Audio, Results, Commerce presentation, and Analytics consume semantic item
  facts without owning them.

## 7. Change log

- **2026-08-24:** Created the working record. Progress is 0 of 12 questions.
- **2026-08-24:** Approved IE-01 through IE-03. Progress is 3 of 12 questions.
  Established item/stack/entitlement identity, atomic presets, and role support.
- **2026-08-24:** Approved IE-04 through IE-06. Progress is 6 of 12 questions.
  Established fixed power derivation, modifier allowlists, and loadout snapshots.
- **2026-08-24:** Approved IE-07 through IE-09. Progress is 9 of 12 questions.
  Established consumable reservation/commit/refund and cosmetic ownership.
- **2026-08-24:** Approved IE-10 through IE-12 and reconciled all twelve
  decisions into canonical `ITEMS_AND_EQUIPMENT.md`.
