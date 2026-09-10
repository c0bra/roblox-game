# Bands Battle Builds and Specialization

- **Status:** Approved
- **Approved:** 2026-08-30
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#62-builds--specialization)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Progression dependency:** [`PROGRESSION.md`](PROGRESSION.md)
- **Items/Loadout dependency:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)
- **Abilities dependency:** [`ABILITIES_AND_COOPERATIVE_ACTIONS.md`](ABILITIES_AND_COOPERATIVE_ACTIONS.md)
- **Player-language contract:** [`NAMING_AND_TONE.md`](NAMING_AND_TONE.md)
- **Decision source:** [`BUILDS_AND_SPECIALIZATION_WORKING.md`](BUILDS_AND_SPECIALIZATION_WORKING.md)
- **Interview plan:** [`BUILDS_AND_SPECIALIZATION_QUESTIONS.md`](BUILDS_AND_SPECIALIZATION_QUESTIONS.md)

## 1. Role and authority

Builds owns the **Battle Style** system: functional **Focus** and option catalogs;
one-**Anthem**/three-**Riff** configuration; beginner Loadout templates and
advanced-editor behavior; unlock consumption; free edit/respec; typed hook
resolution; power budgets, caps, stacking, incompatibility, and synergy;
resolved build modifiers; validation; and option/template lifecycle/migration.

It does not own gear, base abilities, Progression awards, the save/apply system
for the three saved Loadouts, gameplay-domain state, economy transactions, raw Rhythm
judgment, persistence implementation, presentation, or player-language rules. It
supplies exact configuration and typed modifier/effect facts to those owners.

## 2. Governing invariants

1. **One Anthem plus three Riffs:** all four slots are required; the same exact
   option cannot repeat.
2. **Universal mixing:** Focuses and options cross instruments; no class,
   pure-role requirement, duplicate-role penalty, or party-composition gate.
3. **One Loadout system:** Battle Style is part of the Current Loadout and each
   saved Loadout, never a second Loadout surface.
4. **Behavior over raw power:** gear carries most direct growth; Battle Styles emphasize
   conditions, tradeoffs, redistribution, support, and hybrids.
5. **Post-score and event-driven only:** no chart, judgment, timing, control, or
   playable-material authority.
6. **One root, no recursion:** contribution is not copied; Battle Style-derived
   effects do not trigger other Battle Style rules.
7. **Guaranteed bases survive:** Battle Styles cannot erase committed Signature/Call,
   Crescendo minimum, Revive/Last Stand, or other owner-domain reliability.
8. **Movement and position remain universal:** no travel/recovery/charge/
   telegraph/graph/risk-ratio change.
9. **Recovery and rewards are protected:** no added attempt, invulnerability,
   auto-revive, reward/eligibility, matchmaking, paid-value, or slot hook.
10. **Budgets are explicit:** every option, synergy, source, Focus/effect category, and global
    cap has visible deterministic cost and order.
11. **Snapshots do not drift:** final lock freezes exact configuration and
    resolved revisions through downing/disconnection/rejoin.
12. **Names are governed:** Battle Style, Focus, Anthem, Riff, beginner Loadout,
    and future option strings follow `NAMING_AND_TONE.md`; rejected/internal
    labels never leak into shipping/localized player strings.

## 3. Battle Style configuration and universal Focuses

A valid first-release Battle Style has exactly one Anthem (the internal major
rule) and three distinct Riffs (the internal supporting rules). All slots are
filled before Apply/Ready. Options have one primary Focus and may carry cross-
Focus tags:

- **Drive:** offense, Momentum, and dangerous-position play;
- **Guard:** Ward, Defend, mitigation, and Revive support;
- **Unity:** teammate support, Band Calls, Crescendos, and Band Harmony; and
- **Pulse:** Hype, Signature, movement-triggered utility, and hybrids.

Any mixture is legal when hooks/unlocks/budgets/incompatibilities validate. An
option is unavailable for an instrument only when its typed gameplay hook is
genuinely absent, with an exact explanation. No option adds a required control.

The Battle Style is one component of the Current Loadout and each of Items' three
saved Loadouts. Beginner Loadout templates are selectable complete
configurations, not additional durable Loadout slots.

## 4. Option definition contract

Every immutable option revision declares:

- stable identity, Anthem/Riff slot (internal major/supporting type), primary
  Focus/tags, and Progression unlock reference;
- trigger root event/source, conditions, eligible owner/target/runtime states,
  and owning domain;
- hook stage, typed effect, target/distribution/fallback, duration/cooldown, and
  once-per-source behavior;
- power-budget cost/category, per-effect/category/global caps, stacking, synergy,
  incompatibility, balance revision, and deterministic order;
- plain-language behavior/tradeoff and exact-value fields; and
- trigger/inactive/fallback reason and multimodal cue keys.

Runtime facts add attempt/snapshot/player, option revision, causal root event,
target, logical time, and idempotency identity.

Options issue typed requests rather than mutating domain state. The owning
domain revalidates and applies/rejects. Invalid/unavailable conditions create no
effect plus an identified reason, never a substitute behavior/target/value.

## 5. Beginner Loadouts, unlocks, and free respec

Onboarding initially offers three complete templates: **Break Through** for
offense, **Guard the Band** for protection/support, and **Ready for Anything**
for a balanced hybrid. Ready for Anything is the safe default when selection is
skipped.

Before advanced-editor unlock, players switch among unlocked templates freely.
Templates use universal options and never bind instrument or irreversible path.

Order Rank opens the editor and all three saved Loadouts together.
Order Rank and Boss Mastery unlock visible options with exact earnable
sources. No functional option is purchase-only.

After unlock, templates remain optional one-click starting points. Preview and
explicit confirmation precede replacement of a customized configuration; other
saved Loadouts remain unchanged.

Edit/template application/respec is free in hub or unlocked staging. A change to
the current staging Loadout clears Ready. Final lock freezes the attempt; edits
affect future snapshots only. Unlocks are account-wide and reusable across
saved Loadouts.

## 6. Anthem (major-rule) contract

The one Anthem is a substantial bounded transformation, never an unconditional
flat-stat upgrade. It declares a meaningful condition, tradeoff, redistribution,
or changed use pattern.

Legal examples include exchanging personal Defend value for shared protection,
emphasizing Attack from dangerous play, changing approved Call distribution, or
changing an allowlisted use of Hype/Signature bonus.

An Anthem splits/redirects only inside one declared total budget and cannot copy a
primary route or re-enter normalization. It preserves committed Signature/Call
bases, Crescendo minimum, Revive/Last Stand reliability, and other guarantees,
adjusting only allowlisted bonus/distribution hooks.

It resolves at most once per causal source. Roster/difficulty may affect declared
targets/caps, not definition or composition. Invalid conditions/targets use the
deterministic baseline-preserving fallback. Trigger/tradeoff/transformation/
fallback/cap receive multimodal feedback; no control is added.

## 7. Riffs (supporting rules) and recursion barrier

Riffs create smaller conditional effects from approved normalized or
semantic roots: contribution, intent change, committed attack/impact, Ward
threshold, completed movement/settling, position, group commit/result, Hype, or
Revive state.

Each processes one root identity once. Battle-Style-derived events are non-triggering.
Declared synergy observes the common original root and snapshot rather than
another option's derived effect.

Rules resolve at declared stages; category/order then stable option identity
orders one stage. Same additive category values add; independently budgeted
multiplicative categories apply once.

Ordinary zero performance stays zero. Separately budgeted non-performance utility
requires a real authoritative event and is not instrument contribution.

Invalid/suppressed/downed/disconnected/unavailable/cooling-down rules omit the
effect with a reason. Missed triggers are not banked/caught up. Musical durations/
cooldowns use the shared clock; nonmusical contexts use server time.

## 8. Power budget, caps, synergy, and incompatibility

First release uses a 100-unit abstract envelope:

- major: at most 40 units;
- support 1: at most 20;
- support 2: at most 20; and
- support 3: at most 20.

Unused capacity does not transfer. Option conditional maximums fit their slot.
Synergies are charged inside participating envelopes and initially cap around
15% of total Battle Style value; they add no unbudgeted layer.

Gear retains a separate budget. Gear/Battle Style/ability sources in one hook category
add together before the shared category/global cap; source order gives no
preference. Independently budgeted categories multiply only at fixed stages.

The editor shows exact resolved values, sources, unused budget, synergy, and cap
saturation. Explicit symmetric incompatibility tags block contradictory/unique
combinations and report every issue without dropping/replacing/weakening choices.
Every other cross-category mixture remains legal. Testing rejects mandatory
options/configurations.

## 9. Combat and Survival hook allowlist

Builds may conditionally affect post-normalized Attack conversion and legal
Momentum handling; Defend mitigation; Ward reinforcement/restoration;
deterministic teammate protection; authentic Revive contribution; returned
Ward; and re-entry protection within typed caps.

A major may include a snapshotted budgeted maximum-Ward tradeoff. Supporting
rules cannot grant unconditional maximum-Ward/direct-stat power.

Revive still requires authentic exclusively redirected performance. Battle Styles
cannot auto-revive, use zero/fabricated contribution, remove opportunity cost, or
bypass target/Activity Map constraints.

Last Stand cannot gain/preserve/refund attempts, bypass/shorten challenge,
fabricate input, prevent failure/downing, or create invulnerability. Only
allowlisted returned Ward/protection values may change inside caps.

One contribution has one route; monotonicity/zero rules hold. Builds cannot
affect unopened Resolve, layer sequencing, finishing, committed attack target/
geometry/time, outcome, guaranteed bases, friendly-fire prohibition, or Survival
authority.

Multi-target protection uses fixed-total division or roster-aware capped per-
recipient value with deterministic eligibility/fallback and no manual target
control.

## 10. Position, Hype, Signature, and group hooks

Builds may observe settled position/risk, authentic dangerous performance, and
completed movement/settling to modify a later effect or request separately
budgeted visible utility.

They cannot change travel/dash/recovery/settling, graph/legality/occupancy,
cover/hazard/telegraph geometry, or universal position Attack/danger/reward
ratios. Passive occupancy produces no performance/reward.

Hype hooks may adjust slow/fast gain or Signature positive-bonus allocation, not
charge count/overflow/auto-fire/state/consumption/refund/reset/Special routing or
guaranteed base/timing.

Band Call hooks may alter personal readiness/share/potency or typed distribution
inside budgets, not uses, reservation, lockout floor, candidates, participation,
or base. Crescendo hooks affect only bounded authentic personal share or separate
legal target effect, not activation/candidate/preview/eligibility/tier thresholds/
Echo/roster representation.

Group/Revive shares remain independent/nonnegative/one-route. Battle Styles never
alter rewards or Order Ally triggers/recipients/functions/shares/suppression/
performance/attribution.

## 11. Immutable runtime resolution

Final lock records configuration/options, definition/balance, budget/synergy/
incompatibility, gear/ability modifiers, category/global caps, and resolved typed
modifier set.

For each root, the owner:

1. validates snapshot and source/target state;
2. computes base conversion/effect;
3. applies major transformation at its stage;
4. combines same-stage gear/Battle Style categories and supporting additions;
5. applies legal position and encounter/target/difficulty stages; and
6. clamps once at the effect cap.

Same categories add; independent categories multiply once. Major precedes
supporting additions at the same stage; stable option ID breaks ties. Network
arrival cannot change musical order.

Committed effects survive downing/disconnect/target/phase change using fallback;
uncommitted triggers stop. Durations/cooldowns continue on the musical clock and
never reset on reconnect. Rejoin restores exact state.

Duplicate instruments do not affect legality/value. Roster/difficulty enter only
declared snapshot inputs. Solo uses human contracts; Order Allies are excluded. All
eligibility/omission/stages/values/caps/fallback/duration/results retain private
root/option attribution.

## 12. Validation and saved Loadout application

The editor may retain incomplete draft state. Apply/Ready requires one unlocked
active major, three distinct unlocked active supports, all hooks, budgets/caps,
and no incompatibility/prohibited effect.

Template/config application to the Current Loadout or one saved Loadout is atomic.
Failure changes nothing and reports every issue. Editing a nonactive saved Loadout does
not clear Ready; a Current Loadout change in staging does.

Items' final lock freezes exact Battle Style/resolved modifier revisions. Duplicate/
stale save/apply/lock/migration commands are idempotent.

## 13. Catalog lifecycle and migration

Ordinary compatible balance updates apply to future snapshots and disclose
material value changes. Behavior-identical revision mapping may migrate
automatically only when semantics/hooks/budget match.

Structural retirement/disablement/replacement makes future configurations
Incomplete until explicit player choice. Retired unlock value maps to an equal-
or-broader replacement unlock without replay/cost, but selection is not silently
substituted.

Active attempts retain locked revision. Only critical integrity/security failure
may invalidate through No Contest, never per-player mid-song mutation. Template
updates affect only future explicit applications and never rewrite saved/custom
configurations.

## 14. Disclosure, accessibility, naming, and privacy

Selection shows trigger/condition, tradeoff, resolved values, budget/caps,
cooldown/duration, targets/distribution, fallback, synergies/incompatibilities,
affected systems, and comparison. Beginner UI provides concise summary plus
expandable details; Advanced exposes full sources/calculation/caps. Drawbacks and
saturation are never hidden.

Runtime uses compact contextual state/trigger/cap/cooldown/fallback cues without
moving rhythm controls or adding buttons/clutter. Repeated inactive notices
summarize.

Critical meaning uses text/icon/shape and optional audio/haptics. Color/sound
alone is insufficient; motion/flash/color/comfort changes presentation only.

Stable internal keys map through `NAMING_AND_TONE.md` to approved localized
Battle Style, Focus, Anthem, Riff, Loadout, and future option strings. Missing or
rejected mappings remain a publication blocker.

Other players see role/Ready/appearance/necessary shared-effect cues, not exact
Battle Style, power, recommendations, unlocks, migration, or evidence.

## 15. Semantic outputs and consumers

Authoritative facts cover draft/edit/template, validation, save/apply/Ready
invalidation, lock, unlock/availability, lifecycle/migration/replacement, runtime
eligibility/trigger/omission, major/support/synergy, source combination,
budget/cap, target/fallback, duration/cooldown/expiry, and effect request/result.

Facts carry player, Loadout/snapshot, option/catalog/balance revision, root event,
attempt/time, source/target, pre/post values, reason, and idempotency.

UI/Audio get presentation-neutral facts; Items gets configuration/modifiers;
domains get typed requests; Player Data stores drafts/configurations/revisions/
migration; Results gets private evidence; Analytics gets privacy-reviewed
option/validation/trigger/cap semantics.

## 16. Catalog, Content Authoring, and verification

Required catalogs define categories/slots, beginner templates, every option
field, unlock, hook schemas, budgets/synergies/caps/incompatibilities, lifecycle/
migration, disclosure/cues, and internal/player-name separation.

Verification covers every option/template/instrument hook, every legal 1+3
combination, representative gear/ability cap edges, all difficulties and rosters
one through six, zero/max performance, invalid targets, down/disconnect/rejoin,
duration/cooldown, and Order Ally exclusion.

Builds are global catalog data. Song packages declare normal roles/hooks and may
not hide Battle Style behavior, overrides, multipliers, bans, or player substitutions.
The completed 2026-09-02
[`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md#14-cross-specification-handoffs-and-reconciliation)
reconciliation preserves that boundary.

## 17. Deferred tuning and technical work

Behavior and the core player-facing hierarchy are complete; these remain
versioned content/balance/architecture work:

- final names/localization for not-yet-authored Anthem/Riff options;
- complete option/template/unlock/migration catalogs under the naming grammar;
- exact values/costs/caps/cooldowns/durations/synergies;
- editor comparison/presentation and runtime cue implementation;
- persistence/migration/concurrency implementation; and
- balance, combination, mandatory-option, accessibility, and telemetry tests.

Tuning cannot add slots/controls, instrument classes, required composition,
unconditional support power, recursion/contribution copies, Rhythm/movement/
recovery-count/reward authority, unbudgeted synergy, Order Ally interaction, silent
substitution, old-revision advantage, or shipping placeholder names.

## 18. Approval and change control

The owner interview resolved BS-01 through BS-12 on 2026-08-30. This document is
the canonical Builds and Specialization design specification.

A material change to configuration shape/categories, Loadout integration, option
contract, major/support behavior, budget/synergy/caps, hook allowlists, runtime
order, validation/migration, disclosure/privacy, or naming gate requires an
explicit amendment citing the superseded rule. Numeric/catalog tuning inside
these boundaries creates a new revision and never changes an active snapshot.
