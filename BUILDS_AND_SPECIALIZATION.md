# Bands Battle Builds and Specialization

- **Status:** Approved
- **Approved:** 2026-08-30
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#62-builds--specialization)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Progression dependency:** [`PROGRESSION.md`](PROGRESSION.md)
- **Items/preset dependency:** [`ITEMS_AND_EQUIPMENT.md`](ITEMS_AND_EQUIPMENT.md)
- **Abilities dependency:** [`ABILITIES_AND_COOPERATIVE_ACTIONS.md`](ABILITIES_AND_COOPERATIVE_ACTIONS.md)
- **Decision source:** [`BUILDS_AND_SPECIALIZATION_WORKING.md`](BUILDS_AND_SPECIALIZATION_WORKING.md)
- **Interview plan:** [`BUILDS_AND_SPECIALIZATION_QUESTIONS.md`](BUILDS_AND_SPECIALIZATION_QUESTIONS.md)

## 1. Role and authority

Builds owns functional category and option catalogs; one-major/three-supporting
configuration; beginner templates and advanced-editor behavior; unlock
consumption; free edit/respec; typed hook resolution; power budgets, caps,
stacking, incompatibility, and synergy; resolved build modifiers; validation;
and option/template lifecycle and migration.

It does not own gear, base abilities, Progression awards, the three full spec-
preset save/apply system, gameplay-domain state, economy transactions, raw Rhythm
judgment, persistence implementation, presentation, or final naming. It supplies
exact configuration and typed modifier/effect facts to those owners.

## 2. Governing invariants

1. **One major plus three supports:** all four slots are required; the same exact
   option cannot repeat.
2. **Universal mixing:** categories and options cross instruments; no class,
   pure-role requirement, duplicate-role penalty, or party-composition gate.
3. **One preset system:** build configuration is part of the current loadout and
   each existing full spec preset, never a second saved-preset surface.
4. **Behavior over raw power:** gear carries most direct growth; builds emphasize
   conditions, tradeoffs, redistribution, support, and hybrids.
5. **Post-score and event-driven only:** no chart, judgment, timing, control, or
   playable-material authority.
6. **One root, no recursion:** contribution is not copied; build-derived effects
   do not trigger build rules.
7. **Guaranteed bases survive:** builds cannot erase committed Signature/Call,
   Crescendo minimum, revival/recovery, or other owner-domain reliability.
8. **Movement and position remain universal:** no travel/recovery/charge/
   telegraph/graph/risk-ratio change.
9. **Recovery and rewards are protected:** no added attempt, invulnerability,
   auto-revive, reward/eligibility, matchmaking, paid-value, or slot hook.
10. **Budgets are explicit:** every option, synergy, source, category, and global
    cap has visible deterministic cost and order.
11. **Snapshots do not drift:** final lock freezes exact configuration and
    resolved revisions through downing/disconnection/rejoin.
12. **Names are not approved:** internal labels never leak into shipping/localized
    player strings before the naming/tone pass.

## 3. Configuration and universal categories

A valid first-release configuration has exactly one major rule and three
distinct supporting rules. All slots are filled before Apply/Ready. Options have
one primary internal category and may carry cross-category tags:

- offense, Momentum, and dangerous-position play;
- Ward, Defend, mitigation, and revival support;
- teammate support, Band Calls, Crescendos, and Cohesion interactions; and
- Hype, Signature, movement-triggered utility, and hybrids.

Any mixture is legal when hooks/unlocks/budgets/incompatibilities validate. An
option is unavailable for an instrument only when its typed gameplay hook is
genuinely absent, with an exact explanation. No option adds a required control.

The configuration is one component of the current loadout and each of Items'
three full spec presets. Beginner templates are selectable complete
configurations, not additional durable preset slots.

All category/system/slot/template/option labels remain internal pending the
dedicated naming/tone pass.

## 4. Option definition contract

Every immutable option revision declares:

- stable identity, major/supporting slot, primary category/tags, and Progression
  unlock reference;
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

## 5. Beginner templates, unlocks, and free respec

Onboarding initially offers three complete working templates: offense-focused,
protection/support-focused, and balanced hybrid. These names are unapproved;
Balanced is the safe default when selection is skipped.

Before advanced-editor unlock, players switch among unlocked templates freely.
Templates use universal options and never bind instrument or irreversible path.

General Progression opens the editor and all three full spec presets together.
General Progression and Boss Mastery unlock visible options with exact earnable
sources. No functional option is purchase-only.

After unlock, templates remain optional one-click starting points. Preview and
explicit confirmation precede replacement of a customized configuration; other
presets remain unchanged.

Edit/template application/respec is free in hub or unlocked staging. A change to
the current staging loadout clears Ready. Final lock freezes the attempt; edits
affect future snapshots only. Unlocks are account-wide and reusable across
presets.

## 6. Major-rule contract

The one major is a substantial bounded transformation, never an unconditional
flat-stat upgrade. It declares a meaningful condition, tradeoff, redistribution,
or changed use pattern.

Legal examples include exchanging personal Defend value for shared protection,
emphasizing Attack from dangerous play, changing approved Call distribution, or
changing an allowlisted use of Hype/Signature bonus.

A major splits/redirects only inside one declared total budget and cannot copy a
primary route or re-enter normalization. It preserves committed Signature/Call
bases, Crescendo minimum, revival/recovery reliability, and other guarantees,
adjusting only allowlisted bonus/distribution hooks.

It resolves at most once per causal source. Roster/difficulty may affect declared
targets/caps, not definition or composition. Invalid conditions/targets use the
deterministic baseline-preserving fallback. Trigger/tradeoff/transformation/
fallback/cap receive multimodal feedback; no control is added.

## 7. Supporting rules and recursion barrier

Supporting rules create smaller conditional effects from approved normalized or
semantic roots: contribution, intent change, committed attack/impact, Ward
threshold, completed movement/settling, position, group commit/result, Hype, or
revival state.

Each processes one root identity once. Build-derived events are non-triggering.
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
15% of total build value; they add no unbudgeted layer.

Gear retains a separate budget. Gear/build/ability sources in one hook category
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
deterministic teammate protection; authentic revival contribution; returned
Ward; and re-entry protection within typed caps.

A major may include a snapshotted budgeted maximum-Ward tradeoff. Supporting
rules cannot grant unconditional maximum-Ward/direct-stat power.

Revival still requires authentic exclusively redirected performance. Builds
cannot auto-revive, use zero/fabricated contribution, remove opportunity cost, or
bypass target/Activity Map constraints.

Solo recovery cannot gain/preserve/refund attempts, bypass/shorten challenge,
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

Group/revival shares remain independent/nonnegative/one-route. Builds never
alter rewards or acolyte triggers/recipients/functions/shares/suppression/
performance/attribution.

## 11. Immutable runtime resolution

Final lock records configuration/options, definition/balance, budget/synergy/
incompatibility, gear/ability modifiers, category/global caps, and resolved typed
modifier set.

For each root, the owner:

1. validates snapshot and source/target state;
2. computes base conversion/effect;
3. applies major transformation at its stage;
4. combines same-stage gear/build categories and supporting additions;
5. applies legal position and encounter/target/difficulty stages; and
6. clamps once at the effect cap.

Same categories add; independent categories multiply once. Major precedes
supporting additions at the same stage; stable option ID breaks ties. Network
arrival cannot change musical order.

Committed effects survive downing/disconnect/target/phase change using fallback;
uncommitted triggers stop. Durations/cooldowns continue on the musical clock and
never reset on reconnect. Rejoin restores exact state.

Duplicate instruments do not affect legality/value. Roster/difficulty enter only
declared snapshot inputs. Solo uses human contracts; acolytes are excluded. All
eligibility/omission/stages/values/caps/fallback/duration/results retain private
root/option attribution.

## 12. Validation and full spec-preset application

The editor may retain incomplete draft state. Apply/Ready requires one unlocked
active major, three distinct unlocked active supports, all hooks, budgets/caps,
and no incompatibility/prohibited effect.

Template/config application to current loadout or one full spec preset is atomic.
Failure changes nothing and reports every issue. Editing a nonactive preset does
not clear Ready; a current staging-loadout change does.

Items' final lock freezes exact build/resolved modifier revisions. Duplicate/
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

All current labels are internal keys. Approved localized player strings from the
naming/tone pass are a publication gate.

Other players see role/Ready/appearance/necessary shared-effect cues, not exact
build, power, recommendations, unlocks, migration, or evidence.

## 15. Semantic outputs and consumers

Authoritative facts cover draft/edit/template, validation, save/apply/Ready
invalidation, lock, unlock/availability, lifecycle/migration/replacement, runtime
eligibility/trigger/omission, major/support/synergy, source combination,
budget/cap, target/fallback, duration/cooldown/expiry, and effect request/result.

Facts carry player, preset/snapshot, option/catalog/balance revision, root event,
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
duration/cooldown, and acolyte exclusion.

Builds are global catalog data. Song packages declare normal roles/hooks and may
not hide build behavior, overrides, multipliers, bans, or player substitutions.
`CONTENT_AUTHORING.md` must preserve that boundary during final reconciliation.

## 17. Deferred tuning and technical work

Behavior is complete; these remain versioned design/balance/architecture work:

- final player-facing system/category/slot/template/option names and localization;
- complete option/template/unlock/migration catalogs;
- exact values/costs/caps/cooldowns/durations/synergies;
- editor comparison/presentation and runtime cue implementation;
- persistence/migration/concurrency implementation; and
- balance, combination, mandatory-option, accessibility, and telemetry tests.

Tuning cannot add slots/controls, instrument classes, required composition,
unconditional support power, recursion/contribution copies, Rhythm/movement/
recovery-count/reward authority, unbudgeted synergy, acolyte interaction, silent
substitution, old-revision advantage, or shipping placeholder names.

## 18. Approval and change control

The owner interview resolved BS-01 through BS-12 on 2026-08-30. This document is
the canonical Builds and Specialization design specification.

A material change to configuration shape/categories, preset integration, option
contract, major/support behavior, budget/synergy/caps, hook allowlists, runtime
order, validation/migration, disclosure/privacy, or naming gate requires an
explicit amendment citing the superseded rule. Numeric/catalog tuning inside
these boundaries creates a new revision and never changes an active snapshot.
