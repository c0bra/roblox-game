# Bands Battle Balance Framework Working Record

- **Status:** Interview complete; 15 of 15 questions resolved and reconciled
- **Started:** 2026-09-04
- **Question plan:** [`BALANCE_FRAMEWORK_QUESTIONS.md`](BALANCE_FRAMEWORK_QUESTIONS.md)
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#48-difficulty--scaling)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Canonical result:** [`BALANCE_FRAMEWORK.md`](BALANCE_FRAMEWORK.md)

## 1. Role of this record

This file preserves the approved answers, refinements, inherited constraints,
and cross-system handoffs from the completed Balance Framework interview. It is
evidence for the canonical specification, not the final authority.

## 2. Inherited boundary

The Balance Framework owns shared numeric relationships, parameter governance,
difficulty/population comparability, cross-system budgets/caps, initial tuning
hypotheses, scenario matrices, evidence requirements, and balance-revision change
control.

Each gameplay/economy system continues to own the semantic meaning and legal use
of its parameters. Content Authoring owns authored song/encounter data; UI/UX and
Audio Presentation own presentation tuning; Multiplayer owns service/session
policy; Player Data owns persistence; Playtest/Analytics owns collection and
research operations; Technical Architecture owns implementation mechanisms.

The complete inherited decision set is recorded in
[`BALANCE_FRAMEWORK_QUESTIONS.md`](BALANCE_FRAMEWORK_QUESTIONS.md#2-fixed-inherited-decisions).

## 3. Decision record

### Checkpoint A - Goals, parameter model, and scaling

#### BF-01 - Balance goals, player cohorts, and acceptable outcome envelopes

- **Status:** Resolved 2026-09-04.
- Balance optimization priority is:
  1. fairness and readable cause-and-effect;
  2. meaningful player skill and choices;
  3. comparable solo and cooperative viability;
  4. satisfying progression and reward pacing; then
  5. long-term mastery challenge.
- A balance profile is not acceptable merely because its aggregate victory rate
  is near a target. Losses and successes must remain attributable, boss identity
  must remain legible, ordinary recovery must matter, and neither stalling nor
  failure farming may become an optimal strategy.
- Initial first-attempt victory hypotheses for the intended cohort are 55–75% on
  Easy and 35–60% on Normal. Hard is not tuned against new-player first attempts.
- Initial learned-play hypotheses are 75–90% on Easy and 55–75% on Normal. For
  the practiced Hard cohort, early serious attempts target 45–65%, rising with
  mastery rather than remaining a permanent coin flip.
- These are playtest hypotheses, not player promises or universal boss quotas.
  They may move through the BF-14 evidence process without changing the approved
  difficulty meanings.
- Evidence separates first attempt, onboarding-complete, learned/competent, and
  practiced/mastery cohorts. It also segments boss, role, device/control profile,
  solo/co-op roster, accessibility configuration, and relevant progression band.
- A cohort outcome is interpreted with downing, recovery, layer timing, finishing,
  abandonment, repeated-failure, and clarity evidence. One blended win rate may
  never conceal a failing subgroup.

#### BF-02 - Balance registry, ownership, units, precision, and parameter states

- **Status:** Resolved 2026-09-04.
- Balance uses one logical registry composed of versioned domain-owned tables.
  Central discovery and revision do not transfer semantic ownership to the
  Balance Framework or permit one system to edit another system's parameters.
- Every parameter entry defines stable identity, semantic owner/domain, units and
  reference frame, formula/table/curve shape, initial/default value, permitted
  test range, dependencies, application stage/order, caps, deterministic
  precision/rounding, compatible content/catalog references, and failure policy.
- Each entry also records evidence, rationale, author/reviewer, revision lineage,
  affected scenarios/consumers, and lifecycle state: `Hypothesis`, `Playtest
  Candidate`, `Approved`, or `Retired`.
- A public attempt references and locks one exact approved balance revision.
  Runtime never silently mixes parameter identities or values from revisions.
- Hypothesis and Playtest Candidate values may run only in explicitly identified
  test contexts approved by BF-14. Debug or developer overrides cannot enter
  ordinary public play, affect durable rewards/progression, or masquerade as an
  approved revision.
- Formula evaluation uses deterministic precision and named rounding boundaries.
  Intermediate display rounding does not feed authoritative calculation.
- Missing, unknown, incompatible, invalid, or out-of-range critical balance data
  blocks deployment before commitment or invokes the owning system's safe
  cancellation/No Contest path. A client, content package, or cached display
  value never supplies an implicit fallback.
- Gameplay/economy balance, content-authored variation, UI/Audio presentation
  tuning, Multiplayer service policy, and technical/operational limits remain
  separately owned even when the registry links their compatible revisions.

#### BF-03 - Difficulty and one-to-six-human scaling model

- **Status:** Resolved 2026-09-04.
- Normal is the reference profile. Easy and Hard change explicitly named axes
  such as chart complexity/windows, Resolve requirements, encounter pressure,
  warning combinations, survival/recovery, Cohesion thresholds, and bounded
  reward modifiers. There is no global difficulty multiplier.
- Difficulty does not alter the common normalized maximum contribution, song,
  individual controls, role identity, accessibility rights, or the fixed Combat
  pipeline. Combat applies difficulty only through the explicit encounter,
  target, or tagged-recovery stage already defined in `COMBAT.md`.
- Retain the starting Resolve relationship by active humans: 1.00, 1.75, 2.50,
  3.25, 4.00, and 4.75 solo-player equivalents for one through six humans.
  These values remain hypotheses subject to the evidence gate.
- Each Resolve layer combines its difficulty base and active-human relationship
  when it opens, then locks. A disconnected human stays counted during grace;
  expiry affects only unopened layers. Acolytes never count as humans.
- Additional-human pressure comes from validated readable target/geometry
  coverage. The initial simultaneous individually targeted count is about half
  the eligible active roster, rounded by the owning profile and never above
  three. It does not accelerate song/attack time, densify individual charts, or
  multiply personal incoming damage.
- Duplicate roles remain neutral. A weak, inactive, downed, or disconnected
  player contributes little or no positive share but never reduces another
  player's earned contribution or creates a duplicate-role penalty.
- Solo acolytes and population scaling target completion rates within roughly ten
  percentage points between solo and cooperative cohorts at comparable measured
  human proficiency. This is an initial comparative envelope, not forced outcome
  equality or permission to fabricate solo performance.
- Party size never changes the underlying personal reward table or divides a
  shared pool. Difficulty and individually earned nonnegative bonuses may affect
  personal results under the Rewards contract.
- BF-13 must separately test one through six humans, duplicates, uneven skill,
  disconnect/grace/rejoin/AFK, down/revival, all-same-role, and solo-acolyte
  scenarios rather than infer population fairness from average party size.

### Checkpoint B - Rhythm, combat, and survival

#### BF-04 - Rhythm proficiency, timing, holds, and normalized contribution

- **Status:** Resolved 2026-09-04.
- Retain the starting judgment windows already approved in Rhythm Gameplay:

  | Difficulty | Perfect | Great | Good | Miss |
  |---|---:|---:|---:|---:|
  | Easy | ±90 ms | ±150 ms | ±230 ms | Beyond ±230 ms |
  | Normal | ±60 ms | ±110 ms | ±170 ms | Beyond ±170 ms |
  | Hard | ±45 ms | ±85 ms | ±135 ms | Beyond ±135 ms |

- Starting contribution factors remain Perfect 100%, Great 80%, Good 50%, and
  Miss 0%. They preserve strict monotonic ordering under every revision.
- Every passage has a hidden maximum of 1,000 normalized contribution units for
  every playable role and difficulty. Event weights may distribute those units
  differently according to authentic material. UI may show percentages or
  summaries rather than exposing the internal scale.
- A hold initially assigns 50% of its event weight to the graded onset and 50%
  linearly across maintained duration. Hold Assist earns the identical model.
  The split may be tuned globally through evidence, not per player or gear.
- An 80 ms global release grace initially absorbs accidental contact loss. A
  release beyond grace stops future duration value without a second judgment.
- An identifiable premature attempt may associate with an upcoming event up to
  300 ms early, capped by nearer unresolved material so it cannot cross or steal
  another event. It resolves as Early Miss when outside that difficulty's Good
  window. Contact-bounce filtering remains technical input tuning.
- Re-entry after suspension requires at least 500 ms of visible preview before
  the first eligible note. If authentic material cannot meet it, re-entry moves
  to the next eligible candidate without fabricating a note or widening judgment.
- A normal encounter initially targets meaningful rhythm participation during
  roughly 65–80% of its duration for an offered role, adjusted for authentic
  rests, dropouts, song function, and evidence. Inactive time must remain
  meaningful but is never filled with fake actions.
- Execution quality, participation coverage, and earned contribution remain
  separate. Sparse/dense roles are compared through normalized opportunity and
  trustworthy coverage, never raw note count.
- Calibration corrects alignment only. Its measurement limits, outlier rules,
  and safe offset bounds require cross-device evidence and cannot change grade
  windows, contribution maximum, difficulty, or rewards.

#### BF-05 - Combat conversion, modifier budgets, caps, and deterministic order

- **Status:** Resolved 2026-09-04.
- Preserve Combat's fixed conversion order:
  1. normalized earned Rhythm value;
  2. versioned base rate for the effective intent/effect;
  3. applicable equipment and build modifiers;
  4. the position modifier legal for that effect;
  5. explicit encounter, target, or difficulty modifiers; then
  6. the effect-specific cap.
- The 1,000-unit passage scale is the common normalized input, not a requirement
  that every intent or effect produces equal output. Domain-owned base rates
  convert it into Resolve pressure, mitigation, reinforcement, restoration,
  Hype eligibility, revival, group, or typed utility units.
- Values in the same declared category add before one category cap. Separately
  budgeted categories multiply only once at fixed stages. A source cannot re-
  enter the pipeline, occupy undeclared stages, recurse as new performance, or
  spend the same normalized portion twice.
- All authoritative calculations use deterministic fixed precision. Intermediate
  display rounding never feeds calculation; final rounding occurs only at the
  named display or discrete-application boundary and retains pre-rounded evidence.
- As an initial first-release guardrail, ordinary Attack produced from the same
  Rhythm contribution caps near 1.6 times the unmodified Middle-position starter
  result after equipment, build, and position effects. BF-10 allocates the
  equipment/build portion within that envelope.
- Committed Signature reliable bases and explicitly budgeted group effects use
  separate caps because they are earned limited resources. They do not raise the
  ordinary Attack ceiling or permit a second primary route.
- Zero ordinary performance remains zero regardless of multipliers or procs.
  Every accepted positive improvement is monotonic until its declared cap.
- Defend, reinforcement, restoration, utility, multi-target, and group packets
  use typed domain caps. A fixed-total multi-target effect divides its budget;
  a per-recipient effect has a roster-aware total cap. Roster size never causes
  unbounded multiplication.
- BF-13 must cover zero, one, threshold, cap, over-cap, invalid-target,
  simultaneous-event, and every modifier-order edge with exact reproducible
  pre/post evidence.

#### BF-06 - Ward, danger, downing, revival, and recovery tuning

- **Status:** Resolved 2026-09-04.
- Reference maximum Ward is 1,000 before approved equipment/effect changes.
  Existing semantic thresholds remain Safe at 50% or above, Fractured below 50%
  to 25%, Critical below 25% to above zero, and Downed at zero.
- No ordinary single readable attack may down a player starting at full normal
  Ward. Clearly combined committed threats remain separately identified and
  ordered rather than disguised as one hit.
- On Normal, a full-Ward player with no mitigation initially should require
  roughly four to six substantial failed threat responses to become Downed.
  Easy permits more accumulated mistakes; Hard may permit fewer while retaining
  readable cause, counterplay, and the no-ordinary-one-shot rule.
- Ordinary combined Defend/reduction mitigation initially caps near 75% for one
  threat. Owning-system cancellation, legal movement avoidance, return
  protection, or compatible cover may still reduce a threat to zero at their
  earlier gates.
- Temporary reinforcement initially caps at 25% above current normal maximum
  Ward. Restoration caps at normal maximum unless an explicitly separate,
  budgeted conversion grants reinforcement. No effect creates a second life bar.
- A competent single revival participant should normally complete the target in
  one short authentic recovery phrase. Additional independent participants
  accelerate it with diminishing returns and a roster-aware cap; acolytes,
  absence, or unearned opportunity never contribute.
- Successful co-op revival returns roughly 35% Ward for one competent helper and
  may rise toward 60% with stronger/multiple help. Solo emergency recovery starts
  near 35% Ward. Consumables cannot add another recovery attempt.
- Return protection/settling remains about two beats, clamped initially to
  roughly 0.75–1.25 seconds, followed by the first event meeting BF-04 preview.
- Downing should result from understandable accumulated danger rather than an
  arbitrary punishment for a single Miss. BF-13 evaluates down frequency,
  repeated-down chains, revival availability/time, all-downed outcomes, solo
  recovery, position, difficulty, roster, and progression bands together.

### Checkpoint C - Boss arc, arena pressure, and cooperative power

#### BF-07 - Resolve layers, Momentum, finishing, and result tiers

- **Status:** Resolved 2026-09-04.
- Each boss profile derives layer targets from the authentic playable opportunity
  before the next authored opening, the locked difficulty/roster relationship,
  and the target proficiency band. It does not assign arbitrary health divorced
  from available song material.
- Initial proficiency bands for resistance and finishing hypotheses are roughly
  55–60% of available normalized contribution on Easy, 70% on Normal, and
  80–85% on Hard. These refine BF-04 and remain evidence-tunable.
- For a competent target cohort, each layer should normally break during the
  latter half of its intended First Clash, Escalation, or Climax region, leaving
  some slack before the next opening or Finishing Cadence. Authored function
  boundaries govern; song-progress percentages are diagnostic, not universal
  timers.
- Earlier-than-target breaks indicate overperformance and feed existing Momentum
  or post-break rules rather than retiming the song. Later breaks preserve the
  sequential layer and may compress future opportunity without skipping it.
- Momentum remains capped initially around 20% of the next layer and can never
  break that layer by itself. Excess is discarded with evidence rather than
  hidden in another bank.
- Third-layer post-break performance initially caps near 15% of total encounter
  Resolve. It may improve result/reward potential and spectacle but cannot satisfy
  Finishing Cadence or produce early victory.
- Finishing Cadence initially uses the same difficulty proficiency targets:
  roughly 55–60% Easy, 70% Normal, and 80–85% Hard, calculated only from its
  eligible human performance.
- Any valid Victory receives the base result tier. A second private tier begins
  around 80% overall eligible performance and a top tier around 90%, with positive
  post-break and finishing evidence included under a versioned formula.
- Higher tiers do not require zero downs, erase a legitimate revival/recovery,
  publicly rank players, change campaign credit, or reverse the exact gameplay
  outcome. Final names remain part of the naming pass.
- Boss profiles may vary pressure and authored opportunity while remaining inside
  the BF-01 cohort envelopes. BF-13 evaluates layer pace, late/unbroken layers,
  finishing misses, early overperformance, result distribution, and sensitivity.
- Invalid/No Contest compensation stays capped at the failure-equivalent amount
  already defined by Rewards. It never implies Victory or supplies ordinary
  result-tier credit.

#### BF-08 - Attack ecology, telegraphs, targeting, movement, and positional risk

- **Status:** Resolved 2026-09-04.
- Starting warning for a major threat is roughly two measures on Easy, one to two
  measures on Normal, and one measure on Hard. Authored rhythm, required action,
  travel distance, device profile, and accessibility evidence may require more
  lead; difficulty never shortens below proven response capacity.
- Easy does not require simultaneous independent answers to two major threats.
  Normal and Hard may combine at most two independent critical responses in one
  window, and only when control, attention, visual, audio, phone, and
  accessibility review proves both readable and legally answerable.
- Candidate weights, family cooldowns, repetition limits, and intensity budgets
  are boss-profile data. They must prevent repeated focus on one player or family
  when equivalent legal alternatives exist and preserve reserved recovery,
  Crescendo, Band Call, and finishing windows.
- Attack damage/effect bands are derived from BF-06's Ward/downing envelope.
  Authors select approved threat families and tags; they do not bypass the shared
  survival model with private damage values.
- Retain one movement charge, roughly 0.75 seconds of travel per edge, and a
  two-beat charge recovery initially clamped to roughly 0.75–1.25 seconds.
  Multi-edge routes pay the same cost per edge, and no balance source grants
  travel invulnerability or changes these universal timings.
- Retain the position hypotheses: Near +25% Attack/+30% incoming danger/+25%
  reward potential; Middle baseline; Rear -20% Attack/-25% danger/no Risk Bonus.
  Gear/builds cannot modify the ratios.
- Total banked personal Risk Bonus caps initially at +25%. Only successful
  normalized settled performance earns it; only the current phrase's unbanked
  value is lost on movement/downing.
- Compatible cover may fully prevent one threat at its earlier gate. It is not
  universal immunity and cannot retroactively retarget a committed attack.
- Every attack/graph/position combination preserves at least one readable legal
  response. Displacement, hazards, graph mutation, and overlapping threats must
  respect route legality, target caps, warning time, and BF-06 survival limits.
- Initial simultaneous individual targeting remains roughly half the eligible
  active roster and never above three under BF-03. More humans widen readable
  coverage, not speed, density, personal damage, or independent critical count.

#### BF-09 - Hype, Signature, Band Call, Crescendo, and solo-acolyte balance

- **Status:** Resolved 2026-09-04.
- Slow Hype from competent engaged ordinary Attack/Defend play initially targets
  one full Signature charge around 60–75% of song progress. It stores no second
  charge and does not guarantee readiness to an inactive or low-coverage player.
- Deliberate fast-Hype routing initially targets full charge after roughly two to
  three strong eligible passages. Every routed portion sacrifices its ordinary
  Attack/Defend effect, preserving a meaningful short-term tradeoff.
- A Signature's guaranteed base initially equals about 50% of one competent
  passage effect in its typed domain. Accepted activation performance may scale
  total value toward one full competent passage effect, subject to target/group/
  effect caps and without changing the ordinary Attack ceiling.
- Band Call readiness should ordinarily arrive around 35–50% of the song for a
  competent engaged player. One committed use prevents recharge. The shared
  lockout remains roughly eight measures and item/build hooks cannot shorten it
  below its global floor.
- Call definitions budget their guaranteed initiator base and each nonnegative
  human share separately. An expert remains useful alone but cannot generate the
  eligible-roster top tier for inactive humans. Exact typed values remain catalog
  data inside the shared group-effect cap.
- Crescendo thresholds initially use eligible-human maximum contribution: Echo
  below 40%, Crescendo from 40% through 74%, and Full Crescendo at 75% or above.
  Echo remains the minimum committed outcome, and a weak share never subtracts.
- Easy's one additional Crescendo evaluation becomes eligible only when projected
  remaining resistance exceeds expected remaining eligible output by about 20%
  and the living group is under meaningful survival pressure. The exact combined
  pressure function remains versioned and cannot activate on Normal/Hard.
- Vanguard initially adds roughly 10% of the triggering competent human Attack
  packet after that packet, clamps before a decisive break, and never spills.
- Warden initially supplies about 5% Ward reinforcement at clean boundaries near
  every eight measures, respecting the shared reinforcement cap and never banking
  a suppressed/unneeded pulse.
- Herald initially increases human-earned readiness by about 10% and supplies at
  most 20% of one target human share at a solo group-action commitment. It creates
  no passive readiness or additional human-equivalent roster member.
- Acolyte suppression remains roughly four measures, does not stack/refresh, and
  provides no catch-up. Acolytes never affect risk, rewards, recovery attempts,
  judgments, contribution records, or player items/builds.
- BF-13 tests these values together against the BF-03 ten-point solo/co-op
  envelope. Support may improve rhythm and recovery texture but cannot make the
  human nondecisive or create superior solo rewards.

### Checkpoint D - Power progression and economy

#### BF-10 - Item, build, loadout-power, and uplift framework

- **Status:** Resolved 2026-09-08.
- A complete starter loadout defines a 100-point multidimensional power reference
  across offense, Ward/defense, and relevant support/utility. The number is an
  internal comparison budget, not a public universal gear score.
- First-release permanent direct-power growth initially caps near 130 points for
  a complete configuration, including General Progression. Gear supplies most
  durable growth; General Progression contributes at most ten points, matching
  its approved ceiling of roughly 10% of a first-tier loadout.
- Power remains a vector. A 130-point offensive, defensive, or support-focused
  setup is not treated as identical in every encounter, and recommended power
  cannot collapse all specializations into damage.
- Each item tier initially has a base plus roughly three upgrade ranks. Each rank
  adds about 3% of that slot's base budget. The next tier's base equals or
  slightly exceeds the preceding tier's maximum rank so an explicit uplift never
  lowers function.
- About 70% of a power item's budget initially belongs to its visible fixed
  primary stat and 30% to its one fixed trait. Trait effects still use typed
  Combat/Ability hooks, categories, costs, and caps; there are no hidden/random
  affixes.
- Builds retain the 100-unit abstract envelope: one major at most 40 and three
  distinct supporting options at most 20 each. Unused slot capacity does not
  transfer.
- Synergy remains charged inside participating option budgets and caps initially
  at 15% of total build value. It cannot become a new unbudgeted multiplier.
- A build's direct-output improvement over equivalent gear should ordinarily
  remain around 10%. Most build value comes from conditions, tradeoffs,
  redistribution, support, timing, or playstyle rather than unconditional stats.
- Gear, builds, abilities, General Progression, and position still meet shared
  hook/category caps. The BF-05 ordinary Attack ceiling near 1.6 times starter
  Middle output applies after equipment/build/position, so a focused combination
  cannot escape through separately named budgets.
- Recommended power evaluates offense, Ward/defensive conversion, and relevant
  ability/support utility. Initial private bands are Prepared at 90% or more of
  the encounter profile, Challenging from 75% through 89%, and Far Below below
  75%. It remains advice, never an access/matchmaking/reward gate.
- Uplift advances exactly one tier, preserves item identity/trait/appearance and
  references, and resets to the nonlosing target-tier base rank. It does not auto-
  scale older or paid items.
- A prepared consumable may not exceed roughly one competent passage effect per
  use, bypass typed effect caps, create another revival/solo-recovery attempt,
  become necessary for baseline victory, or alter the first-release two-slot cap.
- BF-13 tests identical paid/earned definitions, every legal item/build mixture,
  starter/endgame power bands, cap saturation, invalid combinations, multiple
  specializations, and no-mandatory-option outcomes.

#### BF-11 - Campaign, General Progression, Mastery, and recommendation pacing

- **Status:** Resolved 2026-09-08.
- One eligible meaningful first Victory on Easy, Normal, or Hard completes that
  campaign node, restores its fragment, and unlocks satisfied next destinations.
  A meaningful Normal Victory separately unlocks that boss's Hard difficulty.
- Initial General Progression awards are 100 per eligible Victory and 35 per
  eligible Defeat. Invalid/No Contest grants no ordinary progress; one explicit
  compensation event may grant no more than 35 to a meaningful participant.
- Advanced build editing and all three full spec-preset slots initially open near
  200 total General Progress, roughly two Victories. This unlock is account-wide
  and never binds the player to one role or path.
- Broader option/consumable access and the ten-point General direct-power allowance
  are distributed through approximately 1,000 total General Progress. The power
  milestones are front-loaded enough to feel useful but finite; later General
  rewards emphasize options rather than endless stats.
- Defeat progress is useful practice credit but remains substantially less
  efficient than Victory and cannot grant campaign nodes/fragments, boss
  material, item rolls, Signature items, first-clear choices, or Hard access.
- Each boss initially has about ten visible Mastery ranks across roughly 1,500
  total Mastery XP. A Victory grants 100 and an eligible Defeat 25 before modest
  difficulty bonuses.
- Normal initially adds about 10% Mastery XP and Hard about 20%. No rank,
  functional sidegrade, deterministic item path, or completion reward requires
  Hard; difficulty mainly changes efficiency and eligible mastery cosmetics.
- Visible milestone cadence begins near ranks 2, 4, 6, 8, and 10, mixing lore,
  cosmetics/titles, recipes, bounded sidegrades, and deterministic guarantees.
  Completion grants a finite capstone rather than hidden prestige XP or stats.
- Difficulty-up guidance requires at least three recent relevant strong attempts
  with trustworthy execution/survival evidence. Difficulty-down guidance likewise
  requires at least three repeated relevant struggles and distinguishes probable
  calibration/device problems. Neither changes selection or becomes public.
- Personal records retain separate compatible categories for execution,
  participation/holds, Combat, Ward/survival, position/Risk, cooperative help,
  and overall performance. They do not become public damage rankings or balance
  authority.
- Catalog additions grant already-satisfied prerequisites retroactively. Returning
  players retain nodes, fragments, ranks, unlocks, records, and equivalent access;
  no revision requires replay or expires earned progress.
- BF-13 tests first-boss onboarding pace, three-boss campaign flow, two-Victory
  advanced-build timing, General cap time, per-boss Mastery time, failure farming,
  difficulty efficiency, and returning-player/grandfathering cases.

#### BF-12 - Rewards, drops, resources, crafting, upgrades, and Commerce parity

- **Status:** Resolved 2026-09-08.
- Initial eligible Easy Victory base is 100 integer general resource, 20 integer
  boss material, exactly one boss-pool roll, and a disclosed 15% complete-item
  chance, plus deterministic-path progress through retained boss material.
- Normal initially grants about 10% more permitted resource quantity and a 20%
  complete-item chance. Hard initially grants about 20% more resources and a 25%
  complete-item chance. All share one functional pool and one roll.
- Personal bonuses retain their starting maxima: performance +15%, banked Risk
  +25%, and Cohesion/solo breadth +15%, with a +40% combined non-difficulty cap.
  Percentages apply relatively to explicitly permitted resource quantity, item
  chance, or starting-rank fields, not as undisclosed percentage points.
- Difficulty and personal bonuses cannot change pool identity, add rolls, alter
  guaranteed/first-clear awards, create campaign/Hard access, or make essential
  function difficulty-exclusive.
- One pinned standard chosen item initially costs 100 boss material plus 200
  general resource, making the boss-material component exactly five baseline
  Easy Victories. A random item never consumes or resets that deterministic path.
- Voluntary duplicate-power-item salvage initially returns 30% of the equivalent
  boss-material crafting cost, below recreation. Cosmetic duplicates use their
  separately disclosed deterministic conversion and never create power.
- Initial three-rank upgrade costs are approximately 40/70/100 general plus
  5/10/15 matching boss material. Each transaction is deterministic, previews
  exact value, preserves identity, and cannot fail randomly or lower/destroy.
- One-tier uplift initially costs about 200 general, 60 current-tier boss
  material, and 20 original-boss material. It preserves lineage/trait/appearance,
  uses the BF-10 nonlosing target base, and applies identically to paid/earned.
- An ordinary earned-shop consumable initially costs about 25 general resource,
  so one baseline Victory can fund several optional uses without making them free
  or baseline requirements.
- Signature combat items may cost more than the standard path but require a
  finite visible Victory-based recipe or Mastery/clear milestone. They never
  require Hard, random success, or Robux.
- An eligible Defeat initially grants about 30 general resource and no boss
  material, roll, deterministic Victory-path progress, campaign credit, or Hard
  unlock. This remains less efficient than Victory and cannot become optimal.
- Invalid/No Contest provides no ordinary outcome loot. Meaningful-participant
  compensation uses the Defeat ceiling, and exact causal consumables are refunded
  once without exceeding committed quantities.
- Paid equipment is the exact same-tier functional definition as a visible
  earnable equivalent, freezes at the player's unlocked quote tier, and uses
  identical rank/uplift costs, caps, tests, and migration. Only appearance/source
  lineage may differ.
- BF-13 simulations cover expected/variance paths, random-plus-deterministic
  interaction, duplicate streaks, resource inflow/sinks, every bonus cap,
  failure/No Contest, old/current boss efficiency, upgrade/uplift loops, and paid
  equivalence. No configuration may create conversion, arbitrage, negative
  balance, random-only function, or failure-optimal farming.

### Checkpoint E - Evidence, governance, and completion

#### BF-13 - Balance scenario matrix, evidence, and acceptance thresholds

- **Status:** Resolved 2026-09-08.
- Every Playtest Candidate first passes deterministic formula, order, cap,
  monotonicity, identity, revision, and invariant tests. A failure blocks human
  balance interpretation because the candidate is not mechanically trustworthy.
- Each core reference scenario runs at least 10,000 reproducible seeded
  simulations before approval review. Simulation records exact input model,
  parameter/content revisions, seed set, distributions, outputs, and failures.
- Changed parameters receive sensitivity runs around ±10% and ±20% where legal.
  Closely coupled parameters vary both independently and together so cliffs,
  unstable feedback loops, cap saturation, and hidden dominant variables appear.
- Economy simulation covers expected and adverse drop streaks, duplicates,
  deterministic material paths, resource inflow/sinks, upgrade/uplift/crafting,
  consumable use, progression pace, failure farming, old/current content, paid
  equivalence, arbitrage, overflow, and long-session/account projections.
- Simulation uses explicit reference-player models rather than a single perfect
  bot: zero/inactive, developing, competent, practiced, mastery, uneven group,
  and targeted failure patterns. Bots are measurement fixtures, never evidence
  that an interaction is understandable or fun.
- Human evidence is mandatory before a candidate becomes Approved. The smallest
  complete prototype retains at least two rounds of roughly 8–12 new target-age
  players ages 10–14, mostly on representative phones/tablets.
- Content-complete first-release testing retains at least 30 new target-age
  players plus returning players, with the GDD comprehension, telegraph, audio,
  Results, voluntary-next-action, accessibility, and impossible-combination gates.
- A material structural change requires two consecutive successful testing
  rounds before its evidence is considered stable.
- Evidence remains segmented by boss/content revision, role, difficulty,
  device/control/output profile, accessibility configuration, progression/power
  band, first attempt versus learned versus mastery cohort, solo/two/three/six
  humans, duplicate/all-same roles, and uneven skill.
- The scenario matrix also covers one through six humans; down/revival/all-down;
  disconnect/grace/rejoin/AFK; movement/cover/hazard/displacement; Hype/Signature/
  Call/Crescendo/acolytes; zero/threshold/cap/overflow; extreme drop streaks;
  invalid content/balance data; and active-snapshot revision changes.
- Balance outcomes include victory/defeat/No Contest, abandonment/retry, layer
  break timing, Momentum/post-break, finishing, execution/coverage/contribution,
  intent use, damage/mitigation/down/recovery, movement/risk, group participation,
  result tiers, progression/economy pace, build diversity, and cap saturation.
- Impossible attacks, unexplained damage/outcomes, staff-only attention, movement
  surprise, ignored/unreadable Defend, empty quiet passages, Easy feeling weaker,
  missing role activity, teammate blame/punishment, routinely missed group
  invitations, and excessive one-off content exceptions are structural findings.
  They return to design/content rather than being hidden by numeric adjustment.
- Numeric tuning is appropriate only when players understand the rule/cause and
  evidence locates the parameter relationship inside its approved range.
- Outliers are not discarded merely for being inconvenient. Exclusion requires a
  predefined data-quality reason and retains excluded count/reason. An aggregate
  never hides a materially failing segment.
- `PLAYTEST_AND_ANALYTICS.md` will define recruitment, consent/safeguarding,
  instrumentation, formal sample/confidence interpretation, data quality,
  retention/access, reporting, and sign-off operations without weakening these
  minimum balance gates.

#### BF-14 - Tuning workflow, versioning, approval, rollout, and rollback

- **Status:** Resolved 2026-09-08.
- Parameter and profile lifecycle is `Hypothesis -> Playtest Candidate ->
  Approved -> Retired`. Retirement preserves identity, history, reproducibility,
  and explicit successor/compatibility rather than deleting old meaning.
- Every proposed change records stable identities; semantic/domain owner;
  previous and proposed values/formulae/ranges; rationale; expected effect;
  affected systems, catalogs, content, cohorts, and scenarios; compatibility and
  migration impact; simulations; human evidence; risks; and reviewer decisions.
- The semantic domain owner approves legality/meaning. A cross-system balance
  review verifies shared caps, scaling, progression/economy, and unintended
  effects. Affected boss/song/content owners approve profile compatibility.
- One person may hold multiple responsibilities during early prototyping. Before
  public release, a release-critical balance revision also requires an
  independent reviewer who was not its sole author.
- Prefer changing one parameter or one clearly coupled set per candidate so
  causality remains interpretable. A larger rebalance must explicitly enumerate
  its coupled hypotheses and cannot use volume to hide unreviewed changes.
- Rollout proceeds through deterministic tests/simulation, internal play, private
  target-age testing, and then limited/full public use where appropriate. Each
  stage has predefined success, stop, and rollback signals.
- Any cohort experiment uses stable assignment, exact revision identity, privacy
  review, and reward/progression treatment that cannot silently disadvantage a
  player or corrupt paid/earned equivalence. Exact experiment operations belong
  to Playtest/Analytics and Technical Architecture.
- Active attempts keep their locked content/catalog/balance revision. A rollout,
  rollback, hotfix, or experiment never changes an in-progress calculation or
  reconstructs its frozen result under new numbers.
- Rollback selects the last exact compatible Approved revision. It does not
  recalculate or revoke confirmed results, rewards, items, progression, personal
  records, purchases, or compensation.
- When a change makes stored player facts incompatible, the owning catalog and
  Player Data use explicit equivalence, migration, replacement choice, or fair
  compensation. They never silently weaken/delete value or invent a choice.
- A serious exploit, corrupt balance bundle, unsafe attack, or transaction-
  integrity failure may immediately pause new affected deployments/transactions
  and select an exact safe revision. Permanent repair still records authority,
  evidence, migration/compensation, and independent review appropriate to risk.
- No revision may create invisible dynamic difficulty, personalized hidden
  reward chances, player-specific punishment, or a mutable paid-product mapping.
- Reproducibility evidence and prior approved data remain available through the
  defined audit/compatibility window. Retention implementation belongs to
  architecture/policy but must support incident and player-value recovery.

#### BF-15 - Canonical outputs, boundaries, and completion audit

- **Status:** Resolved 2026-09-08.
- Canonical `BALANCE_FRAMEWORK.md` defines semantic authority, governing
  invariants, balance goals/cohorts, parameter-entry contract, units/precision/
  rounding, formula order, shared caps/budgets, difficulty/population matrices,
  starting hypotheses, evidence gates, revision lifecycle, and change control.
- It includes the Rhythm proficiency/contribution model; Combat/Survival/
  Encounter/Positioning/Ability/Solo Support envelopes; Item/Build/power budgets;
  Progression/Mastery/recommendation pacing; Rewards/economy/Commerce parity;
  scenario/invariant/sensitivity/exploit matrices; and consumer handoffs.
- A future versioned machine-readable balance catalog is the runtime numeric
  source of truth. It conforms to the canonical semantic contract and produces
  generated human-readable snapshots/diffs. A runtime value may not exist only
  in source code, a chart, UI text, or an undocumented service setting.
- Each boss-specific profile contains exact boss/content/catalog/balance
  identities; target cohorts; available opportunity; layer/finishing targets;
  difficulty/roster transforms; attack/candidate/arena values; survival/group/
  solo settings; result/reward links; compatible parameter range; evidence;
  lifecycle state; and approvals.
- Actual profiles for the first three bosses must pass the framework and
  Content Authoring gates before those packages publish. Song-specific values
  that cannot yet be evidenced remain explicitly `Hypothesis`; they are not
  fabricated to make the framework appear numerically complete.
- Unproven numbers are valid design starting points only when their Hypothesis
  status, range, owner, consumers, and required evidence are explicit. Public
  runtime requires an Approved compatible catalog/profile revision.
- UI/UX retains layout, interaction, setting, accessibility-presentation, and
  player-facing formatting. Audio retains loudness/mix/duck/concurrency/device
  tuning. Their semantic thresholds may reference balance facts but do not
  become balance-owned presentation values.
- Multiplayer retains invitation/queue/load/rejoin/AFK/ping service policy unless
  a value directly participates in a defined gameplay fairness relationship.
  Player Data retains save/lease/capacity/SLO behavior and stores references/
  outcomes without recalculation.
- Content Authoring owns authentic musical/encounter opportunity and exports
  exact profile references, not hidden numeric overrides. Playtest/Analytics owns
  collection/research/reporting. Technical Architecture owns schemas, services,
  transport, loading, validation, security, and deployment mechanisms.
- Final audit must find no orphaned parameter, ambiguous unit, competing source
  of truth, duplicated semantic owner, missing revision/reference, unbounded
  combination, active-snapshot drift, inaccessible/publicly punitive distinction,
  paid mismatch, or contradiction with the GDD/system specifications.
- After canonical approval, `PLAYTEST_AND_ANALYTICS.md` is the next supporting-
  document interview. Technical Architecture follows the completed supporting
  evidence contract and the already approved system designs.

## 4. Cross-spec reconciliation register

- Every approved value must preserve the source system's invariants and retain
  its semantic owner. Central balance data does not make Balance Framework the
  owner of gameplay decisions.
- Difficulty, roster, item/build, ability/group, progression, and reward
  transforms require one deterministic order and exact locked revision.
- Content packages may reference compatible balance identities but never embed a
  private competing formula or silently override a shared parameter.
- UI/UX and Audio Presentation receive semantic ranges/tiers and evidence needs,
  not permission to infer outcomes from displayed or audible values.
- Player Data stores locked revisions and frozen outcomes without becoming a
  calculator. Playtest/Analytics measures approved definitions without changing
  difficulty, recommendations, or player state automatically.

## 5. Confirmed downstream handoffs

- `PLAYTEST_AND_ANALYTICS.md` will operationalize the approved cohorts, metrics,
  research safeguards, evidence thresholds, and data-quality requirements.
- Technical Architecture will choose balance-data formats, loading, authority,
  validation, rollout, rollback, and observability mechanisms.
- Content production will create first-three-boss profiles inside the approved
  envelopes and prove them through the reconciled authoring/review gates.
- The naming/tone pass will replace working labels without changing parameter
  identity, meaning, history, or balance behavior.

## 6. Change log

- **2026-09-04:** Resolved BF-01 through BF-03. Progress is 3 of 15 questions.
- **2026-09-04:** Resolved BF-04 through BF-06. Progress is 6 of 15 questions.
- **2026-09-04:** Resolved BF-07 through BF-09. Progress is 9 of 15 questions.
- **2026-09-08:** Resolved BF-10 through BF-12. Progress is 12 of 15 questions.
- **2026-09-08:** Resolved BF-13 through BF-15. The interview is complete at 15
  of 15 questions and ready for canonical reconciliation.
- **2026-09-08:** Published canonical `BALANCE_FRAMEWORK.md` and completed its
  cross-specification reconciliation and audit.
- **2026-09-04:** Created the working record. Progress is 0 of 15 questions.
