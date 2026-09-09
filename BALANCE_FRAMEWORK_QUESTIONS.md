# Bands Battle Balance Framework Questions

- **Status:** Interview complete; 15 of 15 questions resolved and reconciled
- **Started:** 2026-09-04
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#48-difficulty--scaling)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **System-specification index:** [`SYSTEM_SPECIFICATIONS_TRACKER.md`](SYSTEM_SPECIFICATIONS_TRACKER.md)
- **Working record:** [`BALANCE_FRAMEWORK_WORKING.md`](BALANCE_FRAMEWORK_WORKING.md)
- **Canonical result:** [`BALANCE_FRAMEWORK.md`](BALANCE_FRAMEWORK.md)

## 1. Interview method

This interview uses five checkpoints of three related questions. It establishes
the balance goals and shared parameter model first, then resolves core rhythm/
combat, encounter/cooperative play, long-term progression/economy, and finally
evidence/change-control/completion.

Answers are persisted after each checkpoint. When all fifteen questions are
resolved, they are reconciled into canonical `BALANCE_FRAMEWORK.md` and audited
against the GDD and all thirteen approved system specifications.

This is a design and tuning framework, not a final immutable spreadsheet or code
architecture. It must define the formulas, units, relationships, safe ranges,
ownership, scenarios, and approval process needed to create versioned balance
data. Initial numbers remain testable hypotheses until the approved evidence gate
promotes them.

## 2. Fixed inherited decisions

The interview does not reopen these approved rules:

- Easy, Normal, and Hard use the same song speed, duration, boss identity,
  story outcome, arena, core phases, controls, and independent accessibility
  options.
- Every difficulty and playable role has the same maximum pre-difficulty combat
  contribution per passage. Fewer notes never make Easy weaker.
- Song-specific roles follow the real arrangement. Duplicate roles are allowed,
  and composition or conventional-instrument requirements are prohibited.
- Easy uses simpler charts, wider timing, lower resistance/pressure, stronger
  recovery, and at most one conditional extra Crescendo. Hard remains fair but
  asks for greater execution/pressure management. Difficulty never changes
  automatically.
- One through six humans use sublinear Resolve scaling. The initial relationship
  is 1.00, 1.75, 2.50, 3.25, 4.00, and 4.75 solo-player equivalents.
- An open Resolve layer locks its roster-scaled threshold. Departure affects only
  unopened layers after rejoin grace expires. NPC acolytes never count as humans.
- Additional population pressure uses broader readable targeting, not faster
  songs, faster attacks, denser individual charts, or multiplied personal damage.
- Uneven performance never subtracts another player's earned contribution.
  Cohesion and group participation are capped nonnegative bonuses.
- Combat uses one deterministic additive/modifier pipeline. Every modifier has a
  category, power-budget cost, cap, order, duration, source, and attribution.
  Recursion, hidden multiplicative escape, and double application are forbidden.
- Position ratios are universal and unaffected by gear/builds. Current hypotheses
  are Near +25% Attack/+30% danger/+25% reward potential, Middle baseline, and
  Rear -20% Attack/-25% danger/no Risk Bonus.
- Ward is individual; a Miss does not directly damage it. Downing follows boss
  effects after mitigation. Co-op revival, one-use solo emergency recovery, and
  fair return protection follow their approved system contracts.
- Hype stores at most one Signature charge. There is at most one Band Call use,
  one normal authored Crescendo, and on Easy at most one conditional extra
  Crescendo. Acolytes never fabricate human performance or earn rewards/risk.
- Gear supplies most bounded direct power. Builds use one major and three
  supporting choices, emphasize tradeoffs/sidegrades, and cannot alter rhythm
  judgment, movement ratios, recovery count, reward authority, or controls.
- General Progression is finite in direct combat power; Boss Mastery is finite
  and per boss. Earned advancement does not expire. Failure cannot grant campaign
  clear credit or become the optimal farming strategy.
- Easy/Normal/Hard share each boss's functional item pool. Higher difficulty may
  improve bounded quantity/quality and grant mastery cosmetics, never exclusive
  essential combat function.
- Rewards are individual, never divided by party size. Bonuses from performance,
  banked Risk, and Cohesion are bounded and nonnegative. Initial hypotheses are
  approximately +15%, +25%, +15%, and +40% combined non-difficulty bonus.
- The deterministic chosen-item path initially targets roughly four to six
  baseline eligible Victories. Economy has no trading, conversion, debt, fees,
  energy, daily pressure, random affixes/rerolls, destructive crafting, or expiry.
- Paid functional items are exact earnable equivalents. Paid rescue, power,
  resources, luck, content, convenience, and superior function are prohibited.
- Accessibility, Hold Assist, calibration, device, privacy, spending, and age do
  not reduce rewards, maximum contribution, progression, or public treatment.
- Every attempt locks exact content, catalog, and balance revisions. A new balance
  revision affects future attempts and never rewrites confirmed history.
- Balance formulae and tables are versioned system data, not private values hidden
  inside song chart notes or client logic.

## 3. Question plan

### Checkpoint A - Goals, parameter model, and scaling

#### BF-01 - Balance goals, player cohorts, and acceptable outcome envelopes

- **Status:** Resolved 2026-09-04.
- **Decision:** Optimize first for fairness/readability, then meaningful skill and
  choice, solo/co-op viability, progression/reward pacing, and mastery. Use
  separate first-attempt and learned outcome hypotheses by difficulty and cohort;
  never tune against one blended win rate.
- **Decision needed:** What experience should the balance optimize for, and how
  do first attempt, learned play, mastery, solo/co-op, and device cohorts differ?
- **Must resolve:** Priority among fairness, clarity, tension, choice, cooperation,
  progression pace, and mastery; target skill/cohort definitions; intended
  Easy/Normal/Hard success envelopes; first-attempt versus learned expectations;
  acceptable boss/content variance; frustration/stall/farming guardrails; and
  which outcomes are hypotheses rather than promises.

#### BF-02 - Balance registry, ownership, units, precision, and parameter states

- **Status:** Resolved 2026-09-04.
- **Decision:** Use one logical, versioned registry of domain-owned parameters.
  Every entry defines identity, owner, units, formula/value/range, dependencies,
  order/caps/rounding, evidence, rationale, and lifecycle state. Attempts lock one
  revision; invalid critical data never falls back to hidden client defaults.
- **Decision needed:** What common contract makes every tunable discoverable,
  reproducible, reviewable, and impossible to override from the wrong system?
- **Must resolve:** Parameter identity/owner/domain; units/reference frame;
  formula/table/curve shape; default and safe range; dependency/order; fixed
  precision/rounding; content/catalog references; hypothesis/candidate/approved/
  retired states; revision and change reason; experiment/debug boundaries;
  missing/invalid behavior; and gameplay, content, economy, presentation, or
  operational ownership distinctions.

#### BF-03 - Difficulty and one-to-six-human scaling model

- **Status:** Resolved 2026-09-04.
- **Decision:** Normal is the reference profile; Easy and Hard modify named axes,
  not one global multiplier. Retain the initial 1.00-to-4.75 Resolve curve,
  layer-open roster locks, readable targeting cap, and neutral duplicate/reward
  rules. Target comparable solo/co-op outcomes within roughly ten points.
- **Decision needed:** How do difficulty and active-human count transform the
  same encounter without breaking equal contribution, clarity, or solo parity?
- **Must resolve:** Normal baseline; Easy/Hard deltas; order of difficulty,
  population, content, and player-power application; Resolve curve; open-layer
  snapshot behavior; target counts/distribution; revival/coordination advantage;
  solo-acolyte comparison; duplicate roles; group/Cohesion thresholds; disconnect/
  rejoin/AFK scenarios; reward neutrality; and segmented outcome targets.

### Checkpoint B - Rhythm, combat, and survival

#### BF-04 - Rhythm proficiency, timing, holds, and normalized contribution

- **Status:** Resolved 2026-09-04.
- **Decision:** Retain the approved grade windows/factors; normalize each passage
  to 1,000 hidden units. Start holds at an even onset/duration split, 80 ms
  release grace, 300 ms bounded Early Miss association, 500 ms re-entry preview,
  and roughly 65–80% meaningful normal-encounter participation.
- **Decision needed:** Which numeric model converts legal chart performance into
  comparable proficiency and contribution across roles and difficulties?
- **Must resolve:** Timing-window hypotheses; grade factors; Early Miss range;
  hold onset/duration allocation and release grace; repeat/alternate handling;
  passage budget unit/rounding; 65-to-80-percent participation hypothesis;
  sparse/dense-role comparability; suspension/re-entry preview; calibration
  bounds; execution/coverage confidence; and proficiency bands used downstream.

#### BF-05 - Combat conversion, modifier budgets, caps, and deterministic order

- **Status:** Resolved 2026-09-04.
- **Decision:** Preserve the six-stage Combat pipeline, additive-within-category
  caps, single multiplication of independent categories, fixed precision, and
  final-boundary rounding. Initially cap ordinary Attack after equipment, build,
  and position near 1.6 times the unmodified Middle-position starter result.
- **Decision needed:** How do normalized performance, intent, equipment, builds,
  position, and group effects produce bounded Combat results?
- **Must resolve:** Canonical internal units; Attack/Defend/Special conversion;
  Resolve pressure, mitigation, reinforcement, restoration, and utility curves;
  additive categories and pipeline order; item/build/shared power budgets;
  per-category/combined caps; diminishing returns; multi-target/group allocation;
  fixed precision/final rounding; monotonicity; and zero/max edge cases.

#### BF-06 - Ward, danger, downing, revival, and recovery tuning

- **Status:** Resolved 2026-09-04.
- **Decision:** Use 1,000 reference Ward, prohibit ordinary full-Ward one-shots,
  target four to six substantial Normal mistakes per down, cap ordinary combined
  mitigation near 75% and reinforcement near 25%, and retain one-phrase revival,
  35–60% return Ward, 35% solo return, and two-beat protected return hypotheses.
- **Decision needed:** What survival envelope creates readable accumulated danger
  and recoverable mistakes without trivializing or abruptly ending encounters?
- **Must resolve:** Ward maximum and starting state; boss damage ranges;
  mitigation/reinforcement/restoration caps; position/difficulty pressure;
  target downing frequency; co-op revival target and scaling; revival opportunity
  coverage; revived Ward and protection; solo emergency challenge/result;
  consumable boundaries; all-downed timing; and skill/roster scenario targets.

### Checkpoint C - Boss arc, arena pressure, and cooperative power

#### BF-07 - Resolve layers, Momentum, finishing, and result tiers

- **Status:** Resolved 2026-09-04.
- **Decision:** Tune layer breaks to the latter half of their authored regions
  using initial Easy/Normal/Hard proficiency targets of roughly 55–60%, 70%, and
  80–85%. Retain 20% Momentum, cap post-break value near 15% total Resolve, use
  matching finishing targets, and keep higher result tiers private and recovery-
  neutral without permitting early victory.
- **Decision needed:** How should progress through one full song feel, and which
  numbers distinguish pressure, recovery, finishing success, and result quality?
- **Must resolve:** Per-layer Resolve targets; expected opening/break ranges;
  overperformance/underperformance behavior; Momentum gain/bank/apply/caps;
  post-break performance; finishing threshold and failure states; result tiers;
  no early victory; outcome sensitivity; boss-to-boss variance; and No Contest
  failure-equivalent compensation ceiling.

#### BF-08 - Attack ecology, telegraphs, targeting, movement, and positional risk

- **Status:** Resolved 2026-09-04.
- **Decision:** Start major warning at two measures Easy, one-to-two Normal, and
  one Hard. Easy has one major response at once; Normal/Hard may combine at most
  two after validation. Preserve repeat protection, Ward-linked damage, one
  movement charge, existing travel/recovery, position ratios, +25% Risk cap,
  compatible full cover, and at least one readable legal response.
- **Decision needed:** How much simultaneous encounter pressure remains fair and
  readable across difficulties, devices, positions, and populations?
- **Must resolve:** Candidate weights/intensity budgets; family cooldown/repeat
  limits; warning/reaction/recovery ranges; damage/effect bands; combinations;
  roughly-half-roster/maximum-three targeting hypothesis; repeat-target
  protection; cover/hazard strength; movement charges/recovery/travel/settling;
  displacement clamps; position multipliers; Risk accrual/loss/cap; and
  phone/accessibility clarity constraints.

#### BF-09 - Hype, Signature, Band Call, Crescendo, and solo-acolyte balance

- **Status:** Resolved 2026-09-04.
- **Decision:** Target one slow-earned Signature around 60–75% song progress or
  fast charge in two-to-three strong sacrificed passages. Start Signature at
  half-to-one passage value, Calls ready around 35–50% with eight-measure lockout,
  Crescendo thresholds at 40%/75%, Easy recovery at a 20% projected shortfall
  plus survival pressure, and bounded 10%/5%/10%/20% acolyte support hypotheses.
- **Decision needed:** How often and how strongly should cooperative/personal
  power appear without replacing ordinary human rhythm performance?
- **Must resolve:** Slow/fast Hype gain and charge timing; Signature guaranteed
  base/performance bonus/caps; Band Call readiness, lead, duration, shares, tiers,
  lockout, maximum delay, and use value; Crescendo selection, lead, tiers, effects,
  Easy behind-state trigger; group-effect caps; Vanguard/Warden/Herald values,
  cadences, suppression, and clamps; and solo/co-op outcome equivalence.

### Checkpoint D - Power progression and economy

#### BF-10 - Item, build, loadout-power, and uplift framework

- **Status:** Resolved 2026-09-08.
- **Decision:** Use a 100-point multidimensional starter-loadout reference and
  initially cap first-release permanent direct power near 130, including at most
  10 General Progression points. Use three roughly 3% ranks per tier, nonlosing
  tier uplift, 70/30 primary-stat/trait budgets, the approved 40/20/20/20 build
  envelope, 15% synergy cap, roughly 10% direct build gain, and typed global caps.
- **Decision needed:** How much durable power may Items and Builds provide while
  preserving skill, role freedom, bounded growth, and exact paid equivalence?
- **Must resolve:** Item tier/rank/primary-stat curves; trait and consumable
  budgets; slot contribution; total loadout/shared build power budget; one-major/
  three-supporting costs and synergy caps; incompatibilities; recommended-power
  calculation; direct-power ceiling; boss relevance; old-item uplift eligibility/
  cost/result; paid/earned comparison tests; and no-mandatory-build thresholds.

#### BF-11 - Campaign, General Progression, Mastery, and recommendation pacing

- **Status:** Resolved 2026-09-08.
- **Decision:** Campaign advances on one meaningful first Victory. Start General
  Progress at 100 per Victory and 35 per eligible Defeat, open advanced builds/
  presets near 200, and distribute its ten power points through roughly 1,000.
  Use about ten boss-mastery ranks across 1,500 XP with 100/25 Victory/Defeat,
  modest Normal/Hard bonuses, five milestones, and three-attempt recommendations.
- **Decision needed:** How quickly should players gain access, options, bounded
  power, mastery expression, and useful private guidance?
- **Must resolve:** Victory/failure General awards and ratio; caps/rank XP;
  campaign fragment/destination timing; Hard unlock; option/system/item unlock
  schedule; direct-power endpoint; free respec/preset access; per-boss Mastery
  curve and finite endpoint; milestone/guarantee cadence; personal-record
  categories; recommendation dimensions/confidence; first-three-boss pacing; and
  returning-player/grandfathering expectations.

#### BF-12 - Rewards, drops, resources, crafting, upgrades, and Commerce parity

- **Status:** Resolved 2026-09-08.
- **Decision:** Start an Easy Victory at 100 general resource, 20 boss material,
  one roll, and 15% complete-item chance; Normal/Hard add bounded quantity and
  20%/25% chances. Retain 15%/25%/15% personal sources and 40% combined cap,
  five-Victory standard crafting, 30% salvage, explicit upgrade/uplift/consumable
  costs, Defeat-only general resource, No Contest ceilings/refunds, and exact
  paid/earned parity.
- **Decision needed:** What economy produces clear earned progress and exciting
  drops without grind traps, farming exploits, random-only access, or paid power?
- **Must resolve:** Outcome base grants; difficulty/performance/Risk/Cohesion
  modifiers and caps; resource sources/sinks/bounds; item pool weights/chances;
  duplicates and guarantees; four-to-six-clear chosen-item target; salvage ratio;
  deterministic recipes; rank/upgrade/uplift/consumable costs; starting-rank
  effects; first-clear/Signature/Mastery guarantees; failure/No Contest grants;
  paid/earned exact equivalence; economy simulation; and exploit guardrails.

### Checkpoint E - Evidence, governance, and completion

#### BF-13 - Balance scenario matrix, evidence, and acceptance thresholds

- **Status:** Resolved 2026-09-08.
- **Decision:** Every candidate passes deterministic invariants, at least 10,000
  seeded runs per core reference scenario, ±10%/±20% sensitivity, and economy/
  exploit simulations before human testing. Retain the GDD's two 8–12-player
  prototype rounds, 30-plus-new-player content-complete test, full segmentation,
  and structural-failure return-to-design rules.
- **Decision needed:** What simulation and human evidence is required before a
  balance revision or boss profile is approved?
- **Must resolve:** Deterministic simulations; scripted bots/reference inputs;
  target-age human playtests; first-attempt/learned/mastery cohorts; device,
  accessibility, role, difficulty, boss, solo/two/three/six-human, duplicate-role,
  uneven-skill, down/revival/disconnect/AFK, zero/max, cap/overflow, and economy
  scenarios; sample/confidence rules; success/down/stall/clarity/pacing metrics;
  exploit/sensitivity tests; outlier policy; and Analytics handoff.

#### BF-14 - Tuning workflow, versioning, approval, rollout, and rollback

- **Status:** Resolved 2026-09-08.
- **Decision:** Use Hypothesis, Playtest Candidate, Approved, and Retired states
  with complete evidence/diffs/compatibility. Domain, cross-system balance, and
  affected-content owners review; public-release approval also has an independent
  reviewer. Roll out in stages, lock active revisions, preserve confirmed history,
  and permit emergency pauses without bypassing recorded repair/compensation.
- **Decision needed:** How does a hypothesis become approved balance, and how may
  it change without invalidating content, history, fairness, or active attempts?
- **Must resolve:** Owner/reviewer roles; hypothesis-to-candidate-to-approved
  lifecycle; baselines and change budgets; rationale/evidence; formula/catalog/
  content compatibility; active snapshot lock; cohort experiments; disclosure;
  staged rollout; anomaly stop; rollback/forward fix; old-item/record/reward
  treatment; reproducibility; retention; and emergency authority.

#### BF-15 - Canonical outputs, boundaries, and completion audit

- **Status:** Resolved 2026-09-08.
- **Decision:** Canonical Markdown owns semantic meaning, constraints, matrices,
  budgets, evidence, lifecycle, and boss-profile template. A future versioned
  machine-readable catalog owns runtime numbers with generated readable diffs.
  First-three-boss profiles gate their packages, while unproven song-specific
  values remain explicit hypotheses. Existing systems retain nonbalance ownership.
- **Decision needed:** Which concrete artifacts make the Balance Framework usable
  by content, playtest, technical architecture, and implementation agents?
- **Must resolve:** Parameter registry; formula/order catalog; difficulty and
  roster matrices; proficiency and survival envelopes; combat/ability/power
  budgets; first-three-boss profiles; Progression/economy models; scenario and
  invariant matrices; machine-readable versus explanatory sources; semantic
  outputs; consumer references; validation tooling; unresolved-number labeling;
  UI/Audio/Multiplayer/Player Data/technical exclusions; architecture handoff;
  and final contradiction/orphan audit.

## 4. Completion criteria

`BALANCE_FRAMEWORK.md` is complete only when:

- BF-01 through BF-15 are resolved;
- every gameplay/economy tunable has one owner, unit, reference frame, safe
  range, revision state, deterministic order, and validation path;
- difficulty and one-to-six-human scaling preserve all approved fairness,
  contribution, role, reward, and active-snapshot invariants;
- Rhythm, Combat, Survival, Encounter, Positioning, Abilities, Solo Support,
  Items, Builds, Progression, Rewards, and Commerce have bounded interoperable
  models without duplicated authority;
- first-release hypotheses are clearly distinguished from evidence-approved
  values and remain reproducible by revision;
- simulation, playtest, sensitivity, exploit, and failure matrices cover every
  approved cohort and critical boundary;
- changes cannot silently alter active attempts, confirmed history, paid/earned
  equivalence, accessibility neutrality, or player-owned value; and
- Content Authoring, Playtest/Analytics, UI/Audio, Player Data, and Technical
  Architecture receive explicit inputs without being absorbed into balance.

## 5. Change log

- **2026-09-04:** Resolved BF-01 through BF-03. Defined balance priorities,
  cohort-specific outcome hypotheses, the versioned parameter registry, and the
  difficulty/population scaling model. Progress is 3 of 15 questions.
- **2026-09-04:** Resolved BF-04 through BF-06. Defined passage contribution
  units, timing/hold/re-entry hypotheses, Combat calculation/cap rules, and the
  Ward/downing/revival/recovery envelope. Progress is 6 of 15 questions.
- **2026-09-04:** Resolved BF-07 through BF-09. Defined Resolve/finishing pacing,
  result tiers, attack/telegraph/position pressure, Hype and group-action pacing,
  and bounded solo-acolyte support. Progress is 9 of 15 questions.
- **2026-09-08:** Resolved BF-10 through BF-12. Defined first-release durable
  power budgets, item/build/rank/uplift relationships, General and Boss Mastery
  pacing, resource/drop/recipe costs, personal reward caps, and paid parity.
  Progress is 12 of 15 questions.
- **2026-09-08:** Resolved BF-13 through BF-15. Defined simulation and human
  evidence gates, structural-versus-numeric findings, tuning lifecycle and
  approval, staged rollout/rollback, canonical artifacts, machine-readable data
  handoff, and system boundaries. The interview is complete at 15 of 15.
- **2026-09-08:** Reconciled all approved decisions into canonical
  `BALANCE_FRAMEWORK.md` and completed the supporting-specification audit.
- **2026-09-04:** Created the 15-question Balance Framework interview from the
  GDD, Systems Map Difficulty & Scaling contract, and deferred balance handoffs
  in all thirteen approved system specifications.
