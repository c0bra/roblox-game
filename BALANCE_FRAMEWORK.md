# Bands Battle Balance Framework

- **Status:** Approved
- **Approved:** 2026-09-08
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#48-difficulty--scaling)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **System specifications:** [`SYSTEM_SPECIFICATIONS_TRACKER.md`](SYSTEM_SPECIFICATIONS_TRACKER.md)
- **Content contract:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)
- **Player-data contract:** [`PLAYER_DATA.md`](PLAYER_DATA.md)
- **Decision source:** [`BALANCE_FRAMEWORK_WORKING.md`](BALANCE_FRAMEWORK_WORKING.md)
- **Interview plan:** [`BALANCE_FRAMEWORK_QUESTIONS.md`](BALANCE_FRAMEWORK_QUESTIONS.md)

## 1. Role and authority

This document defines the shared balance language and first-release tuning
framework for Bands Battle. It turns the approved system behaviors into explicit
units, relationships, caps, ranges, starting hypotheses, scenario matrices,
evidence gates, and change-control rules.

It owns:

- cross-system balance priorities and cohort/outcome targets;
- the common parameter-registry and revision contract;
- difficulty and one-to-six-human scaling relationships;
- shared proficiency units, budgets, caps, and comparison envelopes;
- the initial numeric hypotheses recorded here;
- simulation, sensitivity, exploit, and human-evidence minimums;
- balance-candidate approval, rollout, rollback, and retirement; and
- the required structure of machine-readable balance data and boss profiles.

Each gameplay or economy system still owns what its parameters mean, when they
apply, and which mutations or effects are legal. Central registration does not
transfer semantic authority to Balance Framework. Content Authoring owns
authentic song/encounter opportunity; UI/UX and Audio Presentation own
presentation tuning; Multiplayer owns session/service policy; Player Data owns
durability; Playtest/Analytics owns research and collection operations; and
Technical Architecture owns implementation.

The framework is approved design. Unless an entry is explicitly backed by an
Approved balance revision, numeric values in this document are **Hypotheses** for
simulation and playtesting, not immutable shipping constants.

## 2. Governing invariants

1. **Understanding before optimization:** numeric tuning cannot conceal an
   unclear rule, unexplained outcome, impossible action, or unreadable cue.
2. **The song remains the clock:** difficulty and population never change song
   speed, duration, structure, chart time, or authored event time.
3. **Equal maximum contribution:** every playable role/difficulty has the same
   maximum pre-difficulty contribution for an equivalent passage.
4. **Easy is not weaker:** simplified charts never lower available maximum
   Combat, support, recovery, group, progression, or reward value.
5. **Named difficulty axes only:** Easy/Hard change declared parameters rather
   than applying a hidden global multiplier.
6. **Sublinear population pressure:** added humans increase total challenge while
   preserving breathing room for movement, recovery, and uneven skill.
7. **No negative teammate value:** weak, inactive, downed, or disconnected play
   omits possible positive contribution but never subtracts another player's.
8. **Roles and duplicates are neutral:** balance never requires conventional
   instruments, party composition, or duplicate-role penalties.
9. **Solo remains human-led:** acolytes are fixed support, never fabricated human
   performance, decisive layer breaks, recovery help, risk, or rewards.
10. **One deterministic pipeline:** every value has declared units, owner, order,
    cap, precision, rounding, source, and revision.
11. **Zero remains zero:** ordinary zero earned performance cannot become positive
    output through multipliers or procs.
12. **No hidden recursion:** an effect cannot re-enter normalization, occupy an
    undeclared stage, copy a primary route, or multiply from roster/message count.
13. **Danger is accumulated and readable:** an ordinary single attack does not
    down a full-Ward player, and a Miss never directly removes Ward.
14. **Power is bounded and multidimensional:** gear supplies most direct growth;
    builds emphasize choices/tradeoffs; progression power is finite.
15. **Rewards remain individual:** party size never divides rewards or improves
    the underlying personal pool, and bonuses are bounded/nonnegative.
16. **Failure is useful but not optimal:** Defeat may grant modest practice/
    general progress but cannot outperform Victory or grant victory-owned value.
17. **Earned and paid function are identical:** paid equipment has one exact
    earnable equivalent and the same caps, upgrades, uplift, and tests.
18. **Accessibility is neutral:** accessibility, Hold Assist, calibration, device,
    age, privacy, and spending never reduce value, access, or public treatment.
19. **Attempts lock revisions:** balance changes never alter active play or
    recalculate confirmed outcomes, rewards, records, purchases, or history.
20. **Evidence is segmented:** one favorable aggregate cannot hide a materially
    failing role, device, cohort, difficulty, boss, or population.
21. **Simulation does not prove experience:** human evidence is required for
    clarity, fairness, attention, cooperation, and fun.
22. **Hypotheses are honest:** unavailable song-specific values remain labeled
    Hypothesis rather than being invented to make a table look complete.

## 3. Balance goals and player cohorts

Balance optimization priority is:

1. fairness and readable cause-and-effect;
2. meaningful player skill and choices;
3. comparable solo and cooperative viability;
4. satisfying progression and reward pacing; then
5. long-term mastery challenge.

An acceptable outcome rate does not excuse unexplained damage, unclear victory/
defeat, unreadable threats, weak choices, empty stretches, or farming exploits.
Evidence combines outcome with layer pace, downing/recovery, abandonment/retry,
comprehension, group use, build diversity, and economy/progression behavior.

### Initial outcome hypotheses

| Difficulty | Intended cohort | First relevant attempt | After learning |
|---|---|---:|---:|
| Easy | Onboarding-complete/new | 55–75% Victory | 75–90% |
| Normal | Developing/competent | 35–60% Victory | 55–75% |
| Hard | Practiced/mastery | Not tuned against new players | 45–65% on early serious attempts, rising with mastery |

These ranges are hypotheses, not universal boss quotas or player promises.
Evidence separates first attempt, onboarding-complete, learned/competent, and
practiced/mastery cohorts. It also segments boss/content revision, role, device/
control profile, accessibility configuration, solo/co-op roster, and progression
or power band.

Hard should not remain a permanent coin flip after mastery. Easy should forgive
more errors without removing meaningful decisions. Normal remains the reference
expression of the intended encounter.

## 4. Versioned balance registry

Balance uses one logical registry composed of domain-owned parameter tables. The
registry may later be split physically, but discovery, validation, compatibility,
revision lock, and audit behave coherently.

Every parameter or curve entry defines:

- stable parameter identity and semantic owner/domain;
- units and reference frame;
- scalar, table, formula, or curve shape;
- initial/default value and permitted test range;
- dependencies, application stage, ordering, and caps;
- deterministic precision and named rounding boundary;
- compatible content, catalog, and schema references;
- missing/invalid/incompatible behavior;
- affected consumers, scenarios, and invariant tests;
- lifecycle state and revision lineage;
- author, reviewers, rationale, evidence, and change timestamp; and
- explicit successor/equivalence when retired.

Parameter lifecycle is:

`Hypothesis -> Playtest Candidate -> Approved -> Retired`

A public attempt requires one exact compatible Approved balance revision.
Hypothesis or Candidate values may run only in explicitly identified approved
test contexts. Developer/debug overrides cannot enter ordinary public play,
produce durable rewards/progression, or masquerade as Approved.

Missing, unknown, out-of-range, corrupt, or incompatible critical balance data
blocks deployment before commitment or invokes the owning system's safe
cancellation/No Contest path. Clients, chart files, cached UI, or audio state
never provide a hidden fallback.

### Precision and rounding

Authoritative calculation uses deterministic fixed precision. Every unit declares
its scale. Intermediate display rounding does not feed calculation. Rounding
occurs only at named display or discrete-application boundaries, and pre-rounded
evidence is retained where results, rewards, or records depend on it.

## 5. Difficulty and population scaling

Normal is the reference profile. Easy and Hard change explicitly named axes:

- chart complexity and judgment windows;
- Resolve requirements and finishing proficiency;
- attack combinations, warning, target pressure, and damage;
- survival, recovery, and Easy's conditional extra Crescendo;
- Cohesion/group thresholds; and
- bounded reward efficiency or mastery-cosmetic eligibility.

Difficulty does not change the common normalized contribution maximum, song,
controls, role identity, accessibility rights, Combat pipeline, or essential
reward pool.

### Initial active-human relationship

| Humans | Solo-player equivalents |
|---:|---:|
| 1 | 1.00 |
| 2 | 1.75 |
| 3 | 2.50 |
| 4 | 3.25 |
| 5 | 4.00 |
| 6 | 4.75 |

Each Resolve layer combines its difficulty base and active-human relationship
when it opens, then locks. A disconnected player remains counted during rejoin
grace. Grace expiry affects only unopened layers. Acolytes never count.

Additional-human pressure uses validated readable target/geometry coverage. The
initial simultaneous individually targeted count is about half the eligible
active roster and never above three. More humans do not accelerate song/attack
time, densify an individual's chart, or multiply personal incoming damage.

Duplicate roles remain neutral. Weak/inactive/downed/disconnected players add
little or no positive share but never reduce another player's value.

Solo acolytes and population scaling initially target solo and cooperative
Victory rates within roughly ten percentage points at comparable measured human
proficiency. This does not force identical play or permit fabricated solo
performance. Party size does not change or divide the personal reward table.

## 6. Rhythm proficiency and contribution

### Initial judgment hypotheses

| Difficulty | Perfect | Great | Good | Miss |
|---|---:|---:|---:|---:|
| Easy | ±90 ms | ±150 ms | ±230 ms | Beyond ±230 ms |
| Normal | ±60 ms | ±110 ms | ±170 ms | Beyond ±170 ms |
| Hard | ±45 ms | ±85 ms | ±135 ms | Beyond ±135 ms |

Initial grade factors are Perfect 100%, Great 80%, Good 50%, and Miss 0%.
Every revision preserves monotonic ordering.

Every passage has a hidden maximum of 1,000 normalized contribution units for
every playable role/difficulty. Approved event weights may distribute those units
differently according to authentic material. UI may show percentages or summaries
rather than the internal scale.

### Holds and premature attempts

A hold initially assigns 50% of its weight to the graded onset and 50% linearly
across maintained duration. Hold Assist earns the identical model. An 80 ms
global release grace initially absorbs accidental contact loss. Release beyond
grace stops future duration value without creating another judgment.

An identifiable premature attempt may associate with an upcoming event up to
300 ms early, capped by nearer unresolved material. Outside the selected
difficulty's Good window, it resolves as Early Miss. It cannot cross or steal
another event. Contact-bounce filtering remains technical input tuning.

### Participation and re-entry

Re-entry after suspension requires at least 500 ms of visible preview before the
first eligible note. If authentic material cannot meet it, play advances to the
next eligible candidate without fabricated notes or wider judgments.

A normal encounter initially targets meaningful rhythm participation during
roughly 65–80% of its duration for an offered role, adjusted for authentic rests,
dropouts, song function, and evidence. Inactive time remains meaningful without
fake performance.

Execution quality, participation coverage, and earned contribution remain
separate. Sparse/dense roles compare through normalized opportunity and
trustworthy coverage, never raw note count. Calibration changes alignment only,
not grade windows, difficulty, maximum contribution, or rewards.

## 7. Combat conversion and caps

Accepted normalized contribution follows Combat's established order:

1. normalized earned Rhythm value;
2. versioned base rate for the effective intent/effect;
3. applicable equipment and build modifiers;
4. the position modifier legal for that effect;
5. explicit encounter, target, or difficulty modifiers; then
6. the effect-specific cap.

The 1,000-unit passage scale is a common input, not a promise that Attack,
Defend, Hype, revival, and support produce equal typed output. Domain base rates
convert into their own explicit units.

Values in one declared category add before one category cap. Separately budgeted
categories multiply once at their fixed stages. Sources cannot re-enter, occupy
undeclared stages, recurse, or spend the same normalized portion twice.

As an initial guardrail, ordinary Attack from identical Rhythm contribution caps
near 1.6 times the unmodified Middle-position starter result after equipment,
build, and position. BF-10's durable power budget must fit inside it.

Committed Signature bases and group effects have separate explicit budgets
because they are limited earned resources. They do not raise ordinary Attack's
cap or create a second primary route.

Zero ordinary performance remains zero. Positive changes remain monotonic until
their declared cap. Fixed-total multi-target effects divide their budget;
per-recipient effects use a roster-aware total cap.

## 8. Ward, danger, and recovery

Reference maximum Ward is 1,000 before approved equipment/effect changes.

| State | Reference range |
|---|---|
| Safe | At least 50% of current maximum |
| Fractured | Below 50% and at least 25% |
| Critical | Below 25% and above zero |
| Downed | Zero |

No ordinary single readable attack may down a player from full normal Ward.
Combined threats remain separately identified/ordered rather than disguised as
one hit.

On Normal, a full-Ward player without mitigation initially requires roughly four
to six substantial failed threat responses to become Downed. Easy permits more;
Hard may permit fewer while retaining counterplay and the no-ordinary-one-shot
rule.

Ordinary combined Defend/reduction mitigation initially caps near 75% for one
threat. Owning-system cancellation, movement avoidance, return protection, or
compatible cover may reduce a threat to zero at their earlier gates.

Temporary reinforcement initially caps at 25% above normal maximum Ward.
Restoration caps at normal maximum unless a separately budgeted conversion grants
reinforcement. No effect creates a second life bar.

One competent co-op revival participant should normally complete recovery in one
short authentic phrase. Additional participants accelerate with diminishing
returns and a roster-aware cap.

Co-op revival initially returns about 35% Ward for one competent helper and may
rise toward 60% with stronger/multiple help. Solo emergency recovery starts near
35%. Consumables cannot add another attempt.

Return protection/settling remains about two beats, initially clamped to
0.75–1.25 seconds, followed by the first event satisfying the 500 ms preview.
Downing remains accumulated understandable danger, not direct punishment for one
Rhythm Miss.

## 9. Resolve, Momentum, finishing, and result tiers

Each boss derives Resolve from authentic opportunity before the next authored
opening, locked difficulty/roster, and target proficiency rather than arbitrary
health unrelated to song material.

Initial resistance and Finishing Cadence proficiency hypotheses are:

| Difficulty | Target share of available normalized contribution |
|---|---:|
| Easy | 55–60% |
| Normal | About 70% |
| Hard | 80–85% |

For the target cohort, each layer should usually break during the latter half of
its intended First Clash, Escalation, or Climax region, leaving slack before the
next opening/Finishing Cadence. Authored functions govern; song-progress
percentages are diagnostics rather than universal timers.

An early break feeds Momentum or post-break play without retiming. A late break
preserves sequential layers and may compress later opportunity without skipping.

Momentum caps initially at 20% of the next layer and cannot break it alone.
Third-layer post-break performance caps initially near 15% of total encounter
Resolve. It may improve result/reward potential and spectacle but never satisfies
Finishing Cadence or creates early Victory.

Any valid Victory earns the base result tier. A second private tier begins near
80% overall eligible performance and a top tier near 90%, including versioned
positive post-break/finishing evidence. Higher tiers do not require zero downs,
erase legitimate recovery, publicly rank players, or change campaign credit.

Invalid/No Contest compensation remains at the Rewards failure-equivalent
ceiling and never implies Victory or ordinary result-tier credit.

## 10. Attack, movement, and positional pressure

### Major warning hypotheses

| Difficulty | Starting major-threat warning |
|---|---|
| Easy | Roughly two measures |
| Normal | Roughly one to two measures |
| Hard | Roughly one measure |

The actual authored rhythm, required response, travel distance, device profile,
and accessibility evidence may require more lead. Difficulty never shortens below
proven response capacity.

Easy does not require simultaneous independent answers to two major threats.
Normal/Hard may combine at most two independent critical responses in one window,
and only after phone/accessibility/control/attention validation.

Candidate weights, family cooldowns, repeat limits, and intensity budgets are
boss-profile data. They prevent repeated focus on one player/family when
equivalent alternatives exist and preserve required recovery, Crescendo, Band
Call, and finishing reservations.

Attack damage derives from section 8's Ward/downing envelope. Content selects
approved threat identities/tags rather than embedding private damage rules.

### Movement hypotheses

- one movement charge;
- roughly 0.75 seconds travel per edge; and
- two-beat charge recovery, clamped initially to 0.75–1.25 seconds.

Multi-edge routes pay the same cost per edge. No balance source changes universal
movement time, charge count/recovery, settling, route cost, or invulnerability.

### Position and Risk hypotheses

| Tier | Attack output | Incoming danger | Reward potential |
|---|---:|---:|---:|
| Near | +25% | +30% | +25% |
| Middle | Baseline | Baseline | Baseline |
| Rear | −20% | −25% | No Risk Bonus |

Gear/builds cannot modify these ratios. Total banked personal Risk Bonus caps
initially at +25%. Only successful normalized settled performance earns it; only
the current phrase's unbanked amount is lost on movement/downing.

Compatible cover may fully prevent one threat at its earlier gate. It is not
universal immunity and cannot retarget a committed attack.

Every attack, graph, position, displacement, hazard, and overlap preserves at
least one readable legal response within target/warning/survival limits.

## 11. Hype, group actions, and solo support

### Hype and Signature

Slow Hype from competent engaged Attack/Defend initially produces one charge
around 60–75% of song progress. It stores no second charge.

Fast-Hype routing initially fills after roughly two to three strong eligible
passages, each of which sacrifices its ordinary Attack/Defend effect.

A Signature's guaranteed base initially equals about 50% of one competent
passage effect in its typed domain. Accepted activation performance may scale it
toward one full competent passage effect, subject to typed caps.

### Band Calls and Crescendos

Band Call readiness normally arrives around 35–50% of the song for a competent
engaged player. One committed use prevents recharge. Shared lockout remains about
eight measures and cannot be shortened below its global floor.

Call initiator base and human shares have separate nonnegative budgets. An expert
remains useful alone but cannot create a larger inactive roster's top tier.

| Crescendo tier | Eligible-human maximum contribution |
|---|---:|
| Echo | Below 40% |
| Crescendo | 40–74% |
| Full Crescendo | At least 75% |

Echo is the minimum committed outcome. Weak play never subtracts another share.

Easy's one additional Crescendo becomes eligible only when projected remaining
resistance exceeds expected remaining eligible output by about 20% and living
players are under meaningful survival pressure. The exact combined function is
versioned and cannot activate on Normal/Hard.

### Solo acolyte hypotheses

- **Vanguard:** about 10% of the triggering competent human Attack packet,
  applied afterward and clamped before a decisive layer break.
- **Warden:** about 5% Ward reinforcement at clean boundaries near every eight
  measures, within the shared reinforcement cap.
- **Herald:** about 10% additional human-earned readiness and at most 20% of one
  target human group share at commitment.
- **Suppression:** about four measures, with no stacking, refresh, extension, or
  catch-up.

Acolytes never affect human judgments, records, Risk, rewards, recovery attempts,
item/build power, or human roster count. Their tuning must remain inside the
rough ten-point solo/co-op comparison envelope while keeping the human decisive.

## 12. Items, builds, and durable power

A complete starter loadout defines a 100-point internal multidimensional power
reference across offense, Ward/defense, and support/utility. It is not a public
universal gear score.

First-release permanent direct-power growth initially caps near 130 points for a
complete configuration, including General Progression. Power remains a vector;
different specializations are not flattened into damage.

Gear supplies most durable growth. General Progression contributes at most ten
points. Shared hook/category caps and the section 7 ordinary Attack ceiling still
apply.

### Item hypotheses

- each tier has a base and roughly three upgrade ranks;
- each rank adds about 3% of that slot's base budget;
- a next-tier base equals or slightly exceeds the previous maximum rank;
- about 70% of an item's budget belongs to its visible primary stat; and
- about 30% belongs to its fixed trait.

Traits remain typed/capped and never become random hidden affixes. Uplift advances
one tier, preserves identity/trait/appearance/references, and resets to the
nonlosing target-tier base. Old or paid items never auto-scale.

### Build hypotheses

The existing 100-unit build envelope remains:

- one major at most 40 units; and
- three distinct supporting options at most 20 units each.

Unused slot capacity does not transfer. Synergy is charged inside participating
budgets and caps initially at 15% of total build value.

A build's direct-output improvement over equivalent gear should ordinarily remain
near 10%. Most value comes from conditions, tradeoffs, redistribution, support,
timing, or playstyle rather than unconditional stats.

Recommended power privately compares offense, Ward/defense, and relevant
ability/support utility:

| Band | Profile comparison |
|---|---:|
| Prepared | At least 90% |
| Challenging | 75–89% |
| Far Below | Below 75% |

It is advice, never an access, matchmaking, social, or reward gate.

A prepared consumable may not exceed roughly one competent passage effect,
bypass typed caps, add recovery attempts, become required for baseline Victory,
or change the two-slot model.

## 13. Campaign, General Progression, and Mastery

One eligible meaningful first Victory on any difficulty completes the boss's
campaign node, restores its fragment, and unlocks satisfied destinations. A
meaningful Normal Victory separately unlocks that boss's Hard difficulty.

### General Progression hypotheses

| Outcome | General Progress |
|---|---:|
| Eligible Victory | 100 |
| Eligible Defeat | 35 |
| Invalid/No Contest | No ordinary grant; explicit compensation at most 35 |

Advanced build editing and all three full spec-preset slots initially open near
200 total progress, roughly two Victories.

Broader option/consumable access and the ten-point permanent-power allowance
spread through approximately 1,000 total progress. Power milestones are useful
and finite; later General rewards emphasize options rather than endless stats.

Defeat progress remains substantially less efficient than Victory and cannot
grant campaign credit, boss material, item rolls, Signature items, first-clear
choices, or Hard access.

### Boss Mastery hypotheses

Each boss has about ten visible ranks across roughly 1,500 total Mastery XP.

| Outcome/difficulty | Mastery treatment |
|---|---:|
| Eligible Victory | 100 base XP |
| Eligible Defeat | 25 base XP |
| Normal | About +10% XP |
| Hard | About +20% XP |

No rank, required function, deterministic item path, or completion reward
requires Hard. Milestones begin near ranks 2, 4, 6, 8, and 10, mixing lore,
cosmetics/titles, recipes, bounded sidegrades, and guarantees. Completion has a
finite capstone and no hidden endless prestige power.

### Recommendations and records

Difficulty-up guidance requires at least three recent relevant strong attempts.
Difficulty-down guidance requires at least three repeated relevant struggles and
distinguishes probable calibration/device problems. Advice remains private and
never changes the player's selection.

Personal-record families remain separate for execution, participation/holds,
Combat, Ward/survival, position/Risk, cooperative help, and overall performance.
They are not public damage rankings or balance authority.

New catalog prerequisites grant retroactively when already satisfied. Returning
players retain earned nodes, fragments, ranks, unlocks, records, and equivalent
access without replay or expiry.

## 14. Rewards and economy

### Outcome base hypotheses

| Eligible outcome | General resource | Boss material | Boss-pool roll | Complete-item chance |
|---|---:|---:|---:|---:|
| Easy Victory | 100 | 20 | Exactly one | 15% |
| Normal Victory | About 110 | About 22 | Exactly one, same pool | 20% |
| Hard Victory | About 120 | About 24 | Exactly one, same pool | 25% |
| Defeat | About 30 | None | None | None |
| Invalid/No Contest | No ordinary loot; failure-ceiling compensation plus causal refunds | None | None | None |

Normal/Hard do not change functional pool identity or roll count.

### Personal bonus hypotheses

| Source | Maximum |
|---|---:|
| Normalized personal performance | +15% |
| Banked personal Risk | +25% |
| Cohesion or solo breadth | +15% |
| Combined non-difficulty bonus | +40% |

Percentages apply relatively to explicitly permitted quantity, complete-item
chance, or starting-rank fields, not as undisclosed percentage points. They do
not change pools, add rolls, alter guarantees, grant campaign/Hard access, or
create difficulty-exclusive essential function.

### Initial costs

| Action | General resource | Boss material |
|---|---:|---:|
| Standard chosen item | 200 | 100 matching boss |
| Upgrade rank 1 | About 40 | About 5 matching boss |
| Upgrade rank 2 | About 70 | About 10 matching boss |
| Upgrade rank 3 | About 100 | About 15 matching boss |
| One-tier uplift | About 200 | About 60 current-tier boss plus 20 original boss |
| Ordinary consumable | About 25 | None |

The standard item's boss-material cost equals five baseline Easy Victories. A
random item never consumes or resets that path.

Voluntary duplicate power-item salvage initially returns 30% of equivalent boss-
material crafting cost and remains below recreation. Cosmetic duplicate
conversion remains separate and power-free.

Signature items may cost more but require a finite visible Victory recipe or
Mastery/clear milestone. They never require Hard, random success, or Robux.

An eligible Defeat grants no boss material, roll, deterministic Victory progress,
campaign credit, or Hard unlock. No Contest uses the Defeat compensation ceiling
plus exact causal consumable refunds once.

Paid equipment uses the exact same-tier functional definition as its visible
earnable equivalent. Quote tier freezes at the player's unlocked tier. Rank,
uplift, caps, tests, and migration are identical; only appearance/source lineage
may differ.

## 15. Scenario and evidence matrix

Every Playtest Candidate first passes deterministic formula, order, identity,
revision, monotonicity, cap, and invariant tests.

Each core reference scenario then runs at least 10,000 reproducible seeded
simulations. Evidence records exact reference-player model, input/content/balance
revisions, seed set, distributions, outputs, and failures.

Changed parameters receive legal sensitivity runs near ±10% and ±20%. Coupled
parameters vary independently and together to reveal cliffs, unstable feedback,
cap saturation, and dominant variables.

Economy simulation covers expected/adverse random streaks, duplicates,
deterministic paths, resources/sinks, upgrades/uplift/crafting, consumables,
failure farming, old/current content, paid equivalence, arbitrage, overflow, and
long-account projections.

Reference-player models include zero/inactive, developing, competent, practiced,
mastery, uneven-group, and targeted-failure patterns. Bots are measurement
fixtures, never comprehension/fun evidence.

### Human evidence minimums

- Smallest complete prototype: at least two rounds of roughly 8–12 new players
  ages 10–14, with most play on representative phones/tablets.
- Content-complete first release: at least 30 new target-age players plus
  returning players.
- Material structural change: two consecutive successful testing rounds before
  evidence is considered stable.

Testing retains all GDD comprehension, telegraph, attention, audio, Results,
voluntary-next-action, accessibility, and impossible-combination gates.

Evidence is segmented by boss/content revision, role, difficulty, device/control/
output, accessibility, progression/power, experience cohort, solo/two/three/six
humans, duplicates/all-same roles, and uneven skill.

The matrix covers down/revival/all-down; disconnect/grace/rejoin/AFK; movement/
cover/hazard/displacement; Hype/Signature/Call/Crescendo/acolytes; zero/threshold/
cap/overflow; economy streaks; invalid data; and snapshot revision boundaries.

### Structural versus numeric findings

The following return to design/content rather than being hidden by tuning:

- impossible attacks or no readable legal response;
- unexplained damage, result, or reward;
- players watching only the staff and missing the boss;
- surprising movement recovery or ignored/unreadable Defend;
- quiet passages with no meaningful attention/decision;
- Easy feeling weaker because it has fewer notes;
- any offered role repeatedly lacking authentic activity;
- weaker players feeling blamed or stronger players feeling punished;
- routinely missed group invitations;
- players unable to explain victory/defeat; or
- excessive one-off content-pipeline exceptions.

Numeric tuning is appropriate only when players understand the behavior and
evidence identifies a parameter relationship inside its approved range.

Outliers require predefined data-quality reasons for exclusion, with count and
reason retained. Aggregates cannot hide a materially failing segment.

[`PLAYTEST_AND_ANALYTICS.md`](PLAYTEST_AND_ANALYTICS.md) operationalizes practical
participant sessions, minimum safeguards, instrumentation, simple sample
interpretation, data quality, retention/access, reports, and sign-off without
weakening these gates.

## 16. Tuning approval, rollout, and rollback

Every proposal records stable identities, owner, before/after values/formulae,
rationale, expected effect, affected systems/content/cohorts/scenarios,
compatibility/migration, simulations, human evidence, risks, and decisions.

The semantic domain owner approves legality and meaning. Cross-system balance
review verifies scaling, caps, power, progression, economy, and unintended
effects. Affected song/boss owners approve content-profile compatibility.

One person may cover multiple responsibilities during prototyping. Before public
release, a release-critical revision also requires an independent reviewer who
was not its sole author.

Prefer one parameter or one clearly coupled set per Candidate. Larger rebalances
list every coupled hypothesis rather than hiding unreviewed changes in volume.

Rollout order is:

1. deterministic tests and simulation;
2. internal play;
3. private target-age testing;
4. limited public use where appropriate; then
5. full Approved use.

Each stage defines success, stop, and rollback signals. Cohort experiments use
stable assignment, exact revision identity, privacy review, and reward/
progression treatment that cannot silently disadvantage players or break paid/
earned parity.

Active attempts retain locked revisions. Rollout, rollback, hotfix, or experiment
cannot alter in-progress calculations or rebuild a frozen result under new data.

Rollback selects the last exact compatible Approved revision and does not
recalculate/revoke confirmed outcomes, rewards, items, Progression, records,
purchases, or compensation.

Incompatible player facts use explicit equivalence, migration, replacement
choice, or fair compensation through their owning systems. No balance change
silently weakens/deletes value or invents a player choice.

A serious exploit, corrupt bundle, unsafe attack, or transaction-integrity defect
may immediately pause new affected attempts/transactions and select an exact safe
revision. Permanent repair still records evidence, authority, migration/
compensation, and review.

No revision introduces invisible dynamic difficulty, personalized hidden reward
chances, player-specific punishment, or mutable paid-product mapping.

## 17. Canonical artifacts and boss profiles

This Markdown document is authoritative for semantic meaning, constraints,
relationships, ranges, evidence, and governance.

A future versioned machine-readable balance catalog is authoritative for runtime
numbers. It conforms to this framework and generates readable snapshots/diffs.
Runtime parameters may not exist only in code, chart files, UI text, or an
undocumented service setting.

Each boss-specific profile contains:

- exact boss, encounter, content, catalog, and balance identities;
- target cohorts and available authentic opportunity;
- layer, Momentum, post-break, finishing, and result hypotheses;
- difficulty and one-to-six-human transforms;
- attack family/weight/intensity/cooldown/target/telegraph/effect ranges;
- arena, movement, cover, hazard, displacement, position, and Risk values;
- Ward/downing/recovery, Hype/Signature/group, and solo-support settings;
- compatible item/build/power, progression, and reward references;
- parameter ranges, scenario/sensitivity results, and human evidence;
- lifecycle state, approvals, compatibility, and supersession.

Actual profiles for the first three bosses must pass this framework and Content
Authoring before their packages publish. Song-specific values without evidence
remain explicit Hypotheses and do not block approval of the framework itself.
They cannot enter ordinary public runtime until their compatible profiles are
Approved.

## 18. Semantic outputs and consumer boundaries

Balance data emits identified facts for:

- balance/catalog/profile revision and lifecycle state;
- difficulty/population transforms and locked targets;
- unit/formula/order/cap/rounding evidence;
- proficiency, contribution, Combat, survival, layer/finishing, position/Risk,
  Ability/group/solo, power, progression, mastery, reward, and economy results;
- cap/saturation/discard, missing/incompatible, validation, and rollback; and
- simulation, sensitivity, cohort test, approval, and supersession evidence.

Every fact retains source, owner, revision, target/content/player/attempt context
where applicable, pre/post value, cap/rounding evidence, and safe reason.

Content Authoring references exact compatible profile identities and owns
musical/encounter opportunity, not private formulas. Gameplay/economy domains
consume their own registered parameters and retain semantic authority.

UI/UX consumes player-safe values, tiers, comparisons, and reasons but owns
layout/formatting/settings. Audio consumes semantic intensity/priority ranges but
owns loudness/mix/duck/device values.

Multiplayer retains invitation/queue/load/rejoin/AFK/ping policy unless a value
directly participates in a declared gameplay fairness formula. Player Data stores
exact references/frozen outcomes but never recalculates balance.

Playtest/Analytics measures approved facts without automatically changing
difficulty, recommendations, rewards, or player state. Technical Architecture
owns schemas, services, transport, loading, validation, security, deployment,
and observability products.

## 19. Deferred content, evidence, and technical work

The framework's behavior and initial hypotheses are complete. Downstream work
still includes:

- machine-readable registry/catalog schemas and validation tooling;
- first-three-boss content packages and profile-specific Approved values;
- complete item, ability, build, progression, milestone, reward, and economy
  catalogs inside these budgets;
- simulation/reference-player implementation and generated reports/diffs;
- target-age playtest execution under `PLAYTEST_AND_ANALYTICS.md`;
- final player-facing names, copy, and localized terminology;
- storage, transport, security, authority, experiment, rollout, rollback, and
  operational implementation; and
- iterative parameter promotion from Hypothesis to Approved evidence.

None may alter song timing, equal maximum contribution, fairness/accessibility,
role freedom, nonnegative cooperation, deterministic ordering, bounded power,
failure/value integrity, paid equivalence, active snapshots, or confirmed history.

## 20. Approval and change control

The owner interview resolved BF-01 through BF-15 on 2026-09-08. This document is
the canonical Balance Framework supporting specification.

A material change to balance priorities, outcome cohorts, parameter governance,
difficulty/population scaling, shared units/order/caps, power ceilings,
progression/economy relationships, evidence gates, lifecycle approval, rollout/
rollback, or source-of-truth boundaries requires an explicit amendment citing the
superseded decision.

Promoting or tuning a numeric Hypothesis inside these approved relationships does
not require redesign, but it does require a new versioned balance Candidate,
evidence, approval, compatibility assessment, and future-attempt-only rollout.
