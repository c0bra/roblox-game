# Bands Battle Combat and Player Survival

- **Status:** Approved
- **Approved:** 2026-08-21
- **Parent systems:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#42-combat) and
  [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#43-player-survival--recovery)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Rhythm dependency:** [`RHYTHM_GAMEPLAY.md`](RHYTHM_GAMEPLAY.md)
- **Decision source:** [`COMBAT_WORKING.md`](COMBAT_WORKING.md)
- **Interview plan:** [`COMBAT_QUESTIONS.md`](COMBAT_QUESTIONS.md)

## 1. Role and authority

This document defines how identified normalized musical performance becomes
combat effects and how those effects change a player's single Ward survival
resource, downed state, revival, and solo recovery opportunity.

Combat owns:

- Attack/Defend/Special selection, queuing, and automatic intent return;
- one-time routing of normalized Rhythm contribution;
- versioned conversion and modifier order;
- Attack-pressure, mitigation, reinforcement, restoration, and effect-request
  calculations;
- identified slow/fast Hype-eligible contribution facts;
- multi-target distribution and source/target attribution; and
- semantic combat-effect output.

Player Survival & Recovery, specified here as an integrated boundary, owns:

- maximum/current/reinforced Ward and readable thresholds;
- application of final incoming, restorative, and protective effects;
- active, downed, recovering, protected/settling, and returned state;
- cooperative revival progress state;
- the one-use solo emergency recovery opportunity; and
- returned Ward and re-entry protection.

These systems do not own Rhythm judgment; charts; boss attacks, Resolve,
Momentum, finishing, or encounter outcome; arena movement/cover state; item,
build, ability, or consumable definitions; persistent loadouts; rewards; or
presentation. Owning systems provide identified facts and consume the semantic
outputs established here.

## 2. Governing invariants

1. **Rhythm is already settled:** Combat consumes normalized contribution and
   never regrades notes or derives power from raw note count.
2. **One primary route:** an identified contribution is spent once and cannot
   simultaneously become multiple full-strength effects.
3. **Monotonic skill:** under identical non-rhythm state, more earned normalized
   contribution cannot produce a weaker result.
4. **Post-score builds:** equipment and specialization affect consequences only;
   they never change musical judgment or chart participation.
5. **Typed, bounded modifiers:** every legal modifier has a stage, category,
   budget, condition, cap, and attribution.
6. **No damage from an ordinary Miss:** Ward changes only through explicit
   incoming/restorative combat effects.
7. **One survival resource:** temporary reinforcement remains part of Ward and
   never becomes a second health system.
8. **Readable danger:** one ordinary failure does not cause unexplained downing;
   explicit boss effects and accumulated defensive consequences do.
9. **No double application:** stable causal identity makes duplicates harmless
   and group-size scaling explicit.
10. **Logical musical time wins:** network arrival time never changes combat
    order or creates retroactive effects.
11. **Confirmed history is immutable:** disconnect, downing, rejoin, and balance
    updates do not reinterpret accepted play.
12. **No public blame ranking:** personal and band facts support learning without
    ranking public players by damage or failure.

## 3. Inputs and immutable encounter configuration

At encounter staging lock, Combat binds exact versions of:

- content and schema revision;
- balance-data revision;
- selected difficulty and human-roster scaling;
- each player's validated loadout, build, Signature Special, Band Call, and
  prepared consumables;
- starting/current boss target opportunities;
- position graph and risk/cover semantics; and
- ability/effect definitions and category caps.

During the encounter, Combat consumes:

- `RHYTHM_GAMEPLAY.md` event/scoring-group identity, normalized intent portions,
  coverage, and participation boundaries;
- Boss Encounter threat, target, Commit/Impact, Resolve/Momentum eligibility,
  cancellation, finishing, and outcome-evaluation facts;
- Position state, route, risk tier, cover, avoidance, and movement boundaries;
- Multiplayer active/absent/returned roster identity;
- legal item/build/ability/consumable modifier sources; and
- current Combat/Survival state snapshots.

A content, balance, encounter, player, source, or target identity mismatch is a
validation failure. Combat never guesses which revision or state was intended.

## 4. Intent state and routing

Attack is the default. Defend and Special use the same instrument chart; they do
not load alternate notes or scoring rules.

Pressing an available intent shows it as queued immediately. It becomes effective
on the next playable beat or note supplied by Rhythm. Contribution before that
boundary keeps its prior intent. While moving or otherwise suspended, a valid
selection waits for the next playable material rather than creating a hidden
effect during inactivity.

An unavailable Special explains its state and does not change intent. Automatic
return from Special uses the stored previous Attack or Defend intent. Baseline
success never requires rapid mid-phrase switching, although exact boundary
attribution permits advanced optimization.

Every normalized portion receives one primary route:

- Attack pressure;
- Defend mitigation/reinforcement;
- accelerated Hype generation;
- committed Signature activation performance;
- cooperative revival;
- Band Call;
- Crescendo; or
- another explicitly exclusive cooperative/ability route.

Separately budgeted readiness or conditional utility may observe an accepted
portion, such as slow passive Hype or Band Call readiness. It cannot copy the
full primary effect, recurse as new normalized performance, or spend the source
again.

## 5. Deterministic effect pipeline

An accepted normalized contribution is converted in this order:

1. normalized earned value from Rhythm;
2. versioned base rate for the effective intent/effect kind;
3. applicable equipment and build modifiers;
4. the position modifier legal for that effect;
5. explicit encounter, target, or difficulty modifiers; and
6. the effect-specific cap.

Values in the same declared modifier category add. Independently budgeted
categories multiply once at their fixed stages. A modifier source cannot re-enter
the pipeline or occupy multiple stages unless its definition contains separately
budgeted effects.

Calculations use deterministic fixed precision and round only at the final
display or discrete-application boundary. Implementations may choose the numeric
representation later but must produce reproducible results from the same inputs
and balance revision.

Ordinary zero performance remains zero regardless of multipliers or procs. The
reliable base of an already-earned and committed Signature Special is the one
explicit exception; non-performance-triggered utility defined by another system
is a separate effect source, not fabricated contribution.

Difficulty usually changes Resolve requirements, target pressure, attack danger,
and explicitly tagged recovery values. It does not mutate Rhythm normalization
or silently amplify every player effect.

## 6. Attack and Boss Encounter handoff

An Attack portion creates one identified Attack-pressure packet containing:

- source player and causal Rhythm group/intent segment;
- content, balance, and encounter identity;
- logical musical time;
- Boss Encounter-supplied destination opportunity;
- pre/post-modifier value and cap evidence; and
- applicable position/risk attribution.

Boss Encounters owns current/future Resolve layers, openings, progress, breaks,
Momentum state/cap, finishing requirements, and application of Attack pressure.
Combat never maintains a shadow copy.

The Boss Encounter consumer applies accepted pressure to the current legal layer,
detects its break, and routes same-packet overflow or valid post-break Attack to
Momentum. A future locked layer cannot be damaged. If neither Resolve nor
Momentum is a valid supplied destination, the legitimate musical performance
still appears in personal results but creates no Attack effect; presentation
must not imply damage occurred.

Band Attack totals sum accepted packets while retaining private source
attribution. Results may show a player's own value and the band aggregate, not a
public player ranking.

## 7. Defend focus and mitigation

Defend automatically focuses the earliest unresolved telegraphed threat capable
of affecting that player. No additional threat-selection control is introduced.
Impact time chooses among overlapping threats; an exact tie uses stable authored
event order.

Once a threat passes Commit, its target set and assigned mitigation remain
locked until the owning system resolves or explicitly cancels it. Defend
contribution:

1. fills that threat's personal mitigation capacity; then
2. converts excess at a weaker bounded rate into temporary Ward reinforcement.

When no applicable threat exists, the whole Defend portion uses the weaker
reinforcement path. Mitigation already assigned to a threat persists through a
later intent selection and ordinary movement.

If movement, cover, cancellation, or another outcome prevents the threat from
damaging the player, unused threat-bound mitigation expires. It is not refunded,
copied, or retroactively turned into Ward. UI and Audio distinguish focused
threat, mitigation fill/cap/use/expiry, and reinforcement.

## 8. Ward model

Ward is the only first-release survival bar. At encounter start, current Ward
equals the calculated normal maximum. It has four player-facing states:

| State | Rule |
|---|---|
| Safe | At least 50% of current maximum |
| Fractured | Below 50%, above or equal to 25% |
| Critical | Below 25%, above zero |
| Empty / Downed | Zero |

Presentation uses geometry, label/motion, sound, and meter change rather than
color alone. Exact threshold crossing emits a semantic event.

### Incoming-effect order

Every identified incoming hit passes these gates:

1. owning-system cancellation or complete avoidance;
2. active return protection or explicit immunity;
3. position danger and attack/encounter/difficulty scaling;
4. applicable cover and other tagged reductions;
5. committed Defend mitigation for that threat;
6. temporary Ward reinforcement; and
7. current Ward.

A gate that reaches zero prevents later Ward loss. A value is consumed only by
its declared gate and cannot apply twice. Same-boundary hits retain separate
identity and follow stable encounter-event order. Current/reinforced Ward clamp
at zero, and one boundary can create only one down transition.

An ordinary rhythm Miss never creates an incoming hit. Its combat consequence is
reduced output, Hype, protection, support, or readiness.

## 9. Restoration, reinforcement, and maximum Ward

**Restoration** refills current Ward to the current normal maximum. Excess is
discarded unless an explicit separately budgeted effect converts it into
reinforcement.

**Reinforcement** is a temporary visually distinct segment of the same Ward
meter. It is consumed before normal Ward, combines only to one shared cap, and
creates no hidden reserve at cap. It persists until consumed, downing, or
encounter end. It does not become a second down threshold or survival resource.

A mid-combat maximum-Ward increase preserves the current absolute amount unless
the effect explicitly grants current Ward. A decrease preserves the absolute
amount where legal, then clamps current Ward to the new maximum. Combat loadouts
cannot be swapped during an active encounter.

Downed players cannot receive ordinary restoration or reinforcement. Revival
and solo recovery have dedicated return effects. Every applied, capped,
discarded, converted, consumed, and expired value retains source and target
attribution.

## 10. Player survival lifecycle

Player survival states are:

- **Active:** ordinary target and contribution eligibility apply.
- **Downed:** Ward is zero; ordinary targeting and performance are suspended.
- **Recovering:** an identified cooperative or solo recovery process is active.
- **Return protected/settling:** Ward has been restored, but ordinary targeting
  and chart participation remain suspended for the defined interval.
- **Returned active:** protection/settling ended and Rhythm has found a fair
  previewed re-entry note.
- **Attempt ended:** the player cannot return within the current attempt.

Downing preserves confirmed performance/combat history and does not request that
Abilities clear Hype. It also preserves unspent prepared resources,
spent-resource state, and the previous location when still legal. Temporary
reinforcement clears. A return may use the nearest legal Middle location when
the prior one is invalid.

Boss Encounters owns all-humans-down and solo-attempt-ended defeat. Survival
publishes authoritative player states after each musical boundary.

## 11. Cooperative revival

Co-op revival may begin after any downing while another human remains active.
There is no arbitrary per-player or band-wide revival count; remaining song time,
valid Activity Map candidates, sacrificed output, and all-humans-down are the
constraints.

At the dynamically selected fair boundary, each participant routes authentic
ordinary-chart contribution exclusively to one identified downed target. It
cannot simultaneously become Attack, Defend, Special, Band Call, Crescendo, or
another revival. Multiple participants add independently normalized progress,
accelerating completion. Weak play reduces only that participant's share.

Progress belongs to the downed target and survives contributor changes until
completion or invalidation by encounter end/all-humans-down. It never advances
from absence, fabricated notes, or acolyte performance.

Completion returns a tuned Ward amount, starting around 35% for one competent
participant and potentially rising toward roughly 60% with stronger/multiple
help. These are playtest hypotheses. The player receives about two beats of
protection/settling, then Rhythm resumes at its first fairly previewed note.

## 12. Solo emergency recovery

Solo has exactly one emergency recovery opportunity per encounter. Downing
triggers dynamic Activity Map selection of a clean instrument-aware boundary; an
urgent universal beat challenge may be used only under the approved authoring
rules when authentic material is unavailable soon enough.

The challenge is an identified temporary scoring stream, not ordinary Attack,
Defend, or Special contribution. Acolytes provide presentation only. Success
returns a tuned Ward amount, initially around 35%, plus return
protection/settling. Failure or any later down ends the solo attempt. Robux,
equipment, and consumables cannot add, purchase, or bypass another attempt.

## 13. Hype and Signature Special lifecycle

Successful ordinary Attack or Defend performance creates an identified
slow-Hype-eligible contribution fact. Selecting Special before full Hype stores
the previous Attack or Defend intent and routes subsequent contribution
exclusively into identified fast-Hype-eligible facts. Abilities applies the
revisioned gain values and owns the resulting resource state.

At full Hype:

- overflow is discarded;
- intent returns to the stored Attack or Defend for next playable material;
- exactly one charge is stored;
- Ready is communicated clearly; and
- the Signature never fires automatically.

Selecting Special while Ready arms the next ordinary scoring group but does not
spend Hype. If downing or another invalid state occurs before that group starts,
the arm cancels and full Hype remains. Once the group starts, the charge is
committed. The effect's guaranteed base resolves at the following valid musical
boundary even if execution is poor or participation ends mid-group. Normalized
performance scales only additional strength, duration, or utility.

Abilities consumes Hype when the committed effect enters guaranteed resolution
and signals Combat to restore the prior intent. Hype persists through ordinary
downing/revival, resets between encounters, stores no second charge, and has no
separate cooldown.

Abilities & Cooperative Actions owns Hype amount and lifecycle state, the
equipped Signature definition, activation commitment/consumption, recipient and
effect behavior, legal resolution boundary, and numeric values. Combat owns
Special intent/input routing, eligible slow/fast contribution facts, normalized
effect conversion, and automatic return to the stored intent. This ownership
split does not change the approved player behavior above.

## 14. Modifier and build contract

Every modifier is immutable/versioned and declares:

- source and effect tags;
- authoritative activation conditions;
- shared power-budget cost;
- pipeline stage and additive category;
- duration/expiration and cap behavior;
- recipient/distribution policy where applicable; and
- source/target/causal attribution.

Gear provides most direct power. Build choices emphasize bounded conditions,
tradeoffs, sidegrades, and hybrids. Category caps and a shared loadout/build
power budget prevent multiplicative escape and mandatory combinations.

Modifiers cannot change:

- charts, timing windows, judgments, calibration, note density, or Hold Assist
  value;
- movement time, charges, settling, telegraph/reaction time, or invulnerability;
- revival counts or solo recovery attempts;
- automatic note correction/autoplay;
- reward eligibility; or
- the authored baseline positional risk/reward ratios.

Position supplies baseline Attack, incoming-danger, and reward facts at their
owned stages. A build may trigger a separately budgeted and visible effect while
the player performs dangerously, but it cannot rewrite the universal position
multiplier or remove its danger.

The typed budget contract may later support traits, sidegrades, sets, sockets,
and advanced configuration without granting rhythm authority or adding required
combat controls.

## 15. Multi-target, group, and acolyte effects

A multi-target effect declares one of two budget forms:

- one fixed total divided deterministically among valid recipients; or
- a per-recipient value constrained by a roster-aware total group cap.

Effect definitions own recipient eligibility, priority, and fallback. First
release adds no manual teammate-target control solely for support. Invalid
recipients are removed deterministically before the defined fallback applies.

Group participants keep independently calculated shares. One player's weak play
never subtracts another's share or cancels an initiator's guaranteed base.
Duplicate roles do not alter eligibility or conversion.

Acolytes create identified fixed NPC packets with explicit solo caps. They have
no chart, grade, timing distribution, player performance/reward attribution, or
position risk multiplier. Vanguard support cannot perform the decisive layer
break; acolytes never progress emergency recovery. Other precise acolyte and
group-action behavior belongs to Abilities & Cooperative Actions.

## 16. Same-boundary ordering

Combat orders by logical musical timestamp, not network receipt. At one time:

1. apply scheduled intent, participation, and eligibility changes;
2. accept/finalize valid Rhythm contribution assigned to the boundary;
3. resolve player, support, revival, and cooperative effects;
4. resolve committed boss impacts/hazards in stable authored order;
5. apply Ward thresholds and down transitions; and
6. expose the completed snapshot for Boss Encounter outcome evaluation.

Each phase publishes one atomic snapshot for the next. Stable event identity is
the final tie-breaker inside a phase.

This order lets genuinely on-beat defense, restoration, protection, or revival
help before impact. A same-beat Resolve break does not cancel an attack that
already passed Commit unless its authored definition explicitly permits that
behavior. Boss Encounters evaluates finishing success, all-humans-down,
song-end, and other outcome rules after the boundary snapshot.

A queued intent effective at the boundary applies before its playable event.
Previously resolved contribution is never reinterpreted.

## 17. Validation, disconnect, and idempotency

Before authoritative application, validate:

- immutable content and balance revisions;
- encounter, player, source-event/scoring-group, intent, target, and effect
  identity;
- logical musical time and bounded delivery allowance;
- participation, survival, connection, and target eligibility;
- legal modifier sources and conditions; and
- deterministically recomputed value and cap evidence.

An accepted source/effect identity applies once. Duplicates are no-ops. An
out-of-order dependency may wait only within its logical-time delivery allowance.
Impossible, mismatched, negative, recursive, or wrong-state player effects are
rejected, not silently normalized into plausible values. Normal designed caps
are successful resolution and emit cap evidence.

Server-confirmed combat history is immutable. Local immediate Rhythm feedback is
not by itself authoritative shared combat application; technical architecture
must preserve responsiveness while validating shared effects.

A contribution logically completed before disconnect may be accepted within the
delivery allowance. The disconnect boundary blocks absent-period output. An
already committed boss impact resolves against the disconnect snapshot; after
that set, the disconnected player becomes ordinarily untargetable and contributes
nothing until safe return. Synchronization suspension produces no inferred
combat effect.

Duplicate, rejection, cap, late acceptance, absence, and invalid-session facts
remain private/system evidence rather than public blame labels.

## 18. Semantic output contract

Combat/Survival exposes causally linked facts for:

- intent queued/unavailable/effective/automatic-return;
- contribution accepted/routed/split/capped/expired/rejected;
- Attack pressure and received Resolve/Momentum disposition;
- Defend focus, mitigation fill/use/expiry, and reinforcement;
- Special routing/automatic return and identified slow/fast Hype-eligible
  contribution facts; Abilities emits Hype state and Signature transitions;
- incoming effect gates, reductions, final damage, and avoidance;
- current/max/reinforced Ward changes and threshold crossings;
- downing, revival target/progress/completion, solo recovery use/outcome,
  return protection, settling, and active restoration; and
- multi-target, group, NPC, source, recipient, and causal attribution.

As applicable, every fact carries content revision, balance revision, encounter,
logical time, source, target, effect definition, causal Rhythm/combat identity,
pre/post value, cap/discard evidence, and final state.

Results can derive personal Attack/Defend/Special contribution,
Resolve/Momentum disposition, Ward loss/restoration/reinforcement, attacks
avoided/defended/absorbed, revival help, group share, position modifiers, and
personal history. UI and Audio consume the same semantics at immediate levels;
Analytics observes without becoming authority. No consumer needs or receives a
public damage ranking.

Accessibility changes cue channels, scale, contrast, sound, haptics, or private
guidance. It never changes combat resolution or attribution.

## 19. Content Authoring reconciliation register

These entries must be reconciled into Content Authoring after specifications 2
through 12 are complete.

| Combat requirement | Semantic data | Required validation | Consumers | Compatibility/support status |
|---|---|---|---|---|
| Threat-focused Defend | Stable threat identity; affected-player eligibility; Telegraph/Commit/Impact/Recovery; mitigation-cap reference; cancellation | Targets lock at Commit; ties have stable order; mitigation cannot bind an impossible target | Combat, UI, Boss Encounters | Encounter baseline exists; runtime field set requires reconciliation |
| Attack destination | Resolve layer/Momentum opportunity identity and eligibility at scoring boundaries | No pressure reaches locked/future layers; overflow disposition is deterministic | Combat, Boss Encounters, Results | New explicit consumer contract |
| Incoming effects | Stable effect/target/geometry identity; damage/effect tags; cover interaction; difficulty reference; order | Gate order is complete; cover/avoidance and simultaneous impacts are reproducible | Combat, Survival, Positioning | Attack stages exist; effect tags/order need confirmation |
| Dynamic return | Recovery candidate/challenge identity, valid roster/role/difficulty, maximum delay, conflicts | Required co-op/solo return remains possible for supported configurations without a fixed trigger timestamp | Survival, Rhythm, Boss Encounters | Activity Map baseline exists |
| Signature/group boundaries | Stable activation/action identity, valid start/end/resolution boundaries, conflicts | Committed group resolves once; invalid pre-start request retains its resource when required | Combat, Abilities, Rhythm | Candidate model exists; effect boundary fields need reconciliation |
| Multi-target authored effect | Eligibility, distribution form, roster-aware cap reference, invalid-recipient fallback | Total application cannot multiply unintentionally with roster size | Combat, Abilities, Results | New explicit validator/contract |
| Same-time order | Stable global event order and exact logical time | Ties reproduce across export/runtime and never use receipt order | Combat, Boss Encounters, Multiplayer | Clock exists; global tie-order support must be confirmed |

Combat formulae, item/build definitions, Ward numbers, and balance tables remain
versioned system data rather than private values embedded in individual chart
notes. Encounter packages reference compatible definitions and expose only their
authored targets, tags, candidates, and boundaries.

## 20. Deferred tuning and technical work

Behavior is complete; these values and representations remain deliberately open
for playtesting or architecture:

- intent conversion and eligible Hype-input classification; Abilities owns
  passive/accelerated gain rates and the charge threshold;
- modifier category caps, shared power budget, and fixed-precision scale;
- Resolve-pressure, Defend-mitigation, and reinforcement conversion rates;
- Ward maximums, damage, reinforcement cap, restoration, and recovery amounts;
- revival target/scaling and return-protection duration;
- Signature base/bonus scaling and group-effect caps;
- delivery allowance, validation tolerance, and anti-cheat architecture;
- balance-data storage/versioning and authoritative transport; and
- analytics persistence and operational anomaly thresholds.

Tuning may not violate monotonicity, one-route spending, Rhythm authority,
positional baseline integrity, single-Ward clarity, recovery limits, deterministic
ordering, or immutable confirmed history.

## 21. Approval and change control

The owner interview resolved CM-01 through CM-12 on 2026-08-21. This document is
the canonical Combat and Player Survival & Recovery design specification.

A material change to conversion order, intent routing, Attack/Defend ownership,
modifier boundaries, Ward semantics, downing/revival/recovery rules,
same-boundary precedence, or authoritative validation requires an explicit
design amendment citing the superseded rule. Numeric balance changes inside the
approved behavior create a new balance revision and begin only with new
encounters.
