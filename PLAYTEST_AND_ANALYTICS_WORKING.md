# Bands Battle Playtest & Analytics Working Record

- **Status:** Interview complete; 15 of 15 questions resolved and reconciled
- **Started:** 2026-09-09
- **Question plan:** [`PLAYTEST_AND_ANALYTICS_QUESTIONS.md`](PLAYTEST_AND_ANALYTICS_QUESTIONS.md)
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#83-analytics--playtest-evidence)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Evidence authority:** [`BALANCE_FRAMEWORK.md`](BALANCE_FRAMEWORK.md)
- **Data boundary:** [`PLAYER_DATA.md`](PLAYER_DATA.md)
- **Canonical result:** [`PLAYTEST_AND_ANALYTICS.md`](PLAYTEST_AND_ANALYTICS.md)

## 1. Role of this record

This file progressively preserves owner answers, refinements, inherited
constraints, and cross-system handoffs from the Playtest & Analytics interview.
It is decision evidence for the future canonical specification, not yet the
final authority.

## 2. Inherited boundary

Playtest & Analytics owns the logical event and metric catalog, reviewed
collection and segmentation rules, target-age research procedure, evidence
synthesis, data-quality qualification, readiness reports, and research consent,
safeguarding, retention, and access boundaries.

Gameplay and product systems continue to own the meanings of the semantic facts
they emit. Player Data owns durable player records and account-data lifecycle.
Content Authoring owns publication validation and human content approval. The
Balance Framework owns numeric hypotheses, candidate lifecycle, simulations,
and tuning invariants. Policy and safety review confirm current external
requirements. Technical Architecture owns concrete schemas, transports,
services, vendors, storage, security, deletion implementation, dashboards,
alerts, and code topology.

Analytics observes approved facts and may support decisions. It never becomes a
hidden source of gameplay truth, changes a player's live or durable state,
creates public blame or skill labels, or promotes a design candidate by itself.

The complete inherited decision set is recorded in
[`PLAYTEST_AND_ANALYTICS_QUESTIONS.md`](PLAYTEST_AND_ANALYTICS_QUESTIONS.md#2-fixed-inherited-decisions).

## 3. Decision record

### Checkpoint A - Decisions, readiness, and evidence coverage

#### PA-01 - Decision stages and readiness authority

- **Status:** Resolved 2026-09-09.
- Playtest & Analytics supports distinct decisions rather than one universal
  readiness score:
  1. exploratory internal learning, which may guide iteration but cannot grant
     approval;
  2. commitment of the smallest complete prototype to full native production;
  3. publication of an exact content package;
  4. promotion of an exact Balance Framework candidate;
  5. readiness for the first public release;
  6. acceptance of a material change to an approved design, content, or balance
     contract; and
  7. ongoing live-health monitoring after public use.
- Every gated decision begins with a named requester and accountable decision
  owner. The evidence lead confirms the protocol and packet, while each affected
  semantic owner retains authority over the meaning and legality of its system.
- A packet identifies the exact design, content, balance, catalog, system, study,
  and instrumentation revisions under review. It defines entry requirements,
  required evidence, pass conditions, immediate stop conditions, and the change
  or additional evidence required before retry.
- Disposition is `Pass`, `Fail`, or `Inconclusive`. Missing mandatory evidence,
  insufficient coverage, unresolved data quality, or material conflict is
  Inconclusive rather than an assumed Pass.
- Passing an earlier or narrower stage never grants a later or broader approval.
  A published content package does not approve a balance candidate; a balance
  promotion does not establish first-public-release readiness; live averages do
  not rewrite either decision.
- A material revision expires only the evidence whose assumptions, meanings,
  coverage, or outcomes it could change. The owner records the impact analysis
  and required revalidation rather than discarding unrelated evidence or silently
  carrying affected approval forward.
- Exploratory dashboards and live monitoring may create a review request or stop
  signal. They cannot approve a candidate, change live gameplay, or replace the
  signed evidence packet.

#### PA-02 - Evidence hierarchy, finding classes, and escalation

- **Status:** Resolved 2026-09-09.
- Readiness evidence uses four complementary sources:
  - authoritative semantic events establish what the game accepted and resolved;
  - direct observation establishes visible behavior, attention, hesitation,
    confusion, workarounds, and context;
  - short neutral player explanations establish what the participant understood;
    and
  - voluntary next actions establish what the participant chose without a forced
    prompt.
- Surveys and ratings may add context but never replace those four sources. A
  proxy event cannot prove comprehension, causality, accessibility need, intent,
  enjoyment, or blame.
- Evidence sources are not collapsed to force agreement. A correct semantic
  outcome with visible confusion, a confident explanation contradicted by
  behavior, or telemetry inconsistent with observation becomes an explicit
  finding and may also trigger a data-quality investigation.
- Findings use one primary class with linked secondary effects:
  `Structural Design/Content`, `Numeric Balance`, `Usability/Accessibility`,
  `Technical Reliability`, `Safety/Privacy`, or `Data Quality`.
- Severity is:
  - `Stop-Blocker` for impossible/unsafe play, privacy-invalid collection,
    outcome corruption, loss of an essential accessibility cue, or violation of
    a mandatory invariant or GDD gate;
  - `Major` for systematic misunderstanding, unfairness, material segment
    failure, or a defect likely to change the decision;
  - `Minor` for bounded friction that does not invalidate the tested task or
    affected gate; and
  - `Observation` for a traceable lead requiring more evidence.
- A Stop-Blocker immediately stops the affected scenario and blocks its decision.
  Mandatory GDD gates, safety/privacy requirements, impossible combinations, and
  corrupted outcomes cannot be waived.
- A nonblocking Major or Minor finding may be explicitly accepted only by the
  accountable decision owner with rationale, affected scope, expiry, risk, and a
  retest or monitoring condition. Release-critical acceptance also requires the
  independent reviewer. Silence or backlog placement is not acceptance.
- Every finding receives stable identity, exact evidence/revisions, severity,
  owner, affected decisions/segments, disposition, and closure criteria. Closure
  requires evidence on the corrected exact revision; code completion or an owner
  assertion alone is insufficient.

#### PA-03 - Readiness matrix and representative coverage

- **Status:** Resolved 2026-09-09.
- Coverage is risk-based. The team does not attempt the unhelpful full Cartesian
  product of every setting, role, device, difficulty, roster, and content state.
- Each applicable study still covers every mandatory segment across its complete
  participant set: boss/content revision, role, difficulty, device/control/audio
  output, accessibility configuration, progression/power band, experience
  cohort, solo/co-op, and required roster sizes.
- The smallest-complete-prototype gate retains two rounds of roughly 8–12 new
  players ages 10–14, with most sessions on representative phones and tablets.
  Its scope concentrates on onboarding, core inputs, attention, telegraphs,
  music-to-combat understanding, voluntary continuation, and impossible play.
- Content-complete evidence retains at least 30 new target-age players plus a
  separately visible returning cohort. Across the full study it covers phone,
  tablet, desktop, gamepad, Easy/Normal/Hard, every first-release role, solo and
  two-/three-/six-human groups, and relevant accessibility combinations.
- Named high-risk intersections receive direct coverage even when they would not
  arise from ordinary quota distribution. They include lower-end phone plus
  touch; phone speakers and headphones; duplicate/all-same roles; uneven skill;
  solo, two, and six humans; accessibility combinations; disconnect/grace/rejoin/
  AFK; down/revival/all-down; quiet passages; maximum warning/input/attention
  overlap; group invitations without voice; and representative failure/No
  Contest/results paths.
- Content publication also retains the exact automated and human review matrix
  in `CONTENT_AUTHORING.md`. Playtest sampling cannot waive a deterministic
  invalid combination or substitute one package's evidence for another package's
  musical and encounter approval.
- A mandatory segment or named high-risk cell with no eligible evidence makes
  the affected conclusion Inconclusive. It cannot be inferred from a convenient
  neighboring segment or hidden in an overall result.
- The evidence lead and affected domain owner may expand the matrix when risk or
  an observed finding warrants it. A reduction requires written rationale that
  the omitted cell cannot materially change the question; release-critical
  reduction also requires the independent reviewer.
- Each matrix is frozen with the evidence packet, including inclusions,
  exclusions, actual counts, missing cells, deviations, and exact revisions.

### Checkpoint B - Semantic events, context, and metrics

#### PA-04 - Semantic event catalog and domain ownership

- **Status:** Resolved 2026-09-09.
- Playtest & Analytics maintains one logical, versioned event catalog. Central
  discovery does not transfer semantic ownership: the domain system defines what
  happened and Analytics defines why, where, and under which safeguards that fact
  may be collected and combined as evidence.
- The initial catalog families are:
  - study, participant, and session lifecycle;
  - onboarding, Help, practice, and observed teaching actions;
  - hub navigation, configuration, loadout/build, difficulty, and voluntary
    selection;
  - party, matching, staging, consent, Ready, connection, and follow-up;
  - encounter/attempt lifecycle, roster snapshots, validity, and outcome;
  - rhythm opportunity, input attempt, judgment, hold, coverage, suspension,
    calibration, and synchronization health;
  - Combat, contribution, modifiers/caps, Ward, downing, revival, recovery, and
    finishing;
  - boss telegraph, Commit, Impact, Recovery, target/geometry, position,
    movement, cover, hazard, and invalidation;
  - Ability, Hype, Signature, Band Call, Crescendo, acolyte, and group-response;
  - audio-layer/role response, cue, caption, haptic, degradation, output-profile,
    and accessibility presentation health;
  - Results comprehension evidence, personal/private breakdown, confirmed
    reward/progression, and voluntary next action; and
  - technical/service/data-quality facts needed to qualify the evidence without
    reclassifying a system failure as player performance.
- Every catalog entry defines stable event identity/version, semantic owner,
  question and approved consumers, precise trigger/eligibility, emitting or
  observing source, required context, privacy class, collection class, required
  or optional status, expected cadence/sampling, cancellation/correction rules,
  and linked metric/report uses.
- Collection classes are:
  - `Production`: minimum privacy-reviewed facts permitted in ordinary public
    play;
  - `Private Study`: additional facts permitted only inside an identified study
    and its permission/retention boundary;
  - `QA/Synthetic`: developer, bot, validator, or deterministic-test facts that
    are visibly excluded from human/player evidence; and
  - `Prohibited`: facts that are never collected through this system.
- Raw continuous input replay, complete permanent judgment streams, raw voice,
  camera, microphone/audio, chat content, precise identity/location, complete
  profiles, unnecessary purchase detail, and private accessibility or settings
  history are not required semantic events and remain prohibited unless a future
  explicit policy-reviewed amendment establishes a narrow research need.
- Required collection means the event is necessary for a named evidence use. Its
  absence invalidates or limits that metric/session under PA-12; it never invites
  Analytics to reconstruct the fact from a misleading proxy. Optional events
  cannot become undeclared gate dependencies.
- Diagnostic signals remain labeled as diagnostic observations. They cannot
  fabricate a gameplay result, overwrite an authoritative event, or be reported
  as player behavior.
- A material change to an event's trigger, eligibility, semantic meaning,
  privacy class, or correction rules creates a new event version and an impact
  review of every metric, report, experiment, and retained comparison that used
  the prior version.

#### PA-05 - Common event context, identity, time, and revision binding

- **Status:** Resolved 2026-09-09.
- Every collected fact uses a minimum reviewed context envelope selected from:
  - event identity/version, catalog revision, and stable instance/idempotency
    identity;
  - privacy-safe participant or production-player pseudonym scoped no wider or
    longer than the approved question requires;
  - study, study round, session, encounter attempt, and experiment assignment
    identities where applicable;
  - causal parent/source/target and correlation identities needed to join an
    opportunity, response, resolution, correction, and report;
  - emitting source and observation authority, distinguishing authoritative
    server/system facts, client-observed presentation/input health, moderator
    observation, and participant explanation;
  - logical sequence plus musical timestamp/beat/section where musical causality
    matters; wall time only at the precision needed for operations, retention,
    or ordering;
  - exact applicable design, content, encounter, chart, balance, catalog, system,
    client, protocol, event, and metric revisions;
  - roster/difficulty/role and approved device/control/audio-output,
    accessibility, progression/power, experience, and study segments only when
    the named evidence question needs them; and
  - collection class, permission/assent state where applicable, and privacy/
    retention class.
- Production-player pseudonyms, research participant identities, Roblox account
  identities, and recruitment/contact records remain separate scopes. Any
  necessary re-identification map is access-limited and retained separately from
  play evidence. None becomes a public identifier or durable player skill label.
- Accessibility context records only the minimum tested configuration category
  needed to verify cue and interaction equivalence. It does not infer or expose a
  diagnosis, health condition, or private reason for using an option.
- An event reports its observed source rather than pretending a client
  presentation observation is an authoritative gameplay resolution. Source
  disagreement is retained for PA-12 reconciliation and never silently chooses
  whichever value helps a gate pass.
- Duplicate, repeated, out-of-order, late/offline, partial, corrected, canceled,
  and invalid facts remain explicitly distinguishable. Corrections preserve
  causal lineage; they do not erase the earlier observation or rewrite a frozen
  gameplay outcome.
- Context is allowlisted per event, not copied wholesale into every payload.
  Unneeded profile, purchase, accessibility, social, geographic, demographic,
  or history fields are omitted even if technically available.
- Technical Architecture will select concrete identifier formats, clock
  synchronization, serialization, batching, transport, retry, storage, security,
  and schema representation. Those choices must preserve this semantic envelope
  and cannot broaden collection by convenience.

#### PA-06 - Metric registry, denominators, funnels, and derived evidence

- **Status:** Resolved 2026-09-09.
- Playtest & Analytics maintains a versioned metric registry. Every definition
  records stable identity/version, accountable owner, evidence question,
  required source events/rubrics, eligible unit, numerator, denominator,
  exclusions and invalid states, segment dimensions, aggregation window,
  uncertainty presentation, known blind spots, and the decisions/reports allowed
  to consume it.
- Gate percentages use the unit named by the GDD question:
  - onboarding, core-input explanation/use, post-Results comprehension, audio
    recognition, group-invitation comprehension, and voluntary continuation use
    unique eligible participants;
  - response to a telegraph uses eligible participant-pattern exposure after the
    protocol-defined first learning exposure where the gate says "after seeing
    each pattern once";
  - impossible combinations, missing required role activity, accessibility cue
    loss, and corrupted outcomes are zero-tolerance findings rather than rates
    diluted by a large denominator; and
  - solo/co-op comparability uses matched revision, difficulty, measured
    proficiency/cohort, and progression-power context rather than all attempts.
- A participant is eligible only when the relevant opportunity was actually
  presented under a valid compatible revision and the required observation,
  explanation, or event evidence was available. Ineligible, missing, excluded,
  and failed-data cases are reported separately with raw counts and cannot be
  removed from the denominator without their declared rule.
- Observation and explanation use predeclared rubrics tied to the specific
  question. A semantic gameplay event may corroborate use or response but cannot
  substitute for an explanation score when comprehension is the gate.
- Deliberately skipping onboarding is distinguished from silent abandonment,
  accidental dismissal, coaching, or inability to proceed. Only the first joins
  successful completion under the inherited gate.
- The voluntary-next-action opportunity begins when Results is fully interactive
  and ends at the participant's first qualifying meaningful action, explicit
  nonqualifying exit, moderator intervention, or session termination. Retry,
  Continue Story, changing difficulty, and a genuine build/loadout experiment
  qualify; forced navigation, reward claiming, Store entry, or following a
  moderator instruction does not.
- Metrics preserve distributions and funnels where averages conceal behavior:
  timing/error direction, telegraph exposure-to-response, warning-to-movement,
  invitation-to-decision, Results-to-next-action, onboarding/task abandonment,
  down/revival/recovery, audio/cue recognition, outcome/reason understanding,
  reward/economy paths, and technical/data-quality loss.
- No metric or model may infer comprehension, intent, enjoyment, accessibility
  need, medical status, causality, hidden skill, or blame from a proxy. Such a
  claim requires its approved direct evidence or remains unknown.
- Missing data never defaults to success or zero. PA-11 determines uncertainty
  and PA-12 determines qualification; reports always expose eligible numerator,
  denominator, missing/excluded counts, and relevant segment cells.
- A material formula, eligibility, denominator, segment, or source-event change
  creates a new metric version. Reports never blend incompatible metric versions
  as one continuous trend without an explicit, reviewed equivalence analysis.

### Checkpoint C - Target-age research and participant safeguards

#### PA-07 - Participant cohorts, recruitment, scheduling, and compensation

- **Status:** Resolved 2026-09-09.
- Testing uses practical volunteers found through the owner's existing contacts,
  families, and communities. A dedicated recruiting service, research panel,
  school partnership, or formal screening organization is not required.
- The session record uses simple experience labels:
  - `New` has not previously played or watched a meaningful Bands Battle session;
  - `Returning` has played an earlier session or build; and
  - `Familiar Helper` identifies a developer, close collaborator, repeated family
    helper, rhythm-game expert recruited for expertise, or anyone with unusual
    advance knowledge that could bias new-player conclusions.
- A friend or family member who genuinely has not seen the game may still count
  as New, but the relationship/recruitment source is noted so the owner can spot
  convenience-sample bias. Familiar Helpers remain useful for exploratory,
  failure, mastery, device, and comparison testing but are not substituted for
  the inherited new-player evidence.
- The GDD's target-age counts remain the minimum evidence for their named gates:
  roughly 8–12 new players ages 10–14 in each of two prototype rounds, and at
  least 30 new target-age players plus returning players for content-complete
  first-release evidence. There is no additional formal quota system.
- The owner spreads available sessions across the required devices, inputs,
  roles, difficulties, and roster cases using the PA-03 checklist. Perfectly
  even cells are unnecessary; missing mandatory coverage remains visible.
- A session normally lasts 20–45 minutes. Players may pause or stop sooner, and
  the owner avoids extending a tired or disengaged session just to complete a
  checklist.
- Compensation is optional. When used, it is a simple fixed amount or gift
  agreed before play and unrelated to success, playtime extension, purchases,
  favorable feedback, or returning for another session.
- The session record needs only experience label, broad target-age eligibility,
  recruitment source/bias note, and the gameplay/device context required by the
  current question. It does not build a participant dossier.

#### PA-08 - Consent, assent, safeguarding, privacy, and withdrawal

- **Status:** Resolved 2026-09-09.
- A parent or guardian coordinates a minor's participation and remains aware of
  how and where the session occurs. There is no private adult-to-child recruiting,
  direct messaging, follow-up, or off-platform relationship created for testing.
- Before play, the player receives a short plain-language explanation: they are
  trying an unfinished game, mistakes help the game improve, the owner will keep
  simple gameplay notes/events, there is no performance consequence, and they
  may refuse a task, take a break, or stop at any time.
- The ordinary record uses a nickname or session code rather than a full name,
  Roblox username, contact information, or precise location. Guardian/contact
  coordination stays separate from gameplay notes.
- By default the process does not record raw voice, video, camera, screen,
  microphone/audio, chat, precise location, personal messages, or unnecessary
  identity. Ordinary observation is written as a brief gameplay finding, not a
  transcript or behavioral profile.
- A recording may occur only under a future separately justified process with
  current permission, policy review, limited purpose/access/retention, and a
  practical reason that notes and events cannot satisfy. It is not part of the
  baseline workflow.
- A participant may stop without penalty. If the participant or guardian asks to
  remove identifiable notes before they have been reduced to an anonymous
  aggregate/decision record, the owner removes them and records only the minimum
  audit fact needed to avoid accidental reuse.
- An obvious safety, privacy, inappropriate-contact, or participant-distress
  problem stops the session. The owner involves the guardian and follows current
  platform/policy escalation rather than attempting to investigate through more
  gameplay or telemetry.
- Before organized testing or public analytics, the owner checks the current
  Roblox/platform, privacy, and child-safety requirements. Exact legal notices,
  account mechanisms, retention periods, and request workflows are policy and
  Technical Architecture deliverables, not rules guessed by this document.

#### PA-09 - Session protocol, moderator behavior, and evidence capture

- **Status:** Resolved 2026-09-09.
- Each session uses one short checklist rather than a formal research script. It
  records:
  - date/session code and exact game, content, balance, event, and protocol
    revisions when available;
  - New/Returning/Familiar Helper, device/control/audio setup, difficulty, role,
    roster size, and only the accessibility configuration relevant to the test;
  - whether the player received an uncoached first try and when help was given;
  - applicable GDD gate checks and obvious structural/blocking observations;
  - the player's brief answers; and
  - the first voluntary meaningful action or explicit choice to stop.
- The player first tries the relevant onboarding, encounter, Results, or other
  flow without coaching. If they become stuck or the experience would cease to
  be useful, the owner may help, but marks what help was needed. A coached success
  does not count as an uncoached pass.
- The owner observes natural play rather than asking the player to think aloud
  during timing-sensitive activity. Useful observation categories include boss
  versus staff attention, hesitation, misunderstood controls/cues/outcomes,
  unexplained or impossible damage, Defend/movement use, quiet passages, group
  invitations, audio recognition, accessibility cue loss, and voluntary choice.
- After the relevant play, the owner asks only the three to five questions that
  fit the session, in plain neutral language. The baseline set is:
  1. What were your controls and choices?
  2. What was the boss warning you about, and what could you do?
  3. How did playing the music affect the fight?
  4. Why did the encounter end that way and what mattered in the result?
  5. What would you choose to do next?
- Questions are not leading and the owner records a short answer or rubric mark,
  not a transcript. Confusion is useful evidence and is not corrected until the
  answer is captured.
- Available semantic events may confirm exposure, accepted actions, outcomes,
  timing, cue/audio health, and voluntary choices. Their absence does not prevent
  early observational testing, but any metric that requires them remains
  unavailable rather than inferred.
- Breaks and stopping are always allowed. A stopped, interrupted, coached, or
  partial session is labeled honestly and may still provide bounded qualitative
  findings without entering an inapplicable percentage gate.
- Ordinary iterative play needs no randomized procedure. When deliberately
  comparing two safe variants, alternate or randomize which one players see
  first, keep the rest of the setup similar, and mark prior exposure. More formal
  experiment rules apply only if PA-13's controlled comparison is actually used.

### Checkpoint D - Segmentation, interpretation, and data quality

#### PA-10 - Segmentation and cohort assignment

- **Status:** Resolved 2026-09-09.
- The default session labels are intentionally short:
  - exact game/build, content/encounter, and balance revisions;
  - `New`, `Returning`, or `Familiar Helper`;
  - device class, control method, and ordinary speaker/headphone output;
  - difficulty, playable role, and solo/group roster size; and
  - an accessibility option only when that option or its interaction is actually
    part of the session question.
- Boss identity is part of the content/encounter reference rather than a second
  conflicting label. Study round and session code provide the remaining grouping
  needed for the evidence summary.
- Progression/power band, duplicate/all-same role, uneven skill, disconnect/
  recovery state, comparison order, or another scenario label is added only when
  the current question or PA-03 high-risk case requires it.
- Labels describe the session context rather than permanently classifying the
  player. A New player naturally becomes Returning after meaningful prior play;
  a Familiar Helper flag describes evidence bias, not ability or worth.
- Accessibility use is private and recorded only as the tested configuration,
  not a diagnosis or reason. Reports use the smallest grouping that demonstrates
  cue/interaction equivalence without identifying a participant.
- Small groups and missing required groups remain visible with their actual
  count. They are not merged into a broad favorable average or padded through a
  hidden inferred category.
- The system does not infer or retain sensitive demographics, health, disability,
  spending class, family circumstance, exact location, blame, toxicity, or a
  permanent hidden skill rating from behavior.

#### PA-11 - Sample interpretation, confidence, thresholds, and synthesis

- **Status:** Resolved 2026-09-09.
- Reports lead with the raw fraction, such as `8 of 10`, and may display the
  corresponding percentage. The exact fraction determines whether a threshold
  passes; display rounding never turns a value below the gate into a pass.
- Each of the two smallest-prototype rounds stands on its own against the
  applicable GDD gates. A failed round is corrected and repeated. It cannot be
  averaged with a later pass to manufacture two successful rounds.
- A material structural change likewise needs two consecutive successful rounds
  on the affected scope, as inherited from GD-34. Unaffected evidence remains
  usable only when PA-01's revision-impact record shows why.
- The content-complete gate uses at least 30 eligible new target-age players.
  Returning players are reported as a separate visible group; this framework
  invents no additional quota, and returning evidence never replaces the 30 new
  players.
- Written observations and player explanations interpret the count. One
  validated impossible attack, essential-cue loss, safety/privacy failure, or
  other inherited zero-tolerance structural finding blocks the decision even if
  a percentage passes.
- Qualitative praise or an appealing explanation cannot erase a failed numeric
  gate. The team fixes/retests or records `Needs More Testing`.
- The disposition is `Pass`, `Fail`, or `Needs More Testing` in the lightweight
  summary; the last corresponds to PA-01's formal `Inconclusive` state.
- `Needs More Testing` applies when eligible exposure is missing, a required
  segment is absent, evidence sources materially conflict, a data problem could
  change the result, or the owner reasonably finds a boundary result too fragile
  to justify the decision. The next step is a few targeted sessions, not
  statistical modeling for its own sake.
- No p-value, confidence interval, power calculation, or statistical specialist
  is required for the ordinary creator workflow. If much larger public decisions
  later justify those tools, they may strengthen but never replace raw counts,
  segmented observation, or the inherited zero-tolerance gates.

#### PA-12 - Data quality, telemetry qualification, and exclusions

- **Status:** Resolved 2026-09-09.
- Before an instrumented playtest, the owner runs a short smoke session that
  confirms the expected build/content/balance/event revisions and checks that the
  few required events trigger with plausible session/attempt correlation and the
  same final outcome shown by the game.
- Developer, bot, automated, validator, and other synthetic traffic is marked
  `QA/Synthetic` and excluded from human-player counts. It remains useful for
  checking the instrumentation itself.
- The one-page session sheet visibly marks any wrong or unknown revision,
  missing/mismatched required event, game-versus-recorded outcome disagreement,
  coaching, partial play, disconnect, restart, Invalid/No Contest, or other
  deviation that affects interpretation.
- A data problem invalidates only the evidence that depends on it when the
  remaining sources are trustworthy. For example, broken judgment events prevent
  a timing-distribution claim but do not erase an observed unreadable button.
- A session with broken required telemetry cannot enter the affected numeric
  metric. Its observation may remain as a clearly labeled qualitative finding;
  it is never silently promoted into a numeric success or failure.
- Partial, coached, disconnected, or No Contest sessions are not automatically
  discarded. They remain eligible only for questions whose opportunity and
  evidence survived, and their special state remains visible.
- Every omitted session or event set keeps one plain reason and the report shows
  included, omitted, and qualitative-only counts. Exclusion does not require a
  committee, but it may not be invented after seeing the result merely to make a
  gate pass.
- Repeated missing required events, revision mismatches, or outcome disagreement
  stop affected numeric testing. The owner fixes the instrumentation, reruns the
  smoke check, and repeats affected sessions rather than estimating the missing
  values.
- The evidence summary freezes the exact event/metric revisions and the retained
  counts/notes needed to reproduce its conclusion. Technical Architecture later
  defines concrete export, query, storage, deduplication, clock, and monitoring
  mechanisms.

### Checkpoint E - Experiments, reports, governance, and completion

#### PA-13 - Candidate experiments, staged rollout, and fairness

- **Status:** Resolved 2026-09-09.
- Formal experimentation is optional. Ordinary game iteration uses a prior build
  and a changed build in separate practical play sessions, with the evidence
  summary recording what changed and what cannot be compared cleanly.
- A deliberate two-version comparison is appropriate only when both candidates
  are independently safe/legal and the owner has one concrete question that a
  side-by-side comparison can answer.
- The comparison records exact game, content, balance, event, metric, and
  protocol versions. It alternates or randomly selects which candidate a player
  sees first and notes prior exposure. Other meaningful setup conditions stay as
  similar as practical.
- A player remains on one candidate for a complete active attempt. Assignment
  never changes midway through a song or rebuilds a frozen result under different
  data.
- Candidate versions preserve the same available content, accessibility rights,
  reward/progression treatment, paid/earned equivalence, privacy, and player
  safety. The comparison does not test coercive purchase, inferior rewards,
  hidden difficulty changes, or intentionally inaccessible/confusing treatment.
- The owner defines the expected improvement and obvious stop conditions in one
  sentence before testing. A broken, impossible, materially confusing, unfair,
  unsafe, privacy-invalid, or outcome-corrupting candidate stops immediately.
- A limited public candidate identifies the last compatible Approved revision
  before exposure. If stop conditions occur, new attempts return to that exact
  prior revision while already completed history remains unchanged.
- There is no requirement to build an experiment-assignment platform, complex
  statistics service, or permanent cohort system. Exploratory patterns may
  suggest another play session but are not presented as a controlled conclusion.

#### PA-14 - Reports, issue lifecycle, sign-off, and release records

- **Status:** Resolved 2026-09-09.
- Each meaningful test round ends with a short evidence summary containing:
  - decision/question and exact relevant revisions;
  - dates plus total, included, omitted, and qualitative-only session counts;
  - New/Returning/Familiar Helper and representative setup/coverage counts;
  - applicable GDD or Balance gates as raw fractions and optional percentages;
  - the few most important observations and participant explanations;
  - missing segments, coaching/deviations, instrumentation problems, and other
    limits;
  - Stop-Blocker/Major/Minor findings and their owning system;
  - `Pass`, `Fail`, or `Needs More Testing`; and
  - next change, retest, or decision.
- The summary uses anonymous session codes and paraphrased examples. It does not
  publish a minor's identity, raw transcript, accessibility reason, spending,
  hidden performance label, or teammate comparison.
- Findings live in a basic list with stable identity, short description, class/
  severity, exact affected version, owner, status, and correction/retest evidence.
  A separate ticket system is optional.
- The game owner may make ordinary exploratory and prototype decisions and may
  cover evidence/design/content roles while the project is small. The record
  still distinguishes which responsibility was exercised.
- Before the first public release, one independent person who was not the sole
  author reviews the release-critical summary, checks mandatory gates/coverage/
  blockers and obvious privacy issues, and records agreement or concerns. This is
  a practical second set of eyes, not a committee or formal signature system.
- Content publication and balance promotion retain the named approvals already
  required by `CONTENT_AUTHORING.md` and `BALANCE_FRAMEWORK.md`; this summary
  supplies evidence without replacing their authority.
- A decision binds its exact revisions. A material change uses PA-01 impact
  analysis and records whether evidence remains applicable, needs a targeted
  retest, or is expired. Rollback or amendment keeps the prior decision record.

#### PA-15 - Canonical artifacts, retention classes, handoffs, and completion

- **Status:** Resolved 2026-09-09.
- The minimum maintained artifact set is:
  1. canonical `PLAYTEST_AND_ANALYTICS.md`, including the logical event/metric
     rules and a reusable one-page session checklist;
  2. a short evidence summary for each gated or otherwise meaningful round; and
  3. a decision/change history, which may live in the canonical document and the
     relevant round summaries.
- A separate event database specification, research handbook, dashboard catalog,
  statistical analysis plan, participant-management system, or experiment
  platform is not required by design.
- Each artifact carries exact relevant revision references and stable session,
  round, finding, and decision codes sufficient to trace a conclusion from its
  approved source facts without exposing participant identity.
- Contact/scheduling information remains separate from gameplay notes and is
  accessible only to the owner/guardian coordinator who needs it. Gameplay notes
  use a nickname or session code.
- Raw identifiable notes are discarded after they are reduced into the anonymous
  round summary, any immediate correction question is settled, and they are no
  longer needed. A participant/guardian removal request is honored under PA-08
  while identifiable material still exists.
- Anonymous observations, counts, finding histories, and decision lineage may be
  retained with the project because they explain why a design/revision was
  accepted, rejected, or changed. They cannot be reverse-mapped into a second
  player profile.
- Exact public-telemetry retention, account export/deletion, security, access
  control, storage region, child/privacy notices, and current legal/platform
  compliance remain policy, Player Data, and Technical Architecture work. Those
  choices may shorten or restrict collection but cannot authorize new data that
  this design prohibits.
- Content Authoring consumes package-specific validation evidence; Balance
  consumes candidate evidence; semantic system owners consume their findings;
  Player Data supplies only approved minimum projections; and release review
  consumes the short readiness summary. No consumer receives raw material merely
  because it is convenient.
- A material change to evidence authority, GDD gate meaning, event/metric
  semantics, collection class, target-age safeguards, public-player treatment,
  structural-blocker handling, or first-public-release review requires an
  explicit amendment citing the superseded rule. Template wording and concrete
  implementation may evolve without redesign while preserving these boundaries.

## 4. Cross-specification reconciliation register

The canonical draft must reconcile at least these authorities:

| Authority | Required reconciliation |
|---|---|
| `GAME_DESIGN.md` GD-34 | Prototype and content-complete cohorts, thresholds, structural-redesign patterns, evidence mix, and two-round stability |
| `BALANCE_FRAMEWORK.md` | Candidate states, deterministic/simulation prerequisites, human evidence, segment matrix, rollout, rollback, and structural-versus-numeric routing |
| `SYSTEMS_MAP.md` | Analytics purpose, ownership, dependencies, consumers, and non-authority over gameplay/public labels |
| `PLAYER_DATA.md` | Minimum reviewed projection, no second durable profile, privacy/access/retention, export/deletion, and operational evidence boundary |
| `CONTENT_AUTHORING.md` | Automated validators, exact revision evidence, named human approval, first-three-package review, and publication blocking |
| Gameplay specifications | Domain-owned semantic facts, required scenarios, invalid/No Contest handling, and private nonblaming evidence |
| `UI_UX.md` | Onboarding, comprehension, Results, voluntary next action, accessibility, focus, input, and representative-device acceptance |
| `AUDIO_PRESENTATION.md` | Responsive role audio, boss-cue recognition, objective measurements, blind listening, phone Roblox review, and accessibility evidence |
| Policy and Technical Architecture | Current child-safety/privacy/platform rules and concrete schemas, transport, storage, security, deletion, dashboards, alerts, and implementation |

## 5. Confirmed downstream handoffs

| Owner | Confirmed handoff |
|---|---|
| Semantic gameplay/product systems | Define the authoritative meaning, trigger, legality, and correction of each emitted fact; receive routed findings without yielding gameplay authority |
| Content Authoring | Retain deterministic package validation and named human content approval; consume exact-revision playtest summaries where required |
| Balance Framework | Run deterministic/simulation/sensitivity prerequisites and own Candidate promotion; consume segmented human evidence without automatic tuning |
| UI/UX and Audio Presentation | Supply reviewed presentation/health facts and implement the one-page study questions through ordinary observable surfaces; receive usability/accessibility findings |
| Player Data | Supply only approved minimum privacy-filtered facts and preserve its account export/deletion contract; never store raw playtest streams as a second profile |
| Policy and safety review | Confirm current Roblox/platform, privacy, child-safety, notice, permission, retention, access, and recording requirements before applicable collection |
| Technical Architecture | Define concrete events/schemas, identifiers, clock/correlation, transport, batching, storage, security, deletion, querying, dashboards, alerts, and implementation tests |
| Game owner / release review | Run lightweight sessions and summaries, route/fix findings, decide ordinary iteration, and obtain one independent evidence review before first public release |

## 6. Change log

- **2026-09-09:** Created the working record with all fifteen questions open and
  checkpoint A ready for owner review.
- **2026-09-09:** Approved PA-01 through PA-03. Recorded distinct decision stages,
  evidence triangulation and blocker classes, and risk-based mandatory coverage.
  Interview progress is 3 of 15 questions.
- **2026-09-09:** Approved PA-04 through PA-06. Recorded the domain-owned logical
  event catalog, minimum privacy-safe context envelope, and versioned metric/
  denominator contract. Interview progress is 6 of 15 questions.
- **2026-09-09:** Owner rejected a strenuous formal-research model for checkpoint
  C. Revised the pending direction to use practical volunteers, short informal
  sessions, a one-page checklist, minimal personal data, and simple guardian/
  communication boundaries. PA-07 through PA-09 remain open.
- **2026-09-09:** Approved lightweight PA-07 through PA-09. Recorded practical
  volunteer cohorts, 20–45-minute sessions, guardian-coordinated minor testing,
  minimal data/no recording by default, and a one-page uncoached-play checklist.
  Interview progress is 9 of 15 questions.
- **2026-09-09:** Approved PA-10 through PA-12. Recorded the short session-label
  set, raw-fraction interpretation without elaborate statistics, and a lightweight
  instrumentation smoke check with honest qualitative-only/exclusion handling.
  Interview progress is 12 of 15 questions.
- **2026-09-09:** Approved PA-13 through PA-15. Recorded optional/simple version
  comparisons, one-page evidence summaries and independent pre-public-release
  review, and the minimum canonical/checklist/summary/change-history artifact set.
  Published and reconciled canonical `PLAYTEST_AND_ANALYTICS.md`; interview
  progress is complete at 15 of 15.
