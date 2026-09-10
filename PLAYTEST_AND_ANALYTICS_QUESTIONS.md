# Bands Battle Playtest & Analytics Questions

- **Status:** Interview complete; 15 of 15 questions resolved and reconciled
- **Started:** 2026-09-09
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#83-analytics--playtest-evidence)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Evidence authority:** [`BALANCE_FRAMEWORK.md`](BALANCE_FRAMEWORK.md)
- **Data boundary:** [`PLAYER_DATA.md`](PLAYER_DATA.md)
- **System-specification index:** [`SYSTEM_SPECIFICATIONS_TRACKER.md`](SYSTEM_SPECIFICATIONS_TRACKER.md)
- **Working record:** [`PLAYTEST_AND_ANALYTICS_WORKING.md`](PLAYTEST_AND_ANALYTICS_WORKING.md)
- **Canonical result:** [`PLAYTEST_AND_ANALYTICS.md`](PLAYTEST_AND_ANALYTICS.md)

## 1. Interview method

This interview uses five checkpoints of three related questions. It first
establishes which decisions evidence must support, then defines semantic events
and metrics, target-age research operations, analysis and data quality, and
finally experiments, reporting, sign-off, and downstream artifacts.

Answers are persisted in `PLAYTEST_AND_ANALYTICS_WORKING.md` at meaningful
checkpoints. When all fifteen questions are resolved, they will be reconciled
into canonical `PLAYTEST_AND_ANALYTICS.md` and audited against the GDD, Balance
Framework, Player Data contract, Systems Map, and all thirteen approved system
specifications.

This is a product evidence and research-operations specification. It defines
what may be measured, how evidence is interpreted, how target-age studies are
conducted safely, and what evidence permits a design or release decision. Exact
Roblox event schemas, analytics vendors, network transport, storage topology,
encryption, dashboards, alerting products, and code layout remain Technical
Architecture work. Current platform, privacy, legal, and child-safety
requirements must still be confirmed before collecting real participant data.

The operating model is deliberately lightweight for a small creator-led project.
It does not assume a research department, paid panel, statistical specialist,
formal laboratory, elaborate moderator training, or enterprise analytics stack.
The specification should require only the smallest repeatable process that helps
the owner notice real problems and make responsible decisions.

## 2. Fixed inherited decisions

The interview does not reopen these approved rules:

- Analytics & Playtest Evidence exists to produce trustworthy,
  privacy-conscious evidence that the game is understandable, fair, accessible,
  and ready. It is a supporting production system with bounded runtime
  instrumentation, not a gameplay authority.
- Gameplay and product systems own the meanings of the facts they emit.
  Analytics owns the reviewed collection, segmentation, derivation,
  quality-checking, and reporting of those facts.
- Analytics never changes difficulty, rewards, progression, matchmaking,
  content, public player labels, or an active attempt automatically.
- Analytics may receive reviewed semantic events or aggregates, never complete
  player profiles. Raw input streams, complete judgment histories, raw audio or
  microphone data, private settings, purchase details beyond an approved safe
  fact, and unnecessary personal history do not become general telemetry.
- Events and reports use stable privacy-safe identities and the exact relevant
  content, balance, catalog, encounter, and system revisions. An attempt's
  locked snapshot remains immutable.
- Accessibility settings, device, age, calibration, privacy choice, and spending
  are never public skill labels and never reduce rewards, contribution,
  progression, access, or evidence credibility.
- There is no public damage leaderboard, teammate blame label, hidden skill
  score exposed to players, or analytics-derived accusation of who caused a
  defeat.
- Evidence combines direct observation, a player's short explanation, semantic
  telemetry, and voluntary next actions. Survey satisfaction or one aggregate
  metric is never sufficient on its own.
- Before committing to full native production, the smallest complete encounter
  prototype receives at least two rounds of roughly 8–12 new players ages
  10–14, with most play on representative phones and tablets.
- Each prototype round retains the initial GDD gates: at least 80% complete or
  deliberately skip onboarding without coaching; at least 80% explain and use
  all three rhythm inputs, Attack, Defend, and movement; at least 75% identify a
  major telegraph and choose a viable response; players can describe boss
  behavior and the music-to-combat relationship; at least half voluntarily take
  another meaningful action; and no observed attack is impossible.
- Content-complete first-release testing includes at least 30 new target-age
  players plus returning players. Coverage spans representative phone, tablet,
  desktop, and gamepad use; Easy, Normal, and Hard; every first-release playable
  role; solo and two-, three-, and six-human groups; and relevant accessibility
  combinations.
- Content-complete gates retain the GDD thresholds: at least 85% identify the
  outcome, exact reason, important reward, and next action within ten seconds;
  at least 80% recognize responsive role audio and critical boss cues; at least
  80% respond correctly to established major telegraphs after seeing each once;
  at least 75% understand group-action accept/decline without voice chat; at
  least half voluntarily choose another meaningful action; and the remaining
  zero-failure, role-coverage, difficulty, solo/co-op, and accessibility gates
  all pass.
- Results remain segmented so an overall average cannot hide a failing device,
  role, difficulty, roster, accessibility surface, content revision, or player
  cohort.
- The structural-redesign patterns in GD-34 and the Balance Framework cannot be
  hidden through numeric tuning. Impossible or unreadable attacks, unexplained
  damage/outcomes, staff-only attention, surprising movement recovery, ignored
  Defend, empty quiet passages, weaker-feeling Easy charts, inauthentic role
  activity, teammate blame, missed group invitations, and excessive one-off
  pipeline exceptions return to their owning design or content system.
- A material structural change passes two consecutive target-age testing rounds
  before its evidence is considered stable.
- Balance candidates first pass deterministic checks, at least 10,000 seeded
  simulations per core reference scenario, and required sensitivity analysis.
  Simulation and bots cannot substitute for comprehension, fairness, comfort,
  accessibility, musicality, or fun evidence from people.
- Cohort experiments use stable assignment, exact revision identity, privacy
  review, fair treatment, and rollback. They cannot silently disadvantage a
  player, corrupt durable history, or break paid/earned equivalence.
- Content Authoring retains automated validation plus named human musical,
  encounter, accessibility, audio, phone-Roblox, and solo/co-op review. Analytics
  evidence complements rather than replaces accountable approval.
- Player Data owns durable player records, account export/deletion behavior, and
  privacy-filtered projections. Analytics does not turn raw event streams into a
  second permanent player profile.
- Technical performance, networking, persistence, security, anti-cheat, data
  safety, and current Roblox/platform-compliance gates remain separately
  required even when this specification reports their evidence.

## 3. Question plan

### Checkpoint A - Decisions, readiness, and evidence coverage

#### PA-01 - Decision stages and readiness authority

- **Status:** Resolved 2026-09-09.
- **Decision:** Keep exploratory learning, full-production commitment, content
  publication, balance promotion, first public release, material-change
  revalidation, and live-health monitoring distinct. Every gated packet binds
  exact revisions, owner, evidence, and pass/stop/retry rules. Inconclusive never
  passes, later stages do not inherit approval automatically, and no dashboard
  can silently approve a decision.
- **Proposed direction:** Maintain distinct evidence packets for exploratory
  design, commitment to full production, content publication, first public
  release, a material post-approval change, and ongoing live health. Every packet
  names the decision owner, exact candidate/revisions, required evidence, pass,
  stop, and retry conditions; passing one stage never implies another passed.
- **Decision needed:** Which product decisions may Playtest & Analytics support,
  and what authority does a passing or failing report have at each stage?
- **Must resolve:** Decision stages; entry and exit conditions; who may request a
  study; which stages are blocking; whether content, balance, technical, and
  public-release readiness remain separate; handling of partial/inconclusive
  evidence; expiration after material revision; and prohibition on silent
  approval from a dashboard number.

#### PA-02 - Evidence hierarchy, finding classes, and escalation

- **Status:** Resolved 2026-09-09.
- **Decision:** Triangulate authoritative semantic facts, direct observation,
  player explanation, and voluntary behavior. Conflict becomes a finding rather
  than being averaged away. Classify and route structural/content, numeric,
  usability/accessibility, technical, safety/privacy, and data-quality findings;
  impossible, unsafe, privacy-invalid, or outcome-corrupting evidence blocks the
  affected decision immediately.
- **Proposed direction:** Require triangulation among observed behavior, player
  explanation, semantic telemetry, and voluntary behavior. Classify findings as
  structural design/content, numeric balance, usability/accessibility,
  technical reliability, safety/privacy, or data-quality issues. Impossible,
  unsafe, privacy-invalid, or outcome-corrupting findings block the affected
  decision immediately; other findings receive a named severity, owner, and
  retest condition.
- **Decision needed:** How should conflicting evidence be interpreted, and which
  finding classes block progress rather than enter an ordinary tuning backlog?
- **Must resolve:** Evidence sources and precedence; qualitative/quantitative
  disagreement; structural versus numeric routing; issue taxonomy and severity;
  stop-testing triggers; repeated isolated versus systemic failures; owner and
  disposition; waivers, if any; and the evidence required to close a finding.

#### PA-03 - Readiness matrix and representative coverage

- **Status:** Resolved 2026-09-09.
- **Decision:** Use a risk-based matrix rather than a full Cartesian product.
  Cover every mandatory segment across the applicable study and directly test
  named high-risk combinations. Missing mandatory coverage is inconclusive, not
  a favorable extrapolation; justified matrix changes require evidence-owner and
  affected-domain approval, plus independent review for release-critical scope.
- **Proposed direction:** Use a risk-based matrix anchored by the exact GD-34 and
  Balance Framework scenarios. Do not require every possible Cartesian product;
  cover every required segment across the study, then deliberately test named
  high-risk combinations such as low-end phone plus touch, duplicate/all-same
  roles, uneven skill, solo/two/six humans, accessibility combinations,
  disconnect/rejoin/AFK, down/revival, quiet passages, and maximum warning/input
  conflicts.
- **Decision needed:** Which combinations must be observed directly before each
  decision, and when is representative coverage sufficient?
- **Must resolve:** Prototype versus content-complete matrices; required devices,
  controls, audio outputs, roles, difficulties, progression/power bands, roster
  sizes, accessibility combinations, content/balance revisions, first-attempt
  and learned cohorts; risk-based combination rules; missing-cell treatment;
  and who approves a justified matrix reduction or expansion.

### Checkpoint B - Semantic events, context, and metrics

#### PA-04 - Semantic event catalog and domain ownership

- **Status:** Resolved 2026-09-09.
- **Decision:** Use one versioned logical event catalog spanning the approved
  domain families. The semantic system owner defines each fact; Analytics defines
  its reviewed evidence use. Every event is `Production`, `Private Study`,
  `QA/Synthetic`, or `Prohibited`, and required/optional collection is explicit.
  Raw behavior replay is unnecessary and prohibited data never enters the
  catalog.
- **Proposed direction:** Maintain one versioned logical catalog organized by
  domain-owned semantic families: study/session; onboarding/practice;
  navigation/configuration; party/matching/staging; attempt lifecycle; rhythm
  opportunity/input/judgment/coverage; combat/survival; boss telegraph/attack/
  position; abilities/group actions; audio/presentation/accessibility health;
  Results/reward/next action; and validity/technical failure. Analytics defines
  collection use but cannot reinterpret an owner's event.
- **Decision needed:** What event families are necessary to answer the approved
  readiness questions without collecting a raw replay of player behavior?
- **Must resolve:** Catalog families; semantic event versus diagnostic signal;
  domain owner; required/optional status; production, private-study, QA-only, and
  prohibited collection classes; trigger meaning; eligibility; causal source;
  corrections/cancellation; event versioning; and review when an owner changes a
  fact's meaning.

#### PA-05 - Common event context, identity, time, and revision binding

- **Status:** Resolved 2026-09-09.
- **Decision:** Bind each collected fact to a minimum privacy-reviewed envelope:
  stable event/catalog identity, pseudonymous scope, session/study/attempt and
  causal correlation where required, observation source, logical/musical time,
  exact relevant revisions, consent/collection class, and only approved segment
  context. Ordering, duplicates, late arrival, correction, and invalidation stay
  explicit; exact transport and schemas remain technical work.
- **Proposed direction:** Every collected fact carries only the minimum reviewed
  context needed to reproduce its meaning: event/catalog version, privacy-safe
  participant/session/attempt identities, causal identity, logical and musical
  time where relevant, exact content/balance/system revisions, study/experiment
  context, and approved segment fields. Order, duplication, late arrival,
  correction, and invalidation remain explicit rather than guessed.
- **Decision needed:** Which shared context is mandatory for evidence to be
  trustworthy, correlatable, and privacy-minimal?
- **Must resolve:** Identity scopes; pseudonymization; session/attempt/
  participant linkage; causal correlation; client/server observation source;
  logical/wall/musical time; revision envelope; roster and device context;
  consent/collection class; event order/idempotency; late/offline delivery;
  invalid/corrected events; and which exact payload/transport choices remain
  Technical Architecture work.

#### PA-06 - Metric registry, denominators, funnels, and derived evidence

- **Status:** Resolved 2026-09-09.
- **Decision:** Use a versioned metric registry whose entries name the question,
  owner, formula, eligible unit, numerator, denominator, exclusions, segments,
  uncertainty, and blind spots. GDD gates use eligible participant-level or
  participant-pattern denominators rather than convenient event totals. Rubric-
  scored observation/explanation and voluntary behavior remain distinct from
  telemetry; no proxy may infer understanding, intent, enjoyment, need, skill,
  causality, or blame.
- **Proposed direction:** Define every metric as a versioned formula with owner,
  question answered, eligible population, numerator, denominator, exclusions,
  segments, revision window, uncertainty display, and known blind spots. Use
  funnels and distributions where averages obscure behavior. Never infer
  comprehension, intent, causality, skill, accessibility need, or blame from a
  proxy event alone.
- **Decision needed:** Which derived measures turn reviewed events into the GDD
  readiness gates and useful design evidence without overstating what telemetry
  proves?
- **Must resolve:** Metric registry fields; gate formulas; attempt/session/player
  units; completion and deliberate-skip meaning; explanation and observation
  scoring; telegraph exposure and learned-response denominators; voluntary-next-
  action window; audio recognition evidence; timing/performance distributions;
  funnel abandonment; repeat-player/cohort rules; missing data; and metric
  revision/change control.

### Checkpoint C - Target-age research and participant safeguards

#### PA-07 - Participant cohorts, recruitment, scheduling, and compensation

- **Status:** Resolved 2026-09-09.
- **Decision:** Use practical volunteers from existing contacts and communities,
  distinguish new, returning, and familiar-helper experience, and record only
  the basic setup facts the current test needs. Retain the inherited GDD player
  counts without an elaborate quota program. Sessions normally last 20–45
  minutes; compensation is optional, fixed, and unrelated to performance or
  feedback.
- **Proposed direction:** Find practical volunteers through the owner's existing
  contacts and communities. Record whether each player is new or returning plus
  the few device/gameplay facts needed for the test. Friends, family, developers,
  and rhythm experts are useful but marked so they are not mistaken for new-player
  evidence. Sessions are normally 20–45 minutes with breaks whenever needed.
  Compensation is optional and fixed when used.
- **Decision needed:** How will representative target-age and returning cohorts
  be found and distinguished without creating a formal recruiting program?
- **Must resolve:** New versus returning meaning; practical recruitment sources;
  minimum participant/device notes; how familiar helpers are labeled; ordinary
  session length; optional compensation; and use of the inherited GDD player
  counts as goals rather than a complex quota system.

#### PA-08 - Consent, assent, safeguarding, privacy, and withdrawal

- **Status:** Resolved 2026-09-09.
- **Decision:** A parent or guardian coordinates a minor's participation; the
  player receives a plain explanation and may stop at any time; private adult-to-
  child recruiting/communication is prohibited. Use a nickname/session code and
  simple gameplay notes by default, with no raw voice, video, camera, microphone,
  chat, location, or unnecessary identity. Current platform/policy requirements
  are checked before organized testing or public analytics.
- **Proposed direction:** Keep the safety rule simple: a parent or guardian
  coordinates a minor's participation, the player gets a plain-language
  explanation and may stop at any time, and there is no private adult-to-child
  recruiting or communication. By default keep only a nickname/session code and
  gameplay notes; do not record voice, video, camera, microphone, chat, precise
  location, or unnecessary identity. Check current Roblox/platform requirements
  before organized testing or public analytics.
- **Decision needed:** What minimum safeguards make target-age research
  voluntary, understandable, private, and operationally safe?
- **Must resolve:** Guardian coordination; plain-language participant agreement;
  no private communication; default no-recording rule; minimal identity; ability
  to stop or remove identifiable notes; handling of an obvious safety/privacy
  problem; and deferral of exact compliance details to current policy review.

#### PA-09 - Session protocol, moderator behavior, and evidence capture

- **Status:** Resolved 2026-09-09.
- **Decision:** Use a one-page session checklist. Record exact game/setup context,
  allow an uncoached first try, note obvious comprehension/fairness/accessibility
  problems, and ask three to five plain questions afterward. Basic semantic
  events support rather than replace notes. Mark any coaching, allow breaks or
  stopping freely, and control ordering only for a deliberate comparison.
- **Proposed direction:** Use a one-page checklist rather than a formal study
  script. Note the game version and basic setup, let the player try without
  coaching, observe obvious confusion or impossible moments, then ask three to
  five plain questions about controls, boss cues, outcomes, and what they want to
  do next. Basic game events may support the notes when available. Only a
  deliberate side-by-side comparison needs randomized order.
- **Decision needed:** How should a session reveal genuine comprehension and
  behavior without turning playtesting into a burdensome research exercise?
- **Must resolve:** One-page session record; game/setup identity; uncoached first
  try; permitted help afterward; a few standard questions; obvious observation
  categories; voluntary follow-up; breaks/stopping; lightweight notes; and when a
  simple comparison actually needs controlled ordering.

### Checkpoint D - Segmentation, interpretation, and data quality

#### PA-10 - Segmentation and cohort assignment

- **Status:** Resolved 2026-09-09.
- **Decision:** Use only a short session label set: exact game/content/balance
  version, New/Returning/Familiar Helper, device/control/audio setup, difficulty,
  role, roster size, and an accessibility option only when tested. Add a temporary
  question-specific label only when needed. Keep small or missing groups visible;
  prohibit sensitive, spending, health, blame, and permanent hidden-skill
  inference.
- **Proposed direction:** Put only a short set of useful labels on the one-page
  session record: exact game/content/balance version, New/Returning/Familiar
  Helper, device/control/audio setup, difficulty, role, roster size, and any
  accessibility option actually being tested. Add progression/power or another
  label only when it answers that session's question. Never infer demographics,
  spending categories, health, blame, or a permanent hidden skill label.
- **Decision needed:** Which segment definitions are mandatory, optional, or
  prohibited without creating a complicated tagging system?
- **Must resolve:** The default short label list; optional question-specific
  labels; temporary experience classification; privacy for accessibility use;
  visibly small/missing groups; and prohibited sensitive or blame-oriented
  inference.

#### PA-11 - Sample interpretation, confidence, thresholds, and synthesis

- **Status:** Resolved 2026-09-09.
- **Decision:** Report raw fractions with optional percentages and apply each GDD
  gate to its eligible players without display rounding. Each prototype round
  stands alone; failures are fixed and retested, not averaged away. Written
  structural findings can block an otherwise passing percentage but cannot erase
  a numeric failure. Close, contradictory, or missing evidence is Inconclusive
  and calls for a few more sessions, not elaborate statistics.
- **Proposed direction:** Use simple counts such as `8 of 10`, with a percentage
  beside them when helpful. The GDD numbers are practical decision gates, not a
  claim of scientific certainty. Each prototype round stands on its own; a failed
  round is fixed and repeated rather than averaged away. If a result is close,
  contradictory, or based on too few relevant players, mark it uncertain and get
  a few more sessions instead of doing elaborate statistics.
- **Decision needed:** How should small-sample thresholds, uncertainty,
  qualitative evidence, repeated rounds, and conflicting segments determine a
  pass, fail, or inconclusive result?
- **Must resolve:** Per-round versus combined use; show-the-fraction rounding;
  minimum eligible players from inherited gates; how written observations affect
  a decision; close/contradictory results; returning-player reporting; missing
  coverage; and when the honest answer is simply to run more sessions.

#### PA-12 - Data quality, telemetry qualification, and exclusions

- **Status:** Resolved 2026-09-09.
- **Decision:** Run a short event/version smoke check before instrumented testing.
  Mark test/bot, wrong-version, missing/mismatched-event, coached, partial,
  disconnected, and No Contest cases. Broken required telemetry cannot support
  its numeric gate, though a valid observation may remain qualitative. Keep a
  simple omission reason/count; repeated instrumentation failure stops affected
  numeric testing until fixed and retested.
- **Proposed direction:** Before a playtest, run one short event smoke check. On
  the session sheet mark obvious problems: missing events, wrong version, test or
  bot traffic, disconnect/No Contest, coaching, partial play, or disagreement
  between the game result and recorded data. Do not use broken telemetry for a
  numeric gate. The owner's observation can remain as a clearly labeled note,
  and every omitted session keeps a simple reason.
- **Decision needed:** What makes a session, event stream, metric, or study round
  trustworthy enough without building a large analytics-quality program?
- **Must resolve:** A small pre-session smoke check; test/bot labeling; obvious
  missing or mismatched data; partial/coached/disconnected/No Contest sessions;
  when a session remains qualitative only; simple omission reasons and counts;
  and when broken instrumentation requires repeating affected tests.

### Checkpoint E - Experiments, reports, governance, and completion

#### PA-13 - Candidate experiments, staged rollout, and fairness

- **Status:** Resolved 2026-09-09.
- **Decision:** Formal experiments are optional; ordinary iteration uses simple
  before/after play sessions. A deliberate two-version comparison records exact
  versions, uses alternating or random first exposure, keeps other conditions
  similar, preserves immutable attempts and all reward/accessibility/paid
  invariants, and stops or restores the prior approved version when the candidate
  is clearly broken, confusing, unfair, or unsafe.
- **Proposed direction:** Treat formal experiments as optional. Most changes use
  ordinary before/after play sessions. For a deliberate two-version comparison,
  alternate or randomly choose which safe version appears first, record the exact
  versions, keep the setup similar, never vary rewards or paid treatment, and
  stop if one version is clearly broken, confusing, unfair, or unsafe. Restore
  the prior approved version if a limited public change goes wrong.
- **Decision needed:** When may the team run a controlled comparison or limited
  rollout, and what small set of rules keeps it fair and reversible?
- **Must resolve:** Ordinary iteration versus deliberate comparison; exact
  version recording; simple order assignment; comparable setup; immutable active
  attempts; reward/accessibility/paid parity; obvious stop conditions; restoration
  of the prior version; and no requirement to build an experiment platform.

#### PA-14 - Reports, issue lifecycle, sign-off, and release records

- **Status:** Resolved 2026-09-09.
- **Decision:** End each meaningful round with a one-page evidence summary of
  exact revisions, player/setup counts, raw gate fractions, key observations,
  missing/bad data, findings, disposition, and next action. Use a basic issue list
  with owner/status/retest evidence. The owner may approve ordinary prototype
  iteration; an independent person reviews release-critical evidence before the
  first public release.
- **Proposed direction:** End each meaningful round with a short evidence summary:
  exact versions, number and kinds of players/setups, simple gate counts, the most
  important observations, bad/missing data, blockers, decision, and next change.
  Track open findings in a basic list with owner, status, and retest result. The
  game owner may make ordinary prototype decisions; before the first public
  release, one independent person reviews the release-critical summary.
- **Decision needed:** What report and approvals are required to commit to
  production, publish content, promote balance, make a first public release, or
  accept a material change?
- **Must resolve:** One-page summary fields; `Pass`, `Fail`, or `Needs More
  Testing`; simple issue list; privacy-safe examples; owner decision during
  iteration; independent review before first public release; retest evidence;
  and preservation of the exact decision/version record.

#### PA-15 - Canonical artifacts, retention classes, handoffs, and completion

- **Status:** Resolved 2026-09-09.
- **Decision:** Keep a small artifact set: canonical specification with logical
  event/metric definitions and a reusable session checklist, short round
  summaries, and decision/change history. Use session codes, keep contact details
  separate, discard identifiable raw notes after summarizing them and when no
  longer needed, and retain anonymous findings/decision lineage. Policy and
  Technical Architecture own current compliance and concrete implementation.
- **Proposed direction:** Keep the artifact set small: one canonical
  specification containing the logical event/metric definitions, one reusable
  session checklist, short round summaries, and a decision/change log. Keep
  contact details separate from gameplay notes, use session codes, discard raw
  identifiable notes after they are summarized and no longer needed, and retain
  anonymous findings/decision history. Technical Architecture later chooses the
  actual analytics, storage, dashboard, security, and deletion implementation.
- **Decision needed:** Which artifacts must exist for this specification to be
  practical, what information is kept or discarded, and what is left for later
  technical work?
- **Must resolve:** Minimal artifact list; exact-version traceability; session
  codes and separate contact details; raw versus summarized notes; anonymous
  long-lived decisions; basic access/correction/removal; Player Data, Content,
  Balance, policy, and Technical Architecture handoffs; and amendment triggers.

## 4. Completion criteria

The Playtest & Analytics interview is complete only when:

- PA-01 through PA-15 are explicitly resolved with no hidden `TBD` decisions;
- every fixed GDD and Balance Framework evidence gate maps to a defined decision,
  cohort, evidence source, denominator, segment set, and disposition rule;
- the event and metric models collect only the minimum reviewed facts needed to
  answer approved questions and preserve semantic-owner authority;
- prototype, content-complete, balance, accessibility, content-publication,
  material-change, and first-public-release evidence paths are distinguishable;
- target-age recruitment, permission/assent, safeguarding, withdrawal,
  moderation, access, and retention boundaries are operationally clear while
  current policy/legal confirmation remains explicit;
- structural, numeric, content, technical, safety/privacy, and data-quality
  findings have unambiguous owners and stop/retest rules;
- segmentation, small samples, missing cells, exclusions, uncertainty, and
  conflicting evidence cannot be hidden by aggregate reporting;
- experiments preserve fixed player, accessibility, economy, reward,
  progression, privacy, and immutable-attempt guarantees;
- reports, issue closure, independent review, approval, expiry, rollback, and
  amendment rules are traceable;
- Player Data, Content Authoring, Balance Framework, every semantic system,
  policy review, and Technical Architecture handoffs have no competing source of
  truth; and
- canonical `PLAYTEST_AND_ANALYTICS.md` passes local link, terminology,
  cross-document ownership, threshold, scenario, privacy, and completeness
  audits.

## 5. Change log

- **2026-09-09:** Created the 15-question interview plan and opened checkpoint A.
- **2026-09-09:** Approved PA-01 through PA-03, completing checkpoint A. Interview
  progress is 3 of 15 questions.
- **2026-09-09:** Approved PA-04 through PA-06, completing checkpoint B. Interview
  progress is 6 of 15 questions.
- **2026-09-09:** Rejected the formal-research interpretation of checkpoint C as
  too strenuous for the project. Reframed PA-07 through PA-09 around lightweight
  creator-led sessions, simple notes, practical volunteers, and minimum necessary
  safety/privacy safeguards. Progress remains 6 of 15 pending owner approval.
- **2026-09-09:** Approved the lightweight PA-07 through PA-09 direction,
  completing checkpoint C. Interview progress is 9 of 15 questions.
- **2026-09-09:** Approved PA-10 through PA-12, completing checkpoint D with a
  short session-label set, simple fraction-based interpretation, and lightweight
  event/data smoke checks. Interview progress is 12 of 15 questions.
- **2026-09-09:** Approved PA-13 through PA-15, completing checkpoint E and the
  interview. Reconciled all fifteen decisions into canonical
  `PLAYTEST_AND_ANALYTICS.md`.
