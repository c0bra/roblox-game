# Bands Battle Playtest & Analytics

- **Status:** Approved
- **Approved:** 2026-09-09
- **Parent system:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#83-analytics--playtest-evidence)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Evidence authority:** [`BALANCE_FRAMEWORK.md`](BALANCE_FRAMEWORK.md)
- **Content contract:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)
- **Player-data contract:** [`PLAYER_DATA.md`](PLAYER_DATA.md)
- **System specifications:** [`SYSTEM_SPECIFICATIONS_TRACKER.md`](SYSTEM_SPECIFICATIONS_TRACKER.md)
- **Decision source:** [`PLAYTEST_AND_ANALYTICS_WORKING.md`](PLAYTEST_AND_ANALYTICS_WORKING.md)
- **Interview plan:** [`PLAYTEST_AND_ANALYTICS_QUESTIONS.md`](PLAYTEST_AND_ANALYTICS_QUESTIONS.md)

## 1. Role and authority

This document defines the first-release playtest, evidence, and product-analytics
contract for Bands Battle. Its purpose is to help the owner discover whether the
game is understandable, fair, accessible, musically convincing, and ready for
the next decision.

It owns:

- the decisions that require playtest or analytics evidence;
- the logical event and metric catalog used for that evidence;
- privacy-conscious collection and lightweight segmentation;
- practical target-age session procedure and safeguards;
- the inherited GDD readiness-gate calculations and reports;
- data-quality, finding, retest, and evidence-disposition rules;
- optional comparison and limited-rollout evidence; and
- the minimum session, round-summary, and decision-history artifacts.

This is intentionally a small creator-led workflow, not a formal research lab.
It does not require a research department, statistical specialist, paid panel,
participant-management system, experiment platform, or enterprise analytics
stack. The ordinary loop is:

```text
Choose one question and exact build
                |
                v
Run a short event smoke check when instrumentation matters
                |
                v
Let practical volunteers play; observe before coaching
                |
                v
Record one-page notes, short answers, and simple counts
                |
                v
Pass, Fail, or Needs More Testing
                |
                v
Fix/retest or preserve the approved decision
```

Gameplay and product systems retain ownership of what their facts mean. Player
Data owns durable player records and account-data lifecycle. Content Authoring
owns package validation and human content approval. The Balance Framework owns
numeric candidates, simulations, and balance invariants. Policy/safety review
confirms current external requirements. Technical Architecture owns concrete
schemas, vendors, transports, storage, security, dashboards, alerts, and code.

Analytics may observe and report approved facts. It never changes live gameplay,
difficulty, content, matchmaking, rewards, progression, purchases, accessibility
treatment, an active attempt, or a public player label.

## 2. Governing invariants

1. **Evidence answers a named question:** no data is collected merely because it
   might someday be useful.
2. **Semantic owners remain authoritative:** Analytics records a domain fact but
   cannot redefine or reconstruct it from a proxy.
3. **Observation and telemetry are different:** the game says what it accepted;
   observation and explanation show what the player noticed and understood.
4. **No single favorable number is readiness:** evidence combines behavior,
   explanation, semantic events, and voluntary choice.
5. **Segments remain visible:** an aggregate cannot hide a failing device, role,
   difficulty, roster, accessibility surface, cohort, or content revision.
6. **Structural problems are not tuned away:** impossible, unreadable,
   unexplained, unsafe, or misleading play returns to design/content.
7. **Small samples are reported honestly:** use raw fractions and practical
   retesting rather than false statistical precision.
8. **Missing data is not success:** broken or absent evidence produces a limited
   conclusion or `Needs More Testing`.
9. **Privacy is the default:** use session codes, minimum facts, separate contact
   information, and no raw recording or behavior dossier by default.
10. **Target-age participation stays supervised and voluntary:** a guardian
    coordinates a minor's participation; private adult-to-child contact is
    prohibited; the player may stop at any time.
11. **No blame or hidden status:** analytics cannot create a public damage rank,
    teammate-fault label, permanent hidden skill label, accessibility inference,
    or spending class.
12. **Accessibility is neutral:** configuration, device, age, calibration,
    privacy choice, and spending never reduce rewards, contribution, access,
    progression, or public treatment.
13. **Exact revisions bind conclusions:** evidence names the game, content,
    balance, event, metric, and protocol revisions it actually tested.
14. **Active attempts are immutable:** comparison, rollout, rollback, or new data
    never changes an attempt in progress or rebuilds confirmed history.
15. **Bots do not prove experience:** automated and synthetic evidence validates
    mechanics/instrumentation but never comprehension, fairness, comfort,
    musicality, accessibility, or fun.
16. **Experiments preserve value:** variants cannot secretly change rewards,
    paid/earned equivalence, accessibility rights, content access, or safety.
17. **Approval is explicit:** a metric, dashboard, or monitoring alert may open a
    review but cannot silently promote a candidate or approve a release.
18. **Current requirements still apply:** organized target-age testing and public
    analytics require current Roblox/platform, privacy, and child-safety review.

## 3. Evidence decisions and dispositions

Playtest & Analytics supports distinct decisions. Passing one does not grant a
broader or later approval.

| Decision | Minimum purpose | Blocking? |
|---|---|---|
| Exploratory learning | Discover problems and promising ideas during ordinary iteration | No; cannot grant approval |
| Full-production commitment | Decide whether the smallest complete encounter is understandable and structurally sound enough for full native production | Yes |
| Content publication | Supply package-specific human evidence alongside Content Authoring validation/approval | Yes for the exact package |
| Balance promotion | Supply human evidence after deterministic tests and simulations for an exact Balance Candidate | Yes for promotion |
| First public release | Decide whether the content-complete first-release product meets the approved GDD gates | Yes |
| Material-change revalidation | Check the affected gates after a material approved-system/content/balance change | Yes for affected scope |
| Live-health monitoring | Notice regressions or questions after public use | May stop/rollback; never auto-approves or auto-tunes |

Every gated decision names:

- the question, requester, accountable owner, and affected semantic owners;
- exact design/content/balance/catalog/system/event/metric/protocol revisions;
- entry requirements and required coverage;
- evidence sources, gates, and known limitations;
- immediate stop conditions;
- `Pass`, `Fail`, or `Needs More Testing`; and
- the next change, retest, rollback, or approved use.

`Needs More Testing` is the lightweight report label for an inconclusive result.
It applies to missing coverage, insufficient eligible exposure, unresolved data
quality, or material conflict. It is never treated as Pass.

A material revision expires only evidence whose assumptions, meanings,
coverage, or outcomes could change. The owner records that impact rather than
discarding unrelated evidence or silently carrying affected approval forward.

## 4. Evidence sources and finding treatment

Readiness uses four complementary sources:

- **Authoritative semantic events:** what the game accepted, resolved, granted,
  canceled, or invalidated.
- **Direct observation:** attention, hesitation, confusion, workarounds,
  apparent unfairness, impossible moments, and physical/device context.
- **Short player explanation:** what the player believes the controls, cues,
  consequences, outcome, and choices mean.
- **Voluntary behavior:** what the player chooses next without coaching or a
  forced/store prompt.

Surveys and ratings may add context but cannot replace those sources. Evidence
is not averaged merely to make conflicts disappear. A correct server outcome
that visibly confuses the player, a confident explanation contradicted by play,
or telemetry inconsistent with the observed result becomes a finding.

Findings use one primary class:

- `Structural Design/Content`;
- `Numeric Balance`;
- `Usability/Accessibility`;
- `Technical Reliability`;
- `Safety/Privacy`; or
- `Data Quality`.

Severity is:

- **Stop-Blocker:** impossible or unsafe play, privacy-invalid collection,
  outcome corruption, essential accessibility-cue loss, or violation of a
  mandatory invariant/GDD gate;
- **Major:** systematic misunderstanding/unfairness, material segment failure,
  or a defect likely to change the decision;
- **Minor:** bounded friction that does not invalidate the task or gate; and
- **Observation:** a traceable lead that needs more evidence.

A Stop-Blocker ends the affected scenario and blocks the decision. Mandatory
GDD gates, safety/privacy requirements, impossible combinations, and corrupted
outcomes cannot be waived.

A nonblocking finding may be accepted only with the accountable owner's written
rationale, affected scope, expiry, risk, and retest/monitoring condition.
Release-critical acceptance also requires the independent reviewer. Every
finding records a short stable code, exact evidence/version, owner, status, and
closure evidence.

## 5. Required readiness evidence

### Smallest complete prototype

Before committing to full native production, run at least two rounds of roughly
8–12 New players ages 10–14 per round. Most play occurs on representative phones
and tablets.

Each round independently demonstrates:

- at least 80% complete or deliberately skip onboarding without coaching;
- at least 80% can explain and use the three rhythm inputs, Attack, Defend, and
  positional movement;
- at least 75% identify a major boss telegraph and choose a viable response;
- players look away from the staff enough to describe boss behavior;
- players understand that musical performance produces combat outcomes;
- at least half voluntarily retry, continue, change difficulty, or experiment
  with a build; and
- zero observed attacks are impossible because of chart activity, movement
  recovery, position state, or unreadable targeting.

A failed round is fixed and repeated. It is not averaged with a later pass to
manufacture two successful rounds.

### Content-complete first release

Use at least 30 eligible New target-age players plus a separately reported
Returning cohort. Across the complete set, cover representative phone, tablet,
desktop, and gamepad play; Easy/Normal/Hard; every first-release playable role;
solo and two-/three-/six-human groups; and relevant accessibility combinations.

Evidence demonstrates:

- at least 85% identify outcome, exact reason, important reward, and next action
  within ten seconds of Results becoming visible;
- at least 80% recognize their role's responsive audio and critical boss cues on
  ordinary phone speakers or headphones;
- at least 80% respond correctly to established major telegraphs after seeing
  each pattern once;
- zero validated impossible attack combinations;
- no first-release role lacks required authentic playable activity or event
  windows;
- Easy is observably easier without reducing maximum combat contribution;
- solo and similarly skilled co-op completion remain within roughly 15
  percentage points;
- at least 75% of invited players understand accept/decline for group actions
  without voice chat;
- at least half voluntarily choose another meaningful action after an encounter;
  and
- accessibility combinations preserve every essential cue and never change
  rewards or public status.

### Balance, content, and material changes

A Balance Candidate first passes the Balance Framework's deterministic formula,
order, identity, revision, monotonicity, cap, invariant, 10,000-seed reference-
scenario, and sensitivity requirements. Human sessions then test understanding,
fairness, attention, cooperation, and experience. Simulation does not replace
human evidence.

Content publication retains every automated validator and named human musical,
encounter, phone-Roblox, device/input, accessibility, audio, and solo/co-op
review required by `CONTENT_AUTHORING.md`. Sampled play does not waive a
deterministically invalid role/roster/event combination.

A material structural change passes two consecutive successful target-age
rounds for its affected scope. Technical-only and numeric-only changes receive
the smallest targeted evidence their impact analysis requires, without claiming
they validated unrelated behavior.

## 6. Risk-based coverage

Coverage does not require the full Cartesian product of every setting and
scenario. Each applicable study covers every mandatory segment across its
participant set, then directly exercises high-risk intersections.

Default segments are:

- exact game, content/encounter, and balance revision;
- New, Returning, or Familiar Helper;
- device class, control method, and speaker/headphone output;
- difficulty, role, and solo/group roster size; and
- an accessibility option when that option or interaction is being tested.

Progression/power band, duplicate/all-same role, uneven skill, comparison order,
or another scenario label is added only for a question that requires it.

Named high-risk cases include:

- lower-end phone plus touch;
- ordinary phone speakers and representative headphones;
- duplicate and all-same roles;
- uneven-skill groups;
- solo, two-human, and six-human play;
- relevant accessibility combinations;
- disconnect, grace, rejoin, AFK, down, revival, and all-down;
- quiet passages and role-specific sparse material;
- maximum warning/input/attention overlap;
- group invitations without voice chat; and
- representative Defeat, Invalid/No Contest, Results, and retry paths.

A missing mandatory segment or named high-risk cell is visible and makes its
affected conclusion `Needs More Testing`; it cannot be inferred from a nearby
group. The evidence owner and affected domain owner may expand coverage whenever
a finding warrants it. A release-critical reduction also requires the
independent reviewer.

## 7. Logical event catalog

The catalog is logical and versioned. Exact runtime event names and payloads are
Technical Architecture decisions; the semantic facts below remain required or
explicitly unavailable.

| Family | Semantic owner | Minimum useful facts |
|---|---|---|
| Study/session | Playtest & Analytics | Session/round start/end, participant class, applicable permission/collection class, protocol version, coaching/partial flags |
| Onboarding/practice | UI/UX and owning lesson system | Step presented, deliberately skipped, action observed, completed, abandoned, Help/practice used |
| Navigation/configuration | Owning UI/domain system | Difficulty/role/loadout/build/settings choice presented, changed, validated, applied, rejected, or exited |
| Party/matching/staging | Multiplayer | Invitation/proposal presented, accepted/declined, Ready/lock, roster, queue, disconnect/grace/rejoin/AFK, follow-up choice |
| Attempt lifecycle | Boss Encounters | Preparing/Running/pause/Resolving/Complete/Invalid, exact roster snapshot, Victory/Defeat/No Contest and frozen reason |
| Rhythm | Rhythm Gameplay | Eligible opportunity, input attempt, unmatched early attempt, judgment/hold result, available/earned weight, suspension/resume, calibration/sync health |
| Combat/survival | Combat | Intent accepted/rejected, settled contribution/effect/cap, Ward change, down, revival help/completion, solo recovery, finish evidence |
| Boss/position | Boss Encounters and Positioning | Telegraph/Commit/Impact/Recovery, source/target/geometry/response, position/movement/cover/hazard state, cancellation/invalidation |
| Abilities/cooperation | Abilities & Cooperative Actions | Hype/Signature readiness/use, Band Call/Crescendo invitation/response/result, effect/cap, acolyte action clearly separated from human action |
| Audio/presentation/accessibility | Audio Presentation and UI/UX | Required cue/layer/request, locally presented/recognized health, caption/haptic/fallback, degradation/failure, output/control profile application |
| Results/value/next action | Results, Rewards, Progression, Items/Builds | Results shown/interactive, confirmed outcome/reason/reward/progress, private breakdown/suggestion presented, qualifying next action or explicit exit |
| Validity/technical health | Owning runtime/platform system | Critical data/cue/service validity, mismatch, cancellation, No Contest, event/data loss or correction needed to qualify evidence |

Each logical event definition identifies its stable version, semantic owner,
exact trigger/eligibility, source authority, minimum context, collection/privacy
class, required/optional status, correction/cancellation behavior, and permitted
metric/report uses.

Collection classes are:

- **Production:** minimum reviewed facts permitted in ordinary public play;
- **Private Study:** additional facts used only in an identified playtest and its
  permission/retention boundary;
- **QA/Synthetic:** developer, bot, validator, or test traffic excluded from
  human evidence; and
- **Prohibited:** facts this system does not collect.

Raw continuous input replay, permanent complete judgment streams, voice, video,
camera, microphone/audio, chat content, precise identity/location, complete
profiles, unnecessary purchase detail, and private settings/accessibility history
are prohibited by default. A future narrow exception requires an explicit
policy-reviewed amendment; convenience is not justification.

## 8. Event context and revision binding

An event carries only allowlisted context needed by its named evidence use:

- event identity/version, catalog revision, and idempotency identity;
- privacy-safe participant/player pseudonym scoped no wider than necessary;
- study/round/session/attempt/comparison identities where applicable;
- causal parent/source/target correlation;
- authoritative server/system, client-observed, moderator-observed, or
  participant-explanation source;
- logical order and musical timestamp/beat/section when musical causality matters;
- exact relevant game/content/chart/encounter/balance/catalog/system/protocol/
  event/metric revisions;
- required roster/difficulty/role and approved session labels; and
- collection class plus applicable permission/privacy/retention class.

Production-player pseudonyms, research session codes, Roblox account identity,
and contact/scheduling details remain separate. No identifier becomes public or
a durable skill label.

Accessibility context records only the tested configuration, not a diagnosis or
reason. A client observation never pretends to be an authoritative server result.
Duplicate, late, partial, corrected, canceled, invalid, and out-of-order facts
remain distinguishable; correction preserves lineage rather than rewriting a
frozen gameplay outcome.

Technical Architecture defines concrete formats, clock synchronization,
serialization, batching, transport, retry, storage, and security without
broadening these facts.

## 9. Logical metric registry

Every metric definition records:

- stable identity/version and accountable owner;
- the evidence question and allowed decisions;
- source events or observation/explanation rubric;
- eligible unit, numerator, denominator, and exclusions;
- required segments and aggregation window;
- raw count/fraction plus optional percentage;
- known missing data, blind spots, and comparisons; and
- impact when its formula or source meaning changes.

GDD gates use these units:

| Gate/evidence | Eligible unit and rule |
|---|---|
| Onboarding completion/deliberate skip | Unique eligible participant; deliberate skip is distinct from accidental dismissal, confusion, coaching, or abandonment |
| Core controls explained and used | Unique eligible participant with uncoached opportunity plus short explanation/observation rubric |
| Telegraph identification/response | Unique eligible participant-pattern exposure; content-complete learned-response gate begins after the protocol records the first learning exposure |
| Boss attention and music-to-combat understanding | Short neutral explanation plus observation; no eye-tracking or input proxy required |
| Results understanding within ten seconds | Unique eligible participant; timer begins when Results are visibly available, scored by short explanation/identification rubric |
| Responsive role audio and boss-cue recognition | Unique eligible participant under recorded ordinary speaker/headphone setup; post-play recognition plus presentation-health evidence |
| Group invitation understanding | Unique invited participant with an actually presented valid invitation and no voice-chat coaching |
| Voluntary meaningful next action | Unique eligible participant from Results becoming interactive until first qualifying action, explicit exit, moderator intervention, or session end |
| Solo/co-op comparability | Completion fractions under matched revision, difficulty, measured experience/proficiency, and progression/power context |
| Impossible combination or essential cue loss | Zero-tolerance finding; never diluted into a rate |

Retry, Continue Story, changing difficulty, and a genuine build/loadout experiment
qualify as voluntary meaningful actions. Forced navigation, reward collection,
Store entry, or following a moderator instruction does not.

No metric infers comprehension, intent, enjoyment, accessibility need, medical
status, causality, hidden skill, or blame from a proxy. Missing data never
defaults to success or zero. A material formula, eligibility, denominator,
segment, or source-event change creates a new metric version.

## 10. Practical participant sourcing and safeguards

Use volunteers available through the owner's contacts, families, and
communities. A recruiting service or formal panel is unnecessary.

Experience labels are:

- **New:** has not previously played or watched a meaningful Bands Battle
  session;
- **Returning:** has played an earlier session/build; and
- **Familiar Helper:** developer, close collaborator, repeated helper, recruited
  rhythm expert, or anyone with unusual advance knowledge.

A genuinely new friend or family member may count as New, with recruitment
source noted. Familiar Helpers remain useful for exploratory, mastery, failure,
device, and comparison testing but do not replace inherited New-player evidence.

Sessions normally last 20–45 minutes and stop sooner when the player wants a
break or is tired/disengaged. Optional compensation is fixed beforehand and
unrelated to performance, duration extension, spending, favorable feedback, or
returning.

For minors:

- a parent/guardian coordinates and remains aware of participation;
- there is no private adult-to-child recruiting, messaging, or follow-up;
- the player hears a plain explanation that this is unfinished, mistakes help,
  simple notes/events may be kept, and they may refuse, pause, or stop;
- notes use a nickname/session code, not full name, account/contact details, or
  location; and
- distress, inappropriate contact, safety, or privacy concerns stop the session
  and return to the guardian/current policy path.

No recording is part of the baseline. A future recording requires a separately
justified, permissioned, access-limited, time-bounded, policy-reviewed process.
Before organized testing or public analytics, check current Roblox/platform,
privacy, and child-safety requirements rather than relying on this static design
document as legal guidance.

## 11. Lightweight session procedure

1. Choose one main question and the exact build/content/balance revisions.
2. If numeric telemetry matters, run the short PA-12 smoke check.
3. Record the session code and basic setup labels.
4. Explain unfinished-game, note/event, and stop/pause expectations plainly.
5. Let the player try the relevant flow without coaching.
6. Observe natural play; do not demand think-aloud during rhythm activity.
7. If the player is stuck, capture that fact, then help and mark where coaching
   began. A coached success does not count as uncoached.
8. After play, ask only three to five relevant neutral questions.
9. Record the first voluntary next choice or the choice to stop.
10. Mark partial/coached/disconnected/data-quality conditions and finish the
    short round summary when the round is meaningful.

Baseline questions are:

1. What were your controls and choices?
2. What was the boss warning you about, and what could you do?
3. How did playing the music affect the fight?
4. Why did the encounter end that way, and what mattered in the result?
5. What would you choose to do next?

Use only the questions relevant to the session. Record a short paraphrase or
rubric mark, not a transcript. Capture confusion before explaining the answer.

Useful observations include boss-versus-staff attention, hesitation,
misunderstood controls/cues/outcomes, unexplained/impossible damage, Defend and
movement use, quiet passages, group invitations, audio recognition,
accessibility-cue loss, and voluntary choices.

## 12. Reusable one-page session checklist

Copy this section for a session; delete fields that do not apply.

```markdown
# Bands Battle Playtest Session

- Session code / date:
- Main question:
- Game / content / balance / event / protocol revisions:
- Player: New / Returning / Familiar Helper
- Device / controls / audio output:
- Difficulty / role / roster size:
- Accessibility option being tested, if any:
- Uncoached first try: Yes / No
- Help given and when:

## What happened

- Applicable gate checks or counts:
- Boss-versus-staff attention:
- Confusing, unfair, impossible, or inaccessible moments:
- Data problems, disconnect, partial play, or No Contest:
- First voluntary next action or stopped:

## Ask three to five

- What were your controls and choices?
- What was the boss warning you about, and what could you do?
- How did playing the music affect the fight?
- Why did the encounter end that way, and what mattered in the result?
- What would you choose to do next?

## Findings

- Stop-Blocker / Major / Minor / Observation:
- Owning system and next change:
- Include numerically / qualitative only / omit, with reason:
```

## 13. Simple interpretation

Reports lead with raw fractions such as `8 of 10`; percentages are optional.
The exact fraction determines whether a threshold passes. Display rounding never
turns a result below a gate into a pass.

Each prototype round stands alone. Written structural findings can block an
otherwise passing percentage, while praise or a plausible explanation cannot
erase a failed numeric gate.

The content-complete gate retains at least 30 eligible New target-age players.
Returning players remain a separate visible group with their actual count; they
do not replace the 30 and this framework invents no extra quota.

Use `Needs More Testing` when:

- an eligible opportunity or mandatory segment is missing;
- evidence sources materially conflict;
- broken data could change the result;
- a zero-tolerance finding is not yet understood/corrected; or
- the owner reasonably finds a small boundary result too fragile for the
  requested decision.

The remedy is a few targeted sessions after a fix or with missing coverage. No
p-value, confidence interval, power calculation, or statistical specialist is
required for the normal workflow. Larger future decisions may add those tools
without replacing raw counts and segmented observation.

## 14. Data-quality smoke check and session qualification

Before an instrumented playtest, run one short smoke session to confirm:

- expected game/content/balance/event revisions;
- required events trigger with plausible session/attempt correlation;
- the event stream and displayed game reach the same final outcome; and
- developer/bot/test traffic is marked `QA/Synthetic`.

The session checklist marks wrong/unknown revision, missing/mismatched event,
outcome disagreement, coaching, partial play, disconnect, restart, Invalid/No
Contest, or other relevant deviation.

A problem invalidates only evidence that depends on it. Broken judgment events
prevent a timing-distribution claim but do not erase an observed unreadable
button. A session with broken required telemetry may remain qualitative; it
never silently enters the affected numerator or denominator.

Partial, coached, disconnected, or No Contest sessions remain eligible only for
questions whose opportunity/evidence survived. Every omitted session keeps one
plain reason, and reports show included, omitted, and qualitative-only counts.
Exclusions cannot be invented after seeing results merely to make a gate pass.

Repeated missing required events, revision mismatch, or outcome disagreement
stops affected numeric testing. Fix instrumentation, rerun the smoke check, and
repeat the affected sessions rather than estimating missing values.

## 15. Optional comparisons and rollback

Formal A/B testing is not required. Most changes use ordinary before/after play
sessions with honest notes about what differed.

A deliberate two-version comparison is used only when both versions are safe and
legal and one concrete question benefits from direct comparison. It:

- records exact versions and the expected improvement in one sentence;
- alternates or randomly chooses first exposure and notes prior experience;
- keeps other meaningful setup conditions as similar as practical;
- keeps one version for an entire active attempt;
- preserves content access, accessibility, reward/progression treatment,
  paid/earned equivalence, privacy, and safety; and
- defines obvious stop and restoration conditions before testing.

Broken, impossible, materially confusing, unfair, unsafe, privacy-invalid, or
outcome-corrupting treatment stops immediately.

A limited public candidate identifies the last exact compatible Approved
revision. When a stop condition occurs, future attempts return to that revision.
Active attempts and completed history remain unchanged. No experiment platform,
permanent cohort system, or complex early-stopping calculation is required.

## 16. Evidence summaries, issues, and approval

Each gated or otherwise meaningful round receives a short summary:

```markdown
# Playtest Round Summary

- Decision/question and disposition: Pass / Fail / Needs More Testing
- Exact game/content/balance/event/metric/protocol revisions:
- Dates and session counts: total / included / qualitative only / omitted
- Player/setup/coverage counts:
- Applicable gates as raw fractions:
- Important observations and short explanation themes:
- Missing coverage, coaching, deviations, or bad data:
- Findings with class/severity/owner:
- Next change, retest, rollback, or approved use:
- Owner decision and independent release review when required:
```

Examples use session codes and paraphrases, never participant identity, raw
transcripts, accessibility reasons, spending, hidden skill, or teammate
comparisons.

Findings may live in the summary or a basic issue list. Each records code,
description, class/severity, affected exact version, owner, status, and
correction/retest evidence. A dedicated ticket system is optional.

The game owner may make ordinary exploratory/prototype decisions and cover
multiple responsibilities while the project is small. The record still names
which responsibility was exercised.

Before the first public release, one independent person who was not the sole
author reviews the release-critical summary, mandatory gates/coverage/blockers,
and obvious privacy issues. This is a practical second set of eyes, not a
committee or elaborate signature process.

Content and Balance retain their own named approvals. This summary supplies
evidence without replacing their authority.

## 17. Retention, access, and minimum artifacts

The minimum maintained artifact set is:

1. this canonical specification, including logical event/metric rules and the
   reusable checklist;
2. a short summary for every gated or otherwise meaningful round; and
3. decision/change history in this document and the relevant summaries.

Contact/scheduling information remains separate from gameplay notes and is
available only to the owner/guardian coordinator who needs it. Gameplay notes
use a nickname/session code.

Discard raw identifiable notes after they are reduced into an anonymous summary,
immediate correction questions are settled, and they are no longer needed. Honor
a participant/guardian removal request while identifiable material still exists,
retaining only the minimum fact needed to avoid accidental reuse.

Anonymous observations, counts, finding history, and decision lineage may remain
with the project because they explain why a revision was accepted, rejected, or
changed. They cannot be reverse-mapped into a second player profile.

Exact production-telemetry retention, public-account export/deletion, security,
storage region, access control, notices, and current compliance remain Player
Data, policy, and Technical Architecture work. Those choices may further restrict
collection; they cannot authorize prohibited data without design amendment.

## 18. Semantic outputs and consumers

Playtest & Analytics outputs privacy-safe evidence facts rather than gameplay
commands:

- session/round identity and tested revisions;
- coverage, included/qualitative-only/omitted counts, and omission reasons;
- metric fractions, segments, and limitations;
- observation/explanation themes without participant identity;
- finding identity, class, severity, owner, state, and closure evidence;
- decision disposition and approved scope;
- evidence expiry/retest/rollback requirement; and
- instrumentation qualification or failure state.

Content Authoring consumes package evidence. Balance consumes candidate evidence.
Semantic system owners consume their findings. Release review consumes the short
readiness summary. Player Data supplies only approved minimum projections.
Results may consume private historical suggestions only through its existing
bounded contract, never raw playtest notes or a new skill label.

No consumer receives raw research material merely because it is convenient.

## 19. Deferred policy, content, and technical work

This design contract is complete. Downstream work still includes:

- running actual prototype/content-complete target-age sessions;
- authoring exact event and metric catalog entries as implementation surfaces
  become available;
- selecting Roblox/platform analytics and operational tools;
- concrete event schemas, identifiers, time synchronization, batching,
  deduplication, transport, retry, storage, querying, dashboards, and alerts;
- authentication, authorization, encryption, deletion/export integration,
  regional handling, and exact retention periods;
- current child-safety, privacy, platform, and legal review before collection;
- technical performance, networking, persistence, security, anti-cheat, and
  operational release gates; and
- optional larger-sample/statistical or experiment infrastructure if the project
  later genuinely needs it.

None may weaken semantic ownership, minimum collection, supervised voluntary
target-age participation, no-blame/accessibility treatment, revision binding,
immutable attempts, structural blocker handling, fair comparisons, or explicit
approval.

## 20. Approval and change control

The owner interview resolved PA-01 through PA-15 on 2026-09-09. This document is
the canonical Playtest & Analytics supporting specification.

A material change to evidence authority, readiness gates, event/metric meaning,
collection classes, target-age safeguards, public-player treatment, structural
blocker handling, comparison fairness, first-public-release review, retention
boundary, or source-of-truth ownership requires an explicit amendment citing the
superseded decision.

Session/checklist wording, report formatting, optional event coverage, and
concrete technical implementation may evolve without redesign while preserving
these approved boundaries and exact-revision traceability.
