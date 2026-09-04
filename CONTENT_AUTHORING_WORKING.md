# Bands Battle Content Authoring Working Record

- **Status:** Interview complete; archived decision record; 24 of 24 resolved;
  final reconciliation completed 2026-09-02
- **Started:** 2026-08-19
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#81-song-chart--encounter-authoring)
- **Interview plan:** [`CONTENT_AUTHORING_QUESTIONS.md`](CONTENT_AUTHORING_QUESTIONS.md)
- **Canonical result:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)

## 1. Role of this record

This document persists owner decisions for the Content Authoring specification
as the interview proceeds. It is an audit trail and drafting source, not yet a
canonical specification. `GAME_DESIGN.md` governs player-facing behavior and
`SYSTEMS_MAP.md` governs system ownership until the finished
`CONTENT_AUTHORING.md` is approved.

Answers are saved in grouped checkpoints. Each resolved question records the
owner answer, resulting design rule, consequences, and any dependency handed to
a later system specification.

## 2. Inherited baseline

### System boundary

- Content Authoring is an offline, platform-neutral production system.
- It owns intake/provenance, analysis suggestions, human chart and difficulty
  authoring, Activity Maps, ensemble coverage, encounter timelines, validation
  aggregation, approval/versioning, and runtime export.
- It does not own Roblox runtime execution, final gameplay-domain semantics, or
  an in-game creator interface.
- Automation and AI may suggest content but may never approve or publish.

### Approved workflow

The GDD requires seven stages: ingest, automated musical analysis, human chart
editing, difficulty generation, encounter authoring, automatic validation, and
in-Roblox review. Automatic validation is necessary but not sufficient.

### Existing maintained implementation baseline

The version-1 `tools/chart-pipeline/` contract currently provides four stems,
three difficulty charts per instrument, notes, simple attack windows, seconds-
based source offset and duration, a schema-versioned manifest, and basic chart
validation. Its drums, vocals, guitar, and bass categories are a starting
implementation catalog, not the approved permanent instrument roster. The tool
is a useful compiler baseline, not the complete approved authoring contract.

The concrete Blackened Crown web level contains twelve derived chart files—one
for each drums/vocals/guitar/bass and Easy/Medium/Hard combination—over the same
188.71-second duration. Each chart contains only role, difficulty, duration,
`time`/`lane`/`duration` notes, and an empty attack list. Its paired stem/backing
audio is useful evidence for independently controlled local response. It does
not yet contain the approved tempo/meter map, musical coordinates, canonical
performance transcription, phrase/passage identities, Activity Maps, or
encounter timeline, so these files are runtime-export examples rather than the
future authoring source of truth.

## 3. Decision record

### Checkpoint A — Authority, intake, and identity

#### CA-01 — Human roles and approval authority

- **Status:** Resolved.
- **Owner answer:** The proposed accountable owner and human approval separation
  are approved.
- **Design rule:** Every song/encounter package has one accountable Content
  Owner. A Music/Chart Author and Encounter Designer create the content; one
  person may fill more than one creation role when the team is small. Release
  requires explicit musical/chart approval and encounter/gameplay approval. At
  least one human other than the primary author must participate in the final
  in-Roblox review before either approval becomes final.
- **Consequences:** Automatic validators, AI, and processing tools provide
  evidence but cannot approve, publish, or satisfy the independent-review rule.
  The approval record must identify the accountable owner, creation roles,
  reviewers, reviewed revision, and decision.

#### CA-02 — Source intake and release eligibility

- **Status:** Resolved.
- **Owner clarification:** Drums, vocals, guitar, and bass must not be mandatory
  for every song. Songs may use synthesizer, piano, or other musically relevant
  roles; instrumental songs may omit vocals; and a muted, atmospheric, or
  effectively absent bass part must not be forced into a conventional bass
  chart merely to satisfy the current tool's categories.
- **Design rule:** The authoring contract must represent song-specific playable
  role availability and must never fabricate a part that the arrangement does
  not support. Every role actually offered for a song still requires authentic
  chart material and independently controllable instrument audio or an approved
  equivalent.
- **Approved intake rule:** Exploratory analysis may begin with only a mixed
  master or incomplete inputs. Promotion to release candidate requires the
  final master; independently controllable audio or an approved equivalent for
  each offered role; rights or generation provenance; lyrics when vocals are
  present; stable song metadata; available arrangement notes; and an encounter
  brief. Missing inputs remain visible and block promotion rather than being
  silently synthesized.
- **Role-availability rule:** The project maintains an extensible global catalog
  of playable roles, while each song declares its own musically valid subset.
  Piano, synthesizer, or later roles may be offered; instrumental songs may omit
  vocals; and absent bass or another conventional part is valid. A sparse or
  atmospheric part qualifies only if human review finds enough authentic
  material for meaningful play and later coverage validation passes. Duplicate
  roles allow multiplayer population without requiring many distinct roles.
- **Consequences:** The exact first-release catalog and whether related
  instruments share a family remain downstream item/equipment decisions. Song
  intake and validation operate on declared roles rather than four hard-coded
  names.

#### CA-03 — Stable identity and revision semantics

- **Status:** Resolved.
- **Owner answer:** The proposed separate identities and immutable revision
  model are approved.
- **Design rule:** A stable song identity represents the musical work/source
  lineage, and a separate stable encounter identity represents the boss scenario
  mapped to it. An immutable package revision binds exact source fingerprints,
  charts, Activity/encounter data, validation evidence, and approvals.
- **Consequences:** Approved content is never edited in place. Any source,
  timing, chart, encounter, validation-relevant, or approval-relevant change
  creates a new revision that explicitly supersedes its predecessor. Consumers
  reference an exact approved package revision rather than resolving `latest`.
  Prior revisions remain available for audit and rollback according to the
  later retention policy.

### Checkpoint B — Musical clock and chart truth

#### CA-04 — Canonical musical clock

- **Status:** Resolved.
- **Owner answer:** The proposed dual exact-time and musical-position clock is
  approved.
- **Design rule:** Exact time begins at the start of the approved full master,
  preserving any intentional silence, count-in, lead-in, or pickup rather than
  forcing the first beat to time zero. A human-approved tempo-and-meter map
  relates measure/beat/subdivision positions to exact audio time across tempo
  changes and meter changes. Authored timing retains both the musical position
  and exact-time mapping.
- **Correction authority:** Automated beat analysis only proposes the map. Human
  correction establishes the approved clock. Editing that approved map marks
  every dependent transcription, difficulty chart, phrase, Activity Map,
  encounter event, validation result, and export stale until it is rebuilt and
  reviewed against the new revision.
- **Handoff:** Runtime clock authority, interpolation, and latency behavior
  belong to `RHYTHM_GAMEPLAY.md`; this system supplies the approved mapping.

#### CA-05 — Detailed chart source of truth

- **Status:** Resolved.
- **Owner answer:** The proposed difficulty-independent canonical transcription
  is approved, with the Blackened Crown charts retained as examples of today's
  derived runtime format.
- **Design rule:** Every playable role declared by a song has one canonical,
  human-approved performance transcription. It represents the authentic audible
  musical material, note and hold identities, rests, dropouts, relevant
  expressive changes, and source relationship before difficulty reduction.
  This source is not named Hard and may contain more musical information than
  any one shipped difficulty needs.
- **Playable-event rule:** Every chart input must trace to an audible musical
  basis in the role's approved audio. Human authors decide which audible events
  become gameplay and how approved material maps to three pads. They may omit or
  simplify material for readability but may not invent a conventional part
  during a real absence or dropout.
- **Consequences:** Easy, Normal, Hard, and later difficulty charts are reviewed
  derivatives linked to canonical event identities instead of unrelated files.
  The current Blackened Crown chart files remain valid examples of possible
  compiled output, not the editable master representation.

#### CA-06 — Phrase and passage boundaries

- **Status:** Resolved.
- **Owner answer:** The proposed human-approved phrase and passage rules are
  approved.
- **Design rule:** Automated analysis may suggest phrase boundaries, but a human
  approves or edits every boundary. A phrase normally spans one or two measures.
  A pickup, unusual meter, long sustain, or clear musical transition may justify
  a shorter or longer exception when readability remains acceptable.
- **Passage rule:** Adjacent phrases may chain into a sustained performance
  passage without a forced gap. Reaching a phrase boundary alone never creates
  downtime; a break must follow an actual rest/dropout, chosen movement,
  recovery, boss transition, or another meaningful authored event.
- **Crossing holds:** A hold may cross a phrase boundary when the source music
  requires it. It remains one continuous musical event, is explicitly identified
  as crossing the boundary for validation, and must not be split merely to make
  storage simpler. Judgment and intent attribution are deferred to
  `RHYTHM_GAMEPLAY.md` and `COMBAT.md`.

### Checkpoint C — Difficulty and participation coverage

#### CA-07 — Difficulty derivation contract

- **Status:** Resolved.
- **Owner answer:** The shared-source and normalization direction makes sense,
  but Normal cannot rigidly preserve an “intended core arrangement” when the
  baseline musical part is itself very difficult. Derivation must respond to
  note density and resulting playability.
- **Approved rules so far:** Easy, Normal, and Hard use the same approved master,
  clock, phrase boundaries, and passage identities. They are human-reviewed
  derivatives of the canonical performance transcription. Automation may
  suggest reductions but may not approve them. All difficulties preserve the
  same maximum pre-difficulty combat contribution per passage; song speed never
  changes.
- **Final design rule:** Difficulty derivation uses a playtested complexity
  envelope rather than a fixed percentage reduction or an assumption that the
  source's core arrangement is automatically suitable for Normal. Canonical
  events carry human-reviewed musical-importance information. Authors preserve
  the most important authentic material that fits the target envelope for local
  density, burst rate, alternation speed, subdivisions, hold complexity, and
  sustained activity.
- **Flexible relationship:** A sparse part may differ little across the three
  difficulties. A dense or virtuosic part may require substantial reduction
  even on Normal, and Hard may still omit detail that exceeds its readable
  envelope. Easy may consolidate repeated notes or simplify a pattern, so
  difficulty charts need semantic lineage to the canonical source but do not
  have to be literal note-for-note subsets.
- **Naming and tuning:** The new logical contract uses `normal`; `medium` remains
  accepted only as a legacy version-1 import label. Exact envelope values are
  playtest-driven inputs from `RHYTHM_GAMEPLAY.md` and `BALANCE_FRAMEWORK.md`, not
  hard-coded authoring doctrine.

#### CA-08 — Activity Map granularity and facts

- **Status:** Resolved.
- **Owner answer:** The proposed map is approved, with deterministic behavior at
  least for the initial system.
- **Design rule:** Given an exact approved source revision and authoring
  configuration, Activity Map generation is deterministic. It records objective
  facts at each beat boundary for every role and difficulty, then provides
  deterministic measure, phrase, and passage summaries.
- **Required facts:** Current and upcoming playable density; time to next event;
  longest rest; dropouts; crossing holds; entries, exits, and solos; authored
  energy/quietness; continuing activity; distance from the finishing cadence;
  known encounter conflicts; and available warning/reaction time.
- **Interpretive labels:** Human authors approve labels such as solo, quiet, or
  high energy. Once approved, those labels are fixed inputs to the package
  revision. Future probabilistic or AI analysis may propose labels, but cannot
  make a canonical build nondeterministic or bypass review.
- **Consequences:** Rebuilding the same approved revision and configuration must
  reproduce the same map. Any human label edit or algorithm/configuration change
  creates new derived evidence and participates in revision/staleness rules.

#### CA-09 — Candidate eligibility rules

- **Status:** Resolved.
- **Owner answer:** The distinction between candidates and actual events is
  approved.
- **Design rule:** A candidate is an approved, validated opportunity from which
  a runtime system may select; it is not proof that the event will occur. Each
  candidate records its type, allowed time or musical-position range,
  compatible roles/difficulties/rosters, required warning and reaction time,
  duration, known conflicts, suppression reasons, and qualifying evidence.
- **Approval rule:** Automation may propose candidates. A human author must
  approve or adjust them, and automatic validators must confirm them against
  downstream domain requirements. Runtime systems may select only among
  approved candidates compatible with current state.
- **Fixed-event distinction:** Phase boundaries, required story beats, and the
  finishing performance are explicit authored timeline events rather than
  optional candidates. Recovery, revival, boss attack, Band Call, Crescendo,
  and similar state-dependent opportunities use candidate coverage.
- **Revival clarification:** A downing is live state, not authored content. An
  eligible player may request or join revival whenever the Survival rules allow.
  The candidate set does not schedule that downing or decide that a revival will
  occur; it identifies clean musical boundaries where a requested revival
  performance can begin without an unfair dropout, hold, impact, or protected
  finishing conflict. The runtime selects the earliest compatible boundary
  within the allowed delay.
- **Handoff:** Exact eligibility thresholds and maximum-delay requirements come
  from the gameplay system that owns each event type; Content Authoring stores,
  validates, and proves coverage against those requirements.

### Checkpoint D — Encounter timeline and ensemble coverage

#### CA-10 — Encounter timeline contract

- **Status:** Resolved.
- **Owner clarification:** Cooperative revival must not be manually mapped as an
  event expected to occur at a predetermined song time. It is a live response to
  someone actually becoming downed.
- **Design direction:** The authored timeline contains fixed song/encounter
  structure and eligible opportunity data. A revival request can occur whenever
  the owning gameplay rules permit, after which the Activity Map supplies the
  earliest validated musical start. The same fixed-versus-dynamic distinction
  applies wherever runtime state determines whether an optional event is needed.
- **Final timeline rule:** A single multi-track timeline aligns song sections;
  Arrival, First Clash, Escalation, Climax, and Finishing Cadence spans;
  resistance-layer availability; fixed phase, story, arena, and finishing
  events; and derived eligibility for state-dependent opportunities. Every
  entry uses the approved musical clock and carries its required lead time,
  duration, dependencies, conflicts, and reference to its owning domain
  definition.
- **Runtime boundary:** Runtime state may choose among approved candidates or
  react differently to a fixed event, but it cannot move, pause, skip, or rewind
  the authored song timeline. This package instantiates or references gameplay
  definitions; it does not take ownership of attack, survival, ability, or
  movement mechanics.

#### CA-11 — Roster-aware ensemble coverage

- **Status:** Resolved.
- **Design rule:** Deterministic validation evaluates every legal song-specific
  role combination for solo and for two through six humans. It includes
  unrestricted duplicate roles, an all-same-role roster, maximally diverse
  rosters, and combinations most exposed to sparse parts or simultaneous
  dropouts. Two-human play is supported through the approved consent flow;
  three-to-six remains the intended cooperative range.
- **Solo rule:** Every offered role/difficulty is evaluated with the fixed
  acolyte support model. Acolytes provide their authored support functions but do
  not create simulated human charts or count as human playable-event coverage.
- **Human review matrix:** In-Roblox review covers every role solo plus
  representative two-human minimum, three-human target, and six-human maximum
  rosters, including all-same-role, mixed-role, and the most dropout-sensitive
  allowed combination. Machine validation covers the combinations not manually
  replayed.
- **Release gate:** An allowed role/roster/difficulty combination cannot ship if
  a required event type repeatedly lacks a valid opportunity within its maximum
  delay. The content must be corrected or the role combination must cease being
  allowed; the validator cannot hide the failure.

#### CA-12 — Conflict and priority policy

- **Status:** Resolved.
- **Design rule:** Every event type participates in an explicit compatibility
  matrix. Events that compete for the same physical control, contextual prompt,
  movement decision, or critical attention channel are incompatible unless a
  later owning specification deliberately approves and validates the overlap.
- **Protected space:** The Finishing Cadence preview and performance reject
  competing major attacks, recovery/revival performances, Band Call invitations,
  and Crescendos. Band Calls and Crescendos cannot compete with one another.
  Fixed transitions may define additional protected space.
- **Permitted overlap:** Ordinary rhythm performance and a boss telegraph may
  overlap because divided attention is part of the game, but only when timing,
  control, visual, audio, and device-specific reaction evidence passes. The same
  standard applies to any intentional multi-system overlap.
- **Failure handling:** An incompatible candidate is filtered and replaced by a
  later approved candidate, an explicitly allowed urgent fallback, or a skip for
  a nonurgent optional event. Incompatible fixed events block validation; the
  runtime may not silently discard one. Exact spacing and urgency thresholds
  come from each event's owning system specification.

### Checkpoint E — Assistance and authoring surface

#### CA-13 — Automation boundaries

- **Status:** Resolved.
- **Design principle:** Automation is classified by whether it changes semantic
  authoring truth, not merely by whether its algorithm is deterministic. A
  reproducible output may still be only a suggestion requiring musical review.
- **Mechanical operations:** Audio conversion/copying, compilation of approved
  charts, deterministic Activity Map calculation from approved inputs,
  validation, and construction of a draft export may run automatically. Their
  inputs, configuration, progress, result, and failures remain visible.
- **Human-acceptance boundary:** Stem-separation quality; beat/meter corrections;
  canonical transcriptions; playable-role selection; difficulty charts;
  phrase/passage boundaries; interpretive musical labels; candidate approvals;
  and fixed encounter events remain unapproved until accepted by a human with
  the appropriate authoring responsibility.
- **Suggestion workflow:** An author can inspect confidence and evidence, preview
  changes against the current draft, and accept or reject suggestions
  individually or in a reviewed batch. A rejected suggestion remains in audit
  history but does not enter canonical content.
- **Regeneration rule:** Rerunning analysis produces new draft artifacts and
  marks affected dependents stale. It never mutates an approved revision,
  silently accepts changed truth, or publishes. Future AI may participate only
  through this same suggestion-and-review boundary.

#### CA-14 — Minimum authoring surface

- **Status:** Resolved.
- **Owner direction:** The initial authoring surface should be a simple internal
  web application. The author selects a song project directory, and the app loads
  the master, all available stems, and existing pipeline artifacts. It presents
  the timeline and Activity Map and can rerun the already-maintained pipeline as
  needed.
- **Design boundary:** This specification owns that required workflow and visible
  capability. The later technical architecture chooses the web framework,
  filesystem bridge, process orchestration, and deployment model.
- **Required information and controls:** The app aligns synchronized master and
  stem waveforms, mute/solo controls, beat/meter grid, canonical transcription,
  difficulty charts, phrases/passages, Activity Map, and fixed/candidate
  encounter tracks. It supports scrub, zoom, range selection, looping, and edits
  to timing, notes, boundaries, labels, candidate approval, and fixed events.
- **Pipeline workflow:** An author may rerun the full maintained pipeline or only
  stages made stale by an edit. Long steps show real progress and permit safe
  cancellation. New results are compared with the preceding draft, and reruns
  never overwrite an approved revision.
- **Recovery and output:** The app preserves drafts, supports undo, and clearly
  communicates loading, processing, saved, unsaved, stale, invalid, failed, and
  completed states. Validation findings link to the affected timeline range.
  The author can create a new versioned runtime package and export it into the
  direct Roblox test workflow.

#### CA-15 — Authoring state and rework loop

- **Status:** Resolved.
- **Lifecycle:** Each package revision progresses through **Intake**, **Analysis
  Draft**, **Authoring Draft**, **Validation**, **In-Roblox Review**,
  **Approved**, **Published/Exported**, and **Retired**. Processing failures,
  stale dependencies, validation findings, and review findings are visible
  conditions on the applicable state rather than hidden side effects.
- **Promotion authority:** The accountable Content Owner advances a revision
  between working stages after required inputs and evidence exist. The musical/
  chart and encounter/gameplay reviewers control their approval gates. No tool
  or automated result advances itself into Approved or Published/Exported.
- **Rework:** A failed validator or review returns the draft to the responsible
  authoring stage with findings attached to the affected content. Successful
  unaffected work remains available; a pipeline failure does not erase the
  preceding draft.
- **Dependency invalidation:**
  - Source-audio or approved musical-clock edits stale all dependent content.
  - Transcription or role edits stale affected difficulties, Activity Maps,
    candidates, validation, and exports.
  - Difficulty, phrase, or passage edits stale affected Activity Maps,
    candidates, coverage, validation, and exports.
  - Encounter edits stale affected conflict, roster-coverage, validation, and
    export evidence.
- **Approved revision rule:** Approved revisions are immutable. Reworking one
  creates a new draft revision based on it; the prior approved revision remains
  available for audit and rollback. Publication/export and retirement are
  explicit human actions tied to an exact approved revision.

### Checkpoint F — Validation and approval

#### CA-16 — Validator classes and severity

- **Status:** Resolved.
- **Validator classes:** The authoring system runs deterministic validators for
  package structure and required inputs; musical clock/alignment; chart
  authenticity and phrase structure; difficulty complexity and normalized
  contribution; Activity Map and roster coverage; encounter timing and conflict
  compatibility; accessibility/presentation metadata; runtime-export
  compatibility; and completeness of rights/provenance evidence.
- **Severity:**
  - **Error** blocks entry to review, approval, or export as applicable.
  - **Warning** remains visible and requires explicit acknowledgement before
    approval.
  - **Information** records evidence or advice without requiring action.
- **Finding context:** Every finding identifies its validator, affected package
  revision, role, difficulty, roster, field, and musical/timeline range where
  applicable; it explains the failure and the authoring action needed to correct
  it.
- **Execution:** Relevant validators run incrementally during editing. A complete
  deterministic validation pass over the exact candidate revision is mandatory
  before in-Roblox review and again before approval when review causes changes.

#### CA-17 — In-Roblox review matrix

- **Status:** Resolved.
- **Phone-first baseline:** Every offered role/difficulty combination receives a
  complete playthrough on the touch-first phone surface. The matrix includes at
  least one lower-capability supported phone profile.
- **Population baseline:** Solo review covers every offered role. Cooperative
  review covers representative two-, three-, and six-human rosters, including
  an all-same-role band, a diverse mixed-role band, and the legal combination
  most exposed to sparse material or overlapping dropouts.
- **Other inputs:** Desktop keyboard and gamepad review each cover every offered
  role at least once and every difficulty at least once. Their full role-by-
  difficulty cross-product becomes mandatory when an input- or presentation-
  specific risk appears.
- **Accessibility:** Risk-based combinations cover reduced motion,
  non-color-dependent cues, scalable staff/UI presentation, supported remapping,
  and Hold Assist. Pairwise/risk-based selection avoids an unhelpful full
  Cartesian product while ensuring every essential cue and interaction is
  exercised under relevant combinations.
- **Evidence:** Each run records exact package revision, device/performance
  profile, input, role, difficulty, roster, accessibility settings, outcome,
  findings, reviewer, and date. Video or captures are required for a defect when
  they materially aid diagnosis, not for every successful run.

#### CA-18 — Approval record and exception policy

- **Status:** Resolved.
- **Approval record:** Final approval binds the exact immutable package revision
  and source fingerprints; complete validator reports; completed review matrix;
  musical/chart approval; encounter/gameplay approval; independent reviewer
  participation; open warnings and exceptions; approver identities; timestamps;
  and the final approve/reject decision.
- **Non-waivable failures:** Missing rights/provenance evidence, corrupt or
  consumer-incompatible packages, musical-clock misalignment, fabricated or
  unsupported chart material, missing required roster/event coverage,
  impossible control/reaction combinations, safety failures, and failures of
  essential accessibility semantics cannot receive an exception.
- **Limited exceptions:** A warning outside those classes may be accepted only
  by the accountable Content Owner and relevant domain reviewer. The record
  states its reason, exact scope, affected revision, risk, expiration or follow-
  up condition, and responsible owner. It does not carry automatically to a new
  revision.
- **Revocation and rollback:** A serious discovered defect allows authorized
  humans to revoke approval/publication for the affected revision. Distribution
  may roll back to the last acceptable approved revision while corrected content
  proceeds through a new draft and the complete required gates.

### Checkpoint G — Package, compatibility, and export

#### CA-19 — Logical runtime-package contents

- **Status:** Resolved.
- **Artifact separation:** A song/encounter authoring project and its approved
  platform-neutral runtime package are related but distinct artifacts. The
  authoring project is the durable production record; the runtime package is a
  compact, immutable consumer contract derived from an approved revision.
- **Authoring project contents:** Final master and source/control layers; rights
  and generation-provenance evidence; lyrics and arrangement notes where
  applicable; raw analysis and rejected suggestions; approved musical clock;
  canonical role transcriptions; all drafts and human edits; validator details;
  review evidence; approval records; tool/configuration history; and source and
  output fingerprints.
- **Runtime package contents:** Stable song and encounter identities; immutable
  content revision and schema version; runtime-ready full mix and controllable
  role audio; declared playable roles and audio mappings; duration, tempo/meter
  map, beat grid, and musical sections; difficulty charts with canonical-event
  lineage; phrases/passages; Activity Maps and approved candidate eligibility;
  fixed encounter timeline; localization/accessibility metadata; validation and
  approval summary; dependency declarations; and integrity hashes.
- **Privacy and size boundary:** Sensitive rights documents, raw model output,
  rejected suggestions, full edit history, and detailed internal review media do
  not ship to runtime consumers. The package retains identifiers and hashes that
  trace its summary claims back to the authoring project.

#### CA-20 — Schema evolution and dependency compatibility

- **Status:** Resolved.
- **Separate identities:** Content revision identifies one immutable approved
  song/encounter build. Schema version identifies the package format and
  semantics understood by tools and runtime consumers. Changing either never
  mutates an existing approved artifact.
- **Compatibility rule:** A safely ignorable additive field or capability may
  remain compatible within the current schema generation. Removing a required
  field, changing meaning or units, changing required interpretation, or using
  an incompatible structure requires a new major schema version.
- **Consumer contract:** Every consumer declares its supported major versions
  and required capabilities. It rejects an incompatible package with a clear
  diagnostic rather than guessing, silently defaulting changed semantics, or
  partially loading content.
- **Migration:** Migration or rebuilding creates a new content revision with
  provenance linking its source package and migration tool/configuration. During
  an intentional transition, authoring/export tooling supports the current and
  immediately previous major schema long enough to rebuild and validate active
  content. Retirement is explicit after consumers and active content migrate.
- **Rollback:** Runtime selection references an exact approved package and may
  return to an earlier consumer-compatible revision without rewriting it.

#### CA-21 — Roblox export adaptation

- **Status:** Resolved.
- **Canonical boundary:** The approved platform-neutral runtime package remains
  canonical. Roblox export produces an adapter-specific artifact tied to its
  exact package revision; it does not create an independent gameplay truth.
- **Permitted adaptation:** The exporter may transcode audio, upload or replace
  local paths with Roblox asset references, compress/chunk/repack data, adapt
  filenames and directory layout, and omit authoring-only evidence. It records
  all settings, generated asset references, and resulting fingerprints.
- **Prohibited adaptation:** Export may not change chart or phrase meaning,
  musical timing, difficulty behavior, role identity, candidate eligibility,
  encounter windows, reaction time, conflict rules, roster coverage, or any
  other approved semantic rule.
- **Equivalence report:** Each export proves required audio layers and durations,
  role/difficulty charts, event/candidate records, identities, references,
  package integrity, and timing tolerances survived adaptation. Missing assets,
  timing drift, unsupported capabilities, or semantic loss block test/release.
- **Platform-limit response:** If a Roblox constraint cannot preserve approved
  semantics, export fails and returns the issue to Content Authoring. Resolution
  creates a new reviewed package revision instead of an exporter-only exception.

### Checkpoint H — Reproducibility, release, and reconciliation

#### CA-22 — Provenance and reproducibility

- **Status:** Resolved.
- **Owner answer:** The project owner creates, controls, uses, and publishes the
  audio. Proving ownership back to the same person would add bureaucracy without
  reducing current risk.
- **Design rule:** For owner-created audio, a simple internal declaration of that
  fact satisfies the authoring ownership gate. Retain stable source identity and
  fingerprints, content revision, useful generation/processing tool and model
  settings, accepted human edits, and output hashes only to the extent they help
  explain, debug, or rebuild the content.
- **Explicit exclusion:** No separate license packet, legal-review workflow, or
  proof-of-rights dossier is required for the owner's own audio. If third-party
  material is introduced later, or a publishing platform mandates evidence,
  that content must satisfy a new explicit intake requirement before approval.
- **Reproduction target:** Deterministic data transforms should reproduce from
  pinned inputs/configuration. Audio encoding may prove timing and semantic
  equivalence rather than byte identity. Nondeterministic suggestions need not
  recur exactly because accepted canonical edits are retained.

#### CA-23 — First-release production policy

- **Status:** Resolved.
- **Design rule:** Produce the first three bosses with the maintained chart
  pipeline plus the approved simple internal web app. Completing every possible
  automation feature is not a prerequisite for authoring real content.
- **Temporary work allowance:** A documented manual step or one-song adapter is
  acceptable when it is reproducible, writes its final result into the canonical
  project/package model, creates no hidden runtime dependency, passes the normal
  validators/review, and does not fork the schema.
- **Generalization trigger:** The second need for the same workaround establishes
  a shared tooling requirement that must be generalized before the third package
  ships. Any workaround affecting correctness, fairness, traceability needed for
  debugging, or approval gates is generalized immediately.

#### CA-24 — Later-spec reconciliation and completion audit

- **Status:** Resolved.
- **Requirement register:** Specifications 2 through 12 record every new
  authored-data handoff with owning system, semantic requirement, validation
  rule, runtime consumer, compatibility impact, and whether the current package
  already supports it.
- **Reconciliation:** After specification 12, collect the register, resolve
  contradictions and ownership gaps, update the logical package/validators/
  workflow, and obtain downstream consumer sign-off. No system may create a
  private competing song-data contract to avoid reconciliation.
- **Completion audit:** The reconciled contract must cover every downstream
  authoring requirement, contain no orphan fields, leave gameplay semantics with
  their owning systems, preserve platform-neutral authority, and support the
  first three content packages through validation and Roblox review.
- **Current status:** The first-pass canonical `CONTENT_AUTHORING.md` is approved
  now as the foundational baseline. Its mandatory reconciliation remains an
  explicit program gate before technical architecture is finalized.

## 4. Cross-specification handoffs

- `ITEMS_AND_EQUIPMENT.md` must support a roster that is broader and more
  extensible than the current pipeline's four categories.
- `MULTIPLAYER.md` and `UI_UX.md` must handle song-specific role availability
  without implying that every song offers every instrument.
- `AUDIO_PRESENTATION.md` must define acceptable controllability and quality for
  every role a song actually offers.

## 5. Open issues

- The current tool uses `medium` while the GDD uses the player-facing label
  Normal. The logical contract must decide whether this is an internal export
  label or should be migrated.
- The exact first-release global instrument catalog and song-selection behavior
  for unavailable roles belong to later specifications; this contract already
  requires song-specific availability.
- The version-1 `attacks` array is retained only as legacy input. The canonical
  package uses the broader encounter-timeline and candidate contracts defined in
  the approved baseline.
- Later specs 2–12 added authored-data and validation requirements. Their
  mandatory final reconciliation was completed in canonical
  `CONTENT_AUTHORING.md` on 2026-09-02.

## 6. Change log

- **2026-08-19:** Created the working record and captured the inherited GDD,
  systems-map, and version-1 pipeline baseline. Progress is 0 of 24 questions.
- **2026-08-19:** Recorded the owner's clarification that the four current
  pipeline instruments are only a starting point and that valid songs may add,
  omit, or de-emphasize roles according to their real arrangements. CA-02
  remains partially resolved; total progress remains 0 of 24.
- **2026-08-19:** Approved the rest of checkpoint A, finalized the flexible-role
  interpretation, and resolved CA-01 through CA-03. Progress is 3 of 24.
- **2026-08-19:** Inspected the Blackened Crown derived runtime charts, approved
  the clock/transcription/phrase recommendations, and resolved CA-04 through
  CA-06. Progress is 6 of 24.
- **2026-08-19:** Resolved CA-08 and CA-09 with deterministic initial Activity
  Maps and approved-candidate semantics. Recorded the owner's density concern
  for CA-07, which remains partially resolved. Progress is 8 of 24.
- **2026-08-19:** Approved the density-aware complexity-envelope rule, finalized
  CA-07, and completed checkpoint C. Progress is 9 of 24.
- **2026-08-20:** Clarified that revival is requested from live downed state and
  uses the Activity Map only to find a fair start boundary; it is not pre-placed
  on the encounter timeline. Also recorded the initial simple internal web-app
  direction for CA-14. Resolved progress remains 9 of 24.
- **2026-08-20:** Approved CA-10 through CA-12 and CA-14, completing checkpoint D
  and defining the minimum local web authoring surface. Progress is 13 of 24.
- **2026-08-20:** Approved CA-13 and CA-15, defining automation boundaries and
  the dependency-aware lifecycle and completing checkpoint E. Progress is 15
  of 24.
- **2026-08-20:** Approved CA-16 through CA-18, defining validator severity,
  the in-Roblox matrix, and auditable approvals/exceptions and completing
  checkpoint F. Progress is 18 of 24.
- **2026-08-21:** Approved CA-19 through CA-21, separating authoring and runtime
  artifacts, defining schema evolution, and bounding Roblox adaptation and
  completing checkpoint G. Progress is 21 of 24.
- **2026-08-21:** Simplified provenance for owner-created audio, approved the
  first-release production and reconciliation policies, resolved CA-22 through
  CA-24, and reconciled the interview into canonical `CONTENT_AUTHORING.md`.
  Progress is 24 of 24.
- **2026-09-02:** Recorded completion of the mandatory reconciliation against
  specifications 2 through 12. Canonical `CONTENT_AUTHORING.md` now consolidates
  every registered runtime field, owner, consumer, compatibility impact,
  validator, fallback, export-equivalence, and human-review requirement.
