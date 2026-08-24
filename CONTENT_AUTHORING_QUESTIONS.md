# Bands Battle Content Authoring Specification Questions

- **Status:** Interview complete; 24 of 24 questions resolved
- **Started:** 2026-08-19
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#81-song-chart--encounter-authoring)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Existing tool:** [`tools/chart-pipeline/`](tools/chart-pipeline/README.md)
- **Working record:** [`CONTENT_AUTHORING_WORKING.md`](CONTENT_AUTHORING_WORKING.md)
- **Canonical result:** [`CONTENT_AUTHORING.md`](CONTENT_AUTHORING.md)

## 1. Interview method

This finite interview defines the design and production contract for turning an
approved song and encounter concept into a reviewed, versioned, platform-neutral
runtime package. Questions are asked in eight checkpoints of three so the owner
can make connected decisions without receiving an overwhelming list.

After each checkpoint, approved answers and their consequences are persisted in
`CONTENT_AUTHORING_WORKING.md`. When all questions are resolved, the working
record is reconciled into `CONTENT_AUTHORING.md` and audited against the GDD,
systems map, and existing chart pipeline.

This is a design specification, not a code architecture. It may define semantic
data and workflow contracts that architecture must later implement, but it does
not prescribe TypeScript modules, Roblox services, network messages, or storage
technology.

## 2. Fixed inherited decisions

The interview does not reopen these approved rules:

- The system is offline and platform-neutral, not an in-Roblox authoring tool.
- It extends the maintained `tools/chart-pipeline/` rather than replacing its
  working behavior without cause.
- Automation and AI may propose content but may never approve or publish it.
- The beat grid is foundational shared data for charts and encounter events.
- Charts use three inputs; first release excludes chords, swipes, flicks, and
  dragging between pads.
- Each supported instrument has Easy, Normal, and Hard authored chart behavior.
- Difficulty preserves musical identity and normalizes maximum contribution per
  passage rather than rewarding raw note count.
- A normal encounter uses one full song and maps five flexible functions:
  Arrival, First Clash, Escalation, Climax, and Finishing Cadence.
- Activity Maps and ensemble coverage must prove viable opportunities for
  gameplay events across supported instruments, difficulties, and rosters.
- Automatic validation is necessary but never sufficient for approval. Human
  musical judgment and in-Roblox review are mandatory.
- User-authored songs and a public creator product are outside first release.
- The initial Content Authoring contract is reconciled after specifications 2
  through 12 and before technical architecture becomes canonical.

## 3. Existing version-1 baseline

The maintained pipeline currently accepts either a mixed song or a stem
directory and exports a platform-neutral bundle containing:

- drums, vocals, guitar, and bass stems;
- Easy, Medium, and Hard charts for each instrument;
- note time, lane, and duration;
- simple attack windows with start, end, and threshold;
- duration and source-offset timing in seconds;
- a schema-versioned manifest; and
- a basic validation report covering counts, grid rejection, duplicates, and
  grid error.

The word **Medium** in the current technical schema and **Normal** in the GDD
describe the same middle difficulty until the contract explicitly resolves the
export label. Drums, vocals, guitar, and bass are also current implementation
categories, not an approved permanent roster. The GDD explicitly defers the
final first-release instrument roster, and a valid arrangement may include
other roles or omit one of these four entirely.

## 4. Question plan

### Checkpoint A — Authority, intake, and identity

#### [x] CA-01 — Human roles and approval authority

- **Decision needed:** Which logical roles create, review, and finally approve a
  song package, and which approvals must come from a different person?
- **Why now:** Automation is already barred from approval, but the human
  separation of duties is unsettled.
- **Must resolve:** Accountability per song, musical/chart approval,
  encounter/gameplay approval, technical validation acceptance, and whether a
  small team may combine authoring roles.
- **Decision:** One Content Owner is accountable per song. Creation roles may be
  combined, but final release requires musical/chart and encounter/gameplay
  approval plus participation by at least one human other than the primary
  author in the final in-Roblox review. Automated validation is evidence, not an
  approver.

#### [x] CA-02 — Source intake and release eligibility

- **Decision needed:** What inputs permit exploratory processing, and what
  minimum source package is mandatory before a song may become a release
  candidate?
- **Why now:** The existing tool can process a master or stems, while the game
  requires controllable musical layers and proven rights/provenance.
- **Must resolve:** Final master, stems or equivalent layers, lyrics,
  rights/generation provenance, arrangement notes, encounter brief, supported
  playable roles, and rules for intentionally absent, sparse, atmospheric,
  missing, or low-quality inputs.
- **Decision:** Exploratory processing may begin from a master or incomplete
  sources. A release candidate requires the final master, independently
  controllable audio for every role it offers, rights/generation provenance,
  lyrics when applicable, song metadata, available arrangement notes, and an
  encounter brief. The global role catalog is extensible and each song declares
  only the musically authentic subset it supports. Absent roles never require
  fabricated notes; sparse or atmospheric material qualifies only when it can
  sustain an authentic playable role.

#### [x] CA-03 — Stable identity and revision semantics

- **Decision needed:** How is a song/encounter identified across source changes,
  chart revisions, validation, export, Roblox review, and release?
- **Why now:** Later systems must reference an exact approved package rather than
  an ambiguous folder or mutable filename.
- **Must resolve:** Stable identity, immutable revisions, source fingerprints,
  supersession, and which revision a boss encounter references.
- **Decision:** Songs and encounters have separate stable identities. An
  immutable package revision binds exact sources, charts, encounter timeline,
  validation, and approvals. Approved revisions are never overwritten; a change
  creates a superseding revision while retaining prior audit and rollback
  history. Runtime content references an exact approved revision, never an
  implicit latest version.

### Checkpoint B — Musical clock and chart truth

#### [x] CA-04 — Canonical musical clock

- **Decision needed:** What constitutes the canonical timebase shared by audio,
  beat grid, charts, Activity Maps, and encounter events?
- **Must resolve:** Tempo changes, meter, pickup/lead-in, source offset, full-song
  duration, seconds versus musical positions, and correction authority.
- **Decision:** The approved full master anchors exact time at the beginning of
  its audio, including intentional silence, count-in, or pickup. A human-approved
  tempo-and-meter map relates musical positions to exact audio times. Authoring
  and packages retain both representations. Human timing correction overrides
  analysis, and a changed map invalidates dependent content until rebuilt and
  reviewed.

#### [x] CA-05 — Detailed chart source of truth

- **Decision needed:** What is the authoritative highest-detail chart for each
  instrument, and how must playable notes relate to audible performance?
- **Must resolve:** Instrument roles, note and hold identity, lane mapping,
  song-specific role availability, keyboards/synthesizers and other future
  roles, dropouts, ambiguous/polyphonic material, and human correction.
- **Decision:** Each playable role has one human-approved canonical performance
  transcription that is independent of difficulty. It records audible musical
  material, holds, rests, dropouts, and relevant expressive changes. Every
  playable event must have an audible basis, but not every audible note must
  become an input. Difficulty charts and their three-pad mappings are reviewed
  derivatives that trace back to this source rather than becoming unrelated
  copies.

#### [x] CA-06 — Phrase and passage boundaries

- **Decision needed:** How are notes grouped into phrases and chained into
  performance passages?
- **Must resolve:** One/two-measure default, allowed exceptions, intent
  boundaries, crossing holds, rests, dropouts, and authored versus suggested
  groupings.
- **Decision:** Analysis may suggest boundaries, but a human approves them.
  Phrases normally span one or two measures, with musically justified exceptions
  for pickups, unusual meter, long sustains, or transitions. Adjacent phrases
  may chain into a passage without downtime. Holds may cross boundaries as one
  continuous event. Phrase endings do not create breaks; actual musical or
  encounter reasons do.

### Checkpoint C — Difficulty and participation coverage

#### [x] CA-07 — Difficulty derivation contract

- **Decision needed:** Which chart is authored first, how are other difficulties
  derived, and what must remain invariant?
- **Must resolve:** Detailed-source difficulty, Easy/Normal/Hard relationship,
  density reduction, musical accents, holds, syncopation, and contribution
  normalization evidence.
- **Initial direction:** All difficulties share the master audio, clock, phrase
  and passage identities, and normalized maximum passage contribution. They are
  human-reviewed derivatives of the canonical transcription; automation only
  suggests reductions. The derivation must be flexible when the baseline
  musical part is already difficult or unusually dense. This direction required
  the density-aware refinement below.
- **Final decision:** Each difficulty applies a playtested complexity envelope
  rather than preserving a fixed percentage or assuming the core arrangement is
  automatically Normal. Human-reviewed musical importance guides which material
  survives under density, burst, alternation, subdivision, hold, and sustained-
  activity limits. Sparse parts may change little; dense parts may be reduced
  substantially even on Normal or Hard. Charts need semantic source lineage but
  need not be literal subsets. The logical label is `normal`; `medium` is a
  legacy version-1 import label.

#### [x] CA-08 — Activity Map granularity and facts

- **Decision needed:** What canonical facts does the Activity Map record and at
  which musical boundaries?
- **Must resolve:** Beat/measure/phrase granularity, density, rests, crossing
  holds, entries/exits/solos, energy, future activity, reaction time, conflicts,
  and distance from the finisher.
- **Decision:** For a fixed approved package revision and configuration, the
  Activity Map is deterministic. It records objective per-beat facts for every
  role/difficulty and deterministic measure/phrase/passage summaries. Human-
  approved interpretive labels become fixed inputs to that revision. Later
  probabilistic analysis may suggest labels, but cannot make canonical output
  vary from run to run.

#### [x] CA-09 — Candidate eligibility rules

- **Decision needed:** How does authored data distinguish a possible event
  opportunity from an approved runtime candidate?
- **Must resolve:** Candidate types, minimum warning/reaction time, instrument
  eligibility, difficulty coverage, maximum delay, and why a candidate may be
  suppressed.
- **Decision:** An approved candidate is a validated opportunity, not a runtime
  event. It records type, time range, compatible roles/difficulties/rosters,
  reaction time, conflicts, and qualifying evidence. Automation may suggest it,
  but a human approves it. Runtime systems choose only among approved candidates;
  fixed phase boundaries and finishing events remain explicit timeline events.
  Recovery or revival may be requested whenever the owning gameplay rules allow;
  candidates only identify the earliest fair musical boundaries where the
  requested performance can begin. They do not predict or pre-script a downing.

### Checkpoint D — Encounter timeline and ensemble coverage

#### [x] CA-10 — Encounter timeline contract

- **Decision needed:** Which encounter functions and event tracks are authored
  against the song, and what must be fixed versus selectable at runtime?
- **Must resolve:** Five functions, resistance windows, boss attacks, movement,
  recovery, revival, Band Calls, Crescendos, solos, and finishing performance.
- **Clarification:** The authored timeline does not place a future downing or
  mandatory revival. It carries fixed encounter events plus derived eligibility
  for dynamic performances. When a player becomes downed, revival may be
  requested immediately; the selector uses the Activity Map to begin it at the
  earliest compatible musical boundary.
- **Decision:** One multi-track timeline aligns song sections; the five
  encounter-function spans; resistance availability; fixed phase, story, arena,
  and finishing events; and eligible opportunity data. Runtime state may select
  among validated opportunities but may not move the song clock. The package
  references owning gameplay definitions instead of redefining their mechanics.

#### [x] CA-11 — Roster-aware ensemble coverage

- **Decision needed:** Which player counts, instrument combinations, and duplicate
  instruments must be proven viable before approval?
- **Must resolve:** Solo acolytes, three-to-six-player co-op, duplicate parts,
  sparse/dropout instruments, current-roster evaluation, and coverage evidence.
- **Decision:** Deterministic validation covers every legal role combination for
  solo and two-to-six-human play, including duplicates, all-same-role bands,
  diverse roles, and dropout-sensitive rosters. Human review uses a smaller
  representative matrix. A package cannot ship when an allowed roster lacks
  required event coverage, and acolytes do not count as human chart coverage.

#### [x] CA-12 — Conflict and priority policy

- **Decision needed:** Which authored events may overlap and which require
  exclusion, spacing, or priority?
- **Must resolve:** Rhythm phrases, movement, boss impacts, recovery, revival,
  group invitations, finishers, telegraph time, and conflict arbitration.
- **Decision:** An explicit compatibility matrix governs overlap. Competing
  controls, prompts, movement decisions, or critical attention channels are
  incompatible; finishing space and competing group actions are protected.
  Rhythm and boss telegraphs may overlap only when reaction/readability evidence
  passes. Invalid candidates defer or use an approved fallback; incompatible
  fixed events block the package rather than being silently dropped.

### Checkpoint E — Assistance and authoring surface

#### [x] CA-13 — Automation boundaries

- **Decision needed:** Which pipeline outputs are suggestions, which are
  deterministic transforms of approved data, and what must always be authored
  or approved by a human?
- **Must resolve:** Confidence/evidence display, accepting or rejecting batches,
  regeneration, audit history, and prohibited autonomous publication.
- **Decision:** Mechanical transforms may run automatically from approved inputs,
  including conversion, compilation, deterministic Activity Maps, validation,
  and draft export. Outputs that establish musical or encounter truth remain
  suggestions until human acceptance. Authors receive preview/diff and may
  accept individually or in batches. Reruns create drafts, mark dependencies
  stale, preserve history, and never replace approved content or publish.

#### [x] CA-14 — Minimum authoring surface

- **Decision needed:** What must the first internal authoring workflow provide
  before a polished creator tool exists?
- **Must resolve:** Waveform/stems, beat grid, lanes, difficulty layers, encounter
  tracks, Activity Map, loop/scrub/edit, validation, comparison, and test export.
- **Owner direction so far:** Start with a simple internal web application. The
  author selects a song project directory; the app loads its master, stems, and
  existing artifacts, displays the timeline and Activity Map, and can rerun the
  maintained processing pipeline as needed. Detailed editing, revision, and
  export behavior remain to be resolved in checkpoint E.
- **Decision:** The minimum surface adds synchronized waveform/stem playback,
  mute/solo, beat and chart layers, phrases/passages, Activity Map and encounter
  tracks, scrub/zoom/loop, editing, full or stale-stage reruns, visible progress
  and cancellation, result comparison, draft preservation and undo, explicit
  saved/stale/invalid states, validation, and direct Roblox test export. Reruns
  never overwrite an approved revision.

#### [x] CA-15 — Authoring state and rework loop

- **Decision needed:** What lifecycle states move a song from intake through
  analysis, editing, validation, Roblox review, approval, and revision?
- **Must resolve:** Ownership handoffs, failed gates, rework targets, stale
  downstream artifacts, and return to earlier stages after source changes.
- **Decision:** Revisions progress through Intake, Analysis Draft, Authoring
  Draft, Validation, In-Roblox Review, Approved, Published/Exported, and Retired.
  Failed gates return the draft to the owning stage with findings. Dependency-
  aware staleness prevents advancement. Approved revisions are immutable;
  rework forks a new draft while retaining audit and rollback history.

### Checkpoint F — Validation and approval

#### [x] CA-16 — Validator classes and severity

- **Decision needed:** Which failures block progress automatically, which require
  human judgment, and whether any rule may be waived?
- **Must resolve:** Structural, musical, fairness, coverage, accessibility,
  compatibility, and technical validator categories plus error/warning policy.
- **Decision:** Deterministic structural, clock/alignment, chart/phrase,
  difficulty/normalization, Activity/roster, encounter/conflict, accessibility,
  compatibility/export, and provenance-completeness validators produce Error,
  Warning, or Information findings. Errors block advancement; warnings require
  acknowledgement; information is advisory. Findings identify their exact
  affected content and run incrementally plus as a full pre-review gate.

#### [x] CA-17 — In-Roblox review matrix

- **Decision needed:** What combinations must be played and approved before a
  package may ship?
- **Must resolve:** Every instrument/difficulty, solo, representative co-op
  rosters, phone/desktop/gamepad order, accessibility combinations, and evidence.
- **Decision:** Phone receives the complete role-by-difficulty playthrough
  matrix. Solo covers every role; representative co-op covers two, three, and
  six humans plus all-same, mixed, and dropout-sensitive rosters. Desktop and
  gamepad cover every role and difficulty without requiring their full cross-
  product unless risk appears. Accessibility uses a risk-based combination
  matrix, and a lower-capability supported phone profile is required.

#### [x] CA-18 — Approval record and exception policy

- **Decision needed:** What approvals and evidence form the auditable release
  decision, and how are exceptional waivers handled?
- **Must resolve:** Named approvers, timestamps, revision, validator reports,
  playtest evidence, open warnings, expiry, and emergency revocation.
- **Decision:** Approval binds the exact immutable revision, source fingerprints,
  validation, completed review matrix, both domain approvals, independent human
  review, warnings/exceptions, identities, and timestamps. Rights/provenance,
  corruption/compatibility, clock alignment, chart authenticity, required
  coverage, impossible interaction, safety, and essential accessibility errors
  cannot be waived. Other warnings need scoped, expiring, revision-specific
  approval. Serious defects permit revocation and rollback.

### Checkpoint G — Package, compatibility, and export

#### [x] CA-19 — Logical runtime-package contents

- **Decision needed:** Which semantic assets and records must one approved
  platform-neutral package contain?
- **Must resolve:** Audio layers, clock/beat grid, charts, phrase/passage data,
  Activity Maps, encounter timeline, provenance, validation, approvals, and
  localization/accessibility metadata.
- **Decision:** The authoring project retains sources, provenance evidence, raw
  analysis, canonical transcriptions, drafts, edit history, and detailed review
  evidence. Its approved platform-neutral runtime package contains exact
  identity/version data, runtime audio and role mappings, musical clock, charts,
  phrases/passages, Activity/candidate data, encounter timeline, localization/
  accessibility metadata, integrity hashes, and validation/approval summaries.
  Sensitive and authoring-only material remains referenced outside the runtime
  package.

#### [x] CA-20 — Schema evolution and dependency compatibility

- **Decision needed:** How do package and component revisions declare breaking
  versus compatible changes?
- **Must resolve:** Schema versioning, immutable builds, consumer requirements,
  migration/rebuild policy, deprecation, and rollback.
- **Decision:** Immutable content revision and package-format schema version are
  separate. Safely ignorable additions may remain compatible; changed meaning,
  units, removals, or incompatible structure require a new major schema.
  Consumers declare supported versions/capabilities and reject incompatibility
  explicitly. Migration creates a new revision, current and prior major tooling
  overlap during transition, and rollback uses an exact prior approved package.

#### [x] CA-21 — Roblox export adaptation

- **Decision needed:** What may a Roblox exporter transform without changing the
  approved design semantics?
- **Must resolve:** Audio encoding/upload references, data packing, asset IDs,
  platform limits, equivalence validation, and separation from Roblox runtime
  implementation.
- **Decision:** The canonical platform-neutral package remains unchanged. A
  Roblox exporter may transcode audio, substitute asset references, repack or
  chunk data, remove authoring-only evidence, and adapt layout, but may not alter
  gameplay semantics. Every export produces an equivalence report. A platform
  limitation requiring semantic change returns to authoring for a new revision.

### Checkpoint H — Reproducibility, release, and reconciliation

#### [x] CA-22 — Provenance and reproducibility

- **Decision needed:** What source, tool, model, configuration, and human-edit
  history must be retained so a package can be explained and rebuilt?
- **Must resolve:** Rights evidence, generated-music provenance, source hashes,
  tool/model versions, parameters, deterministic steps, and manual-edit history.
- **Decision:** Because the project owner creates, controls, uses, and publishes
  the audio, internal provenance is intentionally lightweight. Record an owner-
  created declaration, source identity/fingerprints, useful tool/model settings,
  content revision, and human edits needed for debugging or rebuilding. Do not
  require a separate license packet or internal legal proof. Any future external
  audio or platform-mandated evidence introduces its own explicit requirement.

#### [x] CA-23 — First-release production policy

- **Decision needed:** What is the acceptable production path for the first three
  bosses before the eventual full authoring surface exists?
- **Must resolve:** Lightweight-tool allowances, one-off manual steps, shared
  contract requirements, exception budget, and the signal that tooling must be
  generalized.
- **Decision:** Use the maintained pipeline plus the simple internal web app.
  Documented manual steps and one-song adapters are allowed when their result
  enters the canonical format and passes normal gates. They may not create a
  runtime dependency or schema fork. A repeated workaround becomes shared work
  and is generalized before the third package; correctness/fairness problems are
  generalized immediately.

#### [x] CA-24 — Later-spec reconciliation and completion audit

- **Decision needed:** How will specs 2–12 add authored-data requirements without
  silently fragmenting the package contract?
- **Must resolve:** Requirement intake, ownership, conflict resolution, final
  reconciliation, consumer sign-off, and criteria for declaring Content
  Authoring complete.
- **Decision:** Specifications 2–12 record each authoring handoff with owner,
  semantic data, validator, consumer, and compatibility impact. After spec 12,
  reconcile all handoffs into this contract, resolve conflicts/gaps, update
  packages and validators, and obtain consumer sign-off. The current canonical
  document is an approved foundational baseline with mandatory reconciliation
  pending.

## 5. Completion criteria

`CONTENT_AUTHORING.md` is complete only when:

- CA-01 through CA-24 are resolved and recorded;
- every inherited GDD requirement is represented by a workflow, semantic data,
  validation, or approval rule;
- existing version-1 behavior is preserved, deliberately superseded, or marked
  as an implementation baseline rather than accidentally contradicted;
- all downstream consumers have an explicit authored-data handshake;
- automation cannot approve or publish content;
- a package cannot ship on structural validation alone;
- the contract works without placing authoring inside Roblox; and
- the post-specification reconciliation obligation remains explicit.

## 6. Change log

- **2026-08-19:** Created the 24-question plan from `SYSTEMS_MAP.md`, GD-10 and
  related GDD decisions, and the maintained version-1 chart bundle.
- **2026-08-19:** Clarified that the pipeline's drums/vocals/guitar/bass model is
  a starting implementation catalog, not a mandatory song or final instrument
  roster. Song-specific role availability remains part of CA-02 and CA-05.
- **2026-08-19:** Approved checkpoint A and resolved CA-01 through CA-03.
  Progress is 3 of 24 questions.
- **2026-08-19:** Reviewed the Blackened Crown runtime-chart example, approved
  checkpoint B, and resolved CA-04 through CA-06. Progress is 6 of 24 questions.
- **2026-08-19:** Approved deterministic Activity Maps and validated event
  candidates, resolving CA-08 and CA-09. CA-07 remains open for a density-aware
  difficulty rule. Progress is 8 of 24 questions.
- **2026-08-19:** Approved the complexity-envelope refinement, resolved CA-07,
  and completed checkpoint C. Progress is 9 of 24 questions.
- **2026-08-20:** Clarified that revival candidates are fair start boundaries,
  not scheduled revival events, and recorded the initial simple local web-app
  direction for CA-14. Resolved progress remains 9 of 24 questions.
- **2026-08-20:** Approved the corrected encounter timeline, deterministic
  roster coverage, compatibility matrix, and minimum local web authoring
  surface. CA-10 through CA-12 and CA-14 are resolved; progress is 13 of 24.
- **2026-08-20:** Approved automation boundaries and the dependency-aware
  authoring lifecycle, resolving CA-13 and CA-15 and completing checkpoint E.
  Progress is 15 of 24.
- **2026-08-20:** Approved validator classes/severity, the in-Roblox review
  matrix, and the auditable exception/revocation policy, resolving CA-16 through
  CA-18 and completing checkpoint F. Progress is 18 of 24.
- **2026-08-21:** Approved the authoring/runtime artifact boundary, schema and
  compatibility policy, and semantics-preserving Roblox adapter, resolving
  CA-19 through CA-21 and completing checkpoint G. Progress is 21 of 24.
- **2026-08-21:** Simplified CA-22 for owner-created audio, approved the
  lightweight first-release production policy and later-spec reconciliation,
  resolved CA-22 through CA-24, and produced canonical `CONTENT_AUTHORING.md`.
  Progress is 24 of 24.
