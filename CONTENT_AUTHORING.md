# Bands Battle Content Authoring

- **Status:** Approved and reconciled
- **Approved:** 2026-08-21
- **Reconciled:** 2026-09-02
- **Parent system map:** [`SYSTEMS_MAP.md`](SYSTEMS_MAP.md#81-song-chart--encounter-authoring)
- **Player-facing authority:** [`GAME_DESIGN.md`](GAME_DESIGN.md)
- **Decision source:** [`CONTENT_AUTHORING_WORKING.md`](CONTENT_AUTHORING_WORKING.md)
- **Interview plan:** [`CONTENT_AUTHORING_QUESTIONS.md`](CONTENT_AUTHORING_QUESTIONS.md)
- **Existing implementation baseline:** [`tools/chart-pipeline/`](tools/chart-pipeline/README.md)

## 1. Role and authority

This document defines how approved music and encounter concepts become reviewed,
versioned, platform-neutral runtime packages. It owns the production contract:
source intake, musical analysis, human chart authoring, difficulty derivatives,
Activity Maps, encounter timelines, validation, review, approval, revision, and
runtime export.

It does not redefine player-facing behavior in `GAME_DESIGN.md`, own Roblox
runtime execution, prescribe client/server architecture, or create an in-game or
public song editor. A system boundary here is a semantic and workflow boundary,
not a required TypeScript module, web framework, Roblox service, or file layout.

This was the first-pass contract required before gameplay specifications. Specs
2 through 12 registered their authored-data needs, and the completed mandatory
reconciliation in section 14 now governs technical-architecture handoff.

## 2. Governing principles

1. **The approved song is the clock.** Charts and encounter events share one
   human-approved musical timing map.
2. **The arrangement is truth.** A playable action must have an audible musical
   basis; the pipeline never invents an absent instrument part.
3. **Roles are song-specific.** Drums, vocals, guitar, and bass are initial tool
   categories, not a permanent or mandatory roster.
4. **Automation proposes; humans approve.** No analysis tool or AI approves or
   publishes musical or encounter truth.
5. **Validation is necessary, not sufficient.** Every package also requires
   musical judgment and in-Roblox review.
6. **Approved revisions are immutable.** Rework creates a new revision with
   explicit history and rollback.
7. **The platform-neutral package is canonical.** Roblox export may adapt
   representation but not semantics.
8. **Build useful tooling, not a creator platform.** The first three bosses use
   a lightweight internal web app built around the maintained pipeline.

## 3. People and approval authority

Every song/encounter package has one accountable **Content Owner**. Logical
creation responsibilities include Music/Chart Author and Encounter Designer. A
small team may combine creation roles.

Final approval requires:

- explicit musical/chart approval;
- explicit encounter/gameplay approval; and
- participation by at least one human other than the primary author in the final
  in-Roblox review.

Automated validation supplies evidence but cannot serve as an approver. Approval
records identify the exact revision, accountable owner, creators, reviewers,
decisions, and timestamps.

## 4. Artifacts, identity, and revisions

### Stable identities

- A stable song identity represents the musical source lineage.
- A separate stable encounter identity represents the boss scenario mapped to
  that song.
- An immutable content revision binds exact source fingerprints, musical clock,
  charts, Activity/encounter data, validation, and approval.
- Consumers reference an exact approved revision, never an implicit `latest`.

Approved content is never edited in place. A source, timing, chart, encounter,
validation-relevant, or approval-relevant change creates a superseding revision.
Prior approved revisions remain available for audit and rollback.

### Authoring project

The durable project directory contains:

- final master and available source/control layers;
- an owner-created audio declaration or any externally required source evidence;
- lyrics and arrangement notes when applicable;
- raw analysis and rejected suggestions;
- approved clock and canonical role transcriptions;
- difficulty charts, Activity Maps, candidates, and encounter timeline;
- drafts, human edits, validation, review, and approval records; and
- useful tool/configuration history plus source and output fingerprints.

### Runtime package

The approved platform-neutral runtime package contains:

- song, encounter, content-revision, and schema identities;
- runtime-ready full mix and controllable role audio;
- declared roles and audio mappings;
- duration, tempo/meter map, beat grid, and musical sections;
- difficulty charts with canonical-event lineage;
- phrase and passage data;
- Activity Maps and approved candidate eligibility;
- fixed encounter timeline;
- localization and accessibility metadata;
- dependency/capability declarations;
- validation and approval summary; and
- integrity hashes.

Raw suggestions, internal review media, full edit history, and any sensitive
source evidence remain in the authoring project rather than the runtime package.

## 5. Intake and playable-role eligibility

Exploratory analysis may begin from a mixed master or incomplete sources. A
revision cannot become a release candidate until it has:

- the final approved master;
- independently controllable audio or an approved equivalent for every offered
  playable role;
- a declaration that the audio is owner-created, or required evidence for any
  future external source;
- lyrics when vocals are present;
- stable song metadata;
- available arrangement notes; and
- an encounter brief.

The project maintains an extensible role catalog. Each song declares only the
subset its real arrangement supports. Piano, synthesizer, percussion, strings,
or later roles may be offered. Instrumental songs may omit vocals. A muted,
atmospheric, sparse, or absent conventional part is not forced into a chart.

A sparse role qualifies only when human review finds enough authentic material
for meaningful play and later coverage validation passes. Duplicate roles are
allowed, so multiplayer capacity does not require many distinct roles.

## 6. Musical clock and canonical transcription

### Clock

Exact time begins at the start of the approved full master, preserving any
silence, count-in, lead-in, or pickup. A human-approved tempo-and-meter map
relates measure/beat/subdivision positions to exact audio time across tempo and
meter changes.

Authoring data retains musical position and exact-time mapping. Automated beat
analysis only proposes the map. Human correction establishes authority. Editing
an approved map marks every dependent transcription, difficulty chart, phrase,
Activity Map, candidate, encounter event, validation result, and export stale.

### Canonical role transcription

Every offered playable role has one human-approved, difficulty-independent
performance transcription. It records authentic musical event identity, timing,
holds, rests, dropouts, relevant expression, and relationship to its audio.

Every chart action traces to an audible musical basis, but not every audible note
must become an input. Authors decide which material is playable and how derived
charts map it to three pads. The canonical transcription is not named Hard and
may contain more detail than any shipped difficulty.

The existing Blackened Crown charts demonstrate a valid derived runtime shape:
role, difficulty, duration, and `time`/`lane`/`duration` notes. They are not the
editable canonical transcription because they lack musical coordinates,
phrases, Activity Maps, and encounter data.

## 7. Phrases, passages, and difficulty

### Phrase and passage rules

- Analysis may suggest phrase boundaries; a human approves or edits each one.
- A phrase normally spans one or two measures.
- Pickups, unusual meter, long sustains, and clear transitions may justify
  shorter or longer exceptions.
- Adjacent phrases may chain into a passage without forced downtime.
- A hold may cross a phrase boundary as one continuous event and is marked for
  validation.
- A phrase ending alone never creates a break. Breaks follow actual musical or
  encounter reasons.

Judgment and combat-intent attribution for crossing holds belong to later Rhythm
and Combat specifications.

### Difficulty derivation

Easy, Normal, and Hard share the same master, clock, phrase boundaries, passage
identities, and maximum pre-difficulty contribution per passage. Song speed
never changes.

Difficulty uses a playtested **complexity envelope**, not a fixed percentage or
the assumption that a source's core arrangement is automatically Normal.
Canonical events carry human-reviewed musical importance. Authors preserve the
most important authentic material that fits limits for local density, burst
rate, alternation, subdivisions, hold complexity, and sustained activity.

A sparse part may change little across difficulties. A dense part may require
substantial reduction even on Normal, and Hard may omit unreadable detail. Easy
may consolidate repeated notes or simplify a pattern. Derived charts require
semantic lineage to the canonical source but need not be literal subsets.

The new logical label is `normal`. The current version-1 `medium` label is
accepted only when importing legacy charts.

## 8. Activity Maps and event candidates

### Deterministic Activity Map

For an exact source revision and configuration, Activity Map generation is
deterministic. It records facts at each beat boundary for every role and
difficulty, with deterministic measure, phrase, and passage summaries.

Required facts include:

- current and upcoming playable density;
- time to the next event and longest rest;
- dropouts and crossing holds;
- entries, exits, and solos;
- approved energy or quietness labels;
- whether meaningful activity continues;
- distance from the Finishing Cadence;
- known encounter conflicts; and
- available warning and reaction time.

Human authors approve interpretive musical labels. Future probabilistic analysis
may suggest labels but cannot make an approved package nondeterministic.

### Candidate semantics

A candidate is a validated opportunity, not a runtime event. It records type,
allowed musical/time range, compatible roles/difficulties/rosters, warning and
reaction time, duration, conflicts, suppression reasons, and evidence.

Automation may suggest candidates. A human approves them, and validators confirm
them against owning gameplay requirements. Runtime systems may select only among
approved candidates compatible with current state.

A downing is not authored. Revival may be requested whenever live Survival rules
allow; candidate data only identifies the earliest fair musical boundary where
the requested performance can begin. The same distinction applies to other
state-dependent recovery and cooperative opportunities.

## 9. Encounter timeline, coverage, and conflicts

### Timeline

One multi-track timeline aligns:

- musical sections;
- Arrival, First Clash, Escalation, Climax, and Finishing Cadence spans;
- resistance-layer availability;
- fixed phase, story, arena, and finishing events; and
- eligibility for state-dependent opportunities.

Entries use the approved musical clock and carry lead time, duration,
dependencies, conflicts, and references to their owning gameplay definitions.
Runtime state may choose candidates or react to fixed events, but it never moves,
pauses, skips, or rewinds the authored song timeline.

### Roster coverage

Deterministic validation evaluates every legal song-specific role combination
for solo and two-to-six-human play. It includes unrestricted duplicates,
all-same-role, diverse-role, and dropout-sensitive rosters. Two-human play uses
the approved consent flow; three-to-six remains the intended co-op range.

Solo evaluation covers every role/difficulty with fixed acolyte support.
Acolytes do not generate human charts or count as human playable coverage.

An allowed configuration cannot ship when a required event type lacks a valid
opportunity within its maximum delay. Content is corrected or the configuration
ceases to be allowed; validation cannot hide the failure.

### Compatibility matrix

Every event type participates in an explicit compatibility matrix. Events
competing for the same control, prompt, movement decision, or critical attention
channel are incompatible unless an owning spec deliberately permits and
validates the overlap.

The Finishing Cadence preview/performance rejects competing major attacks,
recovery/revival performances, Band Call invitations, and Crescendos. Band Call
invitation/performance windows and Crescendo preview/performance windows are
mutually exclusive; the guaranteed Crescendo has higher reservation priority.
Ordinary rhythm and boss telegraphs may overlap only when timing, control,
visual, audio, and device readability evidence passes.

An incompatible candidate defers, uses an approved urgent fallback, or is
skipped when optional. Incompatible fixed events block validation and cannot be
silently dropped at runtime.

## 10. Authoring workflow and internal web app

### Lifecycle

Each revision progresses through:

1. Intake
2. Analysis Draft
3. Authoring Draft
4. Validation
5. In-Roblox Review
6. Approved
7. Published/Exported
8. Retired

Failed gates return the draft to the responsible stage with findings attached.
Source/clock changes stale all downstream data. Transcription/role changes stale
affected difficulties, Activity Maps, candidates, validation, and exports.
Difficulty/phrase changes stale downstream map, coverage, and export evidence.
Encounter edits stale conflict, coverage, validation, and exports.

### Automation boundary

Mechanical operations may run automatically from approved inputs: conversion,
compilation, deterministic Activity Maps, validation, and draft export.

Human acceptance remains required for stem quality, clock corrections,
transcriptions, playable roles, difficulty charts, phrase/passage boundaries,
musical labels, candidates, and fixed encounter events. Suggestions may be
accepted individually or as a reviewed batch after preview and diff. Reruns
create drafts and never replace approved content or publish.

### Minimum internal web application

The initial local web app lets an author select a song project directory and
loads its master, stems, and existing artifacts. It provides:

- synchronized master/stem waveforms with mute and solo;
- beat/meter grid, transcription, difficulty, phrase/passage, Activity Map, and
  encounter tracks on one timeline;
- scrub, zoom, range selection, and loop;
- editing of timing, notes, boundaries, labels, candidate approval, and fixed
  events;
- full or stale-stage pipeline reruns with real progress and cancellation;
- comparison with the prior draft;
- drafts, undo, and clear saved/unsaved/stale/invalid/failed states;
- validation findings linked to their timeline ranges; and
- versioned runtime package and direct Roblox test export.

The later architecture chooses the framework, filesystem bridge, process
orchestration, and deployment model.

## 11. Validation and human review

### Validator classes and severity

Deterministic validators cover structure/inputs, clock/alignment,
chart/phrases, difficulty/normalization, Activity/roster coverage,
encounter/conflicts, accessibility metadata, export compatibility, and required
source declarations.

- **Error:** blocks the applicable review, approval, or export gate.
- **Warning:** remains visible and requires explicit acknowledgement.
- **Information:** advisory evidence requiring no action.

Each finding identifies its validator, revision, affected role/difficulty/
roster/field/timeline range, cause, and corrective action. Relevant validators
run incrementally; a full pass runs before Roblox review and again after any
review-driven change.

### In-Roblox review matrix

- Phone: complete playthrough for every offered role/difficulty plus at least one
  lower-capability supported phone profile.
- Solo: every offered role.
- Co-op: representative two-, three-, and six-human rosters, including all-same,
  mixed, and most dropout-sensitive combinations.
- Keyboard and gamepad: every role and every difficulty at least once; full
  cross-product only when risk appears.
- Accessibility: risk-based combinations covering reduced motion, non-color
  cues, scaling, supported remapping, and Hold Assist.

Each run records exact revision, device/profile, input, role, difficulty, roster,
accessibility settings, outcome, findings, reviewer, and date. Captures are
required when they materially help diagnose a defect, not for every success.

### Approval and exceptions

Approval binds the exact revision/fingerprints, validator reports, review matrix,
two domain approvals, independent reviewer participation, warnings/exceptions,
identities, and timestamps.

Corruption/incompatibility, clock misalignment, fabricated chart material,
missing required coverage, impossible interaction, safety, and essential
accessibility failures are non-waivable. For owner-created audio, the required
source declaration is sufficient; missing third-party or platform-mandated
evidence becomes non-waivable only when such content is actually introduced.

Other warnings need a revision-specific, scoped, expiring exception approved by
the Content Owner and relevant reviewer. A serious discovered defect permits
revocation and rollback to the last acceptable revision.

## 12. Schema evolution and Roblox adaptation

Content revision and schema version are separate. Safely ignorable additive
fields may remain compatible. Removed fields, changed meaning or units, changed
required interpretation, or incompatible structure require a new major schema.

Consumers declare supported versions and capabilities and reject incompatible
packages explicitly. Migration creates a new content revision with linked
source/configuration. Tooling supports the current and immediately prior major
schema during an intentional transition. Rollback selects an exact compatible
approved revision.

A Roblox exporter may transcode audio, upload/substitute asset references,
compress/chunk/repack data, adapt layout, and omit authoring-only evidence. It
may not change musical timing, chart/phrase meaning, difficulty behavior, role
identity, candidate eligibility, event windows, reaction time, conflicts, or
roster coverage.

Every Roblox export produces an equivalence report covering audio duration and
alignment, roles/charts/events, references, integrity, capabilities, and timing
tolerances. A platform limit that cannot preserve semantics returns to authoring
and requires a new approved revision.

## 13. Practical traceability and first-release production

The current project uses owner-created audio. Internal traceability therefore
exists to reproduce and debug content, not to prove the owner's rights back to
the owner. Record the owner-created declaration, source identity/fingerprints,
useful tool/model/configuration details, content revision, accepted edits, and
output hashes. Do not require a separate license dossier or legal workflow.

Deterministic data transforms should reproduce from pinned inputs/configuration.
Audio encodes may prove semantic equivalence rather than byte identity.
Nondeterministic suggestions need not recur exactly because canonical human edits
are retained.

The first three bosses use the maintained pipeline and internal web app.
Documented manual steps and one-song adapters are allowed when they enter the
canonical format, create no hidden runtime dependency or schema fork, and pass
normal validation/review. A second need for the same workaround creates a shared
tooling requirement that is generalized before the third package. Correctness or
fairness workarounds are generalized immediately.

## 14. Cross-specification handoffs and reconciliation

The mandatory reconciliation against canonical specifications 2 through 12 was
completed on 2026-09-02. This section is now the authoritative consolidation of
their song-, chart-, encounter-, audio-, cue-, and validation-data needs.

### Ownership boundary

The platform-neutral content package owns authored music, timing, mapping,
candidate, encounter, arena, cue-reference, and review evidence. Runtime systems
own gameplay meaning and mutable state. Global system catalogs own formulae,
balance values, item/build/ability definitions, reward pools, Progression rules,
settings, and persistence behavior.

Content may reference exact compatible catalog and balance revisions. It may not
copy private system values into chart notes, hide build behavior or bans, define
paid/economy outcomes, store player state, or create a competing runtime schema.

Every runtime field declares:

- stable identity and owning domain;
- exact units, clock/reference frame, and content/schema revision;
- required or optional status plus legal fallback/degradation behavior;
- permitted consumers and privacy/access class;
- compatibility and migration impact; and
- the validator/review evidence required for publication.

### Identity, compatibility, and global ordering

The package preserves stable identities for song, encounter, content revision,
role, difficulty, canonical event, derived event, phrase, passage, candidate,
timeline event, attack, arena node/edge/state, cue, audio asset/layer, and
practice/teaching trigger. Identities never collide across revisions.

Every exact-time or musical-boundary event carries a stable global tie order.
Runtime replay and export never use file, network, or receipt order to resolve
ties.

Each revision declares compatible/incompatible relationships to prior revisions,
including personal-record categories affected by chart, scoring, duration,
phrase, passage, role, or encounter changes. Campaign, mastery, reward, build,
item, and recommendation catalogs remain system data referenced through stable
boss/encounter and catalog identities.

### Rhythm and contribution data

For every playable role and difficulty, the runtime package includes:

- legal lane/action type, exact musical/time position, hold span, repeat or
  alternate grouping, phrase/passage lineage, and canonical-event lineage;
- per-event normalized weight whose passage sum equals the shared maximum
  contribution budget regardless of difficulty or note density;
- deterministic onset/duration allocation and intent segmentation for holds
  crossing phrase, passage, or effective scoring boundaries;
- Activity Map lookahead required to evaluate fair suspension/re-entry; and
- dynamic recovery candidate identity, challenge kind, role/difficulty/roster
  support, conflict data, earliest legal start, and maximum delay.

Release grace, judgment windows/factors, global Early Miss association,
calibration bounds, drift/resync thresholds, and other runtime tuning remain
versioned Rhythm/operational configuration rather than song-authored values.

Publication rejects overlapping holds, another event on a held lane, physically
infeasible release/repress density, fabricated musical material, unequal passage
budgets, ambiguous crossing allocation, or supported transitions without fair
re-entry/recovery material.

### Encounter, combat, and arena data

The encounter contract includes:

- ordered Arrival, First Clash, Escalation, Climax, and Finishing Cadence regions;
- three ordered Resolve openings, finishing identity/boundary, and explicit
  Momentum/post-break eligibility ranges and bank kinds;
- reproducible candidate family, seed weight, cooldown/repeat group, intensity
  cost, variants, preview/duration, priority, maximum delay, fallback, and
  future-capacity effect;
- attack Telegraph/Commit/Impact/Recovery/cancellation boundaries, target and
  geometry eligibility, effect/hazard tags, cover interaction, difficulty
  reference, mitigation-cap reference, cues, and earned Recovery advantage;
- Resolve-layer or Momentum destination eligibility and deterministic overflow;
- multi-target distribution form, roster-aware cap reference, and invalid-target
  fallback;
- arena locations, edges, geometry/anchors, starts/fallbacks, cover/hazard/risk
  tags, graph transformations, routes, travel/multi-edge timing, displacement,
  and mutation constraints; and
- difficulty and one-to-six-human Resolve, target, geometry, and coverage
  variants without roster-created cue ambiguity.

The compatibility/reservation matrix covers every fixed event and candidate.
It proves required future events remain feasible, targets lock at Commit,
committed routes remain legal, graph states stay reachable, lead/reaction time is
fair, simultaneous effects reproduce, and critical data has an approved
cancel/No Contest path when it cannot survive runtime/export.

Combat formulae, Ward values, modifier caps, movement constants, candidate
selection algorithms, and reward values remain versioned system/balance data.

### Abilities, cooperative actions, and solo support

Song/encounter packages reference exact compatible Ability definitions and
provide authentic musical opportunity data for:

- Signature legal boundary, target/fallback, typed effect reference, and cues;
- Band Call readiness opportunities, maximum delay, lead, duration, conflict,
  role/difficulty/roster coverage, effect/fallback reference, and cues;
- two to four valid Crescendo candidates or an approved short-song exception,
  including selection inputs, conflicts, preview/performance boundaries, tier/
  effect references, Easy recovery evaluation, and one-normal-use proof; and
- solo formations/locations, Vanguard clamp opportunities, Warden cadence/pulse,
  Herald opportunity references, suppression/recovery, and cue mappings.

Global Hype gain/thresholds, ability power, group caps, cooldowns, acolyte fixed
contribution, and typed effect behavior remain Ability/balance catalogs. Packages
do not grant acolytes charts, human judgments, performance credit, risk rewards,
Resolve breaks, revival help, or player substitutions.

Band Call invitation/performance and Crescendo preview/performance are mutually
exclusive; the required Crescendo has priority. Dynamic recovery remains a live
request matched to a fair candidate, never a fixed authored revival time.

### Roles, multiplayer, and configuration

Each package declares its supported subset of stable global role identities,
with explicit playable/unavailable state, chart/audio/equivalent mappings, and
compatibility for solo and every allowed two-to-six-human roster. Validation
includes unrestricted duplicate and all-same-role cases. It never infers a role
from item names or assumes drums, vocals, guitar, and bass must exist.

Builds, Items, loadouts, three full spec presets, matchmaking, Ready, active
roster, disconnect, AFK, rejoin, and Player Data remain runtime/system state.
The package declares only stable compatible role/hook/capability references and
cannot hide modifiers, composition locks, substitutions, or player-specific data.

### Audio, UI, accessibility, and teaching data

The runtime audio map includes:

- one approved full-mix reference and, for every playable role, its backing plus
  authentic controllable layer(s) or approved equivalent;
- asset/layer identities, alignment, duration, channel layout, neutral level,
  response capabilities, routes, preload/cache/fallback class, and exact revision;
- neutral and maximum-response reconstruction evidence;
- role-appropriate response definitions for judgment, hold, suspension, down,
  return, pause/resume, and rejoin;
- versioned buses/routes and dynamic-range, loudness, peak/headroom, ducking,
  device, mono, spatial, and concurrency/eviction constraints; and
- stable semantic audio definitions and lifecycle/cancel/idempotency behavior.

Authored encounter/world mappings identify cue family, stage, priority,
source/target/direction, timing, cancellation, coalescing, ducking, caption/source
metadata, alternative modalities, and bounded optional haptic request for attacks,
positions, graph changes, group/recovery windows, story/function transitions,
landmarks, outcomes, dialogue/subtitles, crowd, ambience, and restoration state.

Practice packages provide real safe content for the six approved onboarding
modules and starter roles, plus calibration and first-boss teaching trigger/cue
coverage. They contain content and stable cue identities, never hidden tutorial,
UI layout, input, settings, or progression logic.

Every required semantic cue survives export/load for all applicable roles,
difficulties, rosters, and device/accessibility profiles. Critical meaning has
visual/caption/optional-haptic alternatives and does not rely on color, stereo,
sub-bass, extreme treble, loudness alone, or an image containing the only text.

### Reconciled publication and review gates

Deterministic publication validation now includes:

- schema, identity, reference, lineage, global-order, and compatibility checks;
- chart legality, physical feasibility, equal normalization, crossing holds,
  Activity Maps, and re-entry/recovery coverage;
- five-function lifecycle, Resolve/finishing, candidate reproducibility,
  reservations/conflicts, attacks, graph/travel, scaling, and critical validity;
- role availability and every allowed solo/two-to-six-human roster, including
  duplicates and dropout-sensitive cases;
- Signature, Band Call, Crescendo, solo-support, and semantic-cue coverage;
- neutral/maximum audio reconstruction, phase/alignment, loudness/headroom/mono,
  concurrency, cache/preload/degradation, captions, and alternatives;
- phone-scale warning/target/geometry/countdown overlap and control/attention
  conflict evidence; and
- Roblox-export equivalence for every required identity, mapping, timing,
  fallback, alternative, and integrity fact.

Human review retains musical, encounter, phone Roblox, device/input,
accessibility, audio A/B, solo/co-op, and independent-review evidence. The first
three boss packages cannot publish until the complete reconciled matrix passes.

### Reconciliation outcome

The final design audit confirms:

- every registered downstream need has one authoritative representation or an
  explicit exact system-catalog reference;
- no private competing chart, encounter, audio, gameplay, or player-data schema
  is permitted;
- every runtime field has an owner, consumer, fallback, compatibility impact,
  and validator/review path, so no intentionally orphaned package field remains;
- gameplay and persistence semantics remain with their owning systems; and
- the platform-neutral package plus equivalence report is the only content
  authority consumed by Roblox runtime.

## 15. Approval and change control

The owner interview resolved CA-01 through CA-24 on 2026-08-21. The mandatory
reconciliation against specifications 2 through 12 completed on 2026-09-02.
This document is the canonical reconciled Content Authoring design specification.

A later design may refine a consumer requirement without silently changing this
contract. New authoring requirements must identify their owner, representation,
consumer, compatibility impact, and validation path before publication. A
material change to system ownership, human approval authority, arrangement
authenticity, platform-neutral authority, immutable revision semantics, or the
reconciled publication gates requires an explicit amendment citing the
superseded decision.
