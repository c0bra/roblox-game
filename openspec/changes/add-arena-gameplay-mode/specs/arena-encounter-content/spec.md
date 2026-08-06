## ADDED Requirements

### Requirement: Arena encounter data is versioned and selection-specific
The system SHALL load Arena encounter documents with an explicit schema version, level identifier, supported instrument, supported difficulty, duration, beat timeline, positions, performance phrases, reposition windows, boss events, Boss Resolve threshold, and phase events.

#### Scenario: Load a supported Arena encounter
- **WHEN** a player starts Arena for a level and difficulty with a valid encounter document
- **THEN** the loader returns typed, seconds-based Arena content whose level identifier, instrument, difficulty, and duration match the selected run

#### Scenario: Selected instrument is unsupported
- **WHEN** the selected level and difficulty have an Arena document but the selected instrument does not match it
- **THEN** Arena remains unavailable for that selection and offers an explicit switch to the supported demo combination without starting audio

#### Scenario: Classic chart is loaded
- **WHEN** a player starts Classic Highway for the same level
- **THEN** the Classic loader does not require, merge, or mutate Arena encounter data

### Requirement: Encounter data is parsed at the boundary
The Arena loader SHALL parse untrusted encounter JSON through a strict schema before the controller receives it and SHALL reject unknown versions or invalid structures with a typed load failure.

#### Scenario: Unknown schema version
- **WHEN** an encounter document declares a version the runtime does not support
- **THEN** the loader rejects the document before gameplay begins and presents the Arena recovery flow

#### Scenario: Invalid semantic action
- **WHEN** a phrase contains an action other than the supported semantic action set
- **THEN** the loader reports the invalid action and does not begin a partial encounter

### Requirement: Encounter references, timelines, and cue collisions are valid
The Arena content validator SHALL require finite non-negative seconds, sorted event times, unique identifiers, valid position references, phrase preview at least two authored beats before execution, telegraph-before-impact ordering, movement deadlines that allow travel before impact, recovery-after-impact ordering, events bounded by encounter duration, and no easy-mode collision that introduces a new phrase preview during a critical attack cue.

#### Scenario: Boss attack references an unknown position
- **WHEN** a boss event targets a position identifier not declared by the encounter
- **THEN** validation fails with the boss event and unknown position identified

#### Scenario: Phrase preview begins after execution
- **WHEN** a phrase's preview start is later than its execution start
- **THEN** validation fails before the encounter is exposed to gameplay code

#### Scenario: Valid coincident beat events
- **WHEN** a downbeat, phrase step, and boss impact intentionally share the same valid timestamp
- **THEN** validation accepts them and preserves their deterministic event types for ordered runtime resolution

#### Scenario: Reposition deadline follows impact
- **WHEN** a reposition window would remain valid after its associated boss impact
- **THEN** validation fails before gameplay begins because late movement cannot retroactively avoid resolved damage

#### Scenario: Easy phrase preview collides with a critical telegraph
- **WHEN** a new performance-phrase preview begins during an authored critical telegraph interval on the prototype difficulty
- **THEN** validation rejects the conflicting cues and identifies both events

### Requirement: Combat tuning is data-driven
The Arena encounter document SHALL configure position profiles, base and position-specific perform steps, reposition decision beats and valid destinations, attack damage, safe and affected positions, grade effects, Boss Resolve threshold, and exposure values without requiring source-code edits for encounter-specific tuning.

#### Scenario: Tune Spotlight exposure
- **WHEN** an authored encounter increases Spotlight's exposure multiplier and the document remains valid
- **THEN** the next run uses the new multiplier without changing the Arena controller implementation

#### Scenario: Author a different safe position
- **WHEN** an attack document changes its safe-position references
- **THEN** the telegraph and impact resolution use the updated valid positions from the document

### Requirement: The prototype includes a complete authored encounter
The change SHALL include a brief nonfatal rehearsal followed by one reviewed 30-to-45-second scored Arena encounter segment for one explicit song and instrument on Easy difficulty containing an intro, ambient beats, all three positions, at least one reposition choice, at least one static playable phrase, two visually distinct boss attacks, a boss opening, a climax, and ward-loss, failed-resolve, and victory paths.

#### Scenario: First-time rehearsal
- **WHEN** a player begins the Arena prototype without having completed its rehearsal in the current session
- **THEN** the encounter demonstrates the three controls, one static phrase, and one reposition choice without allowing the rehearsal to cause a scored defeat

#### Scenario: Play the prototype segment successfully
- **WHEN** a player completes the authored segment without losing all ward and satisfies the Boss Resolve threshold at the final cadence
- **THEN** every required event category occurs and the run reaches its authored climax and victory resolution

#### Scenario: Fail the prototype segment
- **WHEN** a player remains vulnerable through enough authored impacts to lose all ward
- **THEN** the encounter reaches a deterministic defeat resolution before its victory event

#### Scenario: Survive without enough boss damage
- **WHEN** the player retains ward through the final cadence but misses the configured Boss Resolve threshold
- **THEN** the encounter reaches the deterministic failed-seal defeat resolution instead of awarding survival victory

### Requirement: A compact deterministic QA encounter exercises the runtime
The change SHALL provide a short QA encounter whose event order, timing, outcomes, and expected checkpoints are deterministic.

#### Scenario: Run Arena with QA enabled
- **WHEN** the application starts Arena with the QA parameter
- **THEN** it loads the compact QA encounter and exposes every semantic action, position transition, boss attack type, opening, victory, and result metric within the bounded QA duration

#### Scenario: Repeat the QA encounter with identical inputs
- **WHEN** the same timestamped inputs are applied to two QA runs
- **THEN** both runs produce the same judgments, state transitions, health, score, exposure, and outcome

### Requirement: Arena content failures are actionable
The loader SHALL distinguish missing, network, parse, version, and semantic-validation failures and SHALL surface a concise user recovery message plus diagnostic context for development.

#### Scenario: Encounter file is missing
- **WHEN** the selected level has no Arena encounter document
- **THEN** the setup or loading flow identifies Arena as unavailable for that selection and offers Classic without starting audio

#### Scenario: Encounter validation fails in development
- **WHEN** a document fails semantic validation during local development or tests
- **THEN** the diagnostic identifies the failing field or event while the user-facing surface avoids exposing raw implementation details
