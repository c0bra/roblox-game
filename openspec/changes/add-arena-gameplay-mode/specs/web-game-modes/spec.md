## ADDED Requirements

### Requirement: Classic Highway remains playable
The web game SHALL preserve Classic Highway as an independently playable mode while Arena is developed and evaluated.

#### Scenario: Existing route without a mode choice
- **WHEN** a player opens the web game without selecting a gameplay mode
- **THEN** the system selects Classic Highway and presents its existing playable flow

#### Scenario: Classic run after Arena is installed
- **WHEN** a player explicitly selects Classic Highway and starts a level
- **THEN** the system loads the existing note chart, highway renderer, controls, and result calculation without requiring Arena encounter data or Arena assets

### Requirement: Players can select a gameplay mode
The web game SHALL present Classic Highway and Arena as distinct gameplay choices and SHALL preserve the current song, instrument, and difficulty selection when the mode changes.

#### Scenario: Select Arena from the setup screen
- **WHEN** a player selects Arena after choosing a song, instrument, and difficulty
- **THEN** the setup screen retains those selections and starts the matching Arena encounter when requested

#### Scenario: Current selection has no Arena encounter
- **WHEN** a player views Arena while the selected song, instrument, and difficulty are not the supported prototype combination
- **THEN** Arena is identified as unavailable for that combination, the current selection is preserved, and the player receives an explicit action to switch to the supported demo selection rather than a silent substitution

#### Scenario: Open a direct Arena QA route
- **WHEN** the application is opened with valid Arena and QA query parameters
- **THEN** the system selects Arena and loads the deterministic Arena QA encounter

### Requirement: Only one mode owns runtime resources
The web game SHALL allow only the active mode to own gameplay listeners, animation loops, audio runs, canvases, Babylon resources, and input state.

#### Scenario: Change modes before starting
- **WHEN** a player changes from Classic to Arena on the setup screen
- **THEN** Classic gameplay resources are not started and Arena becomes the only controller eligible to begin a run

#### Scenario: Leave an Arena run
- **WHEN** a player exits an Arena run and returns to setup
- **THEN** the system stops the audio run and disposes Arena animation frames, listeners, scene resources, effects, and pressed-input state before another mode starts

### Requirement: Arena failures have a Classic recovery path
The web game SHALL provide an explicit recovery path when Arena cannot load required encounter or WebGL resources.

#### Scenario: Arena model or WebGL initialization fails
- **WHEN** Arena cannot initialize its scene or required runtime asset
- **THEN** the system shows a static Arena fallback presentation with actions to retry Arena or launch the same selection in Classic Highway

#### Scenario: Retry succeeds
- **WHEN** a player chooses retry after a transient Arena load failure and required resources then load
- **THEN** the system begins a fresh synchronized Arena countdown without stale scene, input, or audio state
