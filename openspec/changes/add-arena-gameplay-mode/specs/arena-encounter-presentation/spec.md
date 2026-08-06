## ADDED Requirements

### Requirement: Arena uses a fixed-camera 2.5D battle composition
Arena SHALL render the boss, player, three tactical positions, cover, lighting, and world effects in one Babylon.js scene while HUD and controls remain readable interface layers.

#### Scenario: Portrait gameplay composition
- **WHEN** Arena runs at the 375 px portrait viewport
- **THEN** the boss remains readable in the upper-center action area, the player remains visible in the lower-center action area, and all three positions and boss-to-player telegraphs remain inside the gameplay composition

#### Scenario: Desktop gameplay composition
- **WHEN** Arena runs at the 1280 px viewport
- **THEN** the core portrait action geometry retains the same relative framing and additional width is used for atmosphere rather than required gameplay targets

### Requirement: The prototype includes an authored player character
The Arena asset set SHALL include one original, rigged Rift Performer with a phone-readable silhouette, coherent materials, an instrument or energy-performance prop, runtime GLB export, preview, and required animation clips.

#### Scenario: Load the player asset
- **WHEN** the Arena scene imports the approved player GLB
- **THEN** it provides named clips for intro or ready, idle, perform, advance dash, retreat dash, brace or ward, hit or stagger, victory, and defeat with documented loop behavior

#### Scenario: Player changes position
- **WHEN** an advance or retreat action resolves
- **THEN** the character travels visibly between the corresponding anchors using the matching clip and faces the boss at arrival

### Requirement: The prototype includes a licensed presentation-ready first boss
The Arena asset set SHALL acquire and adapt one externally sourced, already rigged and animated first boss with a clear phone-scale silhouette, a license compatible with the intended browser delivery and repository visibility, runtime GLB export, preview, browser-safe materials, and animations that visually distinguish each playable combat state. Existing repository boss models and GLBs SHALL NOT be used as the Arena boss.

#### Scenario: Boss animation inventory
- **WHEN** the approved boss asset is inspected
- **THEN** it provides named clips for intro, idle, at least two distinct telegraphs, two corresponding attacks, hit reaction, stagger or opening, phase transition, and defeat

#### Scenario: Marketplace listing claims animation readiness
- **WHEN** a candidate is described as rigged or animation ready but its delivered package does not contain an explicit playable clip inventory
- **THEN** the candidate is rejected rather than accepted on the strength of marketplace copy or preview media

#### Scenario: Commercial asset cannot be redistributed through the repository
- **WHEN** the selected license permits incorporation into the web game but prohibits public redistribution of the source package
- **THEN** the protected package remains in license-controlled storage and only a license-permitted optimized runtime derivative, manifest record, entitlement reference, and reproducible adaptation instructions enter the project workflow

#### Scenario: Telegraphs are distinguishable without effects
- **WHEN** particles and post-processing are disabled in the presentation showcase
- **THEN** each boss attack remains identifiable from pose, motion, target geometry, and timing

### Requirement: The arena environment communicates positions and danger
The Arena asset set SHALL include a cohesive stage or ruined-threshold kit with three physically distinct anchors or cover treatments, supporting props, backdrop, shadow treatment, and lighting that preserve gameplay readability.

#### Scenario: Compare position silhouettes
- **WHEN** Shelter, Midline, and Spotlight are shown without colored VFX
- **THEN** a player can distinguish them by spatial location, cover silhouette, shape, and label

#### Scenario: Position becomes unsafe
- **WHEN** a boss event targets a position
- **THEN** the anchor and surrounding world present a readable danger state without making the player, boss pose, or active phrase illegible

### Requirement: VFX have complete semantic coverage
Arena SHALL provide purposeful visual effects for beat, downbeat, phrase preview, active phrase step, safe and dangerous anchors, dash, successful perform, boss target path, charge, impact, ward damage, boss damage, opening, climax, victory, and defeat.

#### Scenario: Successful perform contact
- **WHEN** a successful perform action reaches its contact time
- **THEN** the player action, lightline or projectile, boss reaction, impact effect, HUD update, and sound align to the same authored moment

#### Scenario: Boss attack charges
- **WHEN** an attack enters its telegraph phase
- **THEN** charge and target-path effects grow according to authored time and terminate or transition exactly at impact

#### Scenario: Performance phrase is displayed
- **WHEN** a phrase is in preview or execution
- **THEN** its complete symbol group remains stationary, only current and next steps receive timing emphasis, and the presentation does not create a scrolling mini-lane

### Requirement: Timing and intent are redundantly communicated
Arena SHALL communicate required timing and attack intent through at least two of shape, position, motion, text or icon, and audio; color alone SHALL NOT encode a required response.

#### Scenario: Color-vision-independent attack
- **WHEN** the scene is viewed without hue distinctions
- **THEN** the player can still identify affected positions, the required semantic response, and the impact moment from geometry, symbols, contrast, and motion or audio

#### Scenario: Audio is muted
- **WHEN** gameplay audio cues are unavailable while visual rendering remains active
- **THEN** the active beat, phrase step, boss target, and impact timing remain visually perceivable

### Requirement: Sound effects have complete semantic coverage
The Arena vertical slice SHALL include the approved P0 and P1 sound families for count-in, phrase reveal, immediate input acknowledgement, Good/Great/Perfect contact, flub, reposition selection and travel, shared and position-specific arrival, two distinct boss telegraph/charge/impact identities, successful evade, ward hit/crack/break, boss hit, stagger/opening, phrase completion, Boss Resolve gain, arena and boss introduction, sparse authored downbeat accents, phase transition, final Resolve success/failure, boss defeat, encounter results, and required UI actions. Optional attack vocals may be omitted when the non-vocal telegraphs are stronger or clearer.

#### Scenario: Perform feedback has two authored moments
- **WHEN** the player presses Perform within a valid judgment window
- **THEN** a short input acknowledgement occurs immediately and the grade-appropriate contact sound occurs at the same authoritative contact time as the player effect and boss reaction without double-triggering

#### Scenario: Boss attacks are recognizable by sound
- **WHEN** the sweep-style and burst-style attacks are presented without their visuals
- **THEN** first-time listeners can distinguish their warning onset, charge behavior, and impact identity without depending on stereo position alone

#### Scenario: Combat state is audible without becoming a metronome
- **WHEN** a player repositions, evades, takes ward damage, opens the boss, gains meaningful Resolve, or reaches an encounter outcome
- **THEN** the corresponding state has concise audio feedback while ordinary beats remain primarily carried by the song

#### Scenario: Repeated effects retain meaning
- **WHEN** frequently repeated input, contact, flub, boss-hit, or ward-hit sounds select among variants
- **THEN** the variants reduce repetition without changing the perceived action, grade, or severity

### Requirement: Generated audio is production-ready and traceable
Arena sound-effect masters SHALL be retained as 48 kHz 24-bit WAV with generation provenance, edit notes, semantic ID, variation group, channel layout, sync or loop markers, license/original-work status, runtime encode, checksum, and mix intent recorded in an audio manifest. Runtime Arena sound effects SHALL target no more than 1.5 MB transferred excluding the song and stems.

#### Scenario: Import a generated sound family
- **WHEN** a generated sound effect is approved for the slice
- **THEN** accidental leading silence is removed, required loop/release boundaries are verified, point-source channel layout is appropriate, peaks do not exceed -1 dBTP before runtime gain, and the manifest traces the master to its runtime encode

#### Scenario: Mix against the selected song
- **WHEN** the P0 and P1 inventory is reviewed on the named phone and desktop audio outputs
- **THEN** boss warnings, immediate input feedback, and gameplay consequences remain intelligible without masking vocals, the selected instrument, or the next timing transient

### Requirement: Animation and effects stay synchronized to song time
Arena SHALL synchronize skeletal clips, camera cues, particle lifetimes, material effects, and HUD timing indicators to encounter events derived from the audio clock.

#### Scenario: Resume during a telegraph
- **WHEN** a paused encounter resumes partway through an authored telegraph
- **THEN** the boss pose, target path, effects, and countdown reflect the resumed song time rather than restarting the attack from the beginning

#### Scenario: Encounter is replayed
- **WHEN** a player starts a replay after results
- **THEN** all actors, clips, particles, camera state, and materials return to their authored initial state before the countdown

### Requirement: Reduced motion preserves essential gameplay information
Arena SHALL provide a reduced-motion presentation that removes camera shake, idle sway, repeated scale pulses, large particle bursts, and nonessential trails while preserving timing and target information.

#### Scenario: Reduced-motion boss attack
- **WHEN** the operating system requests reduced motion and a boss attack telegraphs
- **THEN** static target geometry, direct contrast changes, the semantic response icon, and linear time-to-impact information remain available without camera shake or repeated pulsing

#### Scenario: Reduced-motion position change
- **WHEN** the player advances or retreats under reduced motion
- **THEN** the system shows a concise readable transition and updated anchor state without a sweeping camera move or long trail

### Requirement: Assets are reproducible, attributable, and license-safe
Every Arena runtime model, texture, concept source, animation, VFX atlas, and audio effect SHALL have a documented source or entitlement reference, license status, permitted storage and distribution scope, adaptation/export path, and runtime manifest entry.

#### Scenario: Inspect a runtime GLB
- **WHEN** a developer traces the player or boss runtime GLB from its manifest
- **THEN** the manifest identifies its Blender source or protected-package entitlement, export settings, axes and scale, clip names, triangle count, texture sizes, output checksum, preview, license, repository/deployment permission, and original or derivative status

#### Scenario: Generated concept art informs an asset
- **WHEN** generated concept art is used during asset development
- **THEN** its provenance is recorded and the final runtime asset is stored as an original project artifact with editable source files rather than an undocumented downloaded binary

### Requirement: Arena meets loading and performance budgets
The Arena vertical slice SHALL load Arena-specific assets on demand, SHALL transfer no more than 12 MB of Arena-specific runtime assets excluding song audio, SHALL avoid textures above 1024 px unless documented, and SHALL meet the defined desktop and mobile frame targets on named QA hardware.

#### Scenario: Start Classic without selecting Arena
- **WHEN** a player starts Classic Highway in a fresh session
- **THEN** the system does not fetch the Arena player, boss-animation, environment, or VFX asset bundle

#### Scenario: Profile the mobile QA viewport
- **WHEN** the complete Arena slice is exercised at 375 px on the named mobile QA hardware
- **THEN** it sustains the 30 FPS floor through both attacks and the climax without removing essential telegraphs or input feedback

#### Scenario: Profile desktop
- **WHEN** the complete Arena slice is exercised on the named desktop QA hardware
- **THEN** it targets 60 FPS and records asset transfer, draw calls, frame timing, and any accepted performance debt

### Requirement: Presentation failures are recoverable
Arena SHALL replace blank or partially initialized presentation states with a static authored fallback and explicit recovery actions.

#### Scenario: Required model import fails
- **WHEN** the player or boss model cannot be imported
- **THEN** the system stops the Arena countdown, displays the fallback poster or silhouette, and offers retry and Classic actions

#### Scenario: WebGL context is unavailable
- **WHEN** Babylon.js cannot create or restore the Arena scene
- **THEN** the system keeps setup and recovery controls usable and does not claim that the Arena encounter started

### Requirement: The slice passes visual and attention validation
Arena SHALL pass an early graybox attention gate before final player/environment asset production, then pass responsive visual QA, interaction-state QA, motion recording review, reduced-motion review, and a final first-time-player attention test before it can become the default mode.

#### Scenario: Graybox attention gate
- **WHEN** three first-time players complete the graybox rehearsal and encounter
- **THEN** the team records phrase accuracy, attack recall, position-choice understanding, player-action recall, and whether attention returns to the battle between phrase glances before authorizing final player/environment production

#### Scenario: Visual QA run
- **WHEN** the completed slice is reviewed at 375, 768, and 1280 px
- **THEN** every required animation, VFX, cue, control state, pause state, result state, and fallback state is captured without clipping, unreadable layering, missing feedback, or flat placeholder presentation

#### Scenario: First-time-player attention test
- **WHEN** five first-time players each complete one easy Arena run
- **THEN** at least four can identify both boss attacks, explain the three-position tradeoff, describe at least one player-character action, achieve at least 60 percent timing accuracy, and satisfy the observer rubric that attention returns to the boss and player between brief phrase glances rather than remaining fixed on the prompt area
