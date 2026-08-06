## ADDED Requirements

### Requirement: Music is the authoritative combat clock
Arena SHALL derive beat state, phrase timing, telegraph timing, impact resolution, animation synchronization, and timing judgments from the active Web Audio run's seconds-based clock.

#### Scenario: Render frames arrive unevenly
- **WHEN** animation frames are delayed or skipped during an active encounter
- **THEN** the next frame derives the correct encounter state from audio time without shifting authored beat, phrase, or impact times

#### Scenario: Pause and resume
- **WHEN** the player pauses or the page is automatically paused after becoming hidden
- **THEN** audio, encounter resolution, character animation, boss animation, and timed effects stop together and resume from the same song position after explicit resume

### Requirement: Arena has no permanent or miniature note highway
Arena SHALL communicate rhythm through an ambient beat pulse, boss/world telegraphs, a static complete phrase preview, and a stationary time-to-hit focus without rendering a continuous or compact scrolling lane of upcoming notes.

#### Scenario: Normal combat between phrases
- **WHEN** no authored phrase is in its preview or execution window
- **THEN** the battle view shows the boss, player, positions, and ambient beat state without a stream of note symbols approaching a strike line

#### Scenario: Phrase execution begins
- **WHEN** an authored phrase reaches its execution window
- **THEN** the complete phrase remains stationary near the boss-player action axis, the current and next perform steps are emphasized according to authored beat times, and no symbols travel toward a strike line

### Requirement: Inputs have stable combat meanings
Arena SHALL use stable semantic actions for `retreat`, `perform`, and `advance`; it SHALL NOT assign arbitrary keys to unrelated prompts during normal combat.

#### Scenario: Same action appears in different phrases
- **WHEN** a perform step appears in two different authored phrases
- **THEN** it uses the same visible label or glyph, keyboard binding, touch control, and combat meaning in both phrases

#### Scenario: Player presses an action
- **WHEN** a player presses retreat, perform, or advance during an active run
- **THEN** the corresponding control immediately acknowledges the input and the player presentation begins the matching readable response

#### Scenario: Movement action has no reachable anchor
- **WHEN** retreat or advance cannot move the player toward a real adjacent anchor
- **THEN** the control communicates the boundary and the input is not repurposed as an arbitrary phrase symbol

### Requirement: Timing judgments affect combat outcomes
Arena SHALL grade authored action inputs using configurable Perfect, Great, Good, and Miss windows and SHALL apply grade-based score and combat effect.

#### Scenario: Perform input inside the Perfect window
- **WHEN** a perform input occurs within the configured Perfect offset of its authored step
- **THEN** the system records a Perfect grade, applies the configured full combat effect, and emits synchronized hit, animation, VFX, and audio feedback

#### Scenario: Early successful perform input
- **WHEN** a successful perform input arrives before its authored beat with enough remaining time for the configured windup
- **THEN** the system acknowledges the input immediately and may schedule the lightline or projectile contact for the authored beat

#### Scenario: Late successful perform input
- **WHEN** a successful perform input arrives after its authored beat but remains inside the Good window
- **THEN** the system acknowledges and grades it immediately, uses immediate or compressed contact feedback, and does not present an effect as though it landed in the past

#### Scenario: Input outside the Good window
- **WHEN** an action input does not match an eligible authored step inside the Good window
- **THEN** the system records a miss or flub, withholds the successful step's score and enhanced combat effect, and gives immediate non-color-only failure feedback

### Requirement: Phrases are static, previewed, and bounded
Arena SHALL reveal the complete ordered performance phrase at least two authored beats before execution, SHALL keep its symbols stationary, SHALL limit the prototype phrase to three through five authored perform steps with optional position-specific bonus steps, and SHALL clear the phrase during recovery or spectacle time.

#### Scenario: Phrase enters preview
- **WHEN** song time reaches a phrase's authored preview start
- **THEN** the system reveals the complete ordered perform sequence without starting its judgments and uses a stationary countdown or contraction to identify when execution will begin

#### Scenario: Phrase completes
- **WHEN** the final phrase step resolves
- **THEN** the phrase prompt clears or transitions to its payoff state and the encounter provides its authored recovery, opening, or spectacle interval

#### Scenario: Spotlight adds phrase demand
- **WHEN** an authored phrase provides optional Spotlight bonus steps and the player begins it at Spotlight
- **THEN** those extra perform steps are included without changing the meaning or ordering of the shared base phrase

### Requirement: Three tactical positions change risk and reward
Arena SHALL provide Shelter, Midline, and Spotlight positions with visibly distinct cover, phrase demand, attack exposure, combat multiplier, song-influence multiplier, and exposure gain configured by encounter data.

#### Scenario: Begin a run
- **WHEN** the Arena countdown completes
- **THEN** the player begins at Midline and can identify Shelter, Midline, and Spotlight by position, shape, and label rather than color alone

#### Scenario: Perform accurately at Spotlight
- **WHEN** a player completes an eligible perform step at Spotlight
- **THEN** the system applies Spotlight's configured higher combat and exposure effect compared with the same grade at Shelter

#### Scenario: Boss changes safe positions
- **WHEN** a boss event targets one or more positions
- **THEN** the system visibly marks affected and safe positions before impact and resolves damage according to the player's position at the authored impact time

#### Scenario: More than one destination is safe
- **WHEN** a reposition window exposes multiple safe destinations with different risk and reward profiles
- **THEN** the player chooses whether to retreat, hold, or advance and the system does not prescribe one movement key as the only valid tactical response

### Requirement: Movement is beat-timed, pre-impact, and physically visible
Arena SHALL expose reposition windows separately from performance phrases, SHALL move the player between adjacent tactical positions using readable advance or retreat travel, and SHALL close each valid movement window early enough for travel to complete before the associated boss impact.

#### Scenario: Successful advance
- **WHEN** the player presses advance within the valid window and a closer position exists
- **THEN** the character visibly dashes to the next closer anchor and arrives in time for the authored resolution

#### Scenario: Movement begins after impact
- **WHEN** a movement input occurs after the associated boss impact has resolved
- **THEN** the movement cannot reverse that damage or retroactively make the previous position safe

#### Scenario: Movement beyond the arena boundary
- **WHEN** the player presses retreat at Shelter or advance at Spotlight
- **THEN** the character remains at the current anchor and the control communicates that no farther position exists

### Requirement: Boss attacks use prepare, impact, and recovery phases
Every playable boss attack SHALL have an authored telegraph phase, one audio-clock impact time, and a recovery or opening phase with a visible relationship between the boss action and affected positions.

#### Scenario: Targeted attack telegraph
- **WHEN** a targeted boss attack enters its telegraph phase
- **THEN** the boss pose, attack path, target geometry, and audio cue communicate the threat and affected position before impact

#### Scenario: Player occupies an unsafe position at impact
- **WHEN** audio time reaches the attack's impact and the player remains in an affected position without satisfying the configured defense response
- **THEN** the system applies the configured ward damage and synchronized boss, player, VFX, audio, and HUD feedback exactly once

#### Scenario: Player avoids the attack
- **WHEN** audio time reaches impact and the player occupies a safe position or completed the configured response
- **THEN** the system records the avoidance or block and enters the authored recovery or boss-opening state without player damage

### Requirement: Performance affects the selected instrument mix
Arena SHALL keep the selected instrument on its own Web Audio channel and SHALL use accurate actions and misses to create audible performance consequences while other stems continue.

#### Scenario: Accurate perform action
- **WHEN** a player earns an eligible successful grade
- **THEN** the selected instrument is present or accented in the mix and the action sound aligns with the visual contact time

#### Scenario: Missed perform action
- **WHEN** a player misses a perform step
- **THEN** the selected instrument receives the configured brief duck or flub treatment without stopping the shared backing song

### Requirement: Boss resolve gates the authored ending
Arena SHALL apply grade- and position-adjusted damage to a Boss Resolve objective, SHALL evaluate the configured resolve threshold at the authored final cadence, and SHALL NOT end the song early when the threshold is reached ahead of that cadence.

#### Scenario: Resolve threshold is met before the final cadence
- **WHEN** successful performance reduces Boss Resolve to its threshold before the authored ending
- **THEN** the song and encounter choreography continue, excess performance contributes score or exposure, and victory waits for the final cadence

#### Scenario: Resolve threshold is missed
- **WHEN** the final cadence arrives while Boss Resolve remains above the configured victory threshold
- **THEN** the encounter produces the failed-seal defeat path even if the player's ward remains above zero

### Requirement: Arena results report combat and attention-relevant metrics
Arena SHALL end in victory or defeat and SHALL report score, timing accuracy, best phrase or streak, Boss Resolve progress, ward damage, and exposure earned.

#### Scenario: Encounter reaches its victory event
- **WHEN** the player survives through the authored final cadence and satisfies the Boss Resolve threshold
- **THEN** the system stops active judgments, plays the victory presentation, and shows Arena result metrics without modifying a loot inventory

#### Scenario: Player ward reaches zero
- **WHEN** resolved boss damage reduces the player's ward to zero
- **THEN** the system stops active judgments, plays the defeat presentation, and offers replay, mode change, and selection recovery actions

### Requirement: Arena supports touch and keyboard play
Arena SHALL expose large, labelled touch controls and complete keyboard bindings for retreat, perform, advance, pause, resume, replay, and exit.

#### Scenario: Portrait touch play
- **WHEN** Arena is played at the 375 px portrait viewport
- **THEN** all three combat controls remain at least 48 px, separated by at least 8 px, clear of safe-area insets, and operable without a precision gesture

#### Scenario: Keyboard-only run
- **WHEN** a player uses only the keyboard
- **THEN** the player can select Arena, complete or fail an encounter, pause, resume, replay, and return to setup with visible focus states
