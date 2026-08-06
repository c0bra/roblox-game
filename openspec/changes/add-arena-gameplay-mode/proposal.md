## Why

The existing highway mode makes accurate play depend on watching a continuous note lane, leaving little attention for the boss, player character, or arena spectacle. An additive Arena V2 prototype will test whether beat-synchronized combat, fixed tactical positions, and world-anchored cues can preserve rhythmic precision while making the battle itself the focus.

## What Changes

- Add an Arena gameplay mode beside the existing highway mode in `roblox/web`; keep the current mode playable as the comparison baseline and share the existing song selection, audio clock, level loading, HUD language, and results flow where appropriate.
- Introduce a phrase-based rhythm-combat loop in which the music provides the global clock, boss animations communicate the threat, and a static three-to-five-step phrase preview plus stationary time-to-hit focus communicates timing without a permanent or miniature scrolling note highway.
- Add three fixed player positions with clear safety, rhythmic-complexity, damage, song-influence, and reward tradeoffs. Reposition windows show safe and dangerous anchors, let the player choose retreat, hold, or advance on authored movement beats, and remain separate from performance phrases so position is a tactical choice rather than a scripted QTE path.
- Add authored Arena encounter data for beat pulses, static phrase windows, reposition choices, boss telegraphs, impacts, safe positions, openings, boss-resolve victory thresholds, and climax events without changing the existing highway charts.
- Build the first Arena vertical slice for one explicit song and instrument on Easy difficulty as a fixed-camera 2.5D scene using the existing Babylon.js/Web Audio stack, with a brief nonfatal rehearsal, a rigged player performer, a licensed or CC0 animated first boss sourced outside the current repository models, a small arena/cover kit, semantic combat controls, camera choreography, animation, lighting, particles, and beat-synchronized visual and audio feedback.
- Establish an asset-acquisition and production pipeline covering marketplace/CC0 research, browser-delivery license review, purchase or download provenance, concept references, modeling or adaptation, retopology where needed, UVs/materials, rigging, animation clips, GLB export, texture/VFX optimization, attribution, and browser/mobile budgets. Commercial source packages remain in license-controlled storage when they cannot be redistributed through the repository.
- Define a prioritized Arena sound-effect inventory and generation contract covering immediate input feedback, successful performance contact, misses, repositioning, two sonically distinct boss attacks, ward and Boss Resolve state, openings, outcomes, UI feedback, optional ambience, variation counts, synchronization, runtime formats, provenance, and mix budgets.
- Provide reduced-motion, color-independent telegraphs, keyboard/touch controls, pause/recovery behavior, rendering fallbacks, deterministic QA encounters, and attention-focused playtest criteria.
- Defer free-roaming movement, production multiplayer networking, multiple complete bosses, progression/loot inventory, and replacement of the classic highway mode until the Arena loop is validated.

## Capabilities

### New Capabilities

- `web-game-modes`: Select and launch Classic Highway or Arena gameplay while both modes share validated levels and remain independently playable.
- `arena-rhythm-combat`: Run the fixed-position, beat-timed combat loop, including semantic actions, phrase cues, boss telegraphs, timing judgments, risk/reward positions, health, scoring, and encounter outcomes.
- `arena-encounter-content`: Define, validate, load, and deterministically exercise Arena-specific beat, phrase, position, boss-attack, opening, and phase data without altering classic note charts.
- `arena-encounter-presentation`: Present the Arena vertical slice with an authored player character, boss, environment, animation set, camera, lighting, VFX, audio-reactive feedback, accessibility variants, and performance-safe fallbacks.

### Modified Capabilities

<!-- No existing OpenSpec capabilities are modified; the repository has no baseline capability specs yet. -->

## Impact

- Affects `roblox/web` application routing, controllers, rendering, input, encounter data, UI templates, styles, tests, QA fixtures, public assets, and the web `DESIGN.md` contract.
- Reuses the existing strict TypeScript, Vite, Babylon.js, Canvas/WebGL, Web Audio, stem playback, song catalog, timing judgments, and build pipeline; any new runtime dependency requires an explicit bundle-cost justification.
- Adds source and exported art assets under the existing Roblox/web asset areas, with reproducible export notes and optimized runtime copies rather than undocumented one-off binaries.
- Requires mobile and desktop performance budgets, deterministic browser QA, motion/reduced-motion recordings, visual-quality review, an early graybox attention gate before final asset production, and final user testing that measures whether players notice boss and character behavior while remaining on beat.
- May add a third-party boss asset under CC0 or a commercial marketplace license. The selected asset must permit the intended web deployment, repository visibility, runtime transformation, and team workflow; source packages, receipts, and runtime derivatives must follow the applicable redistribution terms.
- Does not remove or rewrite the current highway implementation during the prototype.
