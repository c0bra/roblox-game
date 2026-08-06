## 1. Baseline and Design Contract

- [x] 1.1 Run and record the current Classic unit tests, type/format checks, production build, and deterministic browser QA route as the regression baseline.
- [x] 1.2 Choose and document the first 30-to-45-second song segment and supported instrument for the Easy slice, rehearsal duration, named desktop/mobile QA hardware, standardized attention-test audio output, and initial 12 MB asset-budget allocation.
- [x] 1.3 Confirm whether the repository and deployed web demo may expose a paid runtime GLB, review the preferred N-Hance Stylized Demon Boss and fallback candidates against their current licenses, and record which source/runtime files may live in public, private, or deployment-only storage.
- [x] 1.4 Obtain explicit purchase approval if the preferred paid candidate remains viable, acquire its source package and license receipt, or acquire the CC0 fallback without treating a marketplace preview as acceptance.
- [x] 1.5 Inspect the acquired candidate structurally and in a minimal Babylon import for actual clip inventory, deformation, materials, triangle/texture/runtime size, phone-scale silhouette, two pose-distinct attacks, hit/stagger, death, axes, and exportability; record the Blender 5.2 crash exception and use the official glTF/glTF Transform path instead.
- [x] 1.6 Extend `roblox/web/DESIGN.md` with Arena research references, the selected boss integration decision, fixed-camera composition, three anchor primitives, static phrase constellation and stationary timing focus, reposition-choice grammar, semantic action/attack glyphs, cue hierarchy/collision rules, camera limits, motion tokens, reduced-motion rules, asset references, and accepted debt before implementing Arena visuals.
- [x] 1.7 Build an Arena primitive/state showcase for the mode selector and unsupported-combination state, combat controls, position/reposition states, complete static phrase preview/current/next step, boss telegraph states, meters, loading, fallback, and results at 375, 768, and 1280 px.
- [x] 1.8 Create `roblox/assets/arena_v2/` original-source/concept/texture/audio-master/preview/output structure, `roblox/web/public/assets/arena/` permitted model/texture/audio runtime structure, protected-package entitlement references, and versioned asset-origin/license/export and audio-manifest templates.

## 2. Mode Boundary and Classic Preservation

- [x] 2.1 Add failing tests for default Classic selection, explicit Arena query selection, invalid mode fallback, preservation of song/instrument/difficulty across mode selection, unsupported Arena combinations, and explicit opt-in switching to the supported demo combination.
- [x] 2.2 Add a typed `classic | arena` mode boundary and query parser without moving or rewriting the Classic gameplay implementation.
- [x] 2.3 Add a mode-aware application bootstrap that instantiates exactly one controller and exposes deterministic `?mode=arena&qa=1` routing.
- [x] 2.4 Add failing lifecycle tests that prove audio, timers, animation frames, pressed input, global listeners, and Babylon resources are released when a mode exits.
- [x] 2.5 Extract explicit removable listener/timer/frame ownership from the existing oversized Classic controller behind a lifecycle contract without changing Classic mechanics.
- [x] 2.6 Update the Classic controller to satisfy the lifecycle contract and prove its existing selection, countdown, tap/sustain, pause, replay, and result behavior still passes.
- [x] 2.7 Add repeated mode-start/exit integration coverage that detects duplicate input handling, multiple audio runs, orphan animation loops, and retained Babylon scenes.

## 3. Arena Encounter Schema and Content

- [x] 3.1 Add failing schema tests for valid encounter data, level/instrument/difficulty mismatch, unknown versions/actions, invalid position references, unsorted/non-finite times, two-beat preview ordering, position-specific bonus steps, reposition deadlines, easy-mode cue collisions, attack ordering, Boss Resolve threshold, duration bounds, and intentional non-conflicting coincident events.
- [x] 3.2 Define strict Zod schemas and inferred readonly TypeScript types for encounter metadata, beats/downbeats, positions, static performance phrases, position-specific bonus steps, reposition windows, boss events, Boss Resolve threshold, rehearsal, and phases.
- [x] 3.3 Implement cross-reference, timeline, movement-deadline, and cue-collision semantic validation with typed development diagnostics and concise user-facing load failures.
- [x] 3.4 Add an Arena loader that selects encounter data by level/instrument/difficulty, validates it once at the boundary, and leaves existing Classic chart loading unchanged.
- [x] 3.5 Add missing/network/parse/version/semantic error mapping plus retry and same-selection Classic recovery data.
- [x] 3.6 Author a compact deterministic Arena QA encounter that exercises all actions, positions, reposition choices, static base/bonus phrases, two attack types, opening, final-cadence Resolve success/failure, ward defeat, and result metrics.
- [x] 3.7 Author and listen/play-review a brief nonfatal rehearsal plus the selected 30-to-45-second production slice with intro, beat grid, at least one three-to-five-step static phrase, optional Spotlight bonus steps, reposition choice, two boss attacks, opening, climax, failed-resolve path, and bounded recovery intervals.

## 4. Pure Arena Combat Model

- [x] 4.1 Add failing tests for Arena state initialization at Midline, legal adjacent movement, boundary movement, multiple valid destination choices, position multipliers, and exposure accumulation.
- [x] 4.2 Implement the pure Arena run state and reducer for phase, song time, position, active reposition window, ward, Boss Resolve, phrase progress, score, accuracy, streak, and exposure.
- [x] 4.3 Add failing tests for Perfect/Great/Good/Miss perform-step matching, immediate flub behavior, early scheduled contact, late compressed contact, one-resolution-only guarantees, and deterministic coincident-event ordering.
- [x] 4.4 Implement perform judgment from the Web Audio seconds clock while reusing the existing timing thresholds through a shared pure timing seam and keeping immediate input acknowledgement separate from contact presentation time.
- [x] 4.5 Add failing tests for complete two-beat phrase preview, stationary current/next focus, position-specific bonus steps, execution, completion/clear, recovery/opening, interruption by defeat, and replay reset.
- [x] 4.6 Implement the bounded static performance-phrase state machine and grade-based combat/score effects for perform steps.
- [x] 4.7 Add failing tests for asymmetric reposition windows, player-selected retreat/hold/advance, travel completion before impact, and inability for post-impact movement to reverse damage.
- [x] 4.8 Implement reposition-choice resolution separately from performance phrases.
- [x] 4.9 Add failing tests for boss prepare/impact/recovery phases, safe/affected positions, configured responses, damage exactly once, successful avoidance, and opening transitions.
- [x] 4.10 Implement boss event resolution, ward damage, Boss Resolve, position targeting, openings, final-cadence threshold victory, failed-resolve defeat, ward defeat, and no early song completion in the pure model.
- [x] 4.11 Add pause/resume, visibility pause, dropped-frame re-derivation, and deterministic replay tests against a controllable audio-clock adapter.

## 5. Graybox Arena Vertical Slice

- [x] 5.1 Create an `ArenaController` and one Babylon scene with a temporary readable player form, the acquired boss candidate or a license-safe proxy, three radial anchors, cover forms, fixed camera, and explicit disposal.
- [x] 5.2 Implement responsive portrait composition so 375, 768, and 1280 px preserve boss/player/anchor geometry and use extra width only for atmosphere.
- [x] 5.3 Add labelled Retreat, Perform, and Advance touch controls plus keyboard bindings, pressed states, focus states, disabled boundary states, pause, resume, replay, and exit.
- [x] 5.4 Render the ambient beat/downbeat clock from audio time on the boss-player axis without a permanent or miniature scrolling note highway.
- [x] 5.5 Render every three-to-five-step phrase simultaneously at least two beats early, keep the current and next steps legible, use a stationary timing focus, show optional position-bonus steps without obscuring the base phrase, and provide immediate early/on-time/late/miss acknowledgement.
- [x] 5.6 Render data-driven safe/danger anchor states, boss target paths, prepare/impact/recovery phases, multiple safe destination choices where authored, asymmetric reposition deadlines, and visible graybox travel completed before impact.
- [x] 5.7 Connect Arena judgments to selected-stem accent/restore, miss duck/flub, Boss Resolve, ward damage, HUD values, callouts, final-cadence victory, failed-resolve defeat, ward defeat, and Arena-specific results.
- [x] 5.8 Add the brief nonfatal rehearsal before the scored slice and verify that it teaches static phrase timing and reposition choices without changing scored results.
- [x] 5.9 Add an end-to-end deterministic graybox scenario that completes success, failed-resolve, and ward-defeat QA paths with timestamped inputs and verifies exact ward, Boss Resolve, score, accuracy, exposure, and outcome.
- [ ] 5.10 Run the three-person graybox attention gate before final player/environment art: each tester must identify the active phrase, both boss attack preparations, all three position meanings, and at least one visible player action while an observer records whether gaze returns to the battle between inputs; revise the cue grammar until the gate passes.

## 6. Visual Direction and 2D Presentation Assets

- [ ] 6.1 Generate two or three distinct Rift Performer/arena silhouette concept sheets and one integration paint-over for the selected online boss using the existing supernatural concert design tokens.
- [ ] 6.2 Select one player/arena direction through phone-scale silhouette, cue-readability, compatibility with the acquired boss, originality, provenance/license, and asset-budget review; record the decision in `DESIGN.md` and the asset manifest.
- [ ] 6.3 Use the selected first-slice instrument to decide whether the performer carries a recognizable instrument-shaped energy focus or a simpler performance prop, and lock the silhouette before modeling.
- [x] 6.4 Create final SVG perform-step, position, reposition-choice, and boss-attack glyphs with shape/text redundancy; include static preview, current, next, early, on-time, late, success, failure, and disabled states in the showcase.
- [x] 6.5 Create the authored Arena fallback poster/silhouette and verify that loading/error/retry/Classic recovery remain readable at all three QA widths.

## 7. Rift Performer Asset and Animation

- [ ] 7.1 Model the approved stylized Rift Performer and performance prop at phone-readable proportions within the assigned triangle budget.
- [ ] 7.2 Create UVs and optimized cyan/gold materials/textures that remain readable under Arena lighting and meet the texture budget.
- [ ] 7.3 Build and weight a portable humanoid rig, then capture front/side/three-quarter rest, deformation, and combat-pose validation previews.
- [ ] 7.4 Author and review intro/ready, beat-aware idle, and perform animation clips with named timing markers.
- [ ] 7.5 Author and review advance dash, retreat dash, brace/ward, and hit/stagger clips with clean interruptible transitions.
- [ ] 7.6 Author and review victory and defeat clips that hold readable final poses for the results transition.
- [ ] 7.7 Export the player as GLB with named `AnimationGroup` clips, create still and MP4 previews using the Blender 5.2 PNG-sequence/ffmpeg workflow, and complete its export manifest/checksum.
- [ ] 7.8 Import the final player into the Babylon showcase and verify scale, facing, materials, every clip, transition, reduced-motion behavior, reset, and disposal.

## 8. Acquired Boss Adaptation and Animation

- [ ] 8.1 Preserve the licensed original package and receipt only in storage permitted by its license, create a working Blender derivative, and link both locations from the protected entitlement/export manifest without committing prohibited source files.
- [x] 8.2 Audit and select source clips for intro, idle, hit reaction, stagger/opening, at least two pose-distinct attacks, phase transition, and defeat; record any missing transition or timing work before animation adaptation begins.
- [ ] 8.3 Optimize the selected boss geometry, materials, texture dimensions, runtime scale, facing, pivots, and clip set within its budget, then capture deformation and phone-scale silhouette previews.
- [ ] 8.4 Adapt and retime the acquired intro, beat-aware idle, hit reaction, and stagger/opening clips to encounter timing, authoring only the missing transitions or poses rather than replacing usable licensed animation.
- [ ] 8.5 Adapt and review a first distinctive telegraph/attack pair whose affected positions remain understandable from pose and target geometry with particles disabled.
- [ ] 8.6 Adapt and review a second mechanically and visually distinct telegraph/attack pair whose affected positions remain understandable from pose and target geometry with particles disabled.
- [ ] 8.7 Adapt and review phase-transition and defeat clips with clear climax silhouettes and authored timing markers.
- [ ] 8.8 Export only the license-permitted optimized runtime boss GLB with named `AnimationGroup` clips, create still and MP4 previews, and complete its provenance/license/export manifest and checksum.
- [ ] 8.9 Import the optimized boss into the Babylon showcase and verify scale, materials, every required clip, pose-only telegraph distinction, synchronization markers, reset, disposal, and that deployed files cannot be mistaken for a redistributable source package.

## 9. Arena Environment, Camera, and Lighting

- [ ] 9.1 Model the ruined threshold/stage disc, Shelter, Midline, Spotlight, and two to four supporting architectural props from the approved concept.
- [ ] 9.2 Give all three anchors distinct spatial, cover, and shape silhouettes that remain identifiable with hue removed and VFX disabled.
- [ ] 9.3 Create optimized arena materials, shadow receiver, atmospheric backdrop layers, and texture outputs within the assigned budget.
- [ ] 9.4 Establish camera framing, restrained intro/climax choreography, focus targets, occlusion rules, and capped semantic camera-impulse tokens in `DESIGN.md`.
- [ ] 9.5 Build cyan player, violet boss, neutral fill, emissive accent, and atmosphere lighting that preserves actor and telegraph contrast in every phase.
- [ ] 9.6 Export and integrate the environment kit, then record triangle, texture, draw-call, and runtime-size measurements in its manifest.
- [ ] 9.7 Replace the graybox scene with approved assets and re-run the anchor, actor, cue-layering, and responsive-composition showcase at all QA widths.

## 10. Semantic VFX and Audio Flavor

- [ ] 10.1 Implement beat and downbeat effects on boss aura, player instrument, and anchor rims with a non-flashing reduced-motion variant.
- [ ] 10.2 Implement the complete static phrase preview, current/next emphasis, stationary time-to-hit indicator, optional position bonus, and early/on-time/late/success/failure effects whose timing and semantic action remain readable without color or scrolling motion.
- [ ] 10.3 Implement safe/danger anchor effects and boss target geometry that preserve the actor silhouettes and active phrase hierarchy.
- [ ] 10.4 Implement advance/retreat dash trails and arrival effects with concise reduced-motion replacements.
- [ ] 10.5 Implement immediate player-input acknowledgement plus scheduled early, on-time, or compressed-late performance contact, boss impact/hit reaction support, and selected-stem accent as one authoritative-song-time feedback event.
- [ ] 10.6 Implement the first boss charge, target path, impact, player ward hit/crack, and recovery effects with matching audio cues.
- [ ] 10.7 Implement the second boss charge, target path, impact, player ward hit/crack, and recovery effects with a clearly different shape and sound signature.
- [ ] 10.8 Implement opening/stagger, phase transition, climax, victory, and defeat effects with bounded lifetimes and reduced-motion variants.
- [ ] 10.9 Lock the sonic palette after selecting the song segment, instrument, key, and boss animations; refine and approve the filename-level ElevenLabs starting prompts in `sound-prompts.md` for the player, boss, UI, the provisional sweep/burst attack identities, mono/stereo use, tonal/noise layers, variation counts, durations, sync markers, loop/release behavior, and the 1.5 MB runtime SFX budget.
- [ ] 10.10 Generate, edit, and review the P0 graybox inventory: count tick/go; phrase reveal; input acknowledgement; Good/Great/Perfect contact; flub; reposition select; retreat/advance dash; shared anchor arrival; sweep warning/charge/impact; burst warning/charge/impact; evade; ward hit/crack/break; boss hit; stagger/opening; phrase complete; Resolve gain; and final Resolve success/failure.
- [ ] 10.11 Generate, edit, and review the P1 finished-slice inventory: arena intro; boss intro and optional attack vocals; phase transition; boss defeat; sparse authored downbeat accents; position-entry identities; victory/defeat stings; and required UI move/confirm/back/error sounds, reusing existing suitable UI assets where verified.
- [ ] 10.12 Evaluate P2 arena ambience, cover debris, streak milestones, and spectral crowd reactions in the active song mix; keep only effects that improve flavor without reducing timing, attack, vocal, or selected-instrument clarity.
- [ ] 10.13 Trim and master approved audio as 48 kHz 24-bit WAV, create clean charge loops and release tails, export browser-tested runtime encodes, and record semantic ID, prompt/settings or source, original/license status, duration, channel layout, markers, variation group, peak, checksum, and mix intent in the audio manifest.
- [ ] 10.14 Integrate deterministic variation selection for QA, position point-source sounds in the scene, keep UI/results centered, and balance every P0/P1 family against the selected song on the named phone and desktop outputs.
- [ ] 10.15 Add synchronization tests and a replay/pause/resume showcase proving skeletal animation, camera, particles, materials, HUD, immediate/scheduled audio, charge loops, and outcome stingers recover from authoritative song time without drift or duplicate triggers.

## 11. Loading, Accessibility, and Failure Recovery

- [x] 11.1 Add on-demand Arena asset loading so a fresh Classic run does not request player, licensed boss runtime, environment, or VFX bundles and cannot request protected marketplace source packages.
- [x] 11.2 Add determinate or staged Arena loading feedback with retry, cancel, and return-to-Classic actions.
- [x] 11.3 Add static-poster recovery for WebGL initialization, context loss, encounter load, and required model failures without starting an incomplete countdown.
- [x] 11.4 Verify touch targets, safe-area spacing, keyboard completion, visible focus, focus transfer, browser zoom, pause overlays, result actions, and interruption recovery.
- [x] 11.5 Verify color-vision-independent positions/attacks, muted-audio visual timing, and non-color-only Perfect/Great/Good/Miss feedback.
- [ ] 11.6 Verify `prefers-reduced-motion` removes camera shake, idle sway, repeated scale pulses, large bursts, and long trails while preserving active-step, target, and time-to-impact information.
- [x] 11.7 Add the normal setup-screen Arena selector and unsupported-combination state only after the direct QA route passes the complete verification gates; preserve the user's current selection, offer an explicit switch to the supported Arena combination, and keep Classic as the default.

## 12. Verification and Acceptance

- [x] 12.1 Run the full Bun tests, Biome/TypeScript checks, and production build; fix Arena regressions and identify any genuinely pre-existing failures without weakening tests.
- [x] 12.2 Re-run the recorded Classic deterministic scenario and compare controls, chart behavior, audio, pause/resume, results, screenshots, and console output with the baseline.
- [x] 12.3 Run deterministic Arena success, failed-resolve defeat, ward defeat, missed/early/late input, selected reposition, late movement, unsupported selection, boundary movement, pause/resume, visibility pause, replay, repeated mode switch, asset failure, and WebGL failure browser scenarios.
- [x] 12.4 Capture Arena at 375, 768, and 1280 px with every required control, complete static phrase state, position/reposition choice, attack, opening, Boss Resolve outcome, result, loading, and fallback state represented.
- [ ] 12.5 Record full-motion and reduced-motion Arena runs that exercise every player clip, boss clip, camera cue, semantic VFX family, and synchronized audio event; also run a blind audio-only recognition pass proving the two attack warnings/charges/impacts remain distinguishable without stereo position.
- [x] 12.6 Run `/visual-qa` until both defect and design-quality review pass for the showcase and complete encounter, including proof that no flat placeholder actor/effect remains.
- [ ] 12.7 Measure Arena-specific transfer size, individual texture dimensions, model triangles, draw calls, shader/particle cost, loading behavior, desktop frame timing, and the 375 px mobile 30 FPS floor on named hardware with the licensed boss runtime included.
- [x] 12.8 Audit every runtime asset against its editable source or protected-package entitlement, provenance/license entry, public-repository and browser-delivery permission, export settings, preview, budget fields, and checksum; resolve undocumented or impermissibly exposed artifacts.
- [ ] 12.9 Conduct the five-person first-time-player test on the documented audio-output condition and record timing accuracy, recall of both boss attacks, understanding of Shelter/Midline/Spotlight, recall of at least one visible player action, and observer-scored evidence that attention returns to the battle between inputs.
- [ ] 12.10 Keep Classic as the default unless at least four of five testers identify both attacks, explain the position tradeoff, recall at least one player action, achieve at least 60 percent Easy timing accuracy, demonstrate battle-directed attention between phrases, and all automated, visual, licensing, and performance gates pass.
- [x] 12.11 Confirm the final repository and deployed bundle contain no paid source package or other prohibited redistribution, retain the license receipt and attribution where required, and document which runtime derivative is authorized for browser delivery.
