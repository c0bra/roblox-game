## Context

`roblox/web` is a portrait-first strict TypeScript/Vite game using Web Audio for authoritative song time, Canvas 2D for the classic note highway, and Babylon.js 8 for the current boss model. The current controller turns dense per-instrument note charts into three-lane tap and sustain play. This is technically functional, but the permanent highway and strike line consume the player's visual attention, so the boss and arena act mostly as scenery.

Arena V2 is an additive gameplay experiment, not a rewrite. It must keep Classic Highway playable while testing a different interaction thesis: the music is the combat clock, the boss communicates intent, and short world-anchored cues communicate the player's response. The first slice must be attractive enough to test that thesis honestly. Placeholder cubes, unrigged characters, silent attacks, and generic particles would make a valid mechanic feel worse than it is and are therefore insufficient for acceptance.

The existing `roblox/web/DESIGN.md` remains the visual contract. Arena extends its supernatural concert identity, cyan player energy, gold climax energy, violet boss corruption, readable HUD typography, reduced-motion rules, and signature lightline. Arena-specific primitives, motion tokens, camera rules, asset references, and accepted debt must be added to that document before Arena UI or VFX implementation.

## Goals / Non-Goals

**Goals:**

- Preserve Classic Highway as a stable, playable comparison mode in the same web application.
- Deliver one polished solo Arena encounter for one explicit song and instrument on Easy difficulty, preceded by a brief nonfatal rehearsal and followed by a 30-to-45-second scored segment.
- Keep the player's gaze centered on the boss, character, and attack path rather than a permanent note lane.
- Make perform, retreat, and advance meaningful combat verbs whose timing is judged against the Web Audio clock.
- Make three fixed positions visibly and mechanically different in safety, phrase demand, damage, song influence, and reward exposure.
- Produce an original, rigged player performer, acquire and adapt a presentation-ready animated first boss whose license permits the web workflow, and produce a small arena kit, animation clips, and beat-synchronized VFX with reproducible or traceable sources and optimized runtime exports.
- Establish a versioned Arena encounter schema, deterministic QA fixture, input mapping, performance budget, reduced-motion behavior, and visual/manual QA gate.
- Keep the architecture compatible with future multiplayer and additional bosses without implementing those systems now.

**Non-Goals:**

- Free-roaming, analog movement, platforming, or a navigable dungeon.
- Production multiplayer networking, matchmaking, remote authority, or synchronized loot.
- Multiple production-complete bosses, character customization, progression, inventory, or a live drop table.
- Automatic generation of complete Arena encounters from every existing note chart.
- Replacing, deleting, or visually redesigning Classic Highway during the prototype.
- Migrating away from Babylon.js, Vite, Web Audio, or the existing level catalog.
- Photorealistic assets or final Roblox production art. Assets must be cohesive and appealing, but scoped as a vertical-slice quality target.

## Decisions

### 1. Arena is a separate mode inside `roblox/web`

The application will expose `classic` and `arena` mode identifiers. Classic remains the default until Arena passes its validation gates. A mode selector on the existing selection screen and a deterministic query parameter such as `?mode=arena&qa=1` will allow direct comparison and browser automation. The first slice supports exactly one documented song and instrument on Easy difficulty. Unsupported selections remain unchanged, show Arena as unavailable, and offer an explicit action to switch to the supported demo selection rather than silently substituting it.

The bootstrap layer will instantiate only the selected mode controller and will dispose its listeners, animation frames, audio run, and Babylon engine when the mode changes. Song, instrument, and difficulty selection remain shared. Classic's current controller and highway renderer remain behaviorally unchanged except for the minimum lifecycle boundary needed by the mode bootstrap.

**Alternatives considered:**

- A separate `roblox/web-v2` application would protect Classic, but would duplicate dependencies, selection UI, audio loading, charts, tests, and assets and would quickly drift.
- An in-place rewrite would be smaller initially, but would destroy the playable comparison baseline and make rollback harder.

### 2. The first slice is fixed-camera 2.5D in Babylon.js

Arena will use one Babylon scene containing the boss, player, anchor markers, cover props, lights, camera, and world VFX. The existing generated stage artwork can remain as a layered background and atmosphere plate. DOM/CSS continues to own HUD, prompts, controls, pause, error, and results. This is 2.5D: real rigged 3D actors and spatial effects inside a deliberately constrained composition, not free camera movement or a fully traversable arena.

The portrait camera frames the boss in the upper center, the player in the lower center, and the three radial positions along the visual axis between them. The visual path from boss to player doubles as the telegraph path, so timing information appears where the action already is. Tablet and desktop retain the same composition with additional atmospheric side space rather than revealing more gameplay area.

**Alternatives considered:**

- Full 3D free movement adds camera, navigation, collision, occlusion, and control risks before the cue grammar is proven.
- Pure 2D sprites are cheaper, but would not adequately test character visibility, boss performance, spatial dashes, or the intended Roblox-adjacent presentation.
- A new engine would add bundle and learning cost while Babylon.js already loads GLB assets and renders the current boss.

### 3. Web Audio time remains authoritative

All beats, phrase steps, telegraphs, impacts, animation launches, and judgments are scheduled from the existing audio run's seconds-based clock. Rendering interpolates from that time on each frame; requestAnimationFrame and animation clip time never become independent gameplay clocks. Pause and visibility changes pause audio, encounter state, animations, and effects together. Resume re-derives presentation from audio time rather than accumulating wall-clock drift.

Arena reuses the current Perfect, Great, and Good thresholds initially. Combat tuning may assign different effects to those grades, but there is one timing calculation and one semantic seconds unit across audio, encounter data, gameplay, and QA.

### 4. Arena data is separate, versioned encounter content

Each supported level will have an Arena encounter document alongside, not inside, the existing classic charts. The document is parsed once with Zod at the loading boundary and contains:

- schema version, level identifier, supported instrument, duration, and supported difficulty;
- beats and downbeats used by the world pulse;
- three named positions and their configurable risk/reward profile;
- static performance-phrase windows with preview time, execution time, perform steps, optional position-specific bonus steps, and optional opening payoff;
- reposition windows with decision beats, movement deadlines, and valid destination positions;
- boss events with telegraph start, impact, recovery, target positions, safe positions, and required response;
- phase and camera events for rehearsal, intro, escalation, climax, victory, and defeat;
- a boss-resolve threshold evaluated at the authored final cadence.

The first encounter is hand-authored from the existing song and beat data. Future tooling may derive candidate phrases, but generated content is not accepted without human play/listen review. A compact deterministic QA encounter exercises every event type without requiring a full song. Semantic validation rejects easy-mode cue collisions that would introduce a new phrase preview during a critical telegraph or require a movement to complete after its associated impact.

### 5. Combat uses a global beat clock plus short semantic phrases

Arena never displays a permanent scrolling highway. Its cue grammar has three layers:

1. **Pulse:** boss aura, anchor rims, player instrument, and optional HUD ring mark beats; downbeats have a stronger but non-flashing accent.
2. **Telegraph:** the boss pose, target path, ground shape, and sound communicate what attack is coming and which positions are affected.
3. **Phrase:** the complete three-to-five-step performance sequence appears at once near the boss/player axis at least two authored beats before execution. It never scrolls. One current step and, secondarily, one next step are emphasized while a stationary ring or contraction at the player action point communicates time to the authored beat.

The prototype actions are `retreat`, `perform`, and `advance`. Their labels, icons, controller bindings, and visual consequences remain stable; Arena does not use arbitrary QTE keys. Performance phrases contain authored `perform` beats, with optional extra perform beats at riskier positions. Retreat and advance occur only during world-readable reposition windows and only when they move the character toward a real anchor; the authored content exposes valid destinations rather than prescribing a single movement key.

Perform inputs always create immediate tactile/visual acknowledgement. Inputs inside the symmetric Good window earn grade-based combat effect and score; off-window inputs produce a readable flub with reduced or no combat effect. An early successful perform may schedule its lightline contact for the authored beat. A late successful perform uses immediate or compressed contact feedback and never pretends an effect landed in the past. Reposition windows are asymmetric and end early enough for visible travel to finish before the associated boss impact; movement after impact cannot undo resolved damage.

Phrases include intentional recovery or spectacle space and clear completely between active windows. Dense 16th-note streams are not a prototype goal. Sustained source notes may become occasional authored channel actions later, but the first slice uses taps and separate reposition windows so the attention test is not confounded.

### 6. Positions are the tactical difficulty system

The three positions are authored as named anchors and rendered as physical places:

- **Shelter:** strongest cover and simplest phrase profile; lowest damage, song influence, and exposure gain.
- **Midline:** balanced default with standard phrases and baseline rewards.
- **Spotlight:** visibly exposed and closest to the boss; stronger combat and exposure gain, harder phrase profile, and greater attack risk.

Exact multipliers remain data-driven so tuning does not require code changes. Exposure is an encounter result metric, not a live random drop roll. It records how long and how accurately the player performed in risky positions and can later feed a loot system. Boss attacks must periodically make different anchors safe or dangerous so position choice is active rather than a permanent difficulty toggle. When more than one destination is safe, the player can deliberately choose the safer or more rewarding option; the prompt never converts that decision into a predetermined retreat/advance sequence.

### 7. Art assets are a defined production track

Arena asset work is part of implementation, not an optional polish phase. Original project source files will live under `roblox/assets/arena_v2/` with concept, source, texture, preview, and output subdirectories. Commercial third-party packages that cannot be redistributed through the repository will remain in license-controlled private storage; the repository will retain their manifest record, receipt or entitlement reference, adaptation/export instructions, and permitted optimized runtime derivative. Runtime-approved exports will be copied to `roblox/web/public/assets/arena/` only after the applicable license is confirmed to permit the intended browser delivery and repository visibility. Every runtime asset will have a source or entitlement reference, license/origin entry, export settings, and preview.

The vertical slice requires:

- **Player:** one original stylized humanoid "Rift Performer" with a clear silhouette at phone scale, one instrument prop or energy instrument, a portable humanoid rig, and coherent cyan/gold materials.
- **Boss:** one externally sourced, already rigged and animated boss. None of the current repository models or GLBs are accepted as the Arena boss. The acquisition gate requires actual playable animation clips, phone-scale silhouette, editable or convertible FBX/Blend/GLB source, browser-safe materials, a death state, hit/stagger state, at least two visually distinct attack candidates, a license compatible with the web deployment, and an achievable optimized runtime budget.
- **Arena kit:** one stage disc or ruined threshold, three readable anchor/cover treatments, two to four ruined architectural props, an atmospheric backdrop, shadow receiver, and lighting probes/materials.
- **2D/UI art:** semantic action glyphs, position glyphs, boss attack glyphs, phrase frames, and fallback poster/silhouette assets.
- **VFX textures/materials:** soft particle, streak, spark, distortion/noise, ward crack, and impact masks or atlases as needed.

Current research shortlist, to be confirmed by hands-on package inspection rather than marketplace copy alone:

- **Preferred paid candidate:** N-Hance Studio's [Stylized Demon Boss](https://www.fab.com/listings/ddb0e2ad-a5c6-43ef-9af2-08c547644399). The listing advertises 57 animations, FBX plus GLB/glTF delivery, hand-painted PBR materials, and purple/violet variants that fit the existing corruption palette. Its animation inventory and formats make it the best current fit, subject to purchase approval, runtime-size inspection, and confirmation that its license permits the planned browser delivery and repository workflow.
- **CC0 fallback:** Quaternius [Ultimate Monsters](https://quaternius.com/packs/ultimatemonsters.html). It supplies 50 animated monsters in Blend, FBX, and glTF under CC0 and can safely support a public prototype, but its playful low-poly styling would require a strong silhouette selection, material overhaul, lighting, and VFX treatment to meet the intended supernatural-concert tone.
- **Alternate paid candidate:** Olegator's [Black Skeleton Warrior](https://www.fab.com/listings/d92bb158-40ee-4884-aa99-287a42ecadb9). It advertises GLB/FBX and more than 40 combat and locomotion animations, but its humanoid warrior silhouette is less singular than the preferred demon and its listing lacks an established review history.

The project rejects any candidate described only as "animation ready," lacking an explicit clip inventory, using an unclear or noncommercial license, requiring public redistribution of a protected source package, exceeding the web budget without a credible optimization path, or failing to distinguish both prototype attacks without particles. Purchase is not acceptance: the package must be imported into Blender, its source and runtime formats inspected, and its clips, deformation, materials, size, and license recorded before the visual direction locks.

Concept work begins with two or three distinct player/arena silhouette sheets and one integration sheet that stages the selected boss candidate against the existing design contract. One direction is selected before player/environment modeling. Generated concept art may accelerate exploration, but original runtime meshes, textures, and effects must be authored or transformed into project assets with provenance recorded; third-party derivatives retain their original license status rather than being mislabelled as original work.

Blender 5.2 is the source tool. Automation uses `BLENDER_EEVEE`; preview animation renders to PNG sequences and uses the `ffmpeg` CLI for MP4 review. Runtime delivery is GLB/glTF 2.0 with embedded or adjacent optimized textures. A checked-in export manifest records scale, forward axis, clip names, durations, loop behavior, material slots, triangle count, texture sizes, and output checksum.

### 8. Animation and VFX communicate gameplay state

The player animation set includes intro/ready, beat-aware idle, perform, advance dash, retreat dash, brace or ward, hit/stagger, victory, and defeat. The acquired boss package must provide or support adaptation into intro, beat-aware idle, at least two distinct telegraphs, two matching attacks, hit reaction, stagger/opening, phase transition, and defeat. Existing package clips are preferred over unnecessary reanimation, but telegraph lead-ins, timing edits, and transitions may be authored in Blender when required for gameplay readability. Imported Babylon `AnimationGroup`s own skeletal playback; gameplay events choose and synchronize groups from audio time.

The required first-pass VFX set includes beat pulse, downbeat accent, anchor safe/danger state, phrase preview/current-step focus, dash trail, performance projectile or lightline, boss target path, charge, impact, player ward hit/crack, boss hit, stagger opening, climax, and victory. Each effect must have a semantic purpose and a defined lifetime. VFX intensity scales down before reducing cue clarity.

Reduced motion removes camera shake, idle sway, repeated scale pulsing, large particle bursts, and nonessential trails. It preserves static target geometry, direct opacity/brightness changes, the active phrase symbol, and linear time-to-impact travel because those communicate required timing. Camera impulses are capped, beat-authored, and never used as the only hit confirmation.

### 9. Audio and mix feedback carry equal weight

The selected instrument stem remains a separate Web Audio channel. Accurate perform actions restore or accent that stem and trigger a short action sound; misses retain the existing short duck/flub principle. Boss telegraphs have concise, frequency-distinct cues, and impacts have synchronized transients. Audio cues reinforce but never replace visible target and timing information.

The sound language separates three jobs. Player sounds are bright, fast, and cyan-coded in character even when heard without visuals. Boss sounds are darker, rougher, and lower, but telegraphs retain enough upper-mid information to survive phone speakers. Interface sounds are short and neutral so they do not compete with either combat side. Per-beat audio is deliberately sparse: the song carries tempo, while sound effects mark decisions, consequences, and exceptional downbeats rather than creating a second metronome.

The first-slice inventory is organized into production tiers. Counts indicate the desired number of meaningfully different variations or intensities, not pitch-shifted copies of one identical render. The final column contains short standalone starting prompts for ElevenLabs Sound Effects. The canonical copy/paste worksheet is [`sound-prompts.md`](sound-prompts.md), where combined families are expanded into filename-level prompts with duration, loop, and take-selection settings. Generate and approve each listed variation independently; duration, trimming, loop repair, layering, and mix approval still follow the production contract below.

| Tier | Sound family / asset ID | Required variants | Gameplay purpose | Short ElevenLabs prompt |
|---|---|---:|---|---|
| P0 | `run_count_tick`, `run_count_go` | 1 each | Starts rehearsal and scored play with an unambiguous count that does not continue as an added metronome. | `Clean magical percussion tick, precise and dry, 0.15 seconds, no reverb.`<br>`Bright magical start hit with a tiny upward shimmer, decisive, 0.4 seconds.` |
| P0 | `arena_phrase_reveal` | 2 | Announces that a complete static phrase is available without counting every step for the player. | `Soft cyan energy constellation unfolds, anticipatory shimmer, clean, 0.6 seconds.` |
| P0 | `player_input_ack` | 3 | Immediate, very short response at button-down before any scheduled projectile contact. | `Tiny crisp energy pluck for instant button feedback, dry, 0.1 seconds.` |
| P0 | `player_perform_contact_good` | 2 | Restrained successful boss contact for a Good judgment. | `Small cyan energy bolt hits a huge monster, restrained impact, 0.3 seconds.` |
| P0 | `player_perform_contact_great` | 2 | Clearer and richer successful contact for a Great judgment. | `Bright cyan magic strike hits a huge monster, punchy layered impact, 0.4 seconds.` |
| P0 | `player_perform_contact_perfect` | 3 | Strongest successful contact, still short enough not to mask the next beat. | `Powerful cyan-gold energy strike hits a giant boss, brilliant clean impact, 0.5 seconds.` |
| P0 | `player_perform_flub` | 3 | Dry, unmistakable miss/off-window response with no triumphant tail. | `Weak sputtering energy pluck, failed magic attack, dry, 0.25 seconds.` |
| P0 | `player_reposition_select` | 2 | Confirms retreat/advance choice before travel begins. | `Quick tactical selection ping with a subtle energy whoosh, clean, 0.2 seconds.` |
| P0 | `player_dash_retreat`, `player_dash_advance` | 2 each | Directionally distinct travel gestures; readable without relying on stereo pan. | `Fast backward spectral dash whoosh, airy energy trail, 0.45 seconds.`<br>`Fast forward spectral dash whoosh, forceful energy surge, 0.45 seconds.` |
| P0 | `player_anchor_arrive` | 2 | Confirms physical arrival at any anchor with a consistent player-owned sound. | `Light armored landing with a compact magical energy settle, dry, 0.35 seconds.` |
| P0 | `boss_sweep_warn` | 1 | Unique onset identifying the first, path/sweep-style attack. | `Ominous monster weapon scrape expanding into a broad sweep warning, 0.7 seconds.` |
| P0 | `boss_sweep_charge_loop` | 1 seamless loop | Sustained rising danger that can be cut exactly at impact. | `Seamless dark-energy sweep charge loop, steadily rising tension, no impact, 1 second.` |
| P0 | `boss_sweep_impact` | 2 | Broad moving impact with a clear transient and short debris tail. | `Huge horizontal dark-energy slash impact, sharp transient and short debris tail, 0.8 seconds.` |
| P0 | `boss_burst_warn` | 1 | Unique onset identifying the second, zone/burst-style attack. | `Hollow supernatural inhale with crystalline pulses, ominous burst warning, 0.7 seconds.` |
| P0 | `boss_burst_charge_loop` | 1 seamless loop | Pulsed or inhaling danger texture, clearly unlike the sweep charge. | `Seamless pulsing void-orb charge loop, inhaling rhythm, no impact, 1 second.` |
| P0 | `boss_burst_impact` | 2 | Compact radial/crystalline burst, clearly unlike the sweep impact. | `Compact radial void explosion with a crystalline snap and heavy center impact, 0.8 seconds.` |
| P0 | `player_evade_success` | 2 | Confirms that the chosen position avoided an impact. | `Fast magical near-miss whoosh resolving into a bright safe chime, 0.45 seconds.` |
| P0 | `player_ward_hit` | 3 intensities | Communicates light, medium, and heavy ward damage. | `Light blow absorbed by a cyan magical shield, small glassy tick, 0.25 seconds.`<br>`Solid blow absorbed by a cyan magical shield, glassy crack and energy thump, 0.4 seconds.`<br>`Massive blow absorbed by a cyan magical shield, deep thump and violent fracture, 0.55 seconds.` |
| P0 | `player_ward_crack` | 1 | One-time low-ward warning; it must not loop or become an alarm. | `Single ominous magical shield fracture, crystalline stress and fading energy, 0.7 seconds.` |
| P0 | `player_ward_break` | 1 | Immediate ward-loss defeat transition. | `Cyan energy shield shatters violently, crystalline burst and low collapse, 0.9 seconds.` |
| P0 | `boss_hit` | 3 | Short randomized reactions for successful performance contact. | `Giant supernatural monster struck by bright magic, heavy body-energy impact, 0.4 seconds.` |
| P0 | `boss_stagger_open` | 1 | Announces a boss opening and creates space for the next phrase. | `Massive monster staggers, armor groan and collapsing dark aura, clear opening cue, 0.9 seconds.` |
| P0 | `phrase_complete` | 2 | Confirms a completed phrase without duplicating the result sting. | `Bright magical resolve flourish, rewarding, compact, nonmelodic, 0.5 seconds.` |
| P0 | `boss_resolve_gain` | 3 intensities | Reinforces meaningful Resolve progress, with the strongest layer reserved for high-value Spotlight play. | `Small crack forms in a dark magical barrier, bright energy accent, 0.3 seconds.`<br>`Dark magical barrier weakens with a sharp energy fracture, 0.45 seconds.`<br>`Dark magical barrier ruptures under cyan-gold energy, powerful fracture accent, 0.6 seconds.` |
| P0 | `final_resolve_success` | 1 | Locks the final-cadence victory condition before the visual climax. | `Supernatural boss seal breaks decisively, triumphant cyan-gold rupture, 1 second, no melody.` |
| P0 | `final_resolve_failure` | 1 | Distinct failed-seal outcome when ward remains but Resolve is insufficient. | `Ritual seal sputters and collapses unresolved, dark descending energy, 1 second, no melody.` |
| P1 | `arena_intro_rift` | 1 | Short encounter reveal layered under the boss entrance. | `Supernatural arena rift tears open, deep air pull and violet crackle, 2 seconds, no music.` |
| P1 | `boss_intro_vocal` | 2 | Boss identity without a long spoken line or music-like melody. | `Huge stylized fantasy monster awakening roar, intimidating, nonverbal, 1.5 seconds.` |
| P1 | `boss_sweep_vocal`, `boss_burst_vocal` | 2 per attack | Optional vocal layers tied to the two telegraphs, never required for recognition. | `Short monster exertion growl for a sweeping attack, aggressive, nonverbal, 0.6 seconds.`<br>`Short monster inhaling roar for a magical burst attack, ominous, nonverbal, 0.7 seconds.` |
| P1 | `boss_phase_transition` | 1 | Marks escalation without implying that the song has ended. | `Giant monster powers up, low roar and expanding corrupted energy, 1.8 seconds, no music.` |
| P1 | `boss_defeat` | 1 | Physical/energy collapse synchronized to the defeat animation. | `Massive supernatural monster collapses and dissolves into dark energy, heavy, 2 seconds.` |
| P1 | `world_downbeat_accent` | 3 | Sparingly randomized environmental accents for selected authored downbeats, not every measure. | `Subtle supernatural arena pulse, low airy thump, short and unobtrusive, 0.25 seconds.` |
| P1 | `position_shelter_enter`, `position_midline_enter`, `position_spotlight_enter` | 1 each | Adds a world-owned layer giving the three risk states distinct sonic identities after the shared arrival sound. | `Protected stone sanctuary activates, warm low shield hum, 0.5 seconds.`<br>`Balanced magical arena anchor activates, neutral energy chime, 0.5 seconds.`<br>`Exposed high-risk spotlight ignites, bright dangerous electric shimmer, 0.5 seconds.` |
| P1 | `result_victory_sting`, `result_defeat_sting` | 1 each | Brief outcome punctuation designed after the song segment and final cadence are chosen. | `Short triumphant supernatural combat sting, cyan-gold energy, no vocals, under 2 seconds.`<br>`Short dark failed-battle sting, unresolved energy collapse, no vocals, under 2 seconds.` |
| P1 | `ui_move`, `ui_confirm`, `ui_back`, `ui_error` | 2, 2, 1, 1 | Arena setup, pause, retry, and result navigation; existing suitable UI sounds may be reused. | `Tiny neutral spectral interface tick, crisp and soft, 0.1 seconds.`<br>`Clean bright spectral confirmation chime, short, 0.25 seconds.`<br>`Soft descending spectral interface whoosh, brief, 0.2 seconds.`<br>`Muted distorted spectral interface buzz, clear but not harsh, 0.25 seconds.` |
| P2 | `arena_ambience_loop` | 1 seamless loop | Very low-level supernatural room tone, omitted if it muddies the song. | `Seamless ruined supernatural arena ambience, distant wind and faint rift hum, no music, 8 seconds.` |
| P2 | `cover_debris` | 3 | Small environmental reactions to nearby impacts. | `Small stone chips and dust falling after a nearby impact, dry and short, 0.6 seconds.` |
| P2 | `streak_milestone` | 3 intensities | Infrequent streak reinforcement, disabled if it competes with phrase timing. | `Tiny rising energy flourish for an early combat streak, nonmelodic, 0.3 seconds.`<br>`Bright rising energy flourish for a strong combat streak, nonmelodic, 0.45 seconds.`<br>`Powerful rising cyan-gold flourish for a major combat streak, nonmelodic, 0.6 seconds.` |
| P2 | `spectral_crowd_react` | 3 | Optional world flavor for opening, climax, and victory only. | `Distant ghostly crowd swell, subtle, nonverbal, no words, 1.2 seconds.` |

P0 is required for the graybox attention gate. P1 is required for the finished vertical slice except where the table explicitly allows reuse. P2 is polish and is cut first if clarity, schedule, or size suffers. The provisional `sweep` and `burst` identities describe sonic/mechanical contrast; their final names and timbres follow the selected boss animations.

Generated masters are delivered as 48 kHz, 24-bit WAV. Point-source player, boss, ward, dash, and debris sounds are dry mono unless a designed stereo component is essential; UI and result stingers may be stereo. Transients begin without accidental leading silence, charge loops include clean loop boundaries and separate release tails where needed, and every file documents intended start/contact/loop markers. Repeating P0 events use variants without changing their semantic identity. Tonal effects remain pitch-light or are delivered as separable tonal/noise layers until the song segment and musical key are locked.

Runtime files use browser-tested compressed encodes and target no more than 1.5 MB transferred for Arena sound effects, excluding the song and stems. Masters, generation prompts/settings, licenses or original-work declarations, edit notes, runtime encodes, durations, channels, loop points, checksum, and mix intent are recorded in the audio manifest. Effects peak no higher than -1 dBTP before runtime gain, but final loudness is judged in the active song mix on the named phone and desktop outputs rather than normalized in isolation. Boss telegraphs, phrase cues, and input feedback must remain intelligible without masking vocals, the selected instrument, or the next timing transient.

The first slice does not add a new music middleware dependency. Effects use Web Audio buffers and the existing run clock. Random variation selection is deterministic in QA runs, and scheduled contact/impact sounds derive from authoritative song time so pause, resume, replay, and dropped frames cannot double-trigger or drift.

### 10. Performance, loading, and fallback are explicit gates

Arena assets load only after the player selects Arena and begins the encounter. The loading state reports meaningful progress and supports retry or return to Classic. The initial target, excluding song audio, is no more than 12 MB transferred for Arena-specific runtime assets, with no texture above 1024 px unless a measured visual need is recorded. The target is 60 FPS on desktop and a sustained 30 FPS floor at the 375 px mobile QA viewport on the available test hardware.

The asset review records mesh and texture budgets, draw calls, shader cost, bundle change, and peak loading behavior. Babylon engine, scene, materials, textures, particles, and animation groups are disposed when leaving Arena. If WebGL or a required model fails, the app shows a static Arena poster and a recoverable choice to retry or launch Classic; it never leaves a blank canvas or silently starts a mechanically incomplete Arena run.

### 11. Boss resolve gates victory at the final cadence

Boss damage fills a deterministic Resolve objective rather than allowing an early kill that would break the authored song. Accurate performance applies grade- and position-adjusted resolve damage. The encounter evaluates the configured resolve threshold at the authored final cadence: meeting it produces the victory climax, while missing it produces a failed-seal defeat even if the player's ward remains above zero. Excess resolve damage becomes score or exposure and does not end the song early.

### 12. Validation measures attention as well as correctness

Automated coverage includes encounter-schema parsing, sorted/bounded event validation, phrase and attack state transitions, timing grades, position effects, pause/resume synchronization, mode isolation, and deterministic end-to-end Arena completion. Browser QA runs at 375, 768, and 1280 px, with keyboard, pointer, reduced motion, load failure, and visibility pause scenarios.

Visual QA must capture stills and a short motion recording that exercises every required animation and VFX state. A three-person graybox attention check occurs before final player/environment asset production; failure to understand the cue grammar or return attention to the battle pauses the art track and revises the mechanic. The finished slice then receives a five-person first-time-player test. At least four testers must identify both boss attacks, explain the three-position tradeoff, describe at least one player-character action, achieve at least 60% timing accuracy on the easy encounter, and satisfy the observer rubric that attention returns to the boss/player between brief phrase glances rather than remaining fixed on the prompt area. Attention sessions use device speakers or wired audio unless latency is calibrated and record the audio-output condition. Failure of the attention measure blocks making Arena the default even if automated tests pass.

## Risks / Trade-offs

- **[Risk] The ground telegraph becomes another note highway** -> Restrict travel paths to boss threats and special phrase resolution, keep phrases short, and reject continuous dense note streams in the first slice.
- **[Risk] Visual flavor obscures timing or targets** -> Build semantic VFX primitives first, test without particles, then layer atmosphere while maintaining shape, position, and contrast redundancy.
- **[Risk] Asset production dominates the experiment** -> Limit the slice to one player, one boss, one arena kit, two attacks, and one climax; pass the graybox attention gate before final player/environment production and approve silhouettes before detailed modeling.
- **[Risk] A marketplace boss looks suitable but its delivered files, clips, or license do not support the web game** -> Inspect the actual package and license before art lock, reject "animation ready" without playable clips, keep a CC0 fallback, and do not commit protected source files to a public repository.
- **[Risk] 3D asset payload or shaders perform poorly on phones** -> Enforce export manifests and budgets, load Arena on demand, profile representative hardware, reduce texture/particle cost before reducing core telegraph clarity.
- **[Risk] Two modes cause controller/listener or audio leaks** -> Instantiate one mode at a time behind a lifecycle boundary and add repeated-switch integration coverage.
- **[Risk] Timing animation drifts from audio** -> Derive encounter and animation presentation from the Web Audio clock on every frame and test pause/resume and throttled-frame recovery.
- **[Risk] The fixed positions feel like a menu rather than an arena** -> Require visible character travel, boss targeting, cover reactions, camera staging, and meaningful safe/danger changes rather than changing only HUD labels.
- **[Risk] Movement prompts erase positional agency** -> Author safe/danger destinations and decision beats, derive retreat/advance from the player's chosen destination, and keep performance phrases separate from reposition windows.
- **[Risk] The compact phrase prompt becomes a miniature highway** -> Display the full phrase statically, use a stationary time-to-hit focus, clear it between phrases, reject dense streams, and test gaze before final asset production.
- **[Trade-off] 2.5D does not prove full free-roaming combat** -> It intentionally proves the higher-risk cue and attention model first while leaving a path to full 3D later.
- **[Trade-off] Classic remains duplicated at the product level** -> The playable baseline is valuable during validation; retirement is a later explicit product decision.

## Migration Plan

1. Select the supported song and instrument for the Easy slice, name QA hardware, and complete the online boss package/license/format feasibility gate.
2. Add Arena-specific `DESIGN.md` primitives and a component/state showcase without changing the Classic screen.
3. Introduce mode identity, shared lifecycle boundaries, and direct Arena QA routing while keeping Classic as the default.
4. Add and validate the Arena encounter schema and deterministic fixture.
5. Build a graybox encounter with the static phrase grammar, reposition choices, semantic controls, and timing.
6. Run the early graybox attention gate; revise the mechanic before final art if the gaze thesis is not trending correctly.
7. Produce and integrate approved player, acquired boss adaptation, arena, animation, audio, and VFX assets in bounded passes.
8. Complete automated, visual, performance, reduced-motion, and final attention QA.
9. Expose Arena in the normal mode selector only after the slice passes; retain Classic as the immediate fallback.

Rollback is disabling the Arena selector/deep link and removing its on-demand asset references. Classic data and implementation are not migrated, so rollback does not require chart or save conversion.

## Open Questions

- Which existing song segment and instrument provide the clearest 30-to-45-second Easy intro, two attack phrases, opening, and climax for the first slice?
- Should the Rift Performer visibly embody that selected instrument, or use a simpler energy-performance focus that preserves the boss and timing silhouettes?
- Does the intended repository and deployment model permit delivery of the preferred paid boss under its marketplace license, and does hands-on inspection confirm that its GLB/FBX contains usable clips and can meet the optimized budget? If not, which CC0 candidate becomes the art-direction base?
- What named test hardware will define the mobile 30 FPS floor and the final asset budget adjustments?
