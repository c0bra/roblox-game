# Bands Battle Design System

The project-wide aesthetic source of truth is [`ART_DIRECTION.md`](../../ART_DIRECTION.md).
This document owns the web implementation tokens, component behavior, accessibility,
and Arena production contract that apply that direction.

## 0. Research Log

- Embedded references: shortlisted PlayStation, Spotify, and RunwayML. Selected the
  project `taste-skill` plus PlayStation because the game needs console-grade product
  restraint around a dramatic hero object, large touch targets, and a clear blue focus
  language.
- Lazyweb: searched `mobile rhythm game gameplay`, `mobile music game instrument
  selection`, and `boss battle game HUD`; viewed Tapotron, MyTunes, and Royal Match.
  Harvested a compact persistent top HUD, immersive selection cards, one dominant start
  action, and oversized bottom controls. No screenshots are shipped or copied.
- UI/UX database: checked dark mobile game palettes, HUD typography, touch spacing, and
  reduced motion. Rejected the suggested clay treatment as tonally wrong; kept the 48px
  touch floor, 8px separation, tabular metrics, and reduced-motion rules.
- Imagen drafts: generated three portrait arenas. Selected
  `exec-f0cc291f-1529-44a4-af04-b222f9ca8d28.png` because its boss silhouette zone and
  dark lower runway remain readable behind a fast three-lane chart. The other drafts
  were brighter and more visually competitive in the playfield.

## 1. Atmosphere & Identity

Bands Battle is a supernatural concert staged at the last threshold between a cold,
ordered heaven and an eroding violet void. Each song is a distinct battle within the
same shared arena and against the same Fallen Choir. The interface feels like a premium console
game compressed into a phone: calm chrome, severe contrast, and explosive feedback only
when a note or attack earns it. The signature is the "light line": every successful tap
pulls a narrow white-gold strike from the player through the highway into the boss.

Design dials: `DESIGN_VARIANCE 7`, `MOTION_INTENSITY 7`, `VISUAL_DENSITY 7`.

## 2. Color

The game is intentionally dark-theme only. Lane colors are semantic input channels and
do not count as decorative accents.

| Role | Token | Value | Usage |
|---|---|---:|---|
| Void | `--void` | `#05070d` | Page and WebGL fallback |
| Stage | `--stage` | `#0a1020` | Battle shell |
| Panel | `--panel` | `rgb(9 14 28 / 0.9)` | Selection and result surfaces |
| Panel strong | `--panel-strong` | `#111a30` | Loading and error states |
| Text | `--text` | `#f1f7ff` | Titles and HUD |
| Muted text | `--muted` | `#a5b4ca` | Instructions and metadata |
| Player | `--cyan` | `#7ce8ff` | Focus, hits, boss damage, left lane |
| Heaven | `--gold` | `#ffe6a3` | Full hype, victory, and center lane |
| Corruption | `--violet` | `#a15cff` | Boss telegraphs, danger, and right lane |
| Danger | `--danger` | `#ff5470` | Health loss and error |
| Edge | `--line` | `rgb(176 215 255 / 0.24)` | Dividers and resting outlines |
| Neutral 3D fill | `--arena-fill` | `#b7c2d9` | Arena hemispheric light only |
| Ice floor | `--ice-floor` | `#78aeb9` | Centered ice-backdrop arena floor and dominant environment hue |
| Ice haze | `--ice-haze` | `#afd8eb` | Far-field floor fade sampled from the panorama foreground ice |
| Selected surface | `--surface-selected` | `rgb(12 33 49 / 0.96)` | Selected Classic cards |
| Meter surface | `--surface-meter` | `rgb(0 0 0 / 0.72)` | Classic meter tracks |
| Tap surface | `--surface-pad` | `rgb(5 9 18 / 0.9)` | Classic lane controls |
| Dialog surface | `--surface-dialog` | `rgb(7 10 20 / 0.98)` | Classic overlays |
| Soft scrim | `--scrim-soft` | `rgb(2 4 10 / 0.22)` | Stage vignette |
| Dialog scrim | `--scrim-dialog` | `rgb(2 4 10 / 0.7)` | Modal isolation |
| Shadow | `--shadow` / `--shadow-deep` | black variants | Text and depth |
| Gold contrast | `--text-on-gold` | `#15110a` | Primary CTA text |

Rules:

- Text meets WCAG 2.2 AA against its rendered surface.
- Danger, judgment, and lanes always include a word or shape, never color alone.
- Player cyan is the resting action accent. Gold appears only at full hype and victory.
- Violet belongs to the boss and selected right lane, distinguished by geometry.
- The centered ice arena multiplies `--ice-floor` through a low-contrast tiled
  albedo and pairs it with subtle normal and roughness maps. Surface detail must
  stay subordinate to the surrounding rock ring and distant mountain silhouette.
- The 24-unit playable ice disc continues as an unmarked visual ground skirt
  beneath the panorama. Beyond the playable radius, linear `--ice-haze` distance
  fog removes texture contrast and a broad radial opacity feather reveals the
  panorama ground before the mesh reaches the panorama wall; no exposed rim or
  hard color arc may identify the skirt boundary.

## 3. Typography

### Font stack

- Display and compact HUD: `Oxanium`, system sans fallback.
- Reading and controls: `Atkinson Hyperlegible`, system sans fallback.
- Both are self-hosted WOFF2 with `font-display: swap` and OFL licensing.

### Scale

| Role | Token | Size | Weight | Line height | Tracking |
|---|---|---:|---:|---:|---:|
| Title | `--type-title` | `clamp(2.5rem, 11vw, 5.5rem)` | 600 | 0.94 | -0.04em |
| Classic brand | `--type-brand` | `clamp(2.75rem, 13vw, 4.4rem)` | 760 | 0.86 | inherited |
| Screen heading | `--type-h1` | `clamp(1.7rem, 7vw, 3rem)` | 500 | 1.05 | -0.025em |
| Card heading | `--type-h2` | `1.25rem` | 600 | 1.15 | 0 |
| HUD value | `--type-hud` | `clamp(0.9rem, 4vw, 1.2rem)` | 600 | 1 | 0.02em |
| Body | `--type-body` | `1rem` | 400 | 1.5 | 0 |
| Compact copy | `--type-copy` | `0.87rem` | 400 | 1.35 | 0 |
| Instrument icon | `--type-icon` | `1.65rem` | 600 | 1 | 0 |
| Label | `--type-label` | `0.82rem` | 600 | 1.2 | 0.06em |
| Micro | `--type-micro` | `0.75rem` | 500 | 1.3 | 0.04em |
| Compact control | `--type-compact` | `0.68rem` | 500–680 | 1 | component-specific |
| Caption | `--type-caption` | `0.58rem` | 500 | 1 | component-specific |
| Combat callout | `--type-callout` | `clamp(1rem, 5vw, 1.55rem)` | 760 | 1.1 | 0.06em |
| Classic battle callout | `--type-battle-callout` | `clamp(1.25rem, 7vw, 2rem)` | 820 | 1 | 0.06em |
| Compact battle callout | `--type-compact-callout` | `clamp(0.92rem, 4.6vw, 1.25rem)` | 820 | 1 | 0.06em |
| Score | `--type-score` | `1.55rem` | 780 | 1 | inherited |
| Result value | `--type-result` | `1.05rem` | 720 | 1 | inherited |
| Overlay heading | `--type-overlay` | `2rem` | inherited heading weight | 1 | inherited |

All numeric HUD values use tabular figures. Body text never falls below 16px.

## 4. Spacing & Layout

Base unit: 4px.

| Token | Value | Usage |
|---|---:|---|
| `--space-1` | 4px | Glyph details |
| `--space-2` | 8px | Touch separation |
| `--space-3` | 12px | Compact groups |
| `--space-4` | 16px | Default padding |
| `--space-5` | 20px | Cards |
| `--space-6` | 24px | Screen groups |
| `--space-8` | 32px | Major separation |
| `--space-10` | 40px | Display rhythm |

Arena panels and elevation also use shared semantic tokens rather than component-local
shadow recipes: `--surface-arena`, `--surface-arena-soft`,
`--surface-cyan-soft`, `--surface-violet-soft`, `--elevation-panel`, and
`--elevation-dialog`. Geometry-specific dimensions remain local because they encode
camera framing, hit targets, or timing rather than reusable layout rhythm.

The app fills `100dvh`, reserves safe-area insets, and never scrolls during battle.
Portrait is primary. At 768px and above the stage is centered inside a maximum 760px
play column with atmospheric side space. Selection and results may scroll vertically.

Battle composition:

- Top 18%: compact boss and player status.
- Upper-middle 32%: boss silhouette and attack telegraph.
- Lower-middle 32%: perspective note highway.
- Bottom 18%: three equal tap pads above the gesture safe area.

Arena composition replaces the Classic highway geometry only while Arena is active:

- Top 16%: Ward, Boss Resolve, phase, score, and pause; no duplicated song metadata.
- Upper 44%: the Quaternius Demon boss, pose-readable telegraph, target geometry, and opening state.
- Lower-middle 24%: the Rift Performer and the Shelter, Midline, and Spotlight anchors on one boss-player axis.
- Bottom 16%: Retreat, Perform, and Advance controls above the safe area.
- The complete static phrase constellation sits on the boss-player axis, never at a scrolling edge. At 375px it remains below the boss hands and above the performer; at 768px and 1280px the same portrait geometry stays centered while side space receives atmosphere only.
- Before the runtime scene loads, setup shows a noninteractive tactical diagram of
  the boss, performer, axis, and three anchors so Arena's spatial promise is visible
  before the player commits.

## 5. Components

### Level Selector

- Structure: a labelled native-button radio group generated from the validated level catalog.
- Content: each option uses the song's display title; level IDs remain internal path-safe keys.
- States: default, selected, hover, focus-visible, and disabled while loading.
- Layout: a two-column row above instrument selection, with a 48px touch floor. Additional
  songs wrap to new rows without changing the shared stage or boss presentation.
- Keyboard: arrow keys wrap through songs; Home and End jump to the first and last song.
- Selection preserves the currently chosen instrument and difficulty.

### Action Button

- Structure: native `button`, label, optional state label.
- Variants: primary cyan, secondary translucent, danger.
- States: default, hover, pressed, focus-visible, disabled, loading.
- Motion: 0.98 press scale and 180ms color/ring transition. Loading swaps label with a
  blur-fade adapted from beui.dev `action-swap`; reduced motion uses opacity only.
- Accessibility: minimum 48px height, visible 3px focus ring, state exposed through
  `aria-busy` and disabled semantics.

### Instrument Card

- Structure: radio input, title, role, intensity meter, note count, lane glyph trio.
- States: default, hover, selected, focus-visible, disabled while loading.
- Layout: a compact 2 × 2 grid at phone and desktop widths so all choices, the primary action, and the instruction remain visible without scrolling at 375 × 667.
- Motion: selected card lifts 4px and gains a cyan inner rim; reduced motion removes lift.

### Difficulty Selector

- Structure: a labelled native-button radio group with Easy, Medium, and Hard options.
- Default: Easy, so a first-time run uses the lowest authored note density.
- States: default, selected, hover, focus-visible, and disabled while loading.
- Copy: each option states its maximum notes per beat; instrument cards describe the
  instrument role and never duplicate generated chart counts that can drift.
- Layout: one compact three-column row between the instrument grid and primary action;
  every option retains the 48px touch floor at 375px, 768px, and 1280px widths.

### Status Meter

- Variants: boss resolve, player health, hype.
- Structure: text label/value plus a meter element; the value is never color-only.
- Motion: fill changes use scale transforms from the left, not animated width.

### Tap Pad

- Structure: native `button`, geometric lane glyph, keyboard hint.
- States: ready, pressed, held, hit, miss, disabled.
- Touch: fills one-third of the bottom control row with an 8px gap; no precision gesture.
- Motion: immediate 0.96 press scale; hit emits an inward ring; miss flashes a static
  danger rim under reduced motion.

### Sustain Note

- Structure: the existing lane glyph forms the note head; a tapered energy ribbon extends
  toward the detected end of the sung or played note.
- Scope: vocals, guitar, and bass notes at least 350ms long. Drum transients remain taps.
- States: approaching, head hit, holding, completed, and broken by early release.
- Input: pointer and keyboard press begin the note; release ends it. Authored melodic charts
  prevent sustain overlap with the next playable note so the one-thumb mobile persona never
  needs to hold one lane while reaching for another.
- Feedback: the ribbon drains into the lightline while held, the pad keeps its energy fill,
  and completion earns a concise `HELD` judgment. Early release produces `HOLD BROKEN` and
  resets combo; duration is never communicated by color alone.
- Motion: adapted from the beui.dev `expanding-arrow-button` hold mechanism: continuous
  state is represented spatially, input stays interruptible, and release immediately
  retargets the state. Reduced motion keeps the linear ribbon travel because it communicates
  authored song time, while removing non-essential glow changes.

### Game Screen

- Variants: selecting, loading, countdown, playing, paused, won, lost, error.
- Each state has one primary action and a visible recovery path where applicable.
- Focus moves to the state heading or primary action after a state transition.

### Mode Selector

- Structure: two native radio-style buttons for Classic Highway and Arena Battle.
- Default: Classic. `?mode=arena` explicitly selects Arena; an invalid value falls back to Classic.
- Changing modes preserves song, instrument, and difficulty. Unsupported Arena selections remain visible and offer a separate “Use Arena demo setup” action; they are never silently replaced.
- States: default, selected, focus-visible, unsupported, loading, and disabled during an active run.

### Arena Anchor

- Shelter is the farthest position: a broad broken arch and shield glyph communicate low exposure and lower damage.
- Midline is the starting position: split standing stones and a balance glyph communicate the neutral profile.
- Spotlight is closest: a narrow open dais and starburst glyph communicate high exposure and amplified performance.
- States: current, reachable-safe, reachable-danger, targeted, traveling, impact, and disabled boundary. Shape, label, distance, and icon remain sufficient with hue removed.

### Phrase Constellation

- Structure: one complete ordered group of three to five stationary Perform symbols, optional Spotlight bonus symbols on a second compact row, and one stationary timing focus.
- Preview begins at least two authored beats early. All steps are visible together; only current and next receive emphasis.
- States: hidden, preview, ready, current, next, early, Perfect, Great, Good, Miss, complete, interrupted, and clear.
- Timing uses a contracting ring at a fixed location. Symbols never translate toward a strike line, so the component cannot become a miniature highway.

### Arena Combat Controls

- Three native buttons keep stable meanings: Retreat, Perform, Advance. Keyboard defaults are `D`/Left Arrow, `Space`/`F`, and `K`/Right Arrow.
- Each button contains a semantic SVG glyph, text label, and key hint. Touch size remains at least 48px with 8px gaps.
- States: ready, pressed, successful, flub, disabled boundary, unavailable outside a movement window, and focus-visible.
- Movement buttons confirm the choice immediately, then the character visibly completes travel before the authored impact. Perform acknowledges input immediately and may schedule contact to the authored beat.

### Boss Telegraph

- `Rift Sweep`: lateral trident wind-up, broad path geometry, scrape warning, rising rough charge, and horizontal impact. It never relies on violet alone.
- `Void Burst`: centered punch/inhalation pose, concentric target rings, crystalline pulse warning, inhaling charge, and radial impact.
- States: dormant, prepare, committed, impact, recovery, opening, phase transition, and defeated.
- Cue collision rule: a new phrase preview cannot begin during the critical final beat of either telegraph on Easy. Attack geometry stays behind phrase symbols and in front of arena atmosphere.

### Arena Fallback

- Static authored silhouette poster with boss, performer, and all three anchor shapes.
- Copy names the failed subsystem without raw diagnostics and offers Retry Arena, Play this selection in Classic, and Return to setup.
- The countdown and audio cannot start while the fallback is visible.

## 6. Motion & Interaction

| Token | Value | Usage |
|---|---|---|
| `--motion-press` | 90ms ease-out | Tap confirmation |
| `--motion-state` | 180ms ease-out | Button and card state |
| `--motion-screen` | 360ms cubic-bezier(0.16, 1, 0.3, 1) | Screen transition |
| `--motion-hit` | 220ms ease-out | Judgment and light strike |
| `--motion-attack` | beat-authored | Boss telegraph and attack |
| `--motion-dash` | 320ms cubic-bezier(0.2, 0.8, 0.2, 1) | Anchor-to-anchor travel that finishes before impact |
| `--motion-cue` | 140ms ease-out | Phrase and target state changes |
| `--motion-impact` | 180ms ease-out | Short semantic camera/material impulse |

Spatial movement is interruptible and uses transforms. Color and opacity use short
easings. Gameplay note travel is linear because it represents time, not decoration.
Sustain ribbons shorten linearly into the strike line and remain visible for the entire
required hold. This timing motion is essential gameplay information and remains enabled
under reduced motion.

Reduced motion removes camera shake, idle bob, particles, scale entrances, and repeated
pulsing. It preserves linear note travel, immediate press feedback, and static attack
telegraphs because those communicate essential gameplay state.

In Arena, reduced motion replaces dash travel with one concise cross-fade plus destination emphasis, removes camera impulses and long trails, and converts repeated beat scaling to direct brightness changes. The stationary timing ring, active symbol, target geometry, readable boss pose, and linear time-to-impact fill remain because they carry required information.

## 7. Depth & Surface

Strategy: mixed tonal depth with energy rims.

- Background depth comes from the generated arena art plus a dark vertical scrim.
- Panels use one inner highlight, one cyan-tinted edge, and one deep stage shadow.
- Gameplay chrome remains nearly flat so notes, attacks, and the boss own the depth.
- Radius system: 8px utility, 16px cards, 24px major panels, pill only for compact meters.
- Z layers: background 0, boss 10, highway 20, HUD 30, controls 40, modal 60.
- The ice-backdrop test places the viewer at the horizontal center of a real 3D
  `--ice-floor` gameplay disc whose matching visual ground continues beyond the
  playable radius and dissolves through `--ice-haze` into the panorama. Nearby
  dark rock and crystalline ice form the outer depth ring; distant mountain layers
  lose contrast and detail into the same glacier-blue color family. The panorama
  never bakes in a second platform, and the visual ground never exposes a hard rim.
- The panorama camera uses a moderately wide 74.5-degree vertical field of view so
  the surrounding rock ring reads spatially without fisheye stretching at the edges.
- Panorama masters remain strict 2:1 equirectangular images with a repaired longitude
  seam and smooth poles; no second dome warp is applied at runtime. The browser uses
  the 4096 × 2048 asset by default and reserves 8192 × 4096 for high-density desktop
  viewports with an 8K-capable WebGL texture limit.
- Babylon renders the backdrop at the device pixel ratio capped at 2×. This preserves
  Retina detail without allowing high-DPI mobile screens to create an unbounded canvas.

## 8. Accessibility Constraints & Accepted Debt

### Constraints

- WCAG 2.2 AA for menus, controls, HUD text, errors, and results.
- Full keyboard operation for selection, pause, gameplay lanes, retry, and results.
- Three lanes differ by label, shape, position, and color.
- Touch targets are at least 48px with 8px gaps and safe-area clearance.
- Browser zoom remains enabled. Selection and result copy reflow at 200% zoom.
- Live results use polite announcements; imminent attacks use concise assertive alerts.
- Automatic pause occurs when the page becomes hidden, with an explicit resume action.
- Audio-load and WebGL failures include recovery. WebGL failure falls back to the boss
  preview image while the Canvas highway remains playable.

### Inclusive personas

- Mobile commuter using one thumb: can reach every lane in portrait without stretching.
- Color-vision-deficient player: can distinguish lanes and judgments without hue.
- Motion-sensitive player: receives no shake, idle sway, or particle storm.
- Keyboard player: can complete the full loop without pointer input.
- Player interrupted by a call or tab switch: returns to a paused, synchronized battle.

### Accepted debt

| Item | Location | Why accepted | Exit |
|---|---|---|---|
| No screen-reader rhythm alternative | Active battle | A timed music chart is not meaningfully playable through serial speech output; menus and results remain accessible | Revisit with a dedicated audio-haptic accessibility mode |
| No user latency calibration | Settings | The first level uses forgiving windows and the Web Audio clock | Add calibration when multiple songs or harder charts ship |

## 9. Arena V2 Production Contract

### First vertical slice

- Song: Heaven's Edge.
- Instrument and difficulty: Drums, Easy.
- Scored segment: audio seconds 0 through 42, containing the authored 14.64–22.68 and 34.31–42.14 combat windows.
- Rehearsal: eight nonfatal seconds before scored audio. It teaches one three-step static Perform phrase and one Midline-to-Shelter reposition choice; rehearsal values reset before scoring.
- Direct deterministic route: `?mode=arena&qa=1`. Classic remains the default until every attention gate passes.

### Selected boss and animation semantics

The first boss is the Quaternius Ultimate Monsters `Demon`, sourced from the creator's official pack and distributed under CC0 1.0. The 1.26 MB self-contained source glTF has one 43-joint skin, a separate trident mesh, 4,261 vertices, 6,712 triangles, one 1024px atlas, and 14 animation clips. Its source package, license, runtime derivative, checksums, and semantic mapping are recorded under `roblox/assets/arena_v2/manifests/`.

Runtime semantic mapping:

| Arena state | Source clip | Reason |
|---|---|---|
| Intro | `Wave` | Broad readable entrance gesture |
| Beat-aware idle | `Idle` | Stable loop with visible silhouette |
| Rift Sweep telegraph | `No` | Lateral upper-body preparation |
| Rift Sweep attack | `Weapon` | Trident-led attack |
| Void Burst telegraph | `Jump_Idle` | Compact centered charge pose |
| Void Burst attack | `Punch` | Mechanically distinct body-led strike |
| Hit | `HitReact` | Short contact response |
| Stagger/opening | `Duck` | Sustained lowered vulnerable silhouette |
| Phase transition | `Yes` | Broad readable escalation gesture |
| Defeat | `Death` | Existing terminal clip |

The legacy `.blend` crashes Blender 5.2 during load, so the official self-contained glTF is the reproducible source of truth. Animation adaptation uses non-destructive runtime retiming and semantic aliases; the CC0 original remains preserved. Accepted visual debt: this pack is a deliberately modest first-slice asset and may be replaced after the mechanic passes attention testing, but it is not treated as a placeholder in the shipped slice.

### Camera, light, and cue hierarchy

- Fixed portrait camera with a 36–42 degree vertical field of view; no free orbit and no required off-screen target.
- Intro and climax may dolly no more than 8% of camera distance. Semantic impact translation is capped at 6px equivalent and never repeats within 180ms.
- Focus order is active phrase or reposition choice, boss telegraph and target, actor silhouettes, HUD consequence, then atmosphere.
- Player key light uses `--cyan`, boss rim uses `--violet`, neutral fill uses `--arena-fill`, and earned climax accents use `--gold`. Babylon reads these same CSS custom properties at runtime; the CSS root is the single color authority.
- Target paths stop at anchors and never run continuously from the boss to the timing focus. When phrase and attack states coexist, the attack target dims before the active phrase symbol does.

### Interaction reference

The existing beui.dev `action-swap` mechanism remains the loading/state-label reference. Arena's expanding/contracting timing focus adapts the interruptible state mechanism already documented for `expanding-arrow-button`: one fixed spatial object changes progress, immediate input retargets it, and reduced motion retains only linear informational progress.

### QA hardware and output

- Desktop: MacBook Pro `Mac17,8`, Apple M5 Pro (18 CPU / 20 GPU cores), 48 GB RAM, Chromium, 1280px QA capture; target 60 FPS.
- Mobile target: iPhone 15-class Safari at 393×852 physical CSS pixels; deterministic development coverage also runs Chromium at 375×812. Physical-device results remain required before default-mode promotion.
- Attention tests: wired USB-C EarPods at a fixed comfortable system volume, with the device and volume recorded per session. Bluetooth audio is excluded unless latency is calibrated.

### Arena-specific transfer budget

| Family | Budget |
|---|---:|
| Boss model, skin, and animations | 3.5 MB |
| Player model, prop, and animations | 2.0 MB |
| Environment geometry and textures | 2.0 MB |
| VFX textures, glyphs, and fallback art | 2.0 MB |
| P0/P1 runtime sound effects | 1.5 MB |
| UI and loading reserve | 1.0 MB |
| **Total** | **12.0 MB** |

### Arena accepted debt

| Item | Location | Why accepted | Exit |
|---|---|---|---|
| Quaternius source blend is not Blender 5.2-loadable | Asset source | Official glTF is complete, self-contained, CC0, and imports in the browser path | Replace or reconstruct an editable Blender derivative only if runtime retiming is insufficient |
| Physical mobile hardware result not yet recorded | QA evidence | Deterministic 375px browser coverage enables development | Run the named iPhone Safari performance and attention passes before default promotion |
| Real-participant attention evidence not yet available | Playtest evidence | Cannot be synthesized honestly | Complete three-person graybox and five-person final protocols before promotion |
