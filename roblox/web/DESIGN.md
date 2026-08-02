# Heaven's Edge Design System

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

Heaven's Edge is a supernatural concert staged at the last threshold between a cold,
ordered heaven and an eroding violet void. The interface feels like a premium console
game compressed into a phone: calm chrome, severe contrast, and explosive feedback only
when a note or attack earns it. The signature is the "light line": every successful tap
pulls a narrow white-gold strike from the player through the highway into the boss.

Design dials: `DESIGN_VARIANCE 7`, `MOTION_INTENSITY 7`, `VISUAL_DENSITY 7`.

## 2. Color

The game is intentionally dark-theme only. Lane colors are semantic input channels and
do not count as decorative accents.

| Role | Token | Value | Usage |
|---|---|---:|---|
| Void | `--surface-void` | `#05070d` | Page and WebGL fallback |
| Stage | `--surface-stage` | `#0a1020` | Battle shell |
| Panel | `--surface-panel` | `rgb(10 16 32 / 0.82)` | Selection and result surfaces |
| Panel strong | `--surface-panel-strong` | `#10182d` | Loading and error states |
| Text | `--text-primary` | `#f7f9ff` | Titles and HUD |
| Muted text | `--text-secondary` | `#b7c2d9` | Instructions and metadata |
| Dim text | `--text-tertiary` | `#7f8ba4` | Disabled and secondary metrics |
| Player | `--energy-player` | `#7ce8ff` | Focus, hits, boss damage |
| Heaven | `--energy-heaven` | `#ffe6a3` | Full hype and victory only |
| Corruption | `--energy-void` | `#a15cff` | Boss telegraphs and danger |
| Danger | `--status-danger` | `#ff5470` | Health loss and error |
| Lane left | `--lane-left` | `#55d8ff` | Diamond lane |
| Lane center | `--lane-center` | `#ffe08a` | Circle lane |
| Lane right | `--lane-right` | `#bf83ff` | Square lane |
| Edge | `--border-energy` | `rgb(124 232 255 / 0.36)` | Focus and selected outlines |
| Veil | `--veil` | `rgb(2 4 10 / 0.72)` | Readability scrim |

Rules:

- Text meets WCAG 2.2 AA against its rendered surface.
- Danger, judgment, and lanes always include a word or shape, never color alone.
- Player cyan is the resting action accent. Gold appears only at full hype and victory.
- Violet belongs to the boss and selected right lane, distinguished by geometry.

## 3. Typography

### Font stack

- Display and compact HUD: `Oxanium`, system sans fallback.
- Reading and controls: `Atkinson Hyperlegible`, system sans fallback.
- Both are self-hosted WOFF2 with `font-display: swap` and OFL licensing.

### Scale

| Role | Token | Size | Weight | Line height | Tracking |
|---|---|---:|---:|---:|---:|
| Title | `--type-title` | `clamp(2.5rem, 11vw, 5.5rem)` | 600 | 0.94 | -0.04em |
| Screen heading | `--type-h1` | `clamp(1.7rem, 7vw, 3rem)` | 500 | 1.05 | -0.025em |
| Card heading | `--type-h2` | `1.25rem` | 600 | 1.15 | 0 |
| HUD value | `--type-hud` | `clamp(0.9rem, 4vw, 1.2rem)` | 600 | 1 | 0.02em |
| Body | `--type-body` | `1rem` | 400 | 1.5 | 0 |
| Label | `--type-label` | `0.82rem` | 600 | 1.2 | 0.06em |
| Micro | `--type-micro` | `0.75rem` | 500 | 1.3 | 0.04em |

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

The app fills `100dvh`, reserves safe-area insets, and never scrolls during battle.
Portrait is primary. At 768px and above the stage is centered inside a maximum 760px
play column with atmospheric side space. Selection and results may scroll vertically.

Battle composition:

- Top 18%: compact boss and player status.
- Upper-middle 32%: boss silhouette and attack telegraph.
- Lower-middle 32%: perspective note highway.
- Bottom 18%: three equal tap pads above the gesture safe area.

## 5. Components

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
- States: ready, pressed, hit, miss, disabled.
- Touch: fills one-third of the bottom control row with an 8px gap; no precision gesture.
- Motion: immediate 0.96 press scale; hit emits an inward ring; miss flashes a static
  danger rim under reduced motion.

### Game Screen

- Variants: selecting, loading, countdown, playing, paused, won, lost, error.
- Each state has one primary action and a visible recovery path where applicable.
- Focus moves to the state heading or primary action after a state transition.

## 6. Motion & Interaction

| Token | Value | Usage |
|---|---|---|
| `--motion-press` | 90ms ease-out | Tap confirmation |
| `--motion-state` | 180ms ease-out | Button and card state |
| `--motion-screen` | 360ms cubic-bezier(0.16, 1, 0.3, 1) | Screen transition |
| `--motion-hit` | 220ms ease-out | Judgment and light strike |
| `--motion-attack` | beat-authored | Boss telegraph and attack |

Spatial movement is interruptible and uses transforms. Color and opacity use short
easings. Gameplay note travel is linear because it represents time, not decoration.

Reduced motion removes camera shake, idle bob, particles, scale entrances, and repeated
pulsing. It preserves linear note travel, immediate press feedback, and static attack
telegraphs because those communicate essential gameplay state.

## 7. Depth & Surface

Strategy: mixed tonal depth with energy rims.

- Background depth comes from the generated arena art plus a dark vertical scrim.
- Panels use one inner highlight, one cyan-tinted edge, and one deep stage shadow.
- Gameplay chrome remains nearly flat so notes, attacks, and the boss own the depth.
- Radius system: 8px utility, 16px cards, 24px major panels, pill only for compact meters.
- Z layers: background 0, boss 10, highway 20, HUD 30, controls 40, modal 60.

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
