import { arenaGlyphs } from "./arena-glyphs";
import { arenaShowcase } from "./showcase-template";

const modeSelector = (): string => `
  <div class="mode-selector" role="radiogroup" aria-label="Gameplay mode">
    <button type="button" role="radio" aria-checked="false" data-mode="classic">
      <span class="mode-index">01</span><b>Classic Highway</b><small>Continuous three-lane chart</small>
    </button>
    <button class="is-selected" type="button" role="radio" aria-checked="true" data-mode="arena">
      <span class="mode-index">02</span><b>Arena Battle</b><small>Static phrases · tactical anchors</small>
    </button>
  </div>`;

const meter = (
  id: string,
  label: string,
  value: number,
  modifier: string,
): string => `
  <div class="arena-meter ${modifier}">
    <span><b>${label}</b><output id="${id}-value">${value}</output></span>
    <div role="progressbar" id="${id}" aria-label="${label}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${value}"><i id="${id}-fill"></i></div>
  </div>`;

const controls = (): string => `
  <div class="arena-controls" aria-label="Arena combat controls">
    <button type="button" data-arena-action="retreat" aria-label="Retreat toward Shelter">
      ${arenaGlyphs.retreat}<span><b>Retreat</b><kbd>D · ←</kbd></span>
    </button>
    <button class="perform-control" type="button" data-arena-action="perform" aria-label="Perform on the active beat">
      ${arenaGlyphs.perform}<span><b>Perform</b><kbd>Space · F</kbd></span>
    </button>
    <button type="button" data-arena-action="advance" aria-label="Advance toward Spotlight">
      ${arenaGlyphs.advance}<span><b>Advance</b><kbd>K · →</kbd></span>
    </button>
  </div>`;

export const arenaShell = (): string => `
  <main class="arena-shell" data-arena-screen="setup">
    <div class="arena-atmosphere" aria-hidden="true"></div>
    <canvas id="arena-canvas" class="arena-canvas" aria-label="The Rift Performer facing a demon boss across three tactical positions"></canvas>

    <section class="arena-setup" id="arena-setup" aria-labelledby="arena-title">
      <p class="arena-kicker">Experimental performance combat</p>
      <h1 id="arena-title" tabindex="-1">Face the <em>Rift</em></h1>
      <p class="arena-lede">Read the boss, choose your distance, and perform short phrases on the beat. The battle stays in view.</p>
      <div class="arena-setup-preview" aria-hidden="true">
        <span class="setup-preview-boss">${arenaGlyphs.burst}</span>
        <i class="setup-preview-axis"></i>
        <div class="setup-preview-anchors">
          <span>${arenaGlyphs.shelter}</span>
          <span class="is-current">${arenaGlyphs.midline}</span>
          <span>${arenaGlyphs.spotlight}</span>
        </div>
        <span class="setup-preview-performer">${arenaGlyphs.perform}</span>
      </div>
      ${modeSelector()}
      <article class="arena-demo-card">
        <span class="arena-demo-number">V2 / 01</span>
        <div><b>Heaven's Edge</b><small>War Drums · Easy · 42-second encounter</small></div>
        <span class="arena-ready">Ready</span>
      </article>
      <div class="arena-unsupported" id="arena-unsupported" hidden>
        <b>Arena is not authored for that setup.</b>
        <p>Your current selection is preserved.</p>
        <button id="arena-use-demo" type="button">Use Arena demo setup</button>
      </div>
      <button class="arena-primary" id="arena-start" type="button">
        <span>Enter the threshold</span><small>Rehearsal begins first</small>
      </button>
      <a class="arena-text-link" id="arena-classic-setup" href="?mode=classic">Return to Classic Highway</a>
    </section>

    <section class="arena-battle" id="arena-battle" tabindex="-1" aria-label="Arena battle" hidden>
      <header class="arena-hud">
        ${meter("arena-resolve", "Boss Resolve", 100, "is-resolve")}
        <div class="arena-phase"><span id="arena-phase-label">I</span><small id="arena-phase-name">Opening</small></div>
        <button class="arena-pause" id="arena-pause" type="button" aria-label="Pause battle"><span></span><span></span></button>
        ${meter("arena-ward", "Performer Ward", 100, "is-ward")}
        <div class="arena-score"><output id="arena-score">000000</output><small id="arena-accuracy">0% accuracy</small></div>
      </header>
      <div class="arena-callout" id="arena-callout" role="status" aria-live="polite"></div>
      <div class="attack-banner" id="arena-attack-banner" hidden>
        <span id="arena-attack-glyph">${arenaGlyphs.sweep}</span>
        <span><b id="arena-attack-name">Rift Sweep</b><small id="arena-attack-response">Choose a safe anchor</small></span>
        <i id="arena-attack-time"></i>
      </div>
      <div class="phrase-constellation" id="arena-phrase" aria-label="Performance phrase" hidden>
        <div class="phrase-heading"><span id="arena-phrase-status">Preview</span><small>Complete phrase</small></div>
        <div class="phrase-steps" id="arena-phrase-steps"></div>
        <div class="phrase-timing" id="arena-phrase-timing" aria-hidden="true"><i></i></div>
      </div>
      <div class="anchor-labels" aria-label="Tactical positions">
        <span data-position="shelter">${arenaGlyphs.shelter}<b>Shelter</b><small>Guarded</small></span>
        <span data-position="midline" class="is-current">${arenaGlyphs.midline}<b>Midline</b><small>Balanced</small></span>
        <span data-position="spotlight">${arenaGlyphs.spotlight}<b>Spotlight</b><small>Exposed · 1.35×</small></span>
      </div>
      ${controls()}
    </section>

    <section class="arena-overlay arena-loading" id="arena-loading" aria-labelledby="arena-loading-title" hidden>
      <p class="arena-kicker">Opening the threshold</p><h2 id="arena-loading-title" tabindex="-1">Loading Arena</h2>
      <div class="arena-load-track" role="progressbar" aria-label="Arena loading progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><i id="arena-load-fill"></i></div>
      <p id="arena-load-stage">Preparing encounter…</p>
      <button class="arena-text-button" id="arena-load-cancel" type="button">Cancel</button>
    </section>

    <section class="arena-overlay" id="arena-pause-overlay" aria-labelledby="arena-pause-title" hidden>
      <p class="arena-kicker">Audio and battle locked</p><h2 id="arena-pause-title" tabindex="-1">Arena paused</h2>
      <button class="arena-primary" id="arena-resume" type="button"><span>Resume on beat</span></button>
      <button class="arena-text-button" id="arena-exit" type="button">Exit to setup</button>
    </section>

    <section class="arena-overlay" id="arena-result" aria-labelledby="arena-result-title" hidden>
      <p class="arena-kicker" id="arena-result-kicker">Threshold sealed</p><h2 id="arena-result-title" tabindex="-1">Victory</h2>
      <p id="arena-result-copy">Your performance broke the demon's resolve.</p>
      <div class="arena-results-grid">
        <span><output id="arena-result-score">0</output><small>Score</small></span>
        <span><output id="arena-result-accuracy">0%</output><small>Accuracy</small></span>
        <span><output id="arena-result-streak">0×</output><small>Best streak</small></span>
        <span><output id="arena-result-resolve">0%</output><small>Resolve broken</small></span>
        <span><output id="arena-result-ward">0</output><small>Ward damage</small></span>
        <span><output id="arena-result-exposure">0</output><small>Exposure</small></span>
      </div>
      <button class="arena-primary" id="arena-replay" type="button"><span>Replay encounter</span></button>
      <button class="arena-text-button" id="arena-result-exit" type="button">Change mode</button>
    </section>

    <section class="arena-overlay arena-fallback" id="arena-fallback" aria-labelledby="arena-fallback-title" hidden>
      <div class="fallback-silhouette" aria-hidden="true"><i></i><span></span><b></b></div>
      <p class="arena-kicker">The threshold did not open</p><h2 id="arena-fallback-title" tabindex="-1">Arena unavailable</h2>
      <p id="arena-fallback-message">The battle scene could not be initialized.</p>
      <button class="arena-primary" id="arena-retry" type="button"><span>Retry Arena</span></button>
      <a class="arena-text-link" id="arena-classic-recovery" href="?mode=classic">Play this setup in Classic</a>
    </section>
  </main>`;

export { arenaShowcase };
