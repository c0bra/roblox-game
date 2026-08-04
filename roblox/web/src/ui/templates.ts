import {
  chartDifficulties,
  difficultyDensity,
  difficultyLabels,
  type Instrument,
  instrumentLabels,
  instruments,
} from "../data/level";
import type { LevelCatalog } from "../data/level-catalog";

const instrumentIcons: Record<Instrument, string> = {
  drums: `<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="18" r="8"/><path d="m8 5 16 13M24 5 8 18"/></svg>`,
  vocals: `<svg viewBox="0 0 32 32" aria-hidden="true"><rect x="11" y="4" width="10" height="16" rx="5"/><path d="M7 16a9 9 0 0 0 18 0M16 25v4M11 29h10"/></svg>`,
  guitar: `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="m8 24 6-6 4 4-6 6H7l-3-3v-5l6-6 4 4M17 15 28 4M23 5l4 4"/></svg>`,
  bass: `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="m16 3 11 13-11 13L5 16 16 3Z"/><path d="M11 16h10M16 10v12"/></svg>`,
};

const instrumentSummaries: Record<Instrument, string> = {
  drums: "percussion · impact",
  vocals: "melody · soaring",
  guitar: "lead · fierce",
  bass: "low end · crushing",
};

const laneLabels = [
  "Left circle lane",
  "Center diamond lane",
  "Right triangle lane",
];

const escapeHtml = (value: string): string =>
  value.replace(
    /[&<>'"]/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "'": "&#39;",
        '"': "&quot;",
      })[character] ?? character,
  );

export const appShell = (catalog: LevelCatalog): string => `
  <main class="game-shell" data-screen="select">
    <div class="stage-art" aria-hidden="true"></div>
    <canvas class="boss-canvas" id="boss-canvas" aria-label="A three-headed supernatural enemy"></canvas>
    <div class="stage-vignette" aria-hidden="true"></div>

    <section class="select-screen" id="select-screen" aria-labelledby="game-title">
      <div class="eyebrow"><span></span> one night · one fate <span></span></div>
      <h1 id="game-title" tabindex="-1">Bands <em>Battle</em></h1>
      <p class="premise">Choose a song, then wield its drums, voice, guitar, or bass against the Fallen Choir.</p>
      <fieldset class="level-field">
        <legend>Choose song</legend>
        <div class="level-grid" role="radiogroup" aria-label="Choose song">
          ${catalog.levels
            .map(
              (level) => `
            <button class="level-option${level.id === catalog.defaultLevelId ? " is-selected" : ""}" type="button" role="radio" aria-checked="${level.id === catalog.defaultLevelId}" tabindex="${level.id === catalog.defaultLevelId ? 0 : -1}" data-level="${level.id}">
              <b>${escapeHtml(level.title)}</b>
            </button>`,
            )
            .join("")}
        </div>
      </fieldset>
      <div class="instrument-grid" role="radiogroup" aria-label="Choose your instrument">
        ${instruments
          .map(
            (instrument, index) => `
          <button class="instrument-card${index === 0 ? " is-selected" : ""}" type="button" role="radio" aria-checked="${index === 0}" tabindex="${index === 0 ? 0 : -1}" data-instrument="${instrument}">
            <span class="instrument-icon">${instrumentIcons[instrument]}</span>
            <span><b>${instrumentLabels[instrument]}</b><small>${instrumentSummaries[instrument]}</small></span>
            <span class="radio-mark"></span>
          </button>`,
          )
          .join("")}
      </div>
      <fieldset class="difficulty-field">
        <legend>Choose difficulty</legend>
        <div class="difficulty-grid" role="radiogroup" aria-label="Choose difficulty">
          ${chartDifficulties
            .map(
              (difficulty, index) => `
            <button class="difficulty-option${index === 0 ? " is-selected" : ""}" type="button" role="radio" aria-checked="${index === 0}" tabindex="${index === 0 ? 0 : -1}" data-difficulty="${difficulty}">
              <b>${difficultyLabels[difficulty]}</b><small>${difficultyDensity[difficulty]}</small>
            </button>`,
            )
            .join("")}
        </div>
      </fieldset>
      <button class="primary-action" id="start-button" type="button">
        <span class="action-label">Enter the breach</span><span class="action-loading">Summoning…</span>
      </button>
      <p class="microcopy"><kbd>Tap or hold</kbd> the three sigils as notes meet the lightline. Long ribbons last for the sung or played note.</p>
    </section>

    <section class="battle-screen" id="battle-screen" aria-label="Battle" tabindex="-1" hidden>
      <header class="hud">
        <div class="health-block boss-health"><span><b>THE FALLEN CHOIR</b><small id="boss-status">IMMORTAL</small></span><div class="meter" id="boss-health" role="progressbar" aria-label="Boss health" aria-valuemin="0" aria-valuemax="100" aria-valuenow="100"><i id="boss-meter"></i></div></div>
        <button class="icon-button" id="pause-button" type="button" aria-label="Pause">Ⅱ</button>
        <div class="health-block player-health"><span><b id="player-name">WAR DRUMS</b><small>WARD</small></span><div class="meter" id="player-health" role="progressbar" aria-label="Player ward" aria-valuemin="0" aria-valuemax="100" aria-valuenow="100"><i id="player-meter"></i></div></div>
      </header>
      <div class="battle-callout" id="battle-callout" role="status" aria-live="polite"></div>
      <div class="scoreline"><span id="combo">0×</span><strong id="score">000000</strong><time id="timer">1:30</time></div>
      <canvas class="highway-canvas" id="highway-canvas"></canvas>
      <div class="tap-pads" aria-label="Note controls">
        ${[0, 1, 2].map((lane) => `<button class="tap-pad lane-${lane}" type="button" data-lane="${lane}" aria-label="${laneLabels[lane]}; tap short notes and hold long notes"><span></span></button>`).join("")}
      </div>
    </section>

    <section class="overlay-card" id="pause-overlay" aria-labelledby="pause-title" hidden>
      <p class="eyebrow">the breach waits</p><h2 id="pause-title" tabindex="-1">Battle paused</h2>
      <button class="primary-action" id="resume-button" type="button"><span>Return to battle</span></button>
      <button class="text-button" id="quit-button" type="button">Abandon run</button>
    </section>

    <section class="overlay-card result-card" id="result-overlay" aria-labelledby="result-title" hidden>
      <p class="eyebrow" id="result-kicker">the edge holds</p><h2 id="result-title" tabindex="-1">Victory</h2>
      <p id="result-copy">Your final chord seals the breach.</p>
      <div class="result-stats"><span><b id="result-score">0</b>score</span><span><b id="result-accuracy">0%</b>accuracy</span><span><b id="result-combo">0×</b>best combo</span></div>
      <button class="primary-action" id="replay-button" type="button"><span>Play again</span></button>
      <button class="text-button" id="change-button" type="button">Change song or instrument</button>
    </section>

    <section class="overlay-card" id="error-overlay" aria-labelledby="error-title" hidden>
      <p class="eyebrow">signal lost</p><h2 id="error-title" tabindex="-1">The song fell silent</h2>
      <p id="error-message">The level could not be loaded.</p>
      <button class="primary-action" id="retry-button" type="button"><span>Try again</span></button>
      <button class="text-button" id="error-back-button" type="button">Choose another song</button>
    </section>
  </main>`;
