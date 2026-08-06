# Classic Highway Regression Baseline

Recorded 2026-08-05 before Arena implementation.

## Automated baseline

| Check | Result |
|---|---|
| `bun test` | 35 passed, 0 failed |
| `bun run check` | Passed; existing Biome configuration deprecation notice only |
| `bun run build` | Passed; existing lazy boss chunk is 6,275 kB raw / 1,371 kB gzip |

## Browser baseline

- Route: `http://127.0.0.1:4173/?qa=1`.
- Browser: Playwright Chromium on MacBook Pro `Mac17,8`, Apple M5 Pro, 48 GB RAM.
- Observed flow: default selection, start, countdown, active Classic battle at 0:07, lane controls, HUD, boss canvas, pause controls.
- Console: Babylon WebGL startup only; no errors.
- Screenshot: `roblox/web/output/playwright/classic-baseline-qa.png`.
- Console artifact: `roblox/web/.playwright-cli/console-2026-08-05T21-01-38-452Z.log`.

Classic behavior is the comparison contract. Arena may add a lifecycle boundary and mode-aware bootstrap, but may not change Classic chart loading, note timing, tap/sustain judgment, attack resolution, selection defaults, or results calculations.
