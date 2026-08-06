import "@fontsource-variable/atkinson-hyperlegible-next";
import "@fontsource-variable/oxanium";
import "./styles.css";
import { arenaShell, arenaShowcase } from "./arena/arena-template";
import { BattleAudio } from "./audio/battle-audio";
import {
  backdropIdFromSearchParams,
  backdropShell,
} from "./backdrop/backdrop-preview";
import { levelCatalog } from "./data/level-catalog";
import { createBrowserRunLoader } from "./data/level-loader";
import { GameController } from "./game/controller";
import {
  arenaDemoSelection,
  modeFromSearchParams,
  selectionFromSearchParams,
} from "./game/game-mode";
import { SelectionControls } from "./game/selection-controls";
import { appShell } from "./ui/templates";

const app = document.getElementById("app");
if (!app) throw new Error("Missing application root");
const params = new URLSearchParams(location.search);
const backdropId = backdropIdFromSearchParams(params);
const mode = modeFromSearchParams(params);

if (backdropId === "ice") {
  app.innerHTML = backdropShell();
  const canvas = document.getElementById("backdrop-canvas");
  const resetButton = document.getElementById("backdrop-reset");
  const root = document.querySelector<HTMLElement>(".backdrop-preview");
  const status = document.getElementById("backdrop-status");
  if (
    !(canvas instanceof HTMLCanvasElement) ||
    !(resetButton instanceof HTMLButtonElement) ||
    !root ||
    !status
  ) {
    throw new Error("Backdrop preview surface is unavailable");
  }
  const { IceBackdropViewer } = await import("./backdrop/backdrop-scene");
  const viewer = new IceBackdropViewer({ canvas, resetButton, root, status });
  window.addEventListener("pagehide", () => viewer.dispose(), { once: true });
} else if (mode === "arena" && params.has("showcase")) {
  app.innerHTML = arenaShowcase();
} else if (mode === "arena") {
  app.innerHTML = arenaShell();
  const canvas = document.getElementById("arena-canvas");
  const root = document.querySelector<HTMLElement>(".arena-shell");
  if (!(canvas instanceof HTMLCanvasElement) || !root) {
    throw new Error("Arena surface is unavailable");
  }
  const { ArenaController } = await import("./arena/arena-controller");
  const { createBrowserArenaLoader } = await import("./arena/encounter-loader");
  const controller = new ArenaController(
    root,
    canvas,
    new BattleAudio(),
    createBrowserArenaLoader(params.has("qa")),
    selectionFromSearchParams(params, arenaDemoSelection),
  );
  await controller.mount();
  window.addEventListener("pagehide", () => controller.dispose(), {
    once: true,
  });
} else {
  app.innerHTML = appShell(levelCatalog);

  const bossCanvas = document.getElementById("boss-canvas");
  const highwayCanvas = document.getElementById("highway-canvas");
  if (
    !(bossCanvas instanceof HTMLCanvasElement) ||
    !(highwayCanvas instanceof HTMLCanvasElement)
  ) {
    throw new Error("Battle canvases are unavailable");
  }

  const audio = new BattleAudio();
  const selection = new SelectionControls(
    levelCatalog,
    selectionFromSearchParams(params, arenaDemoSelection),
  );
  const loadRun = createBrowserRunLoader(levelCatalog, audio, params.has("qa"));
  const controller = new GameController(
    bossCanvas,
    highwayCanvas,
    selection,
    audio,
    loadRun,
  );
  document
    .getElementById("arena-mode-button")
    ?.addEventListener("click", () => {
      const current = selection.current;
      const query = new URLSearchParams({
        mode: "arena",
        level: current.levelId,
        instrument: current.instrument,
        difficulty: current.difficulty,
      });
      location.href = `?${query.toString()}`;
    });
  await controller.mount();
  window.addEventListener("pagehide", () => controller.dispose(), {
    once: true,
  });
}
