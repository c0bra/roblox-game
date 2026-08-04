import "@fontsource-variable/atkinson-hyperlegible-next";
import "@fontsource-variable/oxanium";
import "./styles.css";
import { BattleAudio } from "./audio/battle-audio";
import { levelCatalog } from "./data/level-catalog";
import { createBrowserRunLoader } from "./data/level-loader";
import { GameController } from "./game/controller";
import { SelectionControls } from "./game/selection-controls";
import { appShell } from "./ui/templates";

const app = document.getElementById("app");
if (!app) throw new Error("Missing application root");
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
const selection = new SelectionControls(levelCatalog);
const loadRun = createBrowserRunLoader(
  levelCatalog,
  audio,
  new URLSearchParams(location.search).has("qa"),
);
const controller = new GameController(
  bossCanvas,
  highwayCanvas,
  selection,
  audio,
  loadRun,
);
void controller.mount();
