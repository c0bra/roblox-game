import "@fontsource-variable/atkinson-hyperlegible-next";
import "@fontsource-variable/oxanium";
import "./styles.css";
import { GameController } from "./game/controller";
import { appShell } from "./ui/templates";

const app = document.getElementById("app");
if (!app) throw new Error("Missing application root");
app.innerHTML = appShell();

const bossCanvas = document.getElementById("boss-canvas");
const highwayCanvas = document.getElementById("highway-canvas");
if (
  !(bossCanvas instanceof HTMLCanvasElement) ||
  !(highwayCanvas instanceof HTMLCanvasElement)
) {
  throw new Error("Battle canvases are unavailable");
}

const controller = new GameController(bossCanvas, highwayCanvas);
void controller.mount();
