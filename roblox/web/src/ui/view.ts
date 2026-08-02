export const element = (id: string): HTMLElement => {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing element #${id}`);
  return found;
};

export interface HudValues {
  duration: number;
  time: number;
  playerHealth: number;
  score: number;
  combo: number;
  finalTriggered: boolean;
  charging: boolean;
}

export const renderHud = (values: HudValues): void => {
  const progress = Math.min(1, values.time / values.duration);
  const bossHealth = Math.max(
    values.finalTriggered ? 0 : 8,
    100 - progress * 92,
  );
  const playerHealth = Math.max(0, values.playerHealth);
  element("boss-meter").style.transform = `scaleX(${bossHealth / 100})`;
  element("player-meter").style.transform = `scaleX(${playerHealth / 100})`;
  element("boss-health").setAttribute(
    "aria-valuenow",
    String(Math.round(bossHealth)),
  );
  element("player-health").setAttribute(
    "aria-valuenow",
    String(Math.round(playerHealth)),
  );
  element("boss-status").textContent = values.finalTriggered
    ? "BANISHED"
    : values.charging
      ? "CHARGING"
      : progress > 0.68
        ? "FALTERING"
        : "IMMORTAL";
  element("combo").textContent = `${values.combo}×`;
  element("score").textContent = String(values.score).padStart(6, "0");
  const remaining = Math.max(0, Math.ceil(values.duration - values.time));
  element("timer").textContent =
    `${Math.floor(remaining / 60)}:${String(remaining % 60).padStart(2, "0")}`;
};

export const showCallout = (message: string, grade: string): void => {
  const callout = element("battle-callout");
  callout.textContent = message;
  callout.dataset.grade = grade;
  callout.classList.toggle("is-compact", message.length > 12);
};

export const showBattle = (): void => {
  const shell = document.querySelector<HTMLElement>(".game-shell");
  if (shell) shell.dataset.screen = "battle";
  element("select-screen").hidden = true;
  element("battle-screen").hidden = false;
  element("result-overlay").hidden = true;
  element("error-overlay").hidden = true;
};

export const showSelect = (): void => {
  const shell = document.querySelector<HTMLElement>(".game-shell");
  if (shell) shell.dataset.screen = "select";
  element("pause-overlay").hidden = true;
  element("result-overlay").hidden = true;
  element("battle-screen").hidden = true;
  element("select-screen").hidden = false;
  element("error-overlay").hidden = true;
  element("game-title").focus();
};

export const showLoadError = (message: string): void => {
  element("select-screen").hidden = true;
  element("battle-screen").hidden = true;
  element("error-message").textContent = message;
  element("error-overlay").hidden = false;
  element("error-title").focus();
};

export const showResult = (
  victory: boolean,
  score: number,
  accuracy: number,
  bestCombo: number,
): void => {
  element("result-kicker").textContent = victory
    ? "the edge holds"
    : "the void breaks through";
  element("result-title").textContent = victory ? "Victory" : "Fallen";
  element("result-copy").textContent = victory
    ? "Your final chord seals the breach."
    : "The Choir shattered your ward. Rise and try again.";
  element("result-score").textContent = score.toLocaleString();
  element("result-accuracy").textContent = `${Math.round(accuracy * 100)}%`;
  element("result-combo").textContent = `${bestCombo}×`;
  element("result-overlay").hidden = false;
  element("result-title").focus();
};
