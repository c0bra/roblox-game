import { arenaGlyphs } from "./arena-glyphs";
import type { ArenaRunState } from "./combat";
import type { ArenaEncounter, ArenaPositionId } from "./encounter";
import { deriveArenaPresentation } from "./presentation";

const required = <T extends Element>(root: ParentNode, selector: string): T => {
  const element = root.querySelector<T>(selector);
  if (!element) throw new Error(`Missing Arena element: ${selector}`);
  return element;
};

const setHidden = (element: HTMLElement, hidden: boolean): void => {
  element.hidden = hidden;
};

const percent = (value: number, maximum: number): number =>
  Math.max(0, Math.min(100, (value / Math.max(1, maximum)) * 100));

const attackCopy = {
  sweep: {
    name: "Rift Sweep",
    response: "Retreat to Shelter",
    glyph: arenaGlyphs.sweep,
  },
  burst: {
    name: "Void Burst",
    response: "Advance to Spotlight",
    glyph: arenaGlyphs.burst,
  },
} as const;

export class ArenaView {
  private readonly setup: HTMLElement;
  private readonly battle: HTMLElement;
  private readonly loading: HTMLElement;
  private readonly pauseOverlay: HTMLElement;
  private readonly result: HTMLElement;
  private readonly fallback: HTMLElement;
  private readonly callout: HTMLElement;
  private readonly phrase: HTMLElement;
  private readonly attack: HTMLElement;
  private calloutTimer = 0;

  constructor(private readonly root: HTMLElement) {
    this.setup = required(root, "#arena-setup");
    this.battle = required(root, "#arena-battle");
    this.loading = required(root, "#arena-loading");
    this.pauseOverlay = required(root, "#arena-pause-overlay");
    this.result = required(root, "#arena-result");
    this.fallback = required(root, "#arena-fallback");
    this.callout = required(root, "#arena-callout");
    this.phrase = required(root, "#arena-phrase");
    this.attack = required(root, "#arena-attack-banner");
  }

  showLoading(percentValue: number, stage: string): void {
    setHidden(this.setup, true);
    setHidden(this.loading, false);
    const track = required<HTMLElement>(this.root, ".arena-load-track");
    track.setAttribute("aria-valuenow", String(Math.round(percentValue)));
    required<HTMLElement>(this.root, "#arena-load-fill").style.transform =
      `scaleX(${percentValue / 100})`;
    required(this.root, "#arena-load-stage").textContent = stage;
  }

  showBattle(): void {
    for (const overlay of [
      this.loading,
      this.pauseOverlay,
      this.result,
      this.fallback,
    ]) {
      setHidden(overlay, true);
    }
    setHidden(this.setup, true);
    setHidden(this.battle, false);
    this.root.dataset.arenaScreen = "battle";
    this.battle.focus();
  }

  showSetup(): void {
    setHidden(this.battle, true);
    for (const overlay of [
      this.loading,
      this.pauseOverlay,
      this.result,
      this.fallback,
    ]) {
      setHidden(overlay, true);
    }
    setHidden(this.setup, false);
    this.root.dataset.arenaScreen = "setup";
    required<HTMLElement>(this.root, "#arena-title").focus();
  }

  showPause(): void {
    setHidden(this.pauseOverlay, false);
    required<HTMLElement>(this.root, "#arena-pause-title").focus();
  }

  hidePause(): void {
    setHidden(this.pauseOverlay, true);
    this.battle.focus();
  }

  showFallback(message: string): void {
    setHidden(this.loading, true);
    setHidden(this.battle, true);
    setHidden(this.fallback, false);
    required(this.root, "#arena-fallback-message").textContent = message;
    required<HTMLElement>(this.root, "#arena-fallback-title").focus();
  }

  showResult(state: ArenaRunState, initialResolve: number): void {
    const won = state.phase === "victory";
    const wardDefeat = state.phase === "ward-defeat";
    required(this.root, "#arena-result-kicker").textContent = won
      ? "Threshold sealed"
      : "The Rift endures";
    required(this.root, "#arena-result-title").textContent = won
      ? "Victory"
      : wardDefeat
        ? "Ward shattered"
        : "Resolve held";
    required(this.root, "#arena-result-copy").textContent = won
      ? "Your performance broke the demon's resolve."
      : wardDefeat
        ? "The demon broke your ward before the final cadence."
        : "The final cadence passed before the demon yielded.";
    this.output("arena-result-score", state.score.toLocaleString());
    this.output(
      "arena-result-accuracy",
      `${Math.round(state.accuracy * 100)}%`,
    );
    this.output("arena-result-streak", `${state.bestStreak}×`);
    this.output(
      "arena-result-resolve",
      `${Math.round(percent(initialResolve - state.bossResolve, initialResolve))}%`,
    );
    this.output("arena-result-ward", String(100 - state.ward));
    this.output("arena-result-exposure", state.exposure.toFixed(1));
    setHidden(this.result, false);
    required<HTMLElement>(this.root, "#arena-result-title").focus();
  }

  render(encounter: ArenaEncounter, state: ArenaRunState, time: number): void {
    const presentation = deriveArenaPresentation(encounter, state, time);
    this.meter("arena-resolve", state.bossResolve, encounter.initialResolve);
    this.meter("arena-ward", state.ward, encounter.initialWard);
    this.output("arena-score", state.score.toString().padStart(6, "0"));
    this.output(
      "arena-accuracy",
      `${Math.round(state.accuracy * 100)}% accuracy`,
    );
    let phaseIndex = 0;
    for (const [index, candidate] of encounter.phases.entries()) {
      if (candidate.start > time) break;
      phaseIndex = index;
    }
    const phase = encounter.phases[phaseIndex];
    required(this.root, "#arena-phase-label").textContent =
      ["I", "II", "III"][phaseIndex] ?? "III";
    required(this.root, "#arena-phase-name").textContent =
      phase?.id ?? "Opening";
    this.renderPositions(presentation.positions, state.position);
    this.renderAttack(presentation.activeAttack);
    this.renderPhrase(encounter, state, time);
  }

  announce(message: string, grade = ""): void {
    window.clearTimeout(this.calloutTimer);
    this.callout.textContent = message;
    this.callout.dataset.grade = grade;
    this.calloutTimer = window.setTimeout(() => {
      this.callout.textContent = "";
      this.callout.dataset.grade = "";
    }, 760);
  }

  dispose(): void {
    window.clearTimeout(this.calloutTimer);
  }

  private renderPositions(
    positions: ReturnType<typeof deriveArenaPresentation>["positions"],
    current: ArenaPositionId,
  ): void {
    for (const position of positions) {
      const element = required<HTMLElement>(
        this.root,
        `[data-position="${position.id}"]`,
      );
      element.className = `${position.current ? "is-current " : ""}is-${position.state}`;
      element.setAttribute(
        "aria-current",
        position.id === current ? "true" : "false",
      );
      const detail = element.querySelector("small");
      if (detail) {
        detail.textContent =
          position.state === "danger"
            ? "Targeted · move"
            : position.state === "safe"
              ? "Safe now"
              : position.current
                ? "Current"
                : position.id === "shelter"
                  ? "Guarded"
                  : position.id === "spotlight"
                    ? "Exposed · 1.35×"
                    : "Balanced";
      }
    }
    const order: readonly ArenaPositionId[] = [
      "shelter",
      "midline",
      "spotlight",
    ];
    const index = order.indexOf(current);
    required<HTMLButtonElement>(
      this.root,
      '[data-arena-action="retreat"]',
    ).disabled = index === 0;
    required<HTMLButtonElement>(
      this.root,
      '[data-arena-action="advance"]',
    ).disabled = index === order.length - 1;
  }

  private renderAttack(
    attack: ReturnType<typeof deriveArenaPresentation>["activeAttack"],
  ): void {
    setHidden(this.attack, !attack);
    if (!attack) return;
    const copy = attackCopy[attack.type];
    required(this.root, "#arena-attack-glyph").innerHTML = copy.glyph;
    required(this.root, "#arena-attack-name").textContent = copy.name;
    required(this.root, "#arena-attack-response").textContent =
      attack.phase === "recovery" ? "Opening · Perform now" : copy.response;
    required<HTMLElement>(this.root, "#arena-attack-time").style.transform =
      `scaleX(${attack.progress})`;
    this.attack.dataset.phase = attack.phase;
  }

  private renderPhrase(
    encounter: ArenaEncounter,
    state: ArenaRunState,
    time: number,
  ): void {
    const progress = state.phraseProgress;
    setHidden(this.phrase, !progress);
    if (!progress) return;
    const phrase = encounter.phrases.find(({ id }) => id === progress.phraseId);
    if (!phrase) return;
    required(this.root, "#arena-phrase-status").textContent = progress.status;
    const steps = [
      ...phrase.steps,
      ...(phrase.positionBonusSteps.find(
        ({ positionId }) => positionId === state.position,
      )?.steps ?? []),
    ].sort((left, right) => left.time - right.time);
    required(this.root, "#arena-phrase-steps").innerHTML = steps
      .map((step) => {
        const resolved = state.resolvedStepIds.includes(step.id);
        const mode = resolved
          ? state.lastJudgment?.stepId === step.id
            ? state.lastJudgment.grade
            : "perfect"
          : step.id === progress.currentStepId
            ? "current"
            : step.id === progress.nextStepId
              ? "next"
              : "preview";
        const bonus = !phrase.steps.some(({ id }) => id === step.id);
        return `<span class="phrase-step is-${mode}${bonus ? " is-bonus" : ""}" aria-label="${bonus ? "Bonus " : ""}Perform step ${mode}">${arenaGlyphs.perform}<b>${bonus ? "Bonus" : mode}</b></span>`;
      })
      .join("");
    const current = steps.find(({ id }) => id === progress.currentStepId);
    const timing = current
      ? 1 - Math.max(0, Math.min(1, current.time - time))
      : 1;
    required<HTMLElement>(this.root, "#arena-phrase-timing").style.setProperty(
      "--timing",
      String(timing),
    );
  }

  private meter(id: string, value: number, maximum: number): void {
    const current = Math.round(percent(value, maximum));
    const element = required<HTMLElement>(this.root, `#${id}`);
    element.setAttribute("aria-valuenow", String(current));
    required<HTMLElement>(this.root, `#${id}-fill`).style.transform =
      `scaleX(${current / 100})`;
    this.output(`${id}-value`, current);
  }

  private output(id: string, value: string | number): void {
    required(this.root, `#${id}`).textContent = String(value);
  }
}
