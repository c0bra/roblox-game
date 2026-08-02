import ky from "ky";
import { BattleAudio } from "../audio/battle-audio";
import {
  type ChartDifficulty,
  chartDifficulties,
  chartPath,
  chartSchema,
  defaultDifficulty,
  type Instrument,
  instrumentLabels,
  instruments,
  type Lane,
  type LevelChart,
} from "../data/level";
import { HighwayRenderer } from "../render/highway";
import {
  element,
  renderHud,
  showBattle,
  showCallout,
  showLoadError,
  showResult,
  showSelect,
} from "../ui/view";
import { judgeTap, resolveAttackWindow, scoreForGrade } from "./judgement";

const isInstrument = (value: string | undefined): value is Instrument =>
  instruments.some((instrument) => instrument === value);

const isDifficulty = (value: string | undefined): value is ChartDifficulty =>
  chartDifficulties.some((difficulty) => difficulty === value);

const laneFrom = (value: string | undefined): Lane | undefined => {
  if (value === "0") return 0;
  if (value === "1") return 1;
  if (value === "2") return 2;
  return undefined;
};

export class GameController {
  private selected: Instrument = "drums";
  private selectedDifficulty: ChartDifficulty = defaultDifficulty;
  private chart: LevelChart | undefined;
  private readonly audio = new BattleAudio();
  private readonly highway: HighwayRenderer;
  private boss:
    | {
        load(): Promise<void>;
        resize(): void;
        setMood(mood: "hit" | "attack" | "defeated", duration?: number): void;
      }
    | undefined;
  private judged = new Set<number>();
  private hits = new Set<number>();
  private playerHealth = 100;
  private score = 0;
  private combo = 0;
  private bestCombo = 0;
  private attackIndex = 0;
  private attackWarned = -1;
  private running = false;
  private finalTriggered = false;
  private frame = 0;

  constructor(
    private readonly bossCanvas: HTMLCanvasElement,
    highwayCanvas: HTMLCanvasElement,
  ) {
    this.highway = new HighwayRenderer(highwayCanvas);
  }

  async mount(): Promise<void> {
    this.bindControls();
    const { BossScene } = await import("../render/boss");
    this.boss = new BossScene(this.bossCanvas);
    await this.boss.load();
    window.addEventListener("resize", () => {
      this.boss?.resize();
      this.highway.resize();
    });
  }

  private bindControls(): void {
    document
      .querySelectorAll<HTMLButtonElement>("[data-instrument]")
      .forEach((button) => {
        button.addEventListener("click", () => this.chooseInstrument(button));
      });
    document
      .querySelectorAll<HTMLButtonElement>("[data-difficulty]")
      .forEach((button) => {
        button.addEventListener("click", () => this.chooseDifficulty(button));
        button.addEventListener("keydown", (event) => {
          const step =
            event.key === "ArrowLeft" || event.key === "ArrowUp"
              ? -1
              : event.key === "ArrowRight" || event.key === "ArrowDown"
                ? 1
                : undefined;
          const edge =
            event.key === "Home"
              ? "first"
              : event.key === "End"
                ? "last"
                : undefined;
          if (step === undefined && edge === undefined) return;
          event.preventDefault();
          this.moveDifficulty(button, step, edge);
        });
      });
    document
      .querySelectorAll<HTMLButtonElement>("[data-lane]")
      .forEach((button) => {
        button.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          const lane = laneFrom(button.dataset.lane);
          if (lane !== undefined) this.tap(lane, button);
        });
      });
    element("start-button").addEventListener("click", () => void this.start());
    element("pause-button").addEventListener("click", () => this.pause());
    element("resume-button").addEventListener("click", () => this.resume());
    element("quit-button").addEventListener("click", () =>
      this.resetToSelect(),
    );
    element("replay-button").addEventListener("click", () => void this.start());
    element("change-button").addEventListener("click", () =>
      this.resetToSelect(),
    );
    element("retry-button").addEventListener("click", () => void this.start());
    element("error-back-button").addEventListener("click", () =>
      this.resetToSelect(),
    );
    document.addEventListener("visibilitychange", () => {
      if (document.hidden && this.running) this.pause();
    });
    window.addEventListener("keydown", (event) => {
      const lane =
        event.key === "d" || event.key === "1"
          ? 0
          : event.key === "f" || event.key === "2"
            ? 1
            : event.key === "k" || event.key === "3"
              ? 2
              : undefined;
      if (lane !== undefined) this.tap(lane);
      if (event.key === "Escape" && this.running) this.pause();
    });
  }

  private chooseInstrument(button: HTMLButtonElement): void {
    if (!isInstrument(button.dataset.instrument)) return;
    this.selected = button.dataset.instrument;
    document
      .querySelectorAll<HTMLButtonElement>("[data-instrument]")
      .forEach((card) => {
        const chosen = card === button;
        card.classList.toggle("is-selected", chosen);
        card.setAttribute("aria-checked", String(chosen));
      });
  }

  private chooseDifficulty(button: HTMLButtonElement): void {
    if (!isDifficulty(button.dataset.difficulty)) return;
    this.selectedDifficulty = button.dataset.difficulty;
    document
      .querySelectorAll<HTMLButtonElement>("[data-difficulty]")
      .forEach((option) => {
        const chosen = option === button;
        option.classList.toggle("is-selected", chosen);
        option.setAttribute("aria-checked", String(chosen));
        option.tabIndex = chosen ? 0 : -1;
      });
  }

  private moveDifficulty(
    button: HTMLButtonElement,
    step: number | undefined,
    edge: "first" | "last" | undefined,
  ): void {
    const options = Array.from(
      document.querySelectorAll<HTMLButtonElement>("[data-difficulty]"),
    );
    const current = options.indexOf(button);
    const target =
      edge === "first"
        ? options[0]
        : edge === "last"
          ? options.at(-1)
          : options[(current + (step ?? 0) + options.length) % options.length];
    if (!target) return;
    this.chooseDifficulty(target);
    target.focus();
  }

  private async start(): Promise<void> {
    const startButton = element("start-button");
    startButton.classList.add("is-loading");
    startButton.setAttribute("aria-busy", "true");
    try {
      element("error-overlay").hidden = true;
      const chartData = await ky
        .get(chartPath(this.selected, this.selectedDifficulty))
        .json();
      const chart = chartSchema.parse(chartData);
      this.chart = new URLSearchParams(location.search).has("qa")
        ? this.qaChart(chart)
        : chart;
      await this.audio.prepare(this.selected);
      this.beginCountdown();
    } catch {
      showLoadError(
        "The level could not be loaded. Check your connection and try again.",
      );
    } finally {
      startButton.classList.remove("is-loading");
      startButton.removeAttribute("aria-busy");
    }
  }

  private beginCountdown(): void {
    this.resetRun();
    this.showBattle();
    element("battle-screen").focus();
    let count = 3;
    const callout = element("battle-callout");
    callout.textContent = String(count);
    const timer = window.setInterval(() => {
      count -= 1;
      callout.textContent = count > 0 ? String(count) : "FIGHT";
      if (count > 0) return;
      window.clearInterval(timer);
      this.audio.start();
      this.running = true;
      this.frame = requestAnimationFrame(() => this.update());
      window.setTimeout(() => callout.replaceChildren(), 500);
    }, 650);
  }

  private resetRun(): void {
    cancelAnimationFrame(this.frame);
    this.audio.stop();
    this.judged = new Set();
    this.hits = new Set();
    this.playerHealth = 100;
    this.score = 0;
    this.combo = 0;
    this.bestCombo = 0;
    this.attackIndex = 0;
    this.attackWarned = -1;
    this.finalTriggered = false;
    this.running = false;
    element("player-name").textContent =
      `${instrumentLabels[this.selected]} · ${this.selectedDifficulty}`.toUpperCase();
    renderHud({
      duration: this.chart?.duration ?? 90,
      time: 0,
      playerHealth: 100,
      score: 0,
      combo: 0,
      finalTriggered: false,
      charging: false,
    });
  }

  private update(): void {
    if (!this.running || !this.chart) return;
    const time = this.audio.time;
    this.markLateNotes(time);
    this.telegraphAttack(time);
    this.resolveAttacks(time);
    if (!this.finalTriggered && time >= this.chart.duration - 2) {
      this.finalTriggered = true;
      this.boss?.setMood("defeated", 4);
      showCallout("FINAL CHORD", "perfect");
    }
    this.highway.draw(this.chart.notes, this.judged, time);
    renderHud({
      duration: this.chart.duration,
      time,
      playerHealth: this.playerHealth,
      score: this.score,
      combo: this.combo,
      finalTriggered: this.finalTriggered,
      charging: this.isAttackCharging(time),
    });
    if (this.playerHealth <= 0 || time >= this.chart.duration) {
      this.finish(this.playerHealth > 0);
      return;
    }
    this.frame = requestAnimationFrame(() => this.update());
  }

  private tap(lane: Lane, button?: HTMLButtonElement): void {
    if (!this.running || !this.chart) return;
    button?.classList.add("is-pressed");
    if (button)
      window.setTimeout(() => button.classList.remove("is-pressed"), 90);
    const result = judgeTap(
      this.chart.notes,
      this.judged,
      this.audio.time,
      lane,
    );
    if (result.noteIndex === undefined) {
      this.combo = 0;
      this.audio.duck();
      showCallout("MISS", "miss");
      return;
    }
    this.judged.add(result.noteIndex);
    this.hits.add(result.noteIndex);
    this.combo += 1;
    this.bestCombo = Math.max(this.bestCombo, this.combo);
    this.score +=
      scoreForGrade(result.grade) *
      Math.max(1, Math.min(4, Math.ceil(this.combo / 10)));
    this.boss?.setMood("hit", 0.2);
    showCallout(result.grade.toUpperCase(), result.grade);
  }

  private markLateNotes(time: number): void {
    if (!this.chart) return;
    for (const [index, note] of this.chart.notes.entries()) {
      if (note.time >= time - 0.17) break;
      if (this.judged.has(index)) continue;
      this.judged.add(index);
      this.combo = 0;
      this.audio.duck();
    }
  }

  private resolveAttacks(time: number): void {
    const chart = this.chart;
    const attack = chart?.attacks[this.attackIndex];
    if (!chart || !attack || time < attack.end) return;
    const noteIndexes = chart.notes.flatMap((note, index) =>
      note.time >= attack.start && note.time <= attack.end ? [index] : [],
    );
    const hitCount = noteIndexes.filter((index) => this.hits.has(index)).length;
    if (
      resolveAttackWindow(hitCount, noteIndexes.length, attack.threshold) ===
      "struck"
    ) {
      this.playerHealth -= 28;
      this.boss?.setMood("attack");
      showCallout("WARD BROKEN −28", "miss");
    } else {
      this.score += 5_000;
      showCallout("ATTACK BLOCKED", "perfect");
    }
    this.attackIndex += 1;
  }

  private telegraphAttack(time: number): void {
    const attack = this.chart?.attacks[this.attackIndex];
    if (
      !attack ||
      time < attack.start ||
      this.attackWarned === this.attackIndex
    )
      return;
    this.attackWarned = this.attackIndex;
    this.boss?.setMood("attack", attack.end - attack.start);
    showCallout("INCOMING · HOLD THE WARD", "great");
  }

  private isAttackCharging(time: number): boolean {
    const attack = this.chart?.attacks[this.attackIndex];
    return Boolean(attack && time >= attack.start && time < attack.end);
  }

  private pause(): void {
    if (!this.running) return;
    this.running = false;
    cancelAnimationFrame(this.frame);
    this.audio.pause();
    element("pause-overlay").hidden = false;
    element("pause-title").focus();
  }

  private resume(): void {
    element("pause-overlay").hidden = true;
    this.audio.resume();
    this.running = true;
    element("battle-screen").focus();
    this.frame = requestAnimationFrame(() => this.update());
  }

  private finish(victory: boolean): void {
    this.running = false;
    this.audio.stop();
    const total = this.chart?.notes.length ?? 0;
    element("battle-screen").hidden = true;
    showResult(
      victory,
      this.score,
      this.hits.size / Math.max(1, total),
      this.bestCombo,
    );
  }

  private showBattle(): void {
    showBattle();
    this.highway.resize();
  }

  private resetToSelect(): void {
    cancelAnimationFrame(this.frame);
    this.audio.stop();
    this.running = false;
    showSelect();
  }

  private qaChart(chart: LevelChart): LevelChart {
    return {
      ...chart,
      duration: 12,
      notes: chart.notes
        .slice(0, 21)
        .map((note, index) => ({ ...note, time: 1.2 + index * 0.42 })),
      attacks: [
        { start: 2.8, end: 4.8, threshold: 0.35 },
        { start: 6.5, end: 8.5, threshold: 0.35 },
      ],
    };
  }
}
