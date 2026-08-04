import type { BattleAudio } from "../audio/battle-audio";
import {
  type ChartNote,
  instrumentLabels,
  type Lane,
  type LevelChart,
} from "../data/level";
import type { RunAssetLoader } from "../data/level-loader";
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
import {
  judgeTap,
  resolveAttackWindow,
  resolveSustain,
  scoreForGrade,
} from "./judgement";
import type { SelectionControls } from "./selection-controls";

const laneFrom = (value: string | undefined): Lane | undefined => {
  if (value === "0") return 0;
  if (value === "1") return 1;
  if (value === "2") return 2;
  return undefined;
};

export class GameController {
  private chart: LevelChart | undefined;
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
  private activeSustains = new Map<Lane, number>();
  private pressedLanes = new Set<Lane>();
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
    private readonly selection: SelectionControls,
    private readonly audio: BattleAudio,
    private readonly loadRun: RunAssetLoader,
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
    this.selection.mount();
    document
      .querySelectorAll<HTMLButtonElement>("[data-lane]")
      .forEach((button) => {
        button.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          const lane = laneFrom(button.dataset.lane);
          if (lane === undefined) return;
          button.setPointerCapture(event.pointerId);
          this.pressLane(lane, button);
        });
        const release = (): void => {
          const lane = laneFrom(button.dataset.lane);
          if (lane !== undefined) this.releaseLane(lane, button);
        };
        button.addEventListener("pointerup", release);
        button.addEventListener("pointercancel", release);
        button.addEventListener("lostpointercapture", release);
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
      const lane = this.laneForKey(event.key);
      if (lane !== undefined && !event.repeat) this.pressLane(lane);
      if (event.key === "Escape" && this.running) this.pause();
    });
    window.addEventListener("keyup", (event) => {
      const lane = this.laneForKey(event.key);
      if (lane !== undefined) this.releaseLane(lane);
    });
    window.addEventListener("blur", () => this.releaseAllLanes());
  }

  private laneForKey(key: string): Lane | undefined {
    if (key === "d" || key === "1") return 0;
    if (key === "k" || key === "3") return 2;
    return undefined;
  }

  private async start(): Promise<void> {
    const startButton = element("start-button");
    startButton.classList.add("is-loading");
    startButton.setAttribute("aria-busy", "true");
    this.selection.setDisabled(true);
    try {
      element("error-overlay").hidden = true;
      this.chart = await this.loadRun(this.selection.current);
      this.beginCountdown();
    } catch {
      showLoadError(
        "The level could not be loaded. Check your connection and try again.",
      );
    } finally {
      this.selection.setDisabled(false);
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
    const selection = this.selection.current;
    cancelAnimationFrame(this.frame);
    this.audio.stop();
    this.judged = new Set();
    this.hits = new Set();
    this.activeSustains = new Map();
    this.releaseAllLanes();
    this.playerHealth = 100;
    this.score = 0;
    this.combo = 0;
    this.bestCombo = 0;
    this.attackIndex = 0;
    this.attackWarned = -1;
    this.finalTriggered = false;
    this.running = false;
    element("player-name").textContent =
      `${instrumentLabels[selection.instrument]} · ${selection.difficulty}`.toUpperCase();
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
    this.resolveActiveSustains(time);
    this.markLateNotes(time);
    this.telegraphAttack(time);
    this.resolveAttacks(time);
    if (!this.finalTriggered && time >= this.chart.duration - 2) {
      this.finalTriggered = true;
      this.boss?.setMood("defeated", 4);
      showCallout("FINAL CHORD", "perfect");
    }
    this.highway.draw(
      this.chart.notes,
      this.judged,
      new Set(this.activeSustains.values()),
      time,
    );
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

  private pressLane(lane: Lane, button?: HTMLButtonElement): void {
    if (!this.running || !this.chart) return;
    if (this.pressedLanes.has(lane)) return;
    this.pressedLanes.add(lane);
    this.laneButton(lane, button)?.classList.add("is-pressed");
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
    const note = this.chart.notes[result.noteIndex];
    if (note && note.duration > 0) {
      this.activeSustains.set(lane, result.noteIndex);
      this.laneButton(lane, button)?.classList.add("is-held");
    }
  }

  private releaseLane(lane: Lane, button?: HTMLButtonElement): void {
    if (!this.pressedLanes.delete(lane)) return;
    this.laneButton(lane, button)?.classList.remove("is-pressed");
    if (this.running) this.resolveActiveSustains(this.audio.time);
  }

  private releaseAllLanes(): void {
    this.pressedLanes.clear();
    document
      .querySelectorAll<HTMLButtonElement>("[data-lane]")
      .forEach((button) => {
        button.classList.remove("is-pressed", "is-held");
      });
  }

  private laneButton(
    lane: Lane,
    button?: HTMLButtonElement,
  ): HTMLButtonElement | undefined {
    return (
      button ??
      document.querySelector<HTMLButtonElement>(`[data-lane="${lane}"]`) ??
      undefined
    );
  }

  private resolveActiveSustains(time: number): void {
    if (!this.chart) return;
    for (const [lane, noteIndex] of this.activeSustains) {
      const note: ChartNote | undefined = this.chart.notes[noteIndex];
      if (!note) continue;
      const result = resolveSustain(note, time, this.pressedLanes.has(lane));
      if (result === "holding") continue;
      this.activeSustains.delete(lane);
      this.laneButton(lane)?.classList.remove("is-held");
      if (result === "complete") {
        this.score += Math.round(note.duration * 500);
        showCallout("HELD", "perfect");
        continue;
      }
      this.hits.delete(noteIndex);
      this.combo = 0;
      this.audio.duck();
      showCallout("HOLD BROKEN", "miss");
    }
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
    this.activeSustains.clear();
    this.releaseAllLanes();
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
    this.releaseAllLanes();
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
    this.releaseAllLanes();
    showSelect();
  }
}
