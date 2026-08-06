import ky from "ky";
import type { LevelAudioUrls } from "../data/level-catalog";
import type { Grade } from "../game/judgement";

export class BattleAudio {
  private context: AudioContext | undefined;
  private backing: AudioBuffer | undefined;
  private stem: AudioBuffer | undefined;
  private backingSource: AudioBufferSourceNode | undefined;
  private stemSource: AudioBufferSourceNode | undefined;
  private stemGain: GainNode | undefined;
  private backingGain: GainNode | undefined;
  private startedAt = 0;
  private pausedAt = 0;
  private playing = false;

  async prepare(urls: LevelAudioUrls): Promise<void> {
    this.context ??= new AudioContext({ latencyHint: "interactive" });
    const [backingData, stemData] = await Promise.all([
      ky.get(urls.backing).arrayBuffer(),
      ky.get(urls.stem).arrayBuffer(),
    ]);
    [this.backing, this.stem] = await Promise.all([
      this.context.decodeAudioData(backingData),
      this.context.decodeAudioData(stemData),
    ]);
  }

  start(offset = 0): void {
    const context = this.context;
    const backing = this.backing;
    const stem = this.stem;
    if (!context || !backing || !stem) return;
    this.stopSources();
    void context.resume();
    this.backingSource = context.createBufferSource();
    this.stemSource = context.createBufferSource();
    this.stemGain = context.createGain();
    this.backingGain = context.createGain();
    this.backingSource.buffer = backing;
    this.stemSource.buffer = stem;
    this.backingSource.connect(this.backingGain).connect(context.destination);
    this.stemSource.connect(this.stemGain).connect(context.destination);
    this.stemGain.gain.value = 1;
    this.backingGain.gain.value = 0.88;
    this.backingSource.start(0, offset);
    this.stemSource.start(0, offset);
    this.startedAt = context.currentTime - offset;
    this.playing = true;
  }

  get time(): number {
    if (!this.playing || !this.context) return this.pausedAt;
    return Math.max(0, this.context.currentTime - this.startedAt);
  }

  pause(): void {
    if (!this.playing) return;
    this.pausedAt = this.time;
    this.stopSources();
    this.playing = false;
  }

  resume(): void {
    if (this.playing) return;
    this.start(this.pausedAt);
  }

  duck(): void {
    const context = this.context;
    const gain = this.stemGain?.gain;
    if (!context || !gain) return;
    const now = context.currentTime;
    gain.cancelScheduledValues(now);
    gain.setValueAtTime(gain.value, now);
    gain.linearRampToValueAtTime(0.25, now + 0.025);
    gain.linearRampToValueAtTime(1, now + 0.35);
    const oscillator = context.createOscillator();
    const flubGain = context.createGain();
    oscillator.type = "sawtooth";
    oscillator.frequency.setValueAtTime(146.83, now);
    oscillator.frequency.exponentialRampToValueAtTime(103.83, now + 0.14);
    flubGain.gain.setValueAtTime(0.0001, now);
    flubGain.gain.exponentialRampToValueAtTime(0.055, now + 0.015);
    flubGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.16);
    oscillator.connect(flubGain).connect(context.destination);
    oscillator.start(now);
    oscillator.stop(now + 0.17);
  }

  accent(grade: Exclude<Grade, "miss">): void {
    const context = this.context;
    const stem = this.stemGain?.gain;
    const backing = this.backingGain?.gain;
    if (!context || !stem || !backing) return;
    const now = context.currentTime;
    const strength =
      grade === "perfect" ? 1.22 : grade === "great" ? 1.15 : 1.1;
    for (const gain of [stem, backing]) {
      gain.cancelScheduledValues(now);
      gain.setValueAtTime(gain.value, now);
    }
    stem.linearRampToValueAtTime(strength, now + 0.015);
    stem.linearRampToValueAtTime(1, now + 0.18);
    backing.linearRampToValueAtTime(0.72, now + 0.015);
    backing.linearRampToValueAtTime(0.88, now + 0.18);
  }

  stop(): void {
    this.stopSources();
    this.playing = false;
    this.pausedAt = 0;
  }

  dispose(): void {
    this.stop();
    void this.context?.close();
    this.context = undefined;
    this.backing = undefined;
    this.stem = undefined;
  }

  private stopSources(): void {
    for (const source of [this.backingSource, this.stemSource]) {
      if (!source) continue;
      source.onended = null;
      try {
        source.stop();
      } catch {}
      source.disconnect();
    }
    this.backingSource = undefined;
    this.stemSource = undefined;
  }
}
