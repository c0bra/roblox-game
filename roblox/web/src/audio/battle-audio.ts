import ky from "ky";
import type { Instrument } from "../data/level";

export class BattleAudio {
  private context: AudioContext | undefined;
  private backing: AudioBuffer | undefined;
  private stem: AudioBuffer | undefined;
  private backingSource: AudioBufferSourceNode | undefined;
  private stemSource: AudioBufferSourceNode | undefined;
  private stemGain: GainNode | undefined;
  private startedAt = 0;
  private pausedAt = 0;
  private playing = false;

  async prepare(instrument: Instrument): Promise<void> {
    this.context ??= new AudioContext({ latencyHint: "interactive" });
    const base = `/levels/heavens-edge/audio/${instrument}`;
    const [backingData, stemData] = await Promise.all([
      ky.get(`${base}-backing.m4a`).arrayBuffer(),
      ky.get(`${base}-stem.m4a`).arrayBuffer(),
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
    this.backingSource.buffer = backing;
    this.stemSource.buffer = stem;
    this.backingSource.connect(context.destination);
    this.stemSource.connect(this.stemGain).connect(context.destination);
    this.stemGain.gain.value = 1;
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

  stop(): void {
    this.stopSources();
    this.playing = false;
    this.pausedAt = 0;
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
