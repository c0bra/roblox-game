import type { ArenaEffect } from "./combat";

type Tone = {
  readonly frequency: number;
  readonly duration: number;
  readonly gain: number;
  readonly type?: OscillatorType;
  readonly endFrequency?: number;
};

export class ArenaSound {
  private context: AudioContext | undefined;

  unlock(): void {
    this.context ??= new AudioContext({ latencyHint: "interactive" });
    void this.context.resume();
  }

  playEffect(effect: ArenaEffect): void {
    switch (effect.type) {
      case "input-ack":
        this.tone({ frequency: 520, duration: 0.04, gain: 0.025 });
        break;
      case "perform-contact":
        this.chord(
          effect.grade === "perfect"
            ? [523, 784]
            : effect.grade === "great"
              ? [440, 659]
              : [392, 523],
          0.11,
        );
        break;
      case "perform-flub":
      case "phrase-miss":
        this.tone({
          frequency: 155,
          endFrequency: 92,
          duration: 0.16,
          gain: 0.04,
          type: "sawtooth",
        });
        break;
      case "move-start":
        this.tone({
          frequency: effect.direction === "advance" ? 310 : 220,
          endFrequency: effect.direction === "advance" ? 620 : 165,
          duration: 0.14,
          gain: 0.035,
        });
        break;
      case "move-arrive":
        this.tone({ frequency: 740, duration: 0.06, gain: 0.025 });
        break;
      case "boundary":
      case "move-unavailable":
        this.tone({
          frequency: 120,
          duration: 0.08,
          gain: 0.025,
          type: "square",
        });
        break;
      case "boss-prepare":
        this.tone({
          frequency: effect.attackType === "sweep" ? 98 : 185,
          endFrequency: effect.attackType === "sweep" ? 196 : 92,
          duration: 0.34,
          gain: 0.045,
          type: effect.attackType === "sweep" ? "sawtooth" : "square",
        });
        break;
      case "boss-impact":
        this.tone({
          frequency: effect.avoided ? 660 : 55,
          duration: effect.avoided ? 0.1 : 0.28,
          gain: 0.065,
          type: "triangle",
        });
        break;
      case "boss-opening":
        this.chord([220, 330, 440], 0.17);
        break;
      case "outcome":
        this.chord(
          effect.outcome === "victory" ? [262, 330, 392, 523] : [196, 165, 131],
          0.42,
        );
        break;
    }
  }

  count(final: boolean): void {
    this.tone({
      frequency: final ? 880 : 440,
      duration: final ? 0.14 : 0.06,
      gain: 0.035,
    });
  }

  dispose(): void {
    void this.context?.close();
    this.context = undefined;
  }

  private chord(frequencies: readonly number[], duration: number): void {
    for (const frequency of frequencies) {
      this.tone({ frequency, duration, gain: 0.018, type: "triangle" });
    }
  }

  private tone(options: Tone): void {
    const context = this.context;
    if (!context) return;
    const now = context.currentTime;
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.type = options.type ?? "sine";
    oscillator.frequency.setValueAtTime(options.frequency, now);
    if (options.endFrequency) {
      oscillator.frequency.exponentialRampToValueAtTime(
        options.endFrequency,
        now + options.duration,
      );
    }
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(options.gain, now + 0.008);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + options.duration);
    oscillator.connect(gain).connect(context.destination);
    oscillator.start(now);
    oscillator.stop(now + options.duration + 0.01);
  }
}
