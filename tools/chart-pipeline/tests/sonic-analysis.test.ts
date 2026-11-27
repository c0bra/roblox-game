import { describe, expect, test } from "bun:test";
import { analyzeSong } from "../src/sonic-analysis";

describe("Sonic Annotator output", () => {
  test("ignores the blank row after a trailing newline", async () => {
    const analysis = await analyzeSong(
      {
        drums: "drums.wav",
        vocals: "vocals.wav",
        guitar: "guitar.wav",
        bass: "bass.wav",
      },
      {
        text: async (command) =>
          command.includes("vamp:beatroot-vamp:beatroot:beats")
            ? "3.12\n3.56\n"
            : "1,0.5,440\n",
        bytes: async () => new ArrayBuffer(0),
      },
    );

    expect(analysis.beatTimes).toEqual([3.12, 3.56]);
    expect(analysis.vocals).toHaveLength(1);
  });
});

test("a low-register detection uses the stem clock and original pitch, with silence excluded", async () => {
  const samples = new Float32Array(8000 * 4);
  samples.fill(0.2, 8000, 16000);
  const result = await analyzeSong(
    { drums: "d", guitar: "g", vocals: "v", bass: "b" },
    {
      text: async (cmd) =>
        cmd.includes("vamp:pyin:pyin:notes") &&
        cmd.at(-1)?.endsWith("octave.wav")
          ? "0.5,0.3,80\n1.5,0.2,80\n"
          : "",
      bytes: async () => samples.buffer,
    },
  );
  expect(result.bassRecovery?.events).toHaveLength(1);
  expect(result.bassRecovery?.events[0]).toMatchObject({
    time: 1,
    duration: 0.6,
  });
  expect(result.bassRecovery?.events[0]?.pitch).toBeCloseTo(
    69 + 12 * Math.log2(40 / 440),
    6,
  );
});

test("locally balanced guitar detection adds a quiet strum and keeps the original attack when both hear it", async () => {
  const pcm = Float32Array.from(
    { length: 16000 },
    (_, i) => 0.12 * Math.sin((2 * Math.PI * 220 * i) / 8000),
  );
  const result = await analyzeSong(
    { drums: "d", vocals: "v", guitar: "g", bass: "b" },
    {
      text: async (cmd) =>
        cmd.includes("vamp:qm-vamp-plugins:qm-onsetdetector:onsets")
          ? cmd.at(-1) === "g"
            ? "0.25\n"
            : "0.23\n1.1\n"
          : "",
      bytes: async () => pcm.buffer,
    },
  );
  expect(result.recovery.guitar.events.map((e) => e.time)).toEqual([0.25, 1.1]);
});
