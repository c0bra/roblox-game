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
