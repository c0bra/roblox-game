import { describe, expect, test } from "bun:test";
import { selectStemCandidate } from "@bands-battle/chart-pipeline/stems";

describe("stem selection", () => {
  test("Given exact and Demucs candidates, when a drum stem is selected, then the exact stem wins", () => {
    expect(
      selectStemCandidate(
        ["/songs/input_(Drums)_htdemucs.wav", "/songs/drum-stem.mp3"],
        "drums",
      ),
    ).toBe("/songs/drum-stem.mp3");
  });

  test("Given a four-stem Demucs export, when guitar is selected, then other is accepted", () => {
    expect(selectStemCandidate(["/songs/other.wav"], "guitar")).toBe(
      "/songs/other.wav",
    );
  });

  test("Given equally ranked candidates, when a stem is selected, then the lexical first path wins", () => {
    expect(
      selectStemCandidate(["/songs/z/drum.wav", "/songs/a/drum.wav"], "drums"),
    ).toBe("/songs/a/drum.wav");
  });

  test("Given no matching candidate, when a stem is selected, then no path is returned", () => {
    expect(
      selectStemCandidate(["/songs/strings.wav"], "drums"),
    ).toBeUndefined();
  });
});
