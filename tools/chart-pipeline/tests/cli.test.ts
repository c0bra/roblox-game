import { describe, expect, test } from "bun:test";
import { runCli } from "../src/cli";

describe("chart CLI", () => {
  test("reports invalid build options without exposing Zod internals", async () => {
    const errors: string[] = [];
    const exitCode = await runCli(["build"], {
      info: () => {},
      error: (message) => errors.push(message),
    });

    expect(exitCode).toBe(1);
    expect(errors).toHaveLength(1);
    expect(errors[0]).toContain(
      "Use exactly one of --song <file> or --stems <directory>.",
    );
    expect(errors[0]).not.toContain("ZodError");
  });
});
