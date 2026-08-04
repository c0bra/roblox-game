import { describe, expect, test } from "bun:test";
import { resolveCommandPath } from "../src/process-runner";

describe("process runner environment", () => {
  test("Given Docker Desktop's helper is installed on macOS, when Docker runs without that directory in PATH, then the helper directory is prepended", () => {
    const path = resolveCommandPath({
      executable: "docker",
      platform: "darwin",
      configuredPath: "/usr/local/bin:/usr/bin",
      dockerDesktopHelperAvailable: true,
    });

    expect(path).toBe(
      "/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:/usr/bin",
    );
  });
});
