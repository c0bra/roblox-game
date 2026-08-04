import { existsSync } from "node:fs";
import { join } from "node:path";

export type ProcessRunner = {
  readonly text: (command: readonly string[]) => Promise<string>;
  readonly bytes: (command: readonly string[]) => Promise<ArrayBuffer>;
};

const dockerDesktopBin = "/Applications/Docker.app/Contents/Resources/bin";

export type CommandPathInput = {
  readonly executable: string | undefined;
  readonly platform: NodeJS.Platform;
  readonly configuredPath: string;
  readonly dockerDesktopHelperAvailable: boolean;
};

export const resolveCommandPath = (input: CommandPathInput): string => {
  const directories = input.configuredPath.split(":");
  return input.executable === "docker" &&
    input.platform === "darwin" &&
    input.dockerDesktopHelperAvailable &&
    !directories.includes(dockerDesktopBin)
    ? [dockerDesktopBin, input.configuredPath].filter(Boolean).join(":")
    : input.configuredPath;
};

const spawnCommand = (command: readonly string[]) => {
  const { PATH: configuredPath = "" } = process.env;
  return Bun.spawn([...command], {
    stdout: "pipe",
    stderr: "pipe",
    env: {
      ...process.env,
      PATH: resolveCommandPath({
        executable: command[0],
        platform: process.platform,
        configuredPath,
        dockerDesktopHelperAvailable: existsSync(
          join(dockerDesktopBin, "docker-credential-desktop"),
        ),
      }),
    },
  });
};

export class CommandFailure extends Error {
  override readonly name = "CommandFailure";

  constructor(
    readonly command: readonly string[],
    readonly exitCode: number,
    readonly stderr: string,
  ) {
    super(`${command[0] ?? "command"} exited ${exitCode}: ${stderr.trim()}`);
  }
}

const runText = async (command: readonly string[]): Promise<string> => {
  const child = spawnCommand(command);
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  if (exitCode !== 0) throw new CommandFailure(command, exitCode, stderr);
  return stdout;
};

const runBytes = async (command: readonly string[]): Promise<ArrayBuffer> => {
  const child = spawnCommand(command);
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).arrayBuffer(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  if (exitCode !== 0) throw new CommandFailure(command, exitCode, stderr);
  return stdout;
};

export const systemProcessRunner: ProcessRunner = {
  text: runText,
  bytes: runBytes,
};
