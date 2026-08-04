export interface CommandRunner {
  run(command: readonly string[]): Promise<void>;
}

export class CommandFailure extends Error {
  override readonly name = "CommandFailure";

  constructor(
    readonly command: readonly string[],
    readonly exitCode: number,
    readonly stderr: string,
  ) {
    super(
      `Command failed with exit code ${exitCode}: ${command.join(" ")}\n${stderr}`,
    );
  }
}

export const systemCommandRunner: CommandRunner = {
  run: async (command) => {
    const process = Bun.spawn([...command], {
      stdout: "ignore",
      stderr: "pipe",
    });
    const [exitCode, stderr] = await Promise.all([
      process.exited,
      new Response(process.stderr).text(),
    ]);
    if (exitCode !== 0) throw new CommandFailure(command, exitCode, stderr);
  },
};

const aacOptions = ["-vn", "-c:a", "aac", "-b:a", "160k"] as const;

export const encodeStemCommand = (
  ffmpeg: string,
  input: string,
  output: string,
): readonly string[] => [
  ffmpeg,
  "-hide_banner",
  "-loglevel",
  "error",
  "-y",
  "-i",
  input,
  "-map",
  "0:a:0",
  ...aacOptions,
  output,
];

export const encodeBackingCommand = (
  ffmpeg: string,
  inputs: readonly [string, string, string],
  output: string,
): readonly string[] => [
  ffmpeg,
  "-hide_banner",
  "-loglevel",
  "error",
  "-y",
  ...inputs.flatMap((input) => ["-i", input]),
  "-filter_complex",
  "[0:a:0][1:a:0][2:a:0]amix=inputs=3:duration=longest:normalize=1[a]",
  "-map",
  "[a]",
  ...aacOptions,
  output,
];
