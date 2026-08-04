import { resolve } from "node:path";
import { ZodError } from "zod";
import { validateBundle } from "./bundle";
import { parsePipelineArgs, UnknownPipelineCommand } from "./options";
import { buildChartBundle } from "./song-pipeline";

const usage = `Chart pipeline

Usage:
  ./chart build --song input.mp3 [--output build/song] [--start 0] [--duration 90]
  ./chart build --stems path/to/stems [--output build/song] [--start 0] [--duration 90]
  ./chart validate build/song

If --output is omitted, charts are written to build/<input-name>.
`;

export interface CliOutput {
  info(message: string): void;
  error(message: string): void;
}

const consoleOutput: CliOutput = {
  info: (message) => console.info(message),
  error: (message) => console.error(message),
};

export const main = async (
  argv: readonly string[],
  output: CliOutput = consoleOutput,
): Promise<void> => {
  const args = parsePipelineArgs(argv);
  switch (args.command) {
    case "build": {
      const manifest = await buildChartBundle(args);
      output.info(
        `Built ${resolve(args.output)} (${manifest.duration}s, schema v${manifest.schemaVersion})`,
      );
      return;
    }
    case "validate": {
      const manifest = await validateBundle(resolve(args.directory));
      output.info(
        `Valid ${resolve(args.directory)} (${manifest.duration}s, schema v${manifest.schemaVersion})`,
      );
      return;
    }
    case "help":
      output.info(usage);
      return;
  }
};

const errorMessage = (argv: readonly string[], error: unknown): string => {
  if (error instanceof ZodError) {
    return argv[0] === "validate"
      ? "Validate requires exactly one bundle directory."
      : "Use exactly one of --song <file> or --stems <directory>.";
  }
  if (error instanceof UnknownPipelineCommand) return error.message;
  if (error instanceof Error) return error.message;
  return "Unexpected chart pipeline failure.";
};

export const runCli = async (
  argv: readonly string[],
  output: CliOutput = consoleOutput,
): Promise<number> => {
  try {
    await main(argv, output);
    return 0;
  } catch (error) {
    output.error(`${errorMessage(argv, error)}\n\n${usage}`);
    return 1;
  }
};

if (import.meta.main) {
  process.exitCode = await runCli(Bun.argv.slice(2));
}
