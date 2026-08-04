import { basename, extname, join } from "node:path";
import { parseArgs } from "node:util";
import { z } from "zod";

const buildBase = {
  output: z.string().min(1).optional(),
  start: z.coerce.number().nonnegative().default(0),
  duration: z.coerce.number().positive().optional(),
  model: z.string().min(1).default("htdemucs.yaml"),
  "snap-ms": z.coerce.number().positive().max(250).default(80),
};

const defaultOutput = (input: string): string => {
  const name = basename(input, extname(input))
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return join("build", name || "song");
};

const buildValuesSchema = z.union([
  z.object({
    ...buildBase,
    song: z.string().min(1),
    stems: z.undefined().optional(),
  }),
  z.object({
    ...buildBase,
    stems: z.string().min(1),
    song: z.undefined().optional(),
  }),
]);

export type BuildOptions =
  | {
      readonly command: "build";
      readonly song: string;
      readonly output: string;
      readonly start: number;
      readonly duration?: number;
      readonly model: string;
      readonly snapMs: number;
    }
  | {
      readonly command: "build";
      readonly stems: string;
      readonly output: string;
      readonly start: number;
      readonly duration?: number;
      readonly model: string;
      readonly snapMs: number;
    };

export type PipelineArgs =
  | BuildOptions
  | { readonly command: "validate"; readonly directory: string }
  | { readonly command: "help" };

const parseBuild = (argv: readonly string[]): BuildOptions => {
  const parsed = parseArgs({
    args: [...argv],
    options: {
      song: { type: "string" },
      stems: { type: "string" },
      output: { type: "string" },
      start: { type: "string" },
      duration: { type: "string" },
      model: { type: "string" },
      "snap-ms": { type: "string" },
    },
    strict: true,
  });
  const values = buildValuesSchema.parse(parsed.values);
  const shared = {
    command: "build" as const,
    output: values.output ?? defaultOutput(values.song ?? values.stems),
    start: values.start,
    model: values.model,
    snapMs: values["snap-ms"],
    ...(values.duration === undefined ? {} : { duration: values.duration }),
  };
  return values.song === undefined
    ? { ...shared, stems: values.stems }
    : { ...shared, song: values.song };
};

export const parsePipelineArgs = (argv: readonly string[]): PipelineArgs => {
  if (argv.includes("--help") || argv.includes("-h"))
    return { command: "help" };
  const command = argv[0];
  switch (command) {
    case "build":
      return parseBuild(argv.slice(1));
    case "validate": {
      const parsed = parseArgs({ args: argv.slice(1), allowPositionals: true });
      const directory = z
        .tuple([z.string().min(1)])
        .parse(parsed.positionals)[0];
      return { command: "validate", directory };
    }
    case "help":
    case "--help":
    case "-h":
    case undefined:
      return { command: "help" };
    default:
      throw new UnknownPipelineCommand(command);
  }
};

export class UnknownPipelineCommand extends Error {
  override readonly name = "UnknownPipelineCommand";

  constructor(readonly command: string) {
    super(`Unknown chart command: ${command}`);
  }
}
