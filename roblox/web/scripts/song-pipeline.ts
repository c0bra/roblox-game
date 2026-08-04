import { runCli } from "@bands-battle/chart-pipeline/cli";

process.exitCode = await runCli(["build", ...Bun.argv.slice(2)]);
