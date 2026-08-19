#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "httpx2[http2,brotli,zstd]",
#     "pydantic>=2.0",
#     "typer>=0.12",
#     "typing-extensions>=4.12",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv or pip install needed):
#      uv run tools/comfyui_batch_audio.py [OPTIONS]
# 3. Or make executable and run:
#      chmod +x tools/comfyui_batch_audio.py && ./tools/comfyui_batch_audio.py
# ─────────────────

from __future__ import annotations

import csv
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, ClassVar, Final, NewType, Protocol, final

import httpx2
import typer
from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel
from typing_extensions import override

PROMPT_TITLE = "Audio Description Prompt"
DURATION_TITLE = "Duration (1-380 seconds)"
COMPLETION_COLUMN: Final = "comfyui_complete"
PromptId = NewType("PromptId", str)
_HTTP_LIMITS: Final = httpx2.Limits(
    max_connections=200,
    max_keepalive_connections=40,
    keepalive_expiry=30.0,
)
_HTTP_TIMEOUT: Final = httpx2.Timeout(
    connect=5.0,
    read=30.0,
    write=10.0,
    pool=10.0,
)
_SOCKET_OPTIONS: Final = ((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),)
_REPO_ROOT: Final = Path(__file__).resolve().parents[1]
_DEFAULT_WORKFLOW: Final = _REPO_ROOT / "Batch Stable Audio Sound Creation.json"
_DEFAULT_CSV: Final = (
    _REPO_ROOT / "openspec/changes/add-arena-gameplay-mode/sound-prompts.csv"
)


@final
@dataclass(frozen=True, slots=True)
class WorkflowFieldError(Exception):
    title: str
    matches: int

    @override
    def __str__(self) -> str:
        return (
            f"workflow must contain exactly one {self.title!r} node; "
            f"found {self.matches}"
        )


class NodeMetadata(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    title: str


class WorkflowNode(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    inputs: dict[str, JsonValue]
    class_type: str
    metadata: NodeMetadata = Field(alias="_meta")


class SoundJob(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    asset_id: str = Field(alias="sound_asset_id", min_length=1)
    variant: str = Field(alias="prompt_variant", min_length=1)
    prompt: str = Field(alias="elevenlabs_prompt", min_length=1)
    duration_seconds: float = Field(gt=0, le=380)
    complete: bool = Field(alias=COMPLETION_COLUMN, default=False)


@dataclass(frozen=True, slots=True)
class TrackedSoundJob:
    sound: SoundJob
    csv_path: Path
    csv_index: int
    fieldnames: tuple[str, ...]


class WorkflowTemplate(RootModel[dict[str, WorkflowNode]]):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    def with_job(self, job: SoundJob) -> WorkflowTemplate:
        prompt_matches = [
            (node_id, node)
            for node_id, node in self.root.items()
            if node.metadata.title == PROMPT_TITLE
        ]
        duration_matches = [
            (node_id, node)
            for node_id, node in self.root.items()
            if node.metadata.title == DURATION_TITLE
        ]
        if len(prompt_matches) != 1:
            raise WorkflowFieldError(title=PROMPT_TITLE, matches=len(prompt_matches))
        if len(duration_matches) != 1:
            raise WorkflowFieldError(
                title=DURATION_TITLE, matches=len(duration_matches),
            )

        prompt_id, prompt_node = prompt_matches[0]
        duration_id, duration_node = duration_matches[0]
        return WorkflowTemplate(
            self.root
            | {
                prompt_id: prompt_node.model_copy(
                    update={"inputs": prompt_node.inputs | {"text": job.prompt}},
                ),
                duration_id: duration_node.model_copy(
                    update={
                        "inputs": duration_node.inputs
                        | {"value": job.duration_seconds},
                    },
                ),
            },
        )


class PromptRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    prompt: WorkflowTemplate


class PromptResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    prompt_id: str


class HistoryResponse(RootModel[dict[str, JsonValue]]):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


@final
@dataclass(frozen=True, slots=True)
class HttpComfyClient:
    http: httpx2.Client
    poll_seconds: float = 1.0

    def queue(self, workflow: WorkflowTemplate) -> PromptId:
        request = PromptRequest(prompt=workflow)
        response = self.http.post(
            "/prompt",
            content=request.model_dump_json(by_alias=True),
            headers={"Content-Type": "application/json"},
        )
        _ = response.raise_for_status()
        return PromptId(PromptResponse.model_validate_json(response.content).prompt_id)

    def wait(self, prompt_id: PromptId) -> None:
        while True:
            response = self.http.get(f"/history/{prompt_id}")
            _ = response.raise_for_status()
            history = HistoryResponse.model_validate_json(response.content)
            if prompt_id in history.root:
                return
            time.sleep(self.poll_seconds)


def create_http_client(base_url: str) -> httpx2.Client:
    transport = httpx2.HTTPTransport(
        http2=True,
        retries=3,
        limits=_HTTP_LIMITS,
        socket_options=_SOCKET_OPTIONS,
    )
    return httpx2.Client(
        transport=transport,
        timeout=_HTTP_TIMEOUT,
        base_url=base_url,
        follow_redirects=True,
    )


def load_jobs(path: Path) -> tuple[TrackedSoundJob, ...]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    jobs: list[TrackedSoundJob] = []
    for index, row in enumerate(rows):
        sound = SoundJob.model_validate(row)
        if sound.complete:
            continue
        jobs.append(
            TrackedSoundJob(
                sound=sound,
                csv_path=path,
                csv_index=index,
                fieldnames=fieldnames,
            ),
        )
    return tuple(jobs)


def mark_complete(job: TrackedSoundJob) -> None:
    with job.csv_path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    fieldnames = list(job.fieldnames)
    if COMPLETION_COLUMN not in fieldnames:
        fieldnames.append(COMPLETION_COLUMN)
    for index, row in enumerate(rows):
        if index == job.csv_index:
            row[COMPLETION_COLUMN] = "true"
        elif not row.get(COMPLETION_COLUMN):
            row[COMPLETION_COLUMN] = "false"
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=job.csv_path.parent,
        delete=False,
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(output.name)
    _ = temporary_path.replace(job.csv_path)


def load_workflow(path: Path) -> WorkflowTemplate:
    return WorkflowTemplate.model_validate_json(path.read_text(encoding="utf-8"))


class ComfyClient(Protocol):
    def queue(self, workflow: WorkflowTemplate) -> PromptId: ...

    def wait(self, prompt_id: PromptId) -> None: ...


def run_batch(
    client: ComfyClient,
    template: WorkflowTemplate,
    jobs: tuple[TrackedSoundJob, ...],
) -> None:
    for index, tracked in enumerate(jobs, start=1):
        job = tracked.sound
        typer.echo(f"[{index}/{len(jobs)}] Generating {job.asset_id} ({job.variant})")
        prompt_id = client.queue(template.with_job(job))
        client.wait(prompt_id)
        mark_complete(tracked)
        typer.echo(f"    completed: {prompt_id}")


def main(
    workflow: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = _DEFAULT_WORKFLOW,
    csv_file: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = _DEFAULT_CSV,
    comfy_url: Annotated[
        str,
        typer.Option(help="ComfyUI server base URL."),
    ] = "http://127.0.0.1:8188",
) -> None:
    template = load_workflow(workflow)
    jobs = load_jobs(csv_file)
    typer.echo(f"Submitting {len(jobs)} jobs sequentially to {comfy_url}")
    with create_http_client(comfy_url.rstrip("/")) as http_client:
        run_batch(HttpComfyClient(http_client), template, jobs)
    typer.echo(f"All {len(jobs)} jobs finished.")


if __name__ == "__main__":
    typer.run(main)
