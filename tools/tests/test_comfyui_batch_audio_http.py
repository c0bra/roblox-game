from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from tools.comfyui_batch_audio import (
    HttpComfyClient,
    PromptId,
    PromptRequest,
    WorkflowTemplate,
    create_http_client,
)


class ServerState:
    requests: list[bytes]
    history_reads: int
    empty_history_reads: int

    def __init__(self, empty_history_reads: int) -> None:
        self.requests = []
        self.history_reads = 0
        self.empty_history_reads = empty_history_reads


@dataclass(frozen=True, slots=True)
class RunningServer:
    state: ServerState
    url: str


@contextmanager
def running_comfy_server(
    empty_history_reads: int = 0,
) -> Generator[RunningServer, None, None]:
    state = ServerState(empty_history_reads=empty_history_reads)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length_header = self.headers.get("Content-Length")
            assert length_header is not None
            state.requests.append(self.rfile.read(int(length_header)))
            body = json.dumps({"prompt_id": f"prompt-{len(state.requests)}"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            _ = self.wfile.write(body)

        def do_GET(self) -> None:
            state.history_reads += 1
            prompt_id = self.path.removeprefix("/history/")
            if state.empty_history_reads > 0:
                state.empty_history_reads -= 1
                body = b"{}"
            else:
                body = json.dumps({prompt_id: {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            _ = self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield RunningServer(
            state=state,
            url=f"http://127.0.0.1:{server.server_port}",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_client_posts_prompt_and_polls_until_history_exists() -> None:
    # Given
    template = WorkflowTemplate.model_validate(
        {
            "13": {
                "inputs": {"value": 0.3},
                "class_type": "Number",
                "_meta": {"title": "Duration (1-380 seconds)"},
            },
            "14": {
                "inputs": {"text": "csv-prompt-token"},
                "class_type": "Text",
                "_meta": {"title": "Audio Description Prompt"},
            },
        },
    )

    # When
    with (
        running_comfy_server(empty_history_reads=1) as comfy,
        create_http_client(comfy.url) as http_client,
    ):
        client = HttpComfyClient(http_client, poll_seconds=0)
        prompt_id = client.queue(template)
        client.wait(prompt_id)

    # Then
    assert prompt_id == PromptId("prompt-1")
    assert comfy.state.history_reads == 2
    payload = PromptRequest.model_validate_json(comfy.state.requests[0])
    assert payload.prompt.root["14"].inputs["text"] == "csv-prompt-token"
    assert payload.prompt.root["13"].inputs["value"] == 0.3


def test_cli_submits_every_csv_row_to_comfyui(tmp_path: Path) -> None:
    # Given
    workflow_path = tmp_path / "workflow.json"
    _ = workflow_path.write_text(
        json.dumps(
            {
                "13": {
                    "inputs": {"value": 4},
                    "class_type": "Number",
                    "_meta": {"title": "Duration (1-380 seconds)"},
                },
                "14": {
                    "inputs": {"text": "template-prompt-token"},
                    "class_type": "Text",
                    "_meta": {"title": "Audio Description Prompt"},
                },
            },
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "jobs.csv"
    _ = csv_path.write_text(
        """sound_asset_id,prompt_variant,elevenlabs_prompt,duration_seconds
first,single_prompt,prompt-token-one,0.2
second,2_of_3,prompt-token-two,1.6
""",
        encoding="utf-8",
    )

    # When
    with running_comfy_server() as comfy:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parents[1] / "comfyui_batch_audio.py"),
                "--workflow",
                str(workflow_path),
                "--csv-file",
                str(csv_path),
                "--comfy-url",
                comfy.url,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    # Then
    assert result.returncode == 0, result.stderr
    assert len(comfy.state.requests) == 2
    payloads = [
        PromptRequest.model_validate_json(request) for request in comfy.state.requests
    ]
    assert [payload.prompt.root["14"].inputs["text"] for payload in payloads] == [
        "prompt-token-one",
        "prompt-token-two",
    ]
    assert [payload.prompt.root["13"].inputs["value"] for payload in payloads] == [
        0.2,
        1.6,
    ]
