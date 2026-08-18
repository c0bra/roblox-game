from __future__ import annotations

from pathlib import Path

from tools.comfyui_batch_audio import (
    PromptId,
    SoundJob,
    WorkflowTemplate,
    load_jobs,
    run_batch,
)


class RecordingComfyClient:
    def __init__(self) -> None:
        self.queued: list[WorkflowTemplate] = []
        self.waited: list[PromptId] = []
        self.events: list[str] = []

    def queue(self, workflow: WorkflowTemplate) -> PromptId:
        self.queued.append(workflow)
        prompt_id = PromptId(f"prompt-{len(self.queued)}")
        self.events.append(f"queue:{prompt_id}")
        return prompt_id

    def wait(self, prompt_id: PromptId) -> None:
        self.waited.append(prompt_id)
        self.events.append(f"wait:{prompt_id}")


def test_workflow_uses_csv_prompt_and_duration_without_mutating_template() -> None:
    # Given
    template = WorkflowTemplate.model_validate(
        {
            "991": {
                "inputs": {"value": 4},
                "class_type": "PixaromaNumber",
                "_meta": {"title": "Duration (1-380 seconds)"},
            },
            "992": {
                "inputs": {"text": "template-prompt-token"},
                "class_type": "PixaromaText",
                "_meta": {"title": "Audio Description Prompt"},
            },
        },
    )
    job = SoundJob.model_validate(
        {
            "sound_asset_id": "asset-token",
            "prompt_variant": "single_prompt",
            "elevenlabs_prompt": "csv-prompt-token",
            "duration_seconds": "0.3",
        },
    )

    # When
    rendered = template.with_job(job)

    # Then
    assert rendered.root["992"].inputs["text"] == "csv-prompt-token"
    assert rendered.root["991"].inputs["value"] == 0.3
    assert template.root["992"].inputs["text"] == "template-prompt-token"
    assert template.root["991"].inputs["value"] == 4


def test_jobs_use_every_csv_row_in_order(tmp_path: Path) -> None:
    # Given
    csv_path = tmp_path / "jobs.csv"
    _ = csv_path.write_text(
        """sound_asset_id,prompt_variant,elevenlabs_prompt,duration_seconds
first,single_prompt,first-prompt-token,0.2
second,2_of_3,second-prompt-token,1.6
""",
        encoding="utf-8",
    )

    # When
    jobs = load_jobs(csv_path)

    # Then
    assert [
        (tracked.sound.asset_id, tracked.sound.variant, tracked.sound.duration_seconds)
        for tracked in jobs
    ] == [
        ("first", "single_prompt", 0.2),
        ("second", "2_of_3", 1.6),
    ]


def test_completed_csv_rows_are_skipped(tmp_path: Path) -> None:
    # Given
    csv_path = tmp_path / "jobs.csv"
    _ = csv_path.write_text(
        """sound_asset_id,prompt_variant,elevenlabs_prompt,duration_seconds,comfyui_complete
first,single_prompt,first-prompt-token,0.2,true
second,2_of_3,second-prompt-token,1.6,false
""",
        encoding="utf-8",
    )

    # When
    jobs = load_jobs(csv_path)

    # Then
    assert [tracked.sound.asset_id for tracked in jobs] == ["second"]


def test_batch_persists_completion_after_waiting(tmp_path: Path) -> None:
    # Given
    csv_path = tmp_path / "jobs.csv"
    _ = csv_path.write_text(
        """sound_asset_id,prompt_variant,elevenlabs_prompt,duration_seconds
first,single_prompt,first-prompt-token,0.2
""",
        encoding="utf-8",
    )
    jobs = load_jobs(csv_path)
    client = RecordingComfyClient()
    template = WorkflowTemplate.model_validate(
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
    )

    # When
    run_batch(client, template, jobs)

    # Then
    assert load_jobs(csv_path) == ()


def test_batch_waits_for_each_job_before_queueing_the_next(tmp_path: Path) -> None:
    # Given
    csv_path = tmp_path / "jobs.csv"
    _ = csv_path.write_text(
        """sound_asset_id,prompt_variant,elevenlabs_prompt,duration_seconds
asset-1,single_prompt,prompt-token-1,1
asset-2,single_prompt,prompt-token-2,2
""",
        encoding="utf-8",
    )
    template = WorkflowTemplate.model_validate(
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
    )
    jobs = load_jobs(csv_path)
    client = RecordingComfyClient()

    # When
    run_batch(client, template, jobs)

    # Then
    assert client.events == [
        "queue:prompt-1",
        "wait:prompt-1",
        "queue:prompt-2",
        "wait:prompt-2",
    ]
    assert [workflow.root["13"].inputs["value"] for workflow in client.queued] == [
        1.0,
        2.0,
    ]
