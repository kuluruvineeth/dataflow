import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import huggingface_hub
import pytest

from dataflow.data import Document
from dataflow.executor.jobs import JobsPipelineExecutor

SEEN: list[int] = []


def record_rank(data, rank=0, world_size=1):
    SEEN.append(rank)
    yield Document(text=f"rank {rank}", id=str(rank))


class FakeJobs(JobsPipelineExecutor):
    """Runs each 'job' in-process; a job listed in `cancel_after` stops after that many of its ranks."""

    def __init__(self, *args, cancel_after: dict[int, int] | None = None, **kwargs):
        super().__init__(*args, poll_interval=0, **kwargs)
        self.cancel_after = cancel_after or {}
        self.submitted: list[list[int]] = []
        self.stages: dict[str, str] = {}

    def prepare(self) -> None:
        pass

    def submit(self, ranks: list[int]) -> str:
        job_id = f"job-{len(self.submitted)}"
        stop = self.cancel_after.pop(len(self.submitted), None)
        self.submitted.append(ranks)
        self.run_ranks(ranks[:stop], workers=1)
        self.stages[job_id] = "COMPLETED" if stop is None else "CANCELED"
        return job_id

    def status(self, job_id: str) -> str:
        return self.stages[job_id]

    def describe(self, ended: dict) -> list[dict]:
        return [{"id": job_id, "stage": self.stages[job_id], "minutes": 1.0, "cost_usd": 0.03} for job_id in ended]


@pytest.fixture(autouse=True)
def reset_seen():
    SEEN.clear()


def test_splits_ranks_into_jobs(tmp_path):
    executor = FakeJobs([record_rank], tmp_path, tasks=5, tasks_per_job=2)
    stats = executor.launch()
    assert executor.submitted == [[0, 1], [2, 3], [4]]
    assert sorted(SEEN) == [0, 1, 2, 3, 4]
    assert len(list((tmp_path / "completions").iterdir())) == 5
    assert stats.to_json() == (tmp_path / "stats.json").read_text()


def test_cancelled_job_is_retried_with_only_its_missing_ranks(tmp_path):
    executor = FakeJobs([record_rank], tmp_path, tasks=4, tasks_per_job=2, cancel_after={0: 1})
    executor.launch()
    assert executor.submitted == [[0, 1], [2, 3], [1]]
    assert sorted(SEEN) == [0, 1, 2, 3]


def test_gives_up_after_max_retries(tmp_path):
    executor = FakeJobs([record_rank], tmp_path, tasks=2, tasks_per_job=2, max_retries=0, cancel_after={0: 1})
    executor.launch()
    assert executor.submitted == [[0, 1]]
    assert not (tmp_path / "completions" / "00001").exists()


def test_rerun_launches_only_missing_ranks(tmp_path):
    FakeJobs([record_rank], tmp_path, tasks=4, tasks_per_job=2, max_retries=0, cancel_after={1: 0}).launch()
    SEEN.clear()
    executor = FakeJobs([record_rank], tmp_path, tasks=4, tasks_per_job=2)
    executor.launch()
    assert executor.submitted == [[2, 3]]
    assert SEEN == [2, 3]


def test_records_minutes_and_cost(tmp_path):
    FakeJobs([record_rank], tmp_path, tasks=4, tasks_per_job=2).launch()
    (record_file,) = (tmp_path / "jobs").iterdir()
    records = json.loads(record_file.read_text())
    assert [record["id"] for record in records] == ["job-0", "job-1"]
    assert sum(record["cost_usd"] for record in records) == pytest.approx(0.06)


def test_cancelled_job_is_billed_until_the_coordinator_saw_it_end(tmp_path, monkeypatch):
    start = datetime(2026, 10, 2, 9, 0, tzinfo=UTC)

    def job(stage, finished_at):
        status = SimpleNamespace(stage=stage)
        return SimpleNamespace(started_at=start, finished_at=finished_at, url="", flavor="cpu-upgrade", status=status)

    jobs = {"done": job("COMPLETED", start + timedelta(minutes=10)), "cancelled": job("CANCELED", None)}
    monkeypatch.setattr(huggingface_hub, "inspect_job", lambda job_id: jobs[job_id])
    monkeypatch.setattr(
        huggingface_hub, "list_jobs_hardware", lambda: [SimpleNamespace(name="cpu-upgrade", unit_cost_usd=0.0005)]
    )
    seen = start + timedelta(minutes=20)
    records = JobsPipelineExecutor([], tmp_path, tasks=1).describe({"done": seen, "cancelled": seen})
    summary = [(record["minutes"], record["cost_usd"], record["estimated"]) for record in records]
    assert summary == [(10.0, 0.005, False), (20.0, 0.01, True)]


def test_inside_a_job_runs_only_its_ranks(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAFLOW_JOB_RANKS", "1,3")
    JobsPipelineExecutor([record_rank], tmp_path, tasks=4, workers_per_job=1).run()
    assert SEEN == [1, 3]
    assert sorted(path.name for path in (tmp_path / "completions").iterdir()) == ["00001", "00003"]


@pytest.mark.parametrize(
    ("logging_dir", "expected"),
    [
        ("hf://buckets/me/pipe/runs/smoke", ("me/pipe", "runs/smoke")),
        ("hf://buckets/me/pipe/a/b/c", ("me/pipe", "a/b/c")),
    ],
)
def test_bucket_location(logging_dir, expected):
    assert JobsPipelineExecutor([], logging_dir, tasks=1).bucket_location() == expected


def test_rejects_logging_dir_outside_a_bucket(tmp_path):
    with pytest.raises(ValueError, match="hf://buckets"):
        JobsPipelineExecutor([], tmp_path, tasks=1).bucket_location()
