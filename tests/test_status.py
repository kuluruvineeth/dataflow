import json
import time

import pytest

from dataflow.data import Document
from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.filters.base import BaseFilter
from dataflow.status import TaskState, collect, estimate_eta, render, trackio_metrics


class DropOdd(BaseFilter):
    name = "drop odd"

    def filter(self, document):
        return True if int(document.id) % 2 == 0 else (False, "odd")


def documents(data, rank=0, world_size=1):
    for i in range(10):
        yield Document(text=f"doc {i}", id=str(i))


def explode(data, rank=0, world_size=1):
    if rank == 1:
        raise RuntimeError("disk on fire")
    yield from data


def write_progress(folder, rank, updated, status="running", documents=5):
    path = folder / "progress" / f"{rank:05d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rank": rank,
                "status": status,
                "started": updated - 60,
                "updated": updated,
                "documents": documents,
                "documents_per_second": 2.0,
                "stats": [],
            }
        )
    )


def test_each_rank_reports_progress_and_the_run_is_described(tmp_path):
    executor = LocalPipelineExecutor([documents, DropOdd()], tmp_path, tasks=2, workers=1)
    executor.run()
    progress = json.loads((tmp_path / "progress/00001.json").read_text())
    assert progress["status"] == "done"
    assert progress["documents"] == 5
    assert progress["stats"][0]["metrics"]["dropped_odd"]["total"] == 5
    assert json.loads((tmp_path / "run.json").read_text())["tasks"] == 2


def test_a_failed_rank_leaves_its_traceback(tmp_path):
    with pytest.raises(RuntimeError):
        LocalPipelineExecutor([documents, explode], tmp_path, tasks=2, workers=1).run()
    assert json.loads((tmp_path / "progress/00001.json").read_text())["status"] == "failed"
    assert "disk on fire" in json.loads((tmp_path / "errors/00001.json").read_text())["traceback"]
    status = collect(tmp_path)
    assert [task.status for task in status.tasks] == ["done", "failed"]
    assert status.errors == {1: "RuntimeError: disk on fire"}


def test_tasks_are_classified_by_markers_and_progress_age(tmp_path):
    now = time.time()
    (tmp_path / "run.json").write_text(json.dumps({"tasks": 4}))
    (tmp_path / "completions").mkdir()
    (tmp_path / "completions/00000").touch()
    write_progress(tmp_path, 1, now - 10)
    write_progress(tmp_path, 2, now - 600)
    status = collect(tmp_path, stale_after=120, now=now)
    assert [task.status for task in status.tasks] == ["done", "running", "stale", "pending"]
    assert status.documents_per_second == 2.0


def test_funnel_and_cost_are_rendered(tmp_path):
    LocalPipelineExecutor([documents, DropOdd()], tmp_path, tasks=2, workers=1).run()
    (tmp_path / "jobs/launched").mkdir(parents=True)
    (tmp_path / "jobs/launched/abc.json").write_text(json.dumps({"id": "abc", "flavor": "cpu-upgrade"}))
    status = collect(tmp_path, job_cost=lambda launched, now: (12.0, 0.006))
    text = render(status)
    assert "2 tasks: 2 done" in text
    assert "drop odd" in text and "kept           10  50.0%" in text and "odd 10" in text
    assert "cost so far $0.0060 (12.0 job-minutes)" in text
    metrics = trackio_metrics(status)
    assert metrics["funnel/Filter: drop odd/forwarded"] == 10 and metrics["cost_usd"] == 0.006


def test_eta_uses_the_average_finished_task():
    now = 1000.0
    tasks = [
        TaskState(0, "done", started=0, updated=100),
        TaskState(1, "running", started=now - 40, updated=now),
        TaskState(2, "pending"),
        TaskState(3, "pending"),
    ]
    assert estimate_eta(tasks, now) == pytest.approx(60 + 2 * 100)
    assert estimate_eta([TaskState(0, "running", started=0, updated=1)], now) is None
