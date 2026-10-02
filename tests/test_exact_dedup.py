import gzip

import orjson
import pytest

from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.dedup import ExactDedupFilter, ExactDedupSignature, ExactFindDedups
from dataflow.pipeline.readers import JsonlReader
from dataflow.pipeline.writers import JsonlWriter

FILES = {
    "a.jsonl": ["apple", "banana", "apple", "cherry"],
    "b.jsonl": ["banana", "date", "elderberry"],
    "c.jsonl": ["fig", "apple", "grape"],
}


def write_input(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, texts in FILES.items():
        (folder / name).write_bytes(b"".join(orjson.dumps({"text": text}) + b"\n" for text in texts))


def texts(folder):
    return sorted(orjson.loads(line)["text"] for path in folder.rglob("*.gz") for line in gzip.open(path))


def run_dedup(tmp_path, tasks, finder_workers):
    write_input(tmp_path / "in")
    LocalPipelineExecutor(
        [JsonlReader(tmp_path / "in"), ExactDedupSignature(tmp_path / "sigs", finder_workers=finder_workers)],
        tmp_path / "logs/1",
        tasks=tasks,
        workers=1,
    ).run()
    LocalPipelineExecutor(
        [ExactFindDedups(tmp_path / "sigs", tmp_path / "dups")], tmp_path / "logs/2", tasks=finder_workers, workers=1
    ).run()
    return LocalPipelineExecutor(
        [
            JsonlReader(tmp_path / "in"),
            ExactDedupFilter(tmp_path / "dups", exclusion_writer=JsonlWriter(tmp_path / "removed")),
            JsonlWriter(tmp_path / "out"),
        ],
        tmp_path / "logs/3",
        tasks=tasks,
        workers=1,
    ).run()


@pytest.mark.parametrize(("tasks", "finder_workers"), [(1, 1), (3, 1), (3, 2), (2, 4)])
def test_each_text_survives_exactly_once(tmp_path, tasks, finder_workers):
    stats = run_dedup(tmp_path, tasks, finder_workers)
    assert texts(tmp_path / "out") == ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    assert texts(tmp_path / "removed") == ["apple", "apple", "banana"]
    dedup_filter = stats.stats[1]
    assert dedup_filter.metrics["dropped"].total == 3
    assert dedup_filter.metrics["forwarded"].total == 7


def test_first_occurrence_is_the_one_kept(tmp_path):
    run_dedup(tmp_path, tasks=3, finder_workers=1)
    kept = [orjson.loads(line) for line in gzip.open(tmp_path / "out/00000.jsonl.gz")]
    assert [row["text"] for row in kept] == ["apple", "banana", "cherry"]


def test_finder_refuses_wrong_number_of_tasks(tmp_path):
    write_input(tmp_path / "in")
    LocalPipelineExecutor(
        [JsonlReader(tmp_path / "in"), ExactDedupSignature(tmp_path / "sigs", finder_workers=2)], tmp_path / "l1"
    ).run()
    with pytest.raises(ValueError, match="tasks=2"):
        LocalPipelineExecutor([ExactFindDedups(tmp_path / "sigs", tmp_path / "dups")], tmp_path / "l2").run()


def test_filter_detects_input_that_changed_between_stages(tmp_path):
    run_dedup(tmp_path, tasks=1, finder_workers=1)
    (tmp_path / "in/c.jsonl").unlink()
    step = ExactDedupFilter(tmp_path / "dups")
    with pytest.raises(RuntimeError, match="input changed"):
        list(step(JsonlReader(tmp_path / "in")()))
