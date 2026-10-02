import gzip

import orjson
import pytest

from dataflow.data import Document
from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.readers import JsonlReader
from dataflow.pipeline.split import HoldoutSplit
from dataflow.pipeline.writers import JsonlWriter

TEXTS = [f"document number {i} about rivers, towns and the people who live there" for i in range(1000)]


def texts(folder):
    return {orjson.loads(line)["text"] for path in folder.rglob("*.gz") for line in gzip.open(path)}


def split(tmp_path, name, tasks, **kwargs):
    folder = tmp_path / "in"
    if not folder.exists():
        folder.mkdir()
        for part in range(4):
            rows = TEXTS[part::4]
            (folder / f"{part}.jsonl").write_bytes(b"".join(orjson.dumps({"text": t}) + b"\n" for t in rows))
    out = tmp_path / name
    pipeline = [JsonlReader(folder), HoldoutSplit(JsonlWriter(out / "valid"), **kwargs), JsonlWriter(out / "train")]
    stats = LocalPipelineExecutor(pipeline, out / "logs", tasks=tasks, workers=1).run()
    return texts(out / "train"), texts(out / "valid"), stats


def test_holdout_is_a_disjoint_split_of_about_the_requested_size(tmp_path):
    train, valid, stats = split(tmp_path, "a", tasks=2, fraction=0.2)
    assert train.isdisjoint(valid) and train | valid == set(TEXTS)
    assert 150 <= len(valid) <= 250
    assert stats.stats[1].metrics["held_out"].total == len(valid)


def test_holdout_does_not_depend_on_task_count_or_order(tmp_path):
    _, one_task, _ = split(tmp_path, "one", tasks=1, fraction=0.2)
    _, four_tasks, _ = split(tmp_path, "four", tasks=4, fraction=0.2)
    assert one_task == four_tasks
    reversed_docs = [Document(text, str(i)) for i, text in enumerate(reversed(TEXTS))]
    step = HoldoutSplit(JsonlWriter(tmp_path / "unused"), fraction=0.2)
    assert {doc.text for doc in reversed_docs if step.in_holdout(doc)} == one_task


def test_salt_picks_a_different_split_and_fraction_bounds(tmp_path):
    _, valid, _ = split(tmp_path, "a", tasks=1, fraction=0.2)
    _, salted, _ = split(tmp_path, "b", tasks=1, fraction=0.2, salt="another")
    assert valid != salted
    assert split(tmp_path, "none", tasks=1, fraction=0)[1] == set()
    assert split(tmp_path, "all", tasks=1, fraction=1)[0] == set()
    with pytest.raises(ValueError, match="fraction"):
        HoldoutSplit(JsonlWriter(tmp_path / "x"), fraction=1.5)
