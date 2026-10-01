import json
import random
import statistics

import orjson
import pytest

from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.filters import LambdaFilter
from dataflow.pipeline.readers import JsonlReader
from dataflow.utils.stats import MetricStats, PipelineStats, Stats

VALUES = [random.Random(0).uniform(0, 100) for _ in range(500)]


def metric_of(values):
    metric = MetricStats()
    for value in values:
        metric.update(value)
    return metric


def test_running_stats_match_exact_formulas():
    metric = metric_of(VALUES)
    assert metric.n == 500
    assert metric.total == pytest.approx(sum(VALUES))
    assert metric.mean == pytest.approx(statistics.mean(VALUES))
    assert metric.variance == pytest.approx(statistics.variance(VALUES))
    assert (metric.min, metric.max) == (min(VALUES), max(VALUES))


@pytest.mark.parametrize("cut", [0, 1, 137, 499, 500])
def test_merging_two_halves_equals_computing_on_everything(cut):
    merged = metric_of(VALUES[:cut]) + metric_of(VALUES[cut:])
    whole = metric_of(VALUES)
    assert merged.n == whole.n
    assert merged.mean == pytest.approx(whole.mean)
    assert merged.variance == pytest.approx(whole.variance)
    assert (merged.min, merged.max) == (whole.min, whole.max)


def test_stats_merge_by_label_and_survive_json():
    a, b = Stats("Filter: x"), Stats("Filter: x")
    a.update("dropped")
    a.update("total", 2)
    b.update("total", 3)
    merged = a + b
    assert merged.metrics["total"].total == 5
    assert merged.metrics["dropped"].total == 1
    restored = PipelineStats.from_json(PipelineStats([merged]).to_json()).stats[0]
    assert restored.to_dict() == merged.to_dict()


def test_executor_saves_per_task_stats_and_merges_them(tmp_path):
    for i in range(3):
        rows = [{"text": "x" * (10 * (j + 1))} for j in range(4)]
        (tmp_path / "in").mkdir(exist_ok=True)
        (tmp_path / f"in/{i}.jsonl").write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    pipeline = [JsonlReader(tmp_path / "in"), LambdaFilter(lambda doc: len(doc.text) > 20)]
    stats = LocalPipelineExecutor(pipeline, tmp_path / "logs", tasks=3, workers=1).run()

    assert sorted(p.name for p in (tmp_path / "logs/stats").iterdir()) == ["00000.json", "00001.json", "00002.json"]
    reader, keep = stats.stats
    assert reader.metrics["documents"].total == 12
    assert reader.metrics["doc_len"].mean == pytest.approx(25)
    assert keep.metrics["total"].total == 12
    assert keep.metrics["forwarded"].total == 6
    assert keep.metrics["dropped"].total == 6

    on_disk = json.loads((tmp_path / "logs/stats.json").read_text())
    assert on_disk[1]["metrics"]["forwarded"]["total"] == 6

    rerun = LocalPipelineExecutor(pipeline, tmp_path / "logs", tasks=3, workers=1).run()
    assert rerun.stats[1].metrics["forwarded"].total == 6
