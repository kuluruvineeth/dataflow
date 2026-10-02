import gzip

import orjson
import pytest

from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.dedup import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
    MinhashDedupSignature,
)
from dataflow.pipeline.dedup.minhash import UnionFind
from dataflow.pipeline.readers import JsonlReader
from dataflow.pipeline.writers import JsonlWriter

RIVER = (
    "the river runs through the old town and the people who live there have built bridges with stone and wood "
    "every spring the water rises to the edge of the market so the traders move their stalls to the hill "
    "children learn to swim in the shallow parts and the fishermen teach them which currents to avoid"
)
MOUNTAIN = (
    "high in the mountains the snow stays until june and the shepherds wait for the passes to open before "
    "they lead their flocks up to the summer pastures where the grass is green and the air is thin and cold"
)
FILES = {
    "a.jsonl": [("river", RIVER), ("mountain", MOUNTAIN), ("short", "too short to shingle")],
    "b.jsonl": [("river-edited", RIVER + " visitors love it"), ("river-copy", RIVER)],
    "c.jsonl": [("mountain-edited", MOUNTAIN + " in winter"), ("other", "a completely unrelated " * 5)],
}


def run_minhash(tmp_path, tasks):
    folder = tmp_path / "in"
    folder.mkdir()
    for name, rows in FILES.items():
        (folder / name).write_bytes(b"".join(orjson.dumps({"id": i, "text": t}) + b"\n" for i, t in rows))
    config = MinhashConfig()
    LocalPipelineExecutor(
        [JsonlReader(folder), MinhashDedupSignature(tmp_path / "sigs", config)],
        tmp_path / "l1",
        tasks=tasks,
        workers=1,
    ).run()
    LocalPipelineExecutor(
        [MinhashDedupBuckets(tmp_path / "sigs", tmp_path / "pairs", config)],
        tmp_path / "l2",
        tasks=config.num_buckets,
        workers=1,
    ).run()
    LocalPipelineExecutor([MinhashDedupCluster(tmp_path / "pairs", tmp_path / "remove")], tmp_path / "l3").run()
    return LocalPipelineExecutor(
        [JsonlReader(folder), MinhashDedupFilter(tmp_path / "remove"), JsonlWriter(tmp_path / "out")],
        tmp_path / "l4",
        tasks=tasks,
        workers=1,
    ).run()


def kept_ids(folder):
    return sorted(orjson.loads(line)["id"] for path in folder.rglob("*.gz") for line in gzip.open(path))


@pytest.mark.parametrize("tasks", [1, 3])
def test_one_document_survives_per_near_duplicate_cluster(tmp_path, tasks):
    stats = run_minhash(tmp_path, tasks)
    assert kept_ids(tmp_path / "out") == ["mountain", "other", "river", "short"]
    assert stats.stats[1].metrics["dropped"].total == 3


def test_union_find_keeps_the_smallest_member_as_root():
    clusters = UnionFind()
    clusters.union((2, 5), (1, 3))
    clusters.union((1, 3), (0, 9))
    clusters.union((4, 0), (5, 1))
    assert {clusters.find(node) for node in [(2, 5), (1, 3), (0, 9)]} == {(0, 9)}
    assert clusters.find((5, 1)) == (4, 0)
    assert sorted(clusters.parent) == [(1, 3), (2, 5), (5, 1)]


def test_bucket_and_cluster_steps_refuse_wrong_task_counts(tmp_path):
    with pytest.raises(ValueError, match="tasks=14"):
        list(MinhashDedupBuckets(tmp_path, tmp_path)(rank=0, world_size=2))
    with pytest.raises(ValueError, match="tasks=1"):
        list(MinhashDedupCluster(tmp_path, tmp_path)(rank=0, world_size=2))
