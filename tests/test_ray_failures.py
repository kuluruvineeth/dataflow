import json
import sys
import threading
import time
from pathlib import Path

import cloudpickle
import pytest
import ray
from ray.cluster_utils import Cluster

from dataflow.data import Document
from dataflow.executor.ray import RayPipelineExecutor
from dataflow.pipeline.classifiers.base import ScoringStep

# Ray workers cannot import this test module, so its functions travel by value (Ray keeps its own registry).
cloudpickle.register_pickle_by_value(sys.modules[__name__])
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

TEST_ENV = {"RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0", "RAY_AUTH_MODE": "disabled"}


@pytest.fixture
def cluster():
    """A head with no CPUs and three worker nodes of 2 CPUs, so every bundle is on a node that a test can remove."""
    with pytest.MonkeyPatch.context() as patch:
        for name, value in TEST_ENV.items():
            patch.setenv(name, value)
        cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 0})
        nodes = {node.node_id: node for node in (cluster.add_node(num_cpus=2) for _ in range(3))}
        ray.init(address=cluster.address)
        cluster.wait_for_nodes()
        yield cluster, nodes
        ray.shutdown()
        cluster.shutdown()


def started_on(folder: Path):
    """Not a generator, so it records the rank's node when the rank starts, not when a later step reads from it."""

    def step(data, rank=0, world_size=1):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{rank:05d}-{time.monotonic_ns()}").write_text(ray.get_runtime_context().get_node_id())
        return data or ()

    return step


def remove_when(cluster: Cluster, nodes: dict, find_node, removed: list) -> threading.Thread:
    """Removes the node that `find_node()` names, as soon as it names one."""

    def act():
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if (node_id := find_node()) in nodes:
                cluster.remove_node(nodes[node_id])
                removed.append(node_id)
                return
            time.sleep(0.05)

    thread = threading.Thread(target=act, daemon=True)
    thread.start()
    return thread


def submissions(logs: Path) -> dict[int, list[dict]]:
    records: dict[int, list[dict]] = {}
    for path in sorted((logs / "submissions").iterdir()):
        record = json.loads(path.read_text())
        records.setdefault(record["rank"], []).append(record)
    return records


def write_after(seconds: float, out: Path):
    def step(data, rank=0, world_size=1):
        time.sleep(seconds)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{rank:05d}").write_text(f"rank {rank}")
        yield from ()

    return step


def test_ranks_on_a_removed_node_are_submitted_again(cluster, tmp_path):
    cluster, nodes = cluster
    started, logs, removed = tmp_path / "started", tmp_path / "logs", []

    def a_node_with_a_running_rank():
        files = list(started.glob("*")) if started.exists() else []
        return files[0].read_text() if files else None

    thread = remove_when(cluster, nodes, a_node_with_a_running_rank, removed)
    pipeline = [started_on(started), write_after(3, tmp_path / "out")]
    RayPipelineExecutor(pipeline, logs, tasks=6, workers=3).run()
    thread.join()
    assert len(removed) == 1
    assert sorted(path.name for path in (logs / "completions").iterdir()) == [f"{r:05d}" for r in range(6)]
    assert {path.name: path.read_text() for path in (tmp_path / "out").iterdir()} == {
        f"{r:05d}": f"rank {r}" for r in range(6)
    }
    records = submissions(logs)
    again = {rank for rank, found in records.items() if len(found) == 2}
    assert again, "no rank was submitted again"
    assert all(len(found) <= 2 for found in records.values())
    for rank in again:
        error = max(records[rank], key=lambda r: r["submission"])["previous_error"]
        assert error.startswith(f"rank {rank} failed: ") and error.endswith("Error")


class LengthScorer(ScoringStep):
    type = "Classifier"
    name = "length"

    def __init__(self, loads: Path):
        self.loads = loads

    def load(self):
        with self.loads.open("a") as file:
            file.write(f"{ray.get_runtime_context().get_node_id()}\n")

    def model_scores(self, texts: list[str]) -> list[float]:
        return [len(text) / 100 for text in texts]

    def run(self, data, rank=0, world_size=1):
        for document in data:
            document.metadata["length"] = self.scores([document.text])[0]
            yield document


def slow_documents(data, rank=0, world_size=1):
    for number in range(15):
        time.sleep(0.2)
        yield Document(text=f"rank {rank} document {number}", id=f"{rank}-{number}")


def write_scores(out: Path):
    def step(data, rank=0, world_size=1):
        scores = [document.metadata["length"] for document in data]
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{rank:05d}").write_text(json.dumps(scores))
        yield from ()

    return step


def test_model_calls_wait_while_the_model_host_restarts_on_another_node(cluster, tmp_path):
    cluster, nodes = cluster
    loads, started, removed = tmp_path / "loads", tmp_path / "started", []

    def the_model_host_node():
        if not started.exists() or not any(started.iterdir()) or not loads.exists():
            return None
        return loads.read_text().splitlines()[0]

    thread = remove_when(cluster, nodes, the_model_host_node, removed)
    pipeline = [started_on(started), slow_documents, LengthScorer(loads), write_scores(tmp_path / "out")]
    RayPipelineExecutor(pipeline, tmp_path / "logs", tasks=4, workers=2).run()
    thread.join()
    hosts = loads.read_text().splitlines()
    assert removed == hosts[:1]
    assert len(hosts) == 2 and hosts[1] != hosts[0]
    expected = {f"{r:05d}": [len(f"rank {r} document {n}") / 100 for n in range(15)] for r in range(4)}
    assert {path.name: json.loads(path.read_text()) for path in (tmp_path / "out").iterdir()} == expected
