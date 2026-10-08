import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

import cloudpickle
import pytest
import ray
from hypothesis import given
from hypothesis import strategies as st

from dataflow.data import Document
from dataflow.executor.local import LocalPipelineExecutor
from dataflow.executor.ray import RankLost, RayPipelineExecutor, run_with_refill
from dataflow.pipeline.classifiers import FastTextClassifier
from dataflow.pipeline.classifiers.base import ScoringStep
from dataflow.pipeline.filters.lambda_filter import LambdaFilter
from dataflow.pipeline.readers.jsonl import JsonlReader
from dataflow.pipeline.writers.jsonl import JsonlWriter
from dataflow.recipes.dclm_classifier import train, training_lines

# Ray workers cannot import this test module, so its functions travel by value. Ray pickles actor arguments with its
# own copy of cloudpickle, which keeps its own registry.
cloudpickle.register_pickle_by_value(sys.modules[__name__])
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

GOOD = [
    "Plants make sugar from light because chlorophyll absorbs red and blue light and passes the energy on.",
    "The bridge stays up because the arch pushes the load sideways into the ground on both banks.",
    "Water boils at a lower temperature on a mountain because the air pressure there is lower.",
]
SPAM = [
    "click here buy now best price free shipping limited offer click here",
    "casino bonus win big today free spins sign up now best casino bonus",
    "cheap deals cheap deals buy now limited offer best price click",
]
# No uv-run worker start (~3 s each) and no auth token written to the home folder.
TEST_ENV = {"RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0", "RAY_AUTH_MODE": "disabled"}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    folder = tmp_path_factory.mktemp("fasttext")
    (folder / "train.txt").write_text("\n".join(training_lines(GOOD * 30, SPAM * 30)) + "\n")
    return train(folder / "train.txt", folder / "model.bin", epoch=25)


@pytest.fixture(scope="module")
def local_ray():
    with pytest.MonkeyPatch.context() as patch:
        for name, value in TEST_ENV.items():
            patch.setenv(name, value)
        ray.init(address="local", num_cpus=8, include_dashboard=False)
        yield
        ray.shutdown()


def write_corpus(folder: Path, files: int, per_file: int = 12) -> Path:
    folder.mkdir(parents=True)
    texts = GOOD + SPAM + ["short"]
    for index in range(files):
        lines = [{"id": f"{index}-{n}", "text": f"{texts[(index + n) % len(texts)]} {n}"} for n in range(per_file)]
        (folder / f"{index:05d}.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    return folder


def scoring_pipeline(corpus: Path, output: Path, model: Path) -> list:
    return [
        JsonlReader(corpus),
        LambdaFilter(lambda document: len(document.text) > 10),
        FastTextClassifier(model, key="quality", batch_size=4),
        JsonlWriter(output),
    ]


def outputs(folder: Path) -> dict[str, bytes]:
    return {path.name: gzip.decompress(path.read_bytes()) for path in sorted(folder.glob("*.gz"))}


def markers(logs: Path) -> list[str]:
    return sorted(path.name for path in (logs / "completions").iterdir())


def submissions(logs: Path) -> dict[int, list[dict]]:
    records: dict[int, list[dict]] = {}
    for path in sorted((logs / "submissions").iterdir()):
        record = json.loads(path.read_text())
        records.setdefault(record["rank"], []).append(record)
    return {rank: sorted(found, key=lambda r: r["submission"]) for rank, found in records.items()}


def stats_without_time(logs: Path) -> list[dict]:
    return [{k: v for k, v in step.items() if k != "time"} for step in json.loads((logs / "stats.json").read_text())]


def touch_rank(folder: Path):
    def step(data, rank=0, world_size=1):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{rank:05d}").write_text(str(os.getpid()))
        yield from ()

    return step


def test_run_starts_a_local_ray_and_stops_it(tmp_path):
    assert not ray.is_initialized()
    with pytest.MonkeyPatch.context() as patch:
        for name, value in TEST_ENV.items():
            patch.setenv(name, value)
        RayPipelineExecutor([touch_rank(tmp_path / "out")], tmp_path / "logs", tasks=2, address="local").run()
    assert not ray.is_initialized()
    assert markers(tmp_path / "logs") == ["00000", "00001"]


def test_same_output_as_the_local_executor(local_ray, tmp_path, model_path):
    corpus = write_corpus(tmp_path / "corpus", files=20)
    local, on_ray = tmp_path / "local", tmp_path / "ray"
    LocalPipelineExecutor(scoring_pipeline(corpus, local / "out", model_path), local / "logs", 20, workers=2).run()
    RayPipelineExecutor(scoring_pipeline(corpus, on_ray / "out", model_path), on_ray / "logs", 20, workers=4).run()
    assert len(outputs(local / "out")) == 20
    assert outputs(on_ray / "out") == outputs(local / "out")
    assert markers(on_ray / "logs") == markers(local / "logs") == [f"{rank:05d}" for rank in range(20)]
    assert stats_without_time(on_ray / "logs") == stats_without_time(local / "logs")


class LengthScorer(ScoringStep):
    type = "Classifier"
    name = "length"

    def __init__(self, loads: Path):
        self.loads = loads

    def load(self):
        with self.loads.open("a") as file:
            file.write(f"{os.getpid()}\n")

    def model_scores(self, texts: list[str]) -> list[float]:
        return [len(text) / 100 for text in texts]

    def run(self, data, rank=0, world_size=1):
        documents = list(data)
        for document, score in zip(documents, self.scores([d.text for d in documents]), strict=True):
            document.metadata["length"] = score
            self.stat_update("scored")
            yield document


def test_a_shared_model_loads_once_for_all_ranks(local_ray, tmp_path):
    corpus = write_corpus(tmp_path / "corpus", files=8)
    scorer = LengthScorer(tmp_path / "loads.txt")
    pipeline = [JsonlReader(corpus), scorer, JsonlWriter(tmp_path / "ray")]
    RayPipelineExecutor(pipeline, tmp_path / "logs", tasks=8, workers=4).run()
    assert len((tmp_path / "loads.txt").read_text().splitlines()) == 1
    assert scorer.scorer is None
    local = [JsonlReader(corpus), LengthScorer(tmp_path / "unused.txt"), JsonlWriter(tmp_path / "local")]
    LocalPipelineExecutor(local, tmp_path / "local-logs", tasks=8, workers=1).run()
    assert outputs(tmp_path / "ray") == outputs(tmp_path / "local")


def test_never_more_than_workers_ranks_in_flight(local_ray, tmp_path):
    running, seen = tmp_path / "running", tmp_path / "seen"
    running.mkdir()
    seen.mkdir()

    def busy(data, rank=0, world_size=1):
        (running / str(rank)).touch()
        (seen / str(rank)).write_text(str(len(list(running.iterdir()))))
        time.sleep(0.3)
        (running / str(rank)).unlink()
        yield from ()

    RayPipelineExecutor([busy], tmp_path / "logs", tasks=8, workers=2).run()
    counts = [int(path.read_text()) for path in seen.iterdir()]
    assert len(counts) == 8
    assert max(counts) <= 2


def fail_unless(flag: Path, rank_to_fail: int, ran: Path):
    def step(data, rank=0, world_size=1):
        ran.mkdir(exist_ok=True)
        (ran / f"{rank}-{time.monotonic_ns()}").touch()
        if rank == rank_to_fail and not flag.exists():
            raise ValueError("boom")
        yield from ()

    return step


def test_a_rank_that_raises_fails_the_run_and_a_second_run_does_only_that_rank(local_ray, tmp_path):
    flag, ran, logs = tmp_path / "fixed", tmp_path / "ran", tmp_path / "logs"
    with pytest.raises(RuntimeError, match="1 of 4 ranks failed, first: rank 2 failed: ValueError: boom"):
        RayPipelineExecutor([fail_unless(flag, 2, ran)], logs, tasks=4, workers=2).run()
    assert markers(logs) == ["00000", "00001", "00003"]
    assert sorted(path.name.split("-")[0] for path in ran.iterdir()) == ["0", "1", "2", "3"]
    flag.touch()
    for path in ran.iterdir():
        path.unlink()
    RayPipelineExecutor([fail_unless(flag, 2, ran)], logs, tasks=4, workers=2).run()
    assert [path.name.split("-")[0] for path in ran.iterdir()] == ["2"]
    assert markers(logs) == ["00000", "00001", "00002", "00003"]


def die_once(attempts: Path, rank_to_kill: int):
    def step(data, rank=0, world_size=1):
        if rank == rank_to_kill and not attempts.exists():
            attempts.touch()
            os._exit(1)
        yield Document(text=f"rank {rank}", id=str(rank))

    return step


def test_the_executor_submits_a_lost_rank_again_and_records_it(local_ray, tmp_path, caplog):
    pipeline = [die_once(tmp_path / "attempted", 1), JsonlWriter(tmp_path / "out")]
    RayPipelineExecutor(pipeline, tmp_path / "logs", tasks=4, workers=2).run()
    assert (tmp_path / "attempted").exists()
    assert markers(tmp_path / "logs") == ["00000", "00001", "00002", "00003"]
    assert outputs(tmp_path / "out")["00001.jsonl.gz"] == b'{"text":"rank 1","id":"1"}\n'
    records = submissions(tmp_path / "logs")
    assert {rank: len(found) for rank, found in records.items()} == {0: 1, 1: 2, 2: 1, 3: 1}
    assert records[1][1]["previous_error"] == "rank 1 failed: WorkerCrashedError"
    assert "rank 1 lost on submission 1: WorkerCrashedError; submission 2" in caplog.text


def die_after_marker(logs: Path, rank_to_kill: int):
    def step(data, rank=0, world_size=1):
        if rank == rank_to_kill:
            (logs / "completions").mkdir(parents=True, exist_ok=True)
            (logs / "completions" / f"{rank:05d}").touch()
            os._exit(1)
        yield from ()

    return step


def test_a_rank_lost_after_its_marker_is_not_submitted_again(local_ray, tmp_path):
    logs = tmp_path / "logs"
    RayPipelineExecutor([die_after_marker(logs, 1)], logs, tasks=3, workers=3).run()
    assert {rank: len(found) for rank, found in submissions(logs).items()} == {0: 1, 1: 1, 2: 1}
    assert markers(logs) == ["00000", "00001", "00002"]


def test_the_placement_group_is_removed_after_the_run(local_ray, tmp_path):
    RayPipelineExecutor([touch_rank(tmp_path / "out")], tmp_path / "logs", tasks=4, workers=2).run()
    groups = ray.util.placement_group_table().values()
    assert groups and all(group["state"] == "REMOVED" for group in groups)


def test_without_a_placement_strategy_there_is_no_group(local_ray, tmp_path):
    before = len(ray.util.placement_group_table())
    pipeline = [touch_rank(tmp_path / "out")]
    RayPipelineExecutor(pipeline, tmp_path / "logs", tasks=2, workers=2, placement_strategy=None).run()
    assert len(ray.util.placement_group_table()) == before
    assert markers(tmp_path / "logs") == ["00000", "00001"]


def test_a_group_that_cannot_fit_stops_the_run(local_ray, tmp_path):
    executor = RayPipelineExecutor([touch_rank(tmp_path / "out")], tmp_path / "logs", tasks=2, workers=20)
    executor.placement_timeout = 1
    with pytest.raises(RuntimeError, match=r"placement group .* \(PACK\) not ready after 1 s"):
        executor.run()
    assert not (tmp_path / "logs" / "completions").exists()


def test_without_retries_a_dead_worker_fails_its_rank(local_ray, tmp_path):
    pipeline = [die_once(tmp_path / "attempted", 1)]
    with pytest.raises(RuntimeError, match="1 of 2 ranks failed, first: rank 1 failed: WorkerCrashedError"):
        RayPipelineExecutor(pipeline, tmp_path / "logs", tasks=2, workers=2, max_retries=0).run()
    assert markers(tmp_path / "logs") == ["00000"]
    assert len(submissions(tmp_path / "logs")[1]) == 1


def test_default_workers_are_the_cluster_cpus_less_the_model_hosts(local_ray, tmp_path):
    assert RayPipelineExecutor([], tmp_path).default_workers(model_hosts=1) == 7
    assert RayPipelineExecutor([], tmp_path, cpus_per_task=2).default_workers(model_hosts=0) == 4


@given(
    ranks=st.lists(st.integers(0, 60), unique=True, max_size=40),
    workers=st.integers(1, 8),
    max_retries=st.integers(0, 3),
    outcomes=st.lists(st.sampled_from(["ok", "ok", "lost", "error"]), min_size=1, max_size=50),
    seed=st.integers(0, 2**32),
)
def test_refill_submits_only_lost_ranks_again_in_their_slot(ranks, workers, max_retries, outcomes, seed):
    order = random.Random(seed)
    running, history, peaks, ended = {}, [], [], []
    outcome_of = {}

    def submit(rank, slot, submission, previous):
        handle = (rank, submission)
        assert slot not in {s for s, _ in running.values()}
        assert (previous is None) == (submission == 1)
        running[handle] = (slot, submission)
        history.append((rank, slot, submission))
        peaks.append(len(running))
        outcome_of[handle] = outcomes[len(history) % len(outcomes)]
        return handle

    def wait(handles):
        assert set(handles) == set(running)
        done = order.choice(sorted(handles))
        del running[done]
        return done

    def result(handle, rank):
        if outcome_of[handle] == "lost":
            raise RankLost(RuntimeError("worker died"))
        if outcome_of[handle] == "error":
            raise ValueError(rank)
        ended.append(rank)

    failures = run_with_refill(ranks, workers, submit, wait, result, max_retries=max_retries)
    failed = [rank for rank, _ in failures]
    assert sorted(ended + failed) == sorted(ranks)
    assert max(peaks, default=0) == min(workers, len(ranks))
    for rank in ranks:
        mine = [(slot, n) for r, slot, n in history if r == rank]
        assert [n for _, n in mine] == list(range(1, len(mine) + 1))
        assert len(mine) <= max_retries + 1
        assert len({slot for slot, _ in mine}) == 1
        results = [outcome_of[(rank, n)] for _, n in mine]
        assert all(outcome == "lost" for outcome in results[:-1])
        assert results[-1] != "lost" or len(mine) == max_retries + 1
        assert (rank in ended) == (results[-1] == "ok")
