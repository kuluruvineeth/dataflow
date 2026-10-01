from pathlib import Path

import pytest

from dataflow.data import Document
from dataflow.executor.local import LocalPipelineExecutor

SEEN: list[int] = []


def record_rank(data, rank=0, world_size=1):
    SEEN.append(rank)
    yield Document(text=f"rank {rank}", id=str(rank))


def fail_on_rank_two(data, rank=0, world_size=1):
    if rank == 2:
        raise RuntimeError("boom")
    yield from data


def write_rank_file(data, rank=0, world_size=1):
    out_dir = Path(next(iter(data)).text)
    (out_dir / f"{rank:05d}.txt").write_text(f"{rank}/{world_size}")
    yield from ()


@pytest.fixture(autouse=True)
def reset_seen():
    SEEN.clear()


def completed(logging_dir: Path) -> list[str]:
    return sorted(path.name for path in (logging_dir / "completions").iterdir())


def test_marks_every_rank_completed(tmp_path):
    LocalPipelineExecutor([record_rank], tmp_path, tasks=3, workers=1).run()
    assert SEEN == [0, 1, 2]
    assert completed(tmp_path) == ["00000", "00001", "00002"]


def test_rerun_skips_completed_ranks(tmp_path):
    LocalPipelineExecutor([record_rank], tmp_path, tasks=3, workers=1).run()
    SEEN.clear()
    LocalPipelineExecutor([record_rank], tmp_path, tasks=3, workers=1).run()
    assert SEEN == []


def test_failed_rank_is_retried_on_rerun(tmp_path):
    with pytest.raises(RuntimeError):
        LocalPipelineExecutor([record_rank, fail_on_rank_two], tmp_path, tasks=4, workers=1).run()
    assert completed(tmp_path) == ["00000", "00001"]
    SEEN.clear()
    LocalPipelineExecutor([record_rank], tmp_path, tasks=4, workers=1).run()
    assert SEEN == [2, 3]


def test_parallel_workers_run_every_rank(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pipeline = [[Document(text=str(out_dir), id="0")], write_rank_file]
    LocalPipelineExecutor(pipeline, tmp_path / "logs", tasks=4, workers=2).run()
    assert sorted(path.read_text() for path in out_dir.iterdir()) == ["0/4", "1/4", "2/4", "3/4"]
    assert completed(tmp_path / "logs") == ["00000", "00001", "00002", "00003"]
