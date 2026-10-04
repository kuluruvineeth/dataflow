import time

import pytest

from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.base import PipelineStep


class KeywordOnlyError(Exception):
    """Like httpx.HTTPStatusError: pickles, but cannot be unpickled, because its arguments are keyword-only."""

    def __init__(self, *, detail: str):
        super().__init__(detail)


class FailsOnRankOne(PipelineStep):
    def run(self, data=None, rank: int = 0, world_size: int = 1):
        if rank == 1:
            raise KeywordOnlyError(detail="403 Forbidden")
        yield from ()


class FailsOnRankZero(PipelineStep):
    def run(self, data=None, rank: int = 0, world_size: int = 1):
        if rank == 0:
            raise ValueError("broken input")
        time.sleep(0.2)
        yield from ()


def test_one_failed_rank_does_not_cancel_the_queued_ones(tmp_path):
    executor = LocalPipelineExecutor([FailsOnRankZero()], str(tmp_path), tasks=10, workers=2)
    with pytest.raises(RuntimeError, match="1 of 10 ranks failed, first: rank 0 failed: ValueError: broken input"):
        executor.run()
    assert executor.completed_ranks() == set(range(1, 10))


def test_an_error_that_cannot_cross_processes_fails_only_its_own_rank(tmp_path):
    executor = LocalPipelineExecutor([FailsOnRankOne()], str(tmp_path), tasks=3, workers=3)
    with pytest.raises(RuntimeError, match="rank 1 failed: KeywordOnlyError: 403 Forbidden"):
        executor.run()
    assert executor.is_rank_completed(0) and executor.is_rank_completed(2)
    assert not executor.is_rank_completed(1)
