import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

from dataflow.data import DocumentsPipeline
from dataflow.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)

Pipeline = list[PipelineStep | Callable[..., DocumentsPipeline] | Sequence]


class PipelineExecutor(ABC):
    def __init__(self, pipeline: Pipeline, logging_dir: str | Path, skip_completed: bool = True):
        self.pipeline = pipeline
        self.logging_dir = Path(logging_dir)
        self.skip_completed = skip_completed

    @property
    @abstractmethod
    def world_size(self) -> int: ...

    @abstractmethod
    def run(self) -> None: ...

    def _run_for_rank(self, rank: int) -> None:
        if self.is_rank_completed(rank):
            logger.info("rank %d already completed, skipping", rank)
            return
        data = None
        for step in self.pipeline:
            if callable(step):
                data = step(data, rank, self.world_size)
            elif isinstance(step, Sequence) and not isinstance(step, str):
                data = step
            else:
                raise ValueError(f"not a pipeline step: {step!r}")
        if data is not None:
            deque(data, maxlen=0)
        self.mark_rank_as_completed(rank)
        logger.info("rank %d completed", rank)

    def _completion_path(self, rank: int) -> Path:
        return self.logging_dir / "completions" / f"{rank:05d}"

    def is_rank_completed(self, rank: int) -> bool:
        return self.skip_completed and self._completion_path(rank).is_file()

    def mark_rank_as_completed(self, rank: int) -> None:
        path = self._completion_path(rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def get_incomplete_ranks(self) -> list[int]:
        return [rank for rank in range(self.world_size) if not self.is_rank_completed(rank)]
