import json
import logging
import multiprocessing
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import repeat

import cloudpickle

from dataflow.data import DocumentsPipeline
from dataflow.executor.progress import ProgressReporter
from dataflow.io import DataFolderLike, get_datafolder
from dataflow.pipeline.base import PipelineStep
from dataflow.utils.stats import PipelineStats

logger = logging.getLogger(__name__)

Pipeline = list[PipelineStep | Callable[..., DocumentsPipeline] | Sequence]


def _run_rank(executor: bytes, rank: int) -> None:
    cloudpickle.loads(executor)._run_for_rank(rank)


class PipelineExecutor(ABC):
    progress_interval: float = 30

    def __init__(self, pipeline: Pipeline, logging_dir: DataFolderLike, skip_completed: bool = True):
        self.pipeline = pipeline
        self.logging_dir = get_datafolder(logging_dir)
        self.skip_completed = skip_completed

    @property
    @abstractmethod
    def world_size(self) -> int: ...

    @abstractmethod
    def run(self) -> PipelineStats: ...

    def _run_for_rank(self, rank: int) -> None:
        if self.is_rank_completed(rank):
            logger.info("rank %d already completed, skipping", rank)
            return
        stats = [step.stats for step in self.pipeline if isinstance(step, PipelineStep)]
        with ProgressReporter(self.logging_dir, rank, stats, self.progress_interval) as progress:
            data = None
            for step in self.pipeline:
                if callable(step):
                    data = step(data, rank, self.world_size)
                elif isinstance(step, Sequence) and not isinstance(step, str):
                    data = step
                else:
                    raise ValueError(f"not a pipeline step: {step!r}")
            for _ in data or ():
                progress.documents += 1
            self.save_rank_stats(rank)
        self.mark_rank_as_completed(rank)
        logger.info("rank %d completed", rank)

    def write_run_info(self, **extra) -> None:
        info = {"executor": type(self).__name__, "tasks": self.world_size, "started": time.time(), **extra}
        with self.logging_dir.open("run.json", "w") as file:
            json.dump(info, file)

    def run_ranks(self, ranks: list[int], workers: int, start_method: str = "spawn") -> None:
        if workers == 1:
            pipeline = self.pipeline
            try:
                for rank in ranks:
                    self.pipeline = deepcopy(pipeline)
                    self._run_for_rank(rank)
            finally:
                self.pipeline = pipeline
            return
        ctx = multiprocessing.get_context(start_method)
        executor = cloudpickle.dumps(self)
        with ProcessPoolExecutor(min(workers, len(ranks)), mp_context=ctx) as pool:
            for _ in pool.map(_run_rank, repeat(executor), ranks):
                pass

    def save_rank_stats(self, rank: int) -> None:
        stats = PipelineStats([step.stats for step in self.pipeline if isinstance(step, PipelineStep)])
        with self.logging_dir.open(f"stats/{rank:05d}.json", "w") as file:
            file.write(stats.to_json())

    def merge_stats(self) -> PipelineStats:
        merged = PipelineStats()
        if self.logging_dir.exists("stats"):
            for path in self.logging_dir.list_files("stats"):
                with self.logging_dir.open(path, "r") as file:
                    merged += PipelineStats.from_json(file.read())
        with self.logging_dir.open("stats.json", "w") as file:
            file.write(merged.to_json())
        return merged

    def _completion_path(self, rank: int) -> str:
        return f"completions/{rank:05d}"

    def is_rank_completed(self, rank: int) -> bool:
        return self.skip_completed and self.logging_dir.isfile(self._completion_path(rank))

    def mark_rank_as_completed(self, rank: int) -> None:
        self.logging_dir.open(self._completion_path(rank), "w").close()

    def get_incomplete_ranks(self) -> list[int]:
        return [rank for rank in range(self.world_size) if not self.is_rank_completed(rank)]
