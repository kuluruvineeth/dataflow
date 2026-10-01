import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from dataflow.executor.base import Pipeline, PipelineExecutor
from dataflow.io import DataFolderLike
from dataflow.utils.stats import PipelineStats

logger = logging.getLogger(__name__)


class LocalPipelineExecutor(PipelineExecutor):
    def __init__(
        self,
        pipeline: Pipeline,
        logging_dir: DataFolderLike,
        tasks: int = 1,
        workers: int = -1,
        skip_completed: bool = True,
        start_method: str = "spawn",
    ):
        super().__init__(pipeline, logging_dir, skip_completed)
        self.tasks = tasks
        self.workers = tasks if workers == -1 else workers
        self.start_method = start_method

    @property
    def world_size(self) -> int:
        return self.tasks

    def run(self) -> PipelineStats:
        ranks = self.get_incomplete_ranks()
        if skipped := self.tasks - len(ranks):
            logger.info("skipping %d already completed tasks", skipped)
        if ranks and self.workers == 1:
            self._run_sequential(ranks)
        elif ranks:
            self._run_parallel(ranks)
        stats = self.merge_stats()
        logger.info("stats for all %d tasks:\n%s", self.tasks, stats)
        return stats

    def _run_sequential(self, ranks: list[int]) -> None:
        pipeline = self.pipeline
        try:
            for rank in ranks:
                self.pipeline = deepcopy(pipeline)
                self._run_for_rank(rank)
        finally:
            self.pipeline = pipeline

    def _run_parallel(self, ranks: list[int]) -> None:
        ctx = multiprocessing.get_context(self.start_method)
        with ProcessPoolExecutor(min(self.workers, len(ranks)), mp_context=ctx) as pool:
            for _ in pool.map(self._run_for_rank, ranks):
                pass
