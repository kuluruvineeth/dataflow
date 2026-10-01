import logging
import multiprocessing
from copy import deepcopy

from dataflow.executor.base import Pipeline, PipelineExecutor
from dataflow.io import DataFolderLike

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

    def run(self) -> None:
        ranks = self.get_incomplete_ranks()
        if skipped := self.tasks - len(ranks):
            logger.info("skipping %d already completed tasks", skipped)
        if not ranks:
            return
        if self.workers == 1:
            self._run_sequential(ranks)
        else:
            self._run_parallel(ranks)

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
        with ctx.Pool(min(self.workers, len(ranks))) as pool:
            for _ in pool.imap_unordered(self._run_for_rank, ranks):
                pass
