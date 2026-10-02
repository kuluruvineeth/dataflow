import logging

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
        if ranks:
            self.write_run_info(workers=self.workers)
            self.run_ranks(ranks, self.workers, self.start_method)
        stats = self.merge_stats()
        logger.info("stats for all %d tasks:\n%s", self.tasks, stats)
        return stats
