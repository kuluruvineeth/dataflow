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
        shard: tuple[int, int] | None = None,
    ):
        """`shard=(index, count)` runs only the ranks with `rank % count == index`, so `count` machines can share one
        run and its completion markers without doing a rank twice."""
        super().__init__(pipeline, logging_dir, skip_completed)
        self.tasks = tasks
        self.workers = tasks if workers == -1 else workers
        self.start_method = start_method
        if shard is not None and not 0 <= shard[0] < shard[1]:
            raise ValueError(f"shard index must be in [0, {shard[1]}), got {shard[0]}")
        self.shard = shard

    @property
    def world_size(self) -> int:
        return self.tasks

    def run(self) -> PipelineStats:
        ranks = self.get_incomplete_ranks()
        if self.shard is not None:
            index, count = self.shard
            ranks = [rank for rank in ranks if rank % count == index]
        if skipped := self.tasks - len(ranks):
            logger.info("skipping %d already completed tasks", skipped)
        if ranks:
            self.write_run_info(workers=self.workers)
            self.run_ranks(ranks, self.workers, self.start_method)
        stats = self.merge_stats()
        logger.info("stats for all %d tasks:\n%s", self.tasks, stats)
        return stats
