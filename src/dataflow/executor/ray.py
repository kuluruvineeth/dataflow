import logging
from collections.abc import Callable, Hashable, Iterable
from itertools import islice

import cloudpickle
import ray
from ray.exceptions import RayError, RayTaskError

from dataflow.executor.base import Pipeline, PipelineExecutor, _run_rank
from dataflow.io import DataFolderLike
from dataflow.pipeline.classifiers.base import ScoringStep
from dataflow.utils.stats import PipelineStats

logger = logging.getLogger(__name__)

MODEL_HOST_CPUS = 1


def run_with_refill(
    ranks: Iterable[int],
    workers: int,
    submit: Callable[[int], Hashable],
    wait: Callable[[list[Hashable]], Hashable],
    result: Callable[[Hashable], None],
) -> list[tuple[int, BaseException]]:
    """Keeps at most `workers` ranks in flight: starts one, `wait`s for any one to end, reads its `result` and starts
    the next. Returns the ranks whose result raised, with the error."""
    queue = iter(ranks)
    running = {submit(rank): rank for rank in islice(queue, workers)}
    failures = []
    while running:
        done = wait(list(running))
        rank = running.pop(done)
        try:
            result(done)
        except Exception as error:
            failures.append((rank, error))
        if (following := next(queue, None)) is not None:
            running[submit(following)] = following
    return failures


class ModelHost:
    """One copy of a scoring step in a Ray actor: the model loads once, and every rank sends it batches of text."""

    def __init__(self, step: ScoringStep):
        self.step = step
        step.load()

    def scores(self, texts: list[str]) -> list[float]:
        return self.step.model_scores(texts)


class RemoteScorer:
    def __init__(self, actor: ray.actor.ActorHandle):
        self.actor = actor

    def __call__(self, texts: list[str]) -> list[float]:
        return ray.get(self.actor.scores.remote(texts))


class RayPipelineExecutor(PipelineExecutor):
    def __init__(
        self,
        pipeline: Pipeline,
        logging_dir: DataFolderLike,
        tasks: int = 1,
        workers: int = -1,
        skip_completed: bool = True,
        address: str | None = None,
        cpus_per_task: float = 1,
        max_retries: int = 3,
        share_models: bool = True,
    ):
        """Runs each rank as one Ray task in a new worker process, with at most `workers` in flight (-1: the cluster's
        CPUs for tasks). Ray runs a rank again after its worker dies, up to `max_retries` times; an exception in a
        rank is not retried. With `share_models`, each `ScoringStep` loads its model once, in a `ModelHost` actor.

        `address` goes to `ray.init` when Ray is not running yet; None follows Ray's own order (`RAY_ADDRESS`, the
        latest local cluster, a new local instance), and "local" always starts a new local instance."""
        super().__init__(pipeline, logging_dir, skip_completed)
        self.tasks = tasks
        self.workers = workers
        self.address = address
        self.cpus_per_task = cpus_per_task
        self.max_retries = max_retries
        self.share_models = share_models

    @property
    def world_size(self) -> int:
        return self.tasks

    def run(self) -> PipelineStats:
        ranks = self.get_incomplete_ranks()
        if skipped := self.tasks - len(ranks):
            logger.info("skipping %d already completed tasks", skipped)
        if ranks:
            started = not ray.is_initialized()
            if started:
                ray.init(address=self.address, include_dashboard=False)
            try:
                self._run_on_ray(ranks)
            finally:
                if started:
                    ray.shutdown()
        stats = self.merge_stats()
        logger.info("stats for all %d tasks:\n%s", self.tasks, stats)
        return stats

    def default_workers(self, model_hosts: int) -> int:
        cpus = ray.cluster_resources().get("CPU", 1) - model_hosts * MODEL_HOST_CPUS
        return max(1, int(cpus // self.cpus_per_task))

    def _run_on_ray(self, ranks: list[int]) -> None:
        shared = [step for step in self.pipeline if isinstance(step, ScoringStep)] if self.share_models else []
        host = ray.remote(ModelHost).options(num_cpus=MODEL_HOST_CPUS, max_restarts=3, max_task_retries=3)
        actors = [host.remote(step) for step in shared]
        try:
            for step, actor in zip(shared, actors, strict=True):
                step.scorer = RemoteScorer(actor)
                logger.info("%r scores on one ModelHost actor", step)
            workers = self.workers if self.workers > 0 else self.default_workers(len(actors))
            self.write_run_info(workers=workers)
            logger.info("running %d ranks on Ray, %d at a time", len(ranks), workers)
            executor = ray.put(cloudpickle.dumps(self))
            task = ray.remote(
                num_cpus=self.cpus_per_task, max_calls=1, max_retries=self.max_retries, retry_exceptions=False
            )(_run_rank)
            failures = run_with_refill(
                ranks,
                workers,
                submit=lambda rank: task.remote(executor, rank),
                wait=lambda refs: ray.wait(refs, num_returns=1)[0][0],
                result=ray.get,
            )
        finally:
            for step in shared:
                step.scorer = None
            for actor in actors:
                ray.kill(actor)
        if failures:
            messages = [describe(rank, error) for rank, error in failures]
            for message in messages:
                logger.error("%s", message)
            raise RuntimeError(f"{len(failures)} of {len(ranks)} ranks failed, first: {messages[0]}")


def describe(rank: int, error: BaseException) -> str:
    if isinstance(error, RayTaskError) and error.cause is not None:
        return str(error.cause)
    if isinstance(error, RayError):
        return f"rank {rank} failed: {type(error).__name__}"
    return f"rank {rank} failed: {type(error).__name__}: {error}"
