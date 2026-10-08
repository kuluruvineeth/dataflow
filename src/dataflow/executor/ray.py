import json
import logging
import time
from collections.abc import Callable, Hashable, Iterable
from itertools import islice

import cloudpickle
import ray
from ray.exceptions import GetTimeoutError, RayError, RayTaskError
from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from dataflow.executor.base import Pipeline, PipelineExecutor, _run_rank
from dataflow.io import DataFolderLike
from dataflow.pipeline.classifiers.base import ScoringStep
from dataflow.utils.stats import PipelineStats

logger = logging.getLogger(__name__)

MODEL_HOST_CPUS = 1


class RankLost(Exception):
    """A rank's worker or node died before the rank committed; `cause` is the error from Ray."""

    def __init__(self, cause: BaseException):
        super().__init__(f"{type(cause).__name__}")
        self.cause = cause


def run_with_refill(
    ranks: Iterable[int],
    workers: int,
    submit: Callable[[int, int, int, BaseException | None], Hashable],
    wait: Callable[[list[Hashable]], Hashable],
    result: Callable[[Hashable, int], None],
    max_retries: int = 0,
) -> list[tuple[int, BaseException]]:
    """Keeps at most `workers` ranks in flight, one in each slot: `submit(rank, slot, submission, previous_error)`
    starts one, `wait` returns any one that ended, and `result(handle, rank)` reads it. When `result` raises
    `RankLost`, the same rank is submitted again at once in the same slot, at most `max_retries` times. Returns the
    ranks whose result raised anything else, or that were lost too often, with the error."""
    queue = iter(ranks)
    running = {}
    for slot, rank in enumerate(islice(queue, workers)):
        running[submit(rank, slot, 1, None)] = (rank, slot, 1)
    failures = []
    while running:
        done = wait(list(running))
        rank, slot, submission = running.pop(done)
        try:
            result(done, rank)
        except RankLost as lost:
            if submission <= max_retries:
                running[submit(rank, slot, submission + 1, lost.cause)] = (rank, slot, submission + 1)
                continue
            failures.append((rank, lost.cause))
        except Exception as error:
            failures.append((rank, error))
        if (following := next(queue, None)) is not None:
            running[submit(following, slot, 1, None)] = (following, slot, 1)
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
        placement_strategy: str | None = "PACK",
        placement_timeout: float = 300,
    ):
        """Runs each rank as one Ray task in a new worker process, with at most `workers` in flight (-1: the cluster's
        CPUs for tasks). When a rank's worker or node dies before the rank's marker exists, the executor submits the
        rank again itself, up to `max_retries` times, and records each submission in `submissions/`; an exception in
        a rank is not retried. With `share_models`, each `ScoringStep` loads its model once, in a `ModelHost` actor.

        The ranks and the actors run in one placement group with `placement_strategy` ("PACK", "SPREAD", ...; None for
        no group), which must be ready within `placement_timeout` seconds.

        `address` goes to `ray.init` when Ray is not running yet; None follows Ray's own order (`RAY_ADDRESS`, the
        latest local cluster, a new local instance), and "local" always starts a new local instance."""
        super().__init__(pipeline, logging_dir, skip_completed)
        self.tasks = tasks
        self.workers = workers
        self.address = address
        self.cpus_per_task = cpus_per_task
        self.max_retries = max_retries
        self.share_models = share_models
        self.placement_strategy = placement_strategy
        self.placement_timeout = placement_timeout

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

    def _reserve(self, workers: int, model_hosts: int) -> PlacementGroup | None:
        if self.placement_strategy is None:
            return None
        bundles = [{"CPU": self.cpus_per_task}] * workers + [{"CPU": MODEL_HOST_CPUS}] * model_hosts
        group = placement_group(bundles, strategy=self.placement_strategy)
        try:
            ray.get(group.ready(), timeout=self.placement_timeout)
        except GetTimeoutError:
            remove_placement_group(group)
            raise RuntimeError(
                f"placement group {bundles} ({self.placement_strategy}) not ready after {self.placement_timeout} s"
            ) from None
        logger.info("placement group ready: %d rank bundles, %d model bundles", workers, model_hosts)
        return group

    def _record(self, rank: int, submission: int, previous: BaseException | None) -> None:
        record = {"rank": rank, "submission": submission, "time": time.time()}
        if previous is not None:
            record["previous_error"] = describe(rank, previous)
            logger.warning("rank %d lost on submission %d: %s; submission %d", rank, submission - 1,
                           type(previous).__name__, submission)  # fmt: skip
        with self.logging_dir.open(f"submissions/{rank:05d}-{submission}.json", "w") as file:
            json.dump(record, file)

    def _run_on_ray(self, ranks: list[int]) -> None:
        shared = [step for step in self.pipeline if isinstance(step, ScoringStep)] if self.share_models else []
        workers = self.workers if self.workers > 0 else self.default_workers(len(shared))
        group = self._reserve(workers, len(shared))

        def in_bundle(index: int) -> dict:
            if group is None:
                return {}
            return {"scheduling_strategy": PlacementGroupSchedulingStrategy(group, placement_group_bundle_index=index)}

        actors = []
        try:
            host = ray.remote(ModelHost)
            for index, step in enumerate(shared):
                options = {"num_cpus": MODEL_HOST_CPUS, "max_restarts": 3, "max_task_retries": -1}
                actors.append(host.options(**options, **in_bundle(workers + index)).remote(step))
                step.scorer = RemoteScorer(actors[-1])
                logger.info("%r scores on one ModelHost actor", step)
            self.write_run_info(workers=workers, placement_strategy=self.placement_strategy)
            logger.info("running %d ranks on Ray, %d at a time", len(ranks), workers)
            executor = ray.put(cloudpickle.dumps(self))
            task = ray.remote(num_cpus=self.cpus_per_task, max_calls=1, max_retries=0)(_run_rank)

            def submit(rank: int, slot: int, submission: int, previous: BaseException | None) -> ray.ObjectRef:
                self._record(rank, submission, previous)
                return task.options(**in_bundle(slot)).remote(executor, rank)

            def result(ref: ray.ObjectRef, rank: int) -> None:
                try:
                    ray.get(ref)
                except RayTaskError:
                    raise
                except RayError as error:
                    if self.logging_dir.isfile(self._completion_path(rank)):
                        logger.warning("rank %d lost after its marker (%s); it is done", rank, type(error).__name__)
                        return
                    raise RankLost(error) from error

            failures = run_with_refill(
                ranks,
                workers,
                submit=submit,
                wait=lambda refs: ray.wait(refs, num_returns=1)[0][0],
                result=result,
                max_retries=self.max_retries,
            )
        finally:
            for step in shared:
                step.scorer = None
            for actor in actors:
                ray.kill(actor)
            if group is not None:
                remove_placement_group(group)
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
