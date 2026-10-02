import json
import logging
import os
import subprocess
import tempfile
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import cloudpickle

import dataflow
from dataflow.executor.base import Pipeline, PipelineExecutor
from dataflow.io import DataFolderLike
from dataflow.utils.stats import PipelineStats

logger = logging.getLogger(__name__)

RANKS_ENV = "DATAFLOW_JOB_RANKS"
UV_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
CODE_MOUNT = "/dataflow-code"
FINISHED = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}


class JobsPipelineExecutor(PipelineExecutor):
    """Runs a pipeline on Hugging Face Jobs.

    The coordinator (this process) uploads the dataflow wheel and the pickled executor next to the logging folder,
    then launches one Job per chunk of ranks. Each Job installs the wheel, unpickles the executor and runs its ranks
    in parallel on its own CPUs. Completion markers make reruns launch only the ranks that are still missing.
    The logging folder must live in a Hugging Face bucket (`hf://buckets/<namespace>/<bucket>/<path>`).
    """

    def __init__(
        self,
        pipeline: Pipeline,
        logging_dir: DataFolderLike,
        tasks: int,
        tasks_per_job: int = 8,
        workers_per_job: int = -1,
        flavor: str = "cpu-upgrade",
        timeout: str = "2h",
        max_jobs: int = -1,
        max_retries: int = 1,
        poll_interval: float = 30,
        job_name: str = "dataflow",
        image: str = UV_IMAGE,
        skip_completed: bool = True,
    ):
        super().__init__(pipeline, logging_dir, skip_completed)
        self.tasks = tasks
        self.tasks_per_job = tasks_per_job
        self.workers_per_job = tasks_per_job if workers_per_job == -1 else workers_per_job
        self.flavor = flavor
        self.timeout = timeout
        self.max_jobs = max_jobs
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.job_name = job_name
        self.image = image
        self.wheel_name: str | None = None

    @property
    def world_size(self) -> int:
        return self.tasks

    def run(self) -> PipelineStats:
        if RANKS_ENV in os.environ:
            ranks = [int(rank) for rank in os.environ[RANKS_ENV].split(",")]
            # fork, so worker processes inherit the job's logging setup
            self.run_ranks(ranks, self.workers_per_job, start_method="fork")
            return PipelineStats()
        return self.launch()

    def launch(self) -> PipelineStats:
        ranks = self.incomplete(list(range(self.tasks)))
        if skipped := self.tasks - len(ranks):
            logger.info("skipping %d already completed tasks", skipped)
        if ranks:
            self.write_run_info(flavor=self.flavor, tasks_per_job=self.tasks_per_job)
            self.prepare()
            chunks = [ranks[i : i + self.tasks_per_job] for i in range(0, len(ranks), self.tasks_per_job)]
            self.record_jobs(self.run_chunks(chunks))
        if missing := self.incomplete(list(range(self.tasks))):
            logger.warning("%d tasks did not complete: %s", len(missing), missing)
        stats = self.merge_stats()
        logger.info("stats for all %d tasks:\n%s", self.tasks, stats)
        return stats

    def incomplete(self, ranks: list[int]) -> list[int]:
        self.logging_dir.fs.invalidate_cache()
        return [rank for rank in ranks if not self.is_rank_completed(rank)]

    def run_chunks(self, chunks: list[list[int]]) -> dict[str, datetime]:
        pending = deque(enumerate(chunks))
        running: dict[str, tuple[int, list[int]]] = {}
        attempts = [0] * len(chunks)
        ended: dict[str, datetime] = {}
        limit = len(chunks) if self.max_jobs == -1 else self.max_jobs
        while pending or running:
            while pending and len(running) < limit:
                index, ranks = pending.popleft()
                job_id = self.submit(ranks)
                self.record_launch(job_id, ranks)
                attempts[index] += 1
                running[job_id] = (index, ranks)
                logger.info("job %s: ranks %s (attempt %d)", job_id, ranks, attempts[index])
            time.sleep(self.poll_interval)
            for job_id, (index, ranks) in list(running.items()):
                stage = self.status(job_id)
                if stage not in FINISHED:
                    continue
                del running[job_id]
                ended[job_id] = datetime.now(UTC)
                missing = self.incomplete(ranks)
                logger.info("job %s finished as %s, %d of its ranks missing", job_id, stage, len(missing))
                if missing and attempts[index] <= self.max_retries:
                    pending.append((index, missing))
        return ended

    def record_launch(self, job_id: str, ranks: list[int]) -> None:
        with self.logging_dir.open(f"jobs/launched/{job_id}.json", "w") as file:
            json.dump({"id": job_id, "ranks": ranks, "flavor": self.flavor, "launched": time.time()}, file)

    def prepare(self) -> None:
        wheel = build_wheel()
        self.wheel_name = wheel.name
        with self.logging_dir.open(f"code/{wheel.name}", "wb") as file:
            file.write(wheel.read_bytes())
        with self.logging_dir.open("executor.pik", "wb") as file:
            cloudpickle.dump(self, file)

    def bucket_location(self) -> tuple[str, str]:
        path = self.logging_dir.path.strip("/")
        if not path.startswith("buckets/") or len(parts := path.split("/", 3)) < 4:
            raise ValueError(f"logging_dir must be hf://buckets/<namespace>/<bucket>/<path>, got {path!r}")
        return f"{parts[1]}/{parts[2]}", parts[3]

    def submit(self, ranks: list[int]) -> str:
        from huggingface_hub import Volume, get_token, run_job

        bucket, prefix = self.bucket_location()
        code = Volume(type="bucket", source=bucket, mount_path=CODE_MOUNT, path=f"{prefix}/code", read_only=True)
        wheel = f"{CODE_MOUNT}/{self.wheel_name}"
        executor = f"hf://buckets/{bucket}/{prefix}/executor.pik"
        launch = ["python", "-m", "dataflow.executor.launch", executor]
        job = run_job(
            image=self.image,
            command=["uv", "run", "--no-project", "--with", wheel, *launch],
            env={RANKS_ENV: ",".join(map(str, ranks)), "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1"},
            secrets={"HF_TOKEN": get_token()},
            flavor=self.flavor,
            timeout=self.timeout,
            labels={"dataflow": self.job_name},
            volumes=[code],
        )
        return job.id

    def status(self, job_id: str) -> str:
        from huggingface_hub import inspect_job

        return str(inspect_job(job_id=job_id).status.stage)

    def describe(self, ended: dict[str, datetime]) -> list[dict]:
        from huggingface_hub import inspect_job, list_jobs_hardware

        price = {hardware.name: hardware.unit_cost_usd for hardware in list_jobs_hardware()}
        records = []
        for job_id, seen_ending in ended.items():
            job = inspect_job(job_id=job_id)
            # cancelled Jobs have no finished_at; fall back to when the coordinator saw them end
            end = job.finished_at or seen_ending
            minutes = (end - job.started_at).total_seconds() / 60 if job.started_at else 0.0
            records.append(
                {
                    "id": job_id,
                    "url": job.url,
                    "flavor": job.flavor,
                    "stage": str(job.status.stage),
                    "minutes": round(minutes, 2),
                    "cost_usd": round(minutes * price.get(job.flavor, 0), 4),
                    "estimated": job.finished_at is None,
                }
            )
        return records

    def record_jobs(self, ended: dict[str, datetime]) -> None:
        records = self.describe(ended)
        with self.logging_dir.open(f"jobs/{int(time.time())}.json", "w") as file:
            file.write(json.dumps(records, indent=2))
        minutes = sum(record["minutes"] for record in records)
        cost = sum(record["cost_usd"] for record in records)
        logger.info("%d jobs, %.1f job-minutes, $%.4f", len(records), minutes, cost)


def build_wheel() -> Path:
    root = Path(dataflow.__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"cannot find the dataflow project to build a wheel from (looked in {root})")
    out = Path(tempfile.mkdtemp(prefix="dataflow-wheel-"))
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out)], cwd=root, check=True, capture_output=True)
    return next(out.glob("dataflow-*.whl"))
