import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from dataflow.io import DataFolder, DataFolderLike, get_datafolder
from dataflow.utils.stats import PipelineStats


@dataclass
class TaskState:
    rank: int
    status: str
    documents: int = 0
    documents_per_second: float = 0.0
    started: float | None = None
    updated: float | None = None
    error: str | None = None


@dataclass
class RunStatus:
    path: str
    tasks: list[TaskState]
    stats: PipelineStats
    errors: dict[int, str] = field(default_factory=dict)
    job_minutes: float | None = None
    cost_usd: float | None = None
    eta_seconds: float | None = None

    def count(self, status: str) -> int:
        return sum(task.status == status for task in self.tasks)

    @property
    def documents_per_second(self) -> float:
        return sum(task.documents_per_second for task in self.tasks if task.status == "running")


def read_json(folder: DataFolder, path: str):
    with folder.open(path, "r") as file:
        return json.load(file)


def read_folder(folder: DataFolder, subdirectory: str) -> dict[int, dict]:
    if not folder.exists(subdirectory):
        return {}
    return {
        int(path.rsplit("/", 1)[-1].split(".")[0]): read_json(folder, path) for path in folder.list_files(subdirectory)
    }


def collect(
    logging_dir: DataFolderLike,
    stale_after: float = 120,
    now: float | None = None,
    job_cost: Callable[[list[dict], float], tuple[float, float]] | None = None,
) -> RunStatus:
    folder = get_datafolder(logging_dir)
    folder.fs.invalidate_cache()
    now = now or time.time()
    info = read_json(folder, "run.json") if folder.exists("run.json") else {}
    progress = read_folder(folder, "progress")
    finished_stats = read_folder(folder, "stats")
    errors = read_folder(folder, "errors")
    completed = (
        {int(path.rsplit("/", 1)[-1]) for path in folder.list_files("completions")}
        if folder.exists("completions")
        else set()
    )
    known = set(progress) | completed | set(errors) | set(finished_stats)
    tasks_total = info.get("tasks", max(known, default=-1) + 1)

    tasks, merged = [], PipelineStats()
    for rank in range(tasks_total):
        state = progress.get(rank, {})
        if rank in completed:
            status = "done"
        elif rank in errors:
            status = "failed"
        elif not state:
            status = "pending"
        elif now - state["updated"] > stale_after:
            status = "stale"
        else:
            status = "running"
        tasks.append(
            TaskState(
                rank=rank,
                status=status,
                documents=state.get("documents", 0),
                documents_per_second=state.get("documents_per_second", 0.0),
                started=state.get("started"),
                updated=state.get("updated"),
                error=state.get("error"),
            )
        )
        source = finished_stats.get(rank) or state.get("stats")
        if source:
            merged += PipelineStats.from_json(json.dumps(source))

    status = RunStatus(
        path=folder.path,
        tasks=tasks,
        stats=merged,
        errors={rank: error["traceback"].strip().splitlines()[-1] for rank, error in errors.items()},
        eta_seconds=estimate_eta(tasks, now),
    )
    launched = list(read_folder_by_name(folder, "jobs/launched").values())
    if launched:
        status.job_minutes, status.cost_usd = (job_cost or hub_job_cost)(launched, now)
    return status


def read_folder_by_name(folder: DataFolder, subdirectory: str) -> dict[str, dict]:
    if not folder.exists(subdirectory):
        return {}
    return {path.rsplit("/", 1)[-1]: read_json(folder, path) for path in folder.list_files(subdirectory)}


def estimate_eta(tasks: list[TaskState], now: float) -> float | None:
    durations = [
        t.updated - t.started for t in tasks if t.status == "done" and t.started is not None and t.updated is not None
    ]
    running = [t for t in tasks if t.status == "running"]
    pending = sum(t.status in ("pending", "stale") for t in tasks)
    if not durations or not (running or pending):
        return None
    average = sum(durations) / len(durations)
    in_flight = max((average - (now - t.started) for t in running if t.started is not None), default=0.0)
    waves = -(-pending // max(len(running), 1))
    return max(in_flight, 0.0) + waves * average


def hub_job_cost(launched: list[dict], now: float) -> tuple[float, float]:
    from huggingface_hub import inspect_job, list_jobs_hardware

    price = {hardware.name: hardware.unit_cost_usd for hardware in list_jobs_hardware()}
    minutes = cost = 0.0
    for record in launched:
        job = inspect_job(job_id=record["id"])
        if not job.started_at:
            continue
        end = job.finished_at.timestamp() if job.finished_at else now
        job_minutes = max(end - job.started_at.timestamp(), 0.0) / 60
        minutes += job_minutes
        cost += job_minutes * price.get(record["flavor"], 0.0)
    return minutes, cost


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def funnel(stats: PipelineStats) -> list[str]:
    lines = []
    for step in stats.stats:
        metrics = {label: metric.total for label, metric in step.metrics.items()}
        if "documents" in metrics:
            lines.append(f"  {step.name:<24} read {metrics['documents']:>12,.0f}")
        elif "forwarded" in metrics:
            total = metrics.get("total", 0)
            kept = metrics["forwarded"]
            reasons = sorted(
                (
                    (label.removeprefix("dropped_"), value)
                    for label, value in metrics.items()
                    if label.startswith("dropped_")
                ),
                key=lambda item: -item[1],
            )
            reasons += [(label, metrics[label]) for label in ("timeout",) if label in metrics]
            detail = ", ".join(f"{reason} {value:,.0f}" for reason, value in reasons[:4])
            share = f"{kept / total:6.1%}" if total else "     -"
            lines.append(
                f"  {step.name:<24} kept {kept:>12,.0f} {share}   dropped {metrics.get('dropped', 0):,.0f}  {detail}"
            )
        elif "total" in metrics:
            lines.append(f"  {step.name:<24} wrote {metrics['total']:>11,.0f}")
    return lines


def render(status: RunStatus, max_tasks: int = 40) -> str:
    counts = " · ".join(f"{status.count(name)} {name}" for name in ("done", "running", "stale", "failed", "pending"))
    lines = [f"{status.path}", f"{len(status.tasks)} tasks: {counts}"]
    summary = f"throughput {status.documents_per_second:,.0f} docs/s · ETA {duration(status.eta_seconds)}"
    if status.cost_usd is not None:
        summary += f" · cost so far ${status.cost_usd:.4f} ({status.job_minutes:.1f} job-minutes)"
    lines.append(summary)
    unfinished = [task for task in status.tasks if task.status != "done"]
    if unfinished:
        lines += ["", "  rank  status      docs out   docs/s  updated"]
        now = time.time()
        for task in unfinished[:max_tasks]:
            updated = f"{now - task.updated:4.0f}s ago" if task.updated else ""
            rate = f"{task.documents_per_second:>8.1f}"
            lines.append(f"  {task.rank:>4}  {task.status:<9} {task.documents:>10,} {rate}  {updated}")
        if len(unfinished) > max_tasks:
            lines.append(f"  ... {len(unfinished) - max_tasks} more")
    if status.stats.stats:
        lines += ["", "funnel:"] + funnel(status.stats)
    if status.errors:
        lines += ["", "errors:"] + [f"  rank {rank}: {message}" for rank, message in sorted(status.errors.items())]
    return "\n".join(lines)


def trackio_metrics(status: RunStatus) -> dict[str, float]:
    metrics = {
        "tasks/done": status.count("done"),
        "tasks/running": status.count("running"),
        "tasks/failed": status.count("failed"),
        "tasks/stale": status.count("stale"),
        "documents_per_second": status.documents_per_second,
    }
    if status.cost_usd is not None:
        metrics["cost_usd"] = status.cost_usd
    for step in status.stats.stats:
        for label in ("documents", "forwarded", "dropped", "total"):
            if label in step.metrics:
                metrics[f"funnel/{step.name}/{label}"] = step.metrics[label].total
    return metrics
