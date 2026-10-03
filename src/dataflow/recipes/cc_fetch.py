import argparse
import logging
from datetime import UTC, datetime

from dataflow.executor.local import LocalPipelineExecutor
from dataflow.executor.remote import schedule_daily
from dataflow.io import get_datafolder
from dataflow.recipes.cc_language import HF_MIRROR, POLITE_REQUESTS_PER_SECOND, by_source
from dataflow.sources.fetch import RangeFetcher

MIRROR_HTTPS = "https://huggingface.co/buckets/commoncrawl/commoncrawl/resolve/"


def fetch(
    selection: str, output: str, crawls: list[str] | None, executor_for, files_per_task: int = 10, run: str = "jobs"
) -> None:
    """Fetch every selected record into WARC files under `output/warc`: mirrored crawls from the HF bucket, the rest
    from Common Crawl at its polite request rate, shared by all concurrent processes."""
    paths = get_datafolder(selection).list_files(glob_pattern="*/*.parquet")
    available = sorted({path.split("/", 1)[0] for path in paths} - {"summary"})
    wanted = [crawl for crawl in available if crawls is None or crawl in crawls]
    for base, group in by_source(wanted):
        files = sum(path.split("/", 1)[0] in group for path in paths)
        if not files:
            continue
        name = "mirror" if base == HF_MIRROR else "commoncrawl"

        def pipeline(processes: int, base=base, group=group):
            if base == HF_MIRROR:
                return [RangeFetcher(selection, f"{output}/warc", group, MIRROR_HTTPS, requests_per_second=20.0)]
            rate = POLITE_REQUESTS_PER_SECOND / processes
            return [RangeFetcher(selection, f"{output}/warc", group, base, requests_per_second=rate)]

        executor_for(pipeline, f"{output}/logs/{name}/{run}", -(-files // files_per_task)).run()


def local(workers: int = 1):
    def executor_for(pipeline, logging_dir: str, tasks: int):
        return LocalPipelineExecutor(pipeline(workers), logging_dir, tasks=tasks, workers=workers)

    return executor_for


def jobs(max_jobs: int = 2, workers_per_job: int = 8, tasks_per_job: int = 100, timeout: str = "6h"):
    from dataflow.executor.jobs import JobsPipelineExecutor

    def executor_for(pipeline, logging_dir: str, tasks: int):
        processes = min(max_jobs * workers_per_job, tasks)
        return JobsPipelineExecutor(
            pipeline(processes), logging_dir, tasks=tasks, tasks_per_job=min(tasks, tasks_per_job),
            workers_per_job=workers_per_job, max_jobs=max_jobs, timeout=timeout, job_name="cc-fetch",
        )  # fmt: skip

    return executor_for


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch selected Common Crawl records by byte range into WARC files")
    parser.add_argument("selection", help="selection tables written by dataflow.recipes.cc_language")
    parser.add_argument("output", help="folder for the WARC files and logs")
    parser.add_argument("--crawls", nargs="*", help="default: every crawl in the selection")
    parser.add_argument("--jobs", action="store_true", help="fan out on Hugging Face Jobs (fast, for mirrored crawls)")
    parser.add_argument("--schedule", metavar="COMMIT", help="create the daily background job at this commit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.schedule:
        arguments = [args.selection, args.output, *(["--crawls", *args.crawls] if args.crawls else [])]
        print(schedule_daily("dataflow.recipes.cc_fetch", arguments, args.schedule, name="cc-fetch"))
        return
    # the selection grows between runs, so task numbers move: each local run keeps its own completion markers, and
    # files already fetched are skipped by the fetcher itself
    run = "jobs" if args.jobs else datetime.now(UTC).strftime("%Y-%m-%d-%H%M")
    fetch(args.selection, args.output, args.crawls, jobs() if args.jobs else local(), run=run)


if __name__ == "__main__":
    main()
