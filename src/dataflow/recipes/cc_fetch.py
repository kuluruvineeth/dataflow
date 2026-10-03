import argparse
import logging

from huggingface_hub import HfApi, get_token

from dataflow.executor.local import LocalPipelineExecutor
from dataflow.io import get_datafolder
from dataflow.recipes.cc_language import HF_MIRROR, POLITE_REQUESTS_PER_SECOND, by_source
from dataflow.sources.fetch import RangeFetcher

UV_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
MIRROR_HTTPS = "https://huggingface.co/buckets/commoncrawl/commoncrawl/resolve/"
SOURCE = "https://github.com/kuluruvineeth/dataflow/archive/{commit}.tar.gz"


def fetch(selection: str, output: str, crawls: list[str] | None, executor_for, files_per_task: int = 10) -> None:
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

        executor_for(pipeline, f"{output}/logs/{name}", -(-files // files_per_task)).run()


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


def schedule(selection: str, output: str, commit: str, crawls: list[str] | None = None, flavor: str = "cpu-basic"):
    """A daily job that continues the fetch for up to 23 hours, never two at once; it skips what is already fetched,
    so it runs until everything is, for as many days as that takes. Runs this repository's code at `commit`."""
    command = ["uv", "run", "--with", f"dataflow @ {SOURCE.format(commit=commit)}", "python", "-m",
               "dataflow.recipes.cc_fetch", selection, output]  # fmt: skip
    if crawls:
        command += ["--crawls", *crawls]
    return HfApi().create_scheduled_job(
        image=UV_IMAGE, command=command, schedule="0 3 * * *", concurrency=False, timeout="23h", flavor=flavor,
        secrets={"HF_TOKEN": get_token()}, name="cc-fetch",
    )  # fmt: skip


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
        print(schedule(args.selection, args.output, args.schedule, args.crawls))
        return
    fetch(args.selection, args.output, args.crawls, jobs() if args.jobs else local())


if __name__ == "__main__":
    main()
