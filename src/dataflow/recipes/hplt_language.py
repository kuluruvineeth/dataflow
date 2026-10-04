import argparse
import json
import logging
import re
import sys

import httpx

from dataflow.executor.local import LocalPipelineExecutor
from dataflow.executor.remote import launch_driver, schedule_daily
from dataflow.io import get_datafolder
from dataflow.pipeline.readers.hplt import HPLT_POOL, HpltPoolReader
from dataflow.pipeline.writers import ParquetWriter

SKIPPED = ("archivebot",)


def pool_batches(skip: tuple[str, ...] = SKIPPED, common_crawl_before: str | None = None) -> list[str]:
    """Every batch of the pool (`<crawl>/<n>`), from each crawl's `.map` file, sorted, minus the skipped crawls and,
    with `common_crawl_before`, minus Common Crawl crawls from that one on (those we can select ourselves)."""
    listing = httpx.get(HPLT_POOL, timeout=60).text
    batches = []
    for crawl in re.findall(r'href="([^"/?]+)\.map"', listing):
        if crawl in skip or (common_crawl_before and crawl.startswith("CC-MAIN") and crawl >= common_crawl_before):
            continue
        lines = httpx.get(f"{HPLT_POOL}{crawl}.map", timeout=60).text.split()
        batches += sorted({line.removeprefix(HPLT_POOL).rsplit("/", 1)[0] for line in lines})
    return sorted(batches)


def frozen_batches(output: str, common_crawl_before: str | None = None) -> list[str]:
    """The batch list of this extraction, written on the first run and reused, so task numbers never move."""
    folder = get_datafolder(output)
    if folder.exists("batches.json"):
        with folder.open("batches.json", "r") as file:
            return json.load(file)
    batches = pool_batches(common_crawl_before=common_crawl_before)
    with folder.open("batches.json", "w") as file:
        json.dump(batches, file)
    return batches


def extract(output: str, language: str = "tel_Telu", workers: int = 8, jobs: int = 0, budget: float | None = None,
            common_crawl_before: str | None = None) -> None:  # fmt: skip
    """One task per batch; completed batches are skipped, so a stopped run continues where it was. With `jobs`, that
    many Jobs at once, `workers` processes each, launching stops once `budget` dollars are spent."""
    batches = frozen_batches(output, common_crawl_before)
    pipeline = [HpltPoolReader(batches, language), ParquetWriter(f"{output}/documents")]
    if not jobs:
        LocalPipelineExecutor(pipeline, f"{output}/logs", tasks=len(batches), workers=workers).run()
        return
    from dataflow.executor.jobs import JobsPipelineExecutor

    JobsPipelineExecutor(
        pipeline, f"{output}/logs", tasks=len(batches), tasks_per_job=5 * workers, workers_per_job=workers,
        max_jobs=jobs, timeout="6h", budget_usd=budget, job_name="hplt-language",
    ).run()  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one language's documents from the HPLT 4.0 pool")
    parser.add_argument("output", help="folder for the batch list, documents and logs")
    parser.add_argument("--language", default="tel_Telu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--schedule", metavar="COMMIT", help="create a daily Job that continues the extraction")
    parser.add_argument("--jobs", type=int, default=0, help="run on this many Jobs at once instead of locally")
    parser.add_argument("--budget", type=float, help="with --jobs: stop launching once this many dollars are spent")
    parser.add_argument("--common-crawl-before", help="leave out Common Crawl crawls from this one on (first run)")
    parser.add_argument("--driver", metavar="COMMIT", help="run this command itself as a Job, from this commit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.schedule:
        arguments = [args.output, "--language", args.language, "--workers", str(args.workers)]
        print(
            schedule_daily("dataflow.recipes.hplt_language", arguments, args.schedule, "hplt-language", "cpu-upgrade")
        )
        return
    if args.driver:
        arguments = [argument for argument in sys.argv[1:] if argument not in ("--driver", args.driver)]
        print(launch_driver("dataflow.recipes.hplt_language", arguments, args.driver, name="hplt-language-driver"))
        return
    extract(args.output, args.language, args.workers, args.jobs, args.budget, args.common_crawl_before)


if __name__ == "__main__":
    main()
