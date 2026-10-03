import argparse
import csv
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import httpx

from dataflow.executor.local import LocalPipelineExecutor
from dataflow.io import get_datafolder
from dataflow.sources.ccindex import COMMON_CRAWL, LanguageSelector, index_files

COLLECTIONS = "https://index.commoncrawl.org/collinfo.json"
LANGUAGES_CSV = "https://raw.githubusercontent.com/commoncrawl/cc-crawl-statistics/master/plots/languages.csv"
FIRST_TAGGED = "CC-MAIN-2018-39"
HF_MIRROR = "hf://buckets/commoncrawl/commoncrawl/"
POLITE_REQUESTS_PER_SECOND = 5.0


def tagged_crawls() -> list[str]:
    """Every crawl whose columnar index has `content_languages` (CC-MAIN-2018-39 onward)."""
    crawls = [collection["id"] for collection in httpx.get(COLLECTIONS, timeout=60).json()]
    return sorted(crawl for crawl in crawls if crawl >= FIRST_TAGGED)


def mirrored_crawls() -> set[str]:
    folder = get_datafolder(f"{HF_MIRROR}crawl-data")
    return {path.rstrip("/").rsplit("/", 1)[-1] for path in folder.ls("", detail=False)}


def by_source(crawls: list[str]) -> list[tuple[str, list[str]]]:
    """Crawls mirrored in the HF bucket are read there; the rest from Common Crawl."""
    mirror = mirrored_crawls()
    return [(HF_MIRROR, [c for c in crawls if c in mirror]), (COMMON_CRAWL, [c for c in crawls if c not in mirror])]


def published_counts(language: str) -> dict[str, int]:
    """Pages per crawl whose primary language is `language`, from Common Crawl's crawl statistics."""
    rows = csv.DictReader(io.StringIO(httpx.get(LANGUAGES_CSV, timeout=60).text))
    return {row["crawl"]: int(row["pages"]) for row in rows if row["primary_language"] == language}


def selection_table(language: str, output: str, crawls: list[str], executor_for, files_per_task: int = 10) -> None:
    """One selection table per crawl under `output/selection`; crawls on the HF mirror are read there, the rest from
    Common Crawl at its polite request rate, shared by all concurrent processes."""
    for base, group in by_source(crawls):
        if not group:
            continue
        paths = [path for crawl in group for path in index_files(crawl, base)]
        tasks = -(-len(paths) // files_per_task)
        name = "mirror" if base == HF_MIRROR else "commoncrawl"

        def pipeline(processes: int, base=base, paths=paths):
            rate = POLITE_REQUESTS_PER_SECOND / processes if base == COMMON_CRAWL else 50.0
            return [LanguageSelector(paths, f"{output}/selection", language, base, requests_per_second=rate)]

        executor_for(pipeline, f"{output}/logs/{name}", tasks).run()


def compare(output: str, language: str) -> list[dict]:
    """Primary-language rows per crawl against the published counts; the run is right when every crawl is within 1%."""
    folder = get_datafolder(f"{output}/selection")

    def read(path: str) -> dict:
        with folder.open(path, "r") as file:
            return json.load(file)

    ours: dict[str, int] = {}
    with ThreadPoolExecutor(16) as pool:
        for summary in pool.map(read, folder.list_files(subdirectory="summary", glob_pattern="*.json")):
            for crawl, counts in summary.items():
                ours[crawl] = ours.get(crawl, 0) + counts["primary"]
    published = published_counts(language)
    return [
        {
            "crawl": crawl,
            "ours": count,
            "published": published.get(crawl),
            "difference": (count - published[crawl]) / published[crawl] if crawl in published else None,
        }
        for crawl, count in sorted(ours.items())
    ]


def local(workers: int = 4):
    def executor_for(pipeline, logging_dir: str, tasks: int):
        return LocalPipelineExecutor(pipeline(workers), logging_dir, tasks=tasks, workers=workers)

    return executor_for


def jobs(max_jobs: int = 2, workers_per_job: int = 8, tasks_per_job: int = 100, timeout: str = "6h"):
    from dataflow.executor.jobs import JobsPipelineExecutor

    def executor_for(pipeline, logging_dir: str, tasks: int):
        processes = min(max_jobs * workers_per_job, tasks)
        return JobsPipelineExecutor(
            pipeline(processes), logging_dir, tasks=tasks, tasks_per_job=min(tasks, tasks_per_job),
            workers_per_job=workers_per_job, max_jobs=max_jobs, timeout=timeout, job_name="cc-language",
        )  # fmt: skip

    return executor_for


def main() -> None:
    parser = argparse.ArgumentParser(description="Select one language's records from Common Crawl's columnar index")
    parser.add_argument("output", help="folder for selection tables, logs and the comparison")
    parser.add_argument("--language", default="tel", help="ISO 639-3 code as Common Crawl writes it")
    parser.add_argument("--crawls", nargs="*", help="default: every language-tagged crawl")
    parser.add_argument("--jobs", action="store_true", help="run on Hugging Face Jobs instead of locally")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not args.compare_only:
        crawls = args.crawls or tagged_crawls()
        selection_table(args.language, args.output, crawls, jobs() if args.jobs else local())
    rows = compare(args.output, args.language)
    with get_datafolder(args.output).open("comparison.json", "w") as file:
        json.dump(rows, file, indent=1)
    for row in rows:
        difference = f"{row['difference']:+.2%}" if row["difference"] is not None else "n/a"
        print(f"{row['crawl']}  ours {row['ours']:>9,}  published {row['published'] or 0:>9,}  {difference}")


if __name__ == "__main__":
    main()
