import argparse
import gzip
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from dataflow.executor.base import Pipeline, PipelineExecutor
from dataflow.executor.local import LocalPipelineExecutor
from dataflow.executor.remote import launch_driver
from dataflow.pipeline.dedup import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
    MinhashDedupSignature,
)
from dataflow.pipeline.extractors import Trafilatura
from dataflow.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from dataflow.pipeline.formatters import PIIFormatter
from dataflow.pipeline.readers import JsonlReader, WarcReader
from dataflow.pipeline.writers import JsonlWriter, ParquetWriter
from dataflow.sources.http import open_url
from dataflow.utils.stats import PipelineStats

ExecutorFactory = Callable[..., PipelineExecutor]
STAGES = ("filter", "signatures", "buckets", "clusters", "dedup")
BIG_MEMORY = "cpu-performance"


def local(workers: int = -1) -> ExecutorFactory:
    def factory(pipeline: Pipeline, logging_dir: str, tasks: int, **_) -> PipelineExecutor:
        return LocalPipelineExecutor(pipeline, logging_dir, tasks=tasks, workers=workers)

    return factory


def jobs(tasks_per_job: int = 10, **kwargs) -> ExecutorFactory:
    """Each stage on Jobs; a stage may override settings, e.g. clustering asks for a big-memory flavor."""
    from dataflow.executor.jobs import JobsPipelineExecutor

    def factory(pipeline: Pipeline, logging_dir: str, tasks: int, **overrides) -> PipelineExecutor:
        per_job = min(tasks, overrides.pop("tasks_per_job", tasks_per_job))
        settings = {**kwargs, **overrides}
        return JobsPipelineExecutor(pipeline, logging_dir, tasks=tasks, tasks_per_job=per_job, **settings)

    return factory


def english_web(
    input_folder: str,
    output_folder: str,
    executor: ExecutorFactory | None = None,
    tasks: int = 1,
    glob_pattern: str | None = None,
    paths: list[str] | None = None,
    url_filter: URLFilter | None = None,
    language_filter: LanguageFilter | None = None,
    stages: tuple[str, ...] = STAGES,
    keep_removed: bool = True,
) -> dict[str, PipelineStats]:
    """The FineWeb recipe for English web text: filter each WARC, MinHash-deduplicate the run, mask PII."""
    executor = executor or local()
    config = MinhashConfig()
    stats: dict[str, PipelineStats] = {}

    def removed(step: str) -> JsonlWriter | None:
        # on a whole dump the removed pages are most of the crawl: terabytes nobody reads
        if not keep_removed:
            return None
        return JsonlWriter(f"{output_folder}/removed/{step}", "${filter_reason}/${rank}.jsonl")

    def run(name: str, pipeline: Pipeline, stage_tasks: int = tasks, **overrides) -> None:
        if name in stages:
            stats[name] = executor(pipeline, f"{output_folder}/logs/{name}", stage_tasks, **overrides).run()

    filtered, signatures = f"{output_folder}/filtered", f"{output_folder}/minhash/signatures"
    buckets, clusters = f"{output_folder}/minhash/buckets", f"{output_folder}/minhash/clusters"
    run(
        "filter",
        [
            WarcReader(input_folder, glob_pattern=glob_pattern, paths=paths),
            url_filter or URLFilter(fineweb_word_lists=True, exclusion_writer=removed("1_url")),
            Trafilatura(favour_precision=True, timeout=5),
            language_filter or LanguageFilter(["en"], threshold=0.65, exclusion_writer=removed("2_language")),
            GopherRepetitionFilter(exclusion_writer=removed("3_gopher_repetition")),
            GopherQualityFilter(exclusion_writer=removed("4_gopher_quality")),
            C4QualityFilter(filter_no_terminal_punct=False, exclusion_writer=removed("5_c4")),
            FineWebQualityFilter(exclusion_writer=removed("6_fineweb_quality")),
            JsonlWriter(filtered),
        ],
    )
    run("signatures", [JsonlReader(filtered), MinhashDedupSignature(signatures, config)])
    run("buckets", [MinhashDedupBuckets(signatures, buckets, config)], config.num_buckets)
    run("clusters", [MinhashDedupCluster(buckets, clusters)], 1, flavor=BIG_MEMORY, timeout="24h")
    run(
        "dedup",
        [
            JsonlReader(filtered),
            MinhashDedupFilter(
                clusters, exclusion_writer=JsonlWriter(f"{output_folder}/removed/7_minhash") if keep_removed else None
            ),
            PIIFormatter(),
            ParquetWriter(f"{output_folder}/output"),
        ],
    )
    return stats


def warc_paths(listing: str) -> list[str]:
    """The WARC paths of a crawl from its `warc.paths.gz` (a URL or a local file)."""
    if listing.startswith(("http://", "https://")):
        with open_url(listing) as file:
            data = file.read()
    else:
        data = Path(listing).read_bytes()
    return gzip.decompress(data).decode().split()


def main() -> None:
    parser = argparse.ArgumentParser(description="English web text from Common Crawl WARC files (FineWeb recipe).")
    parser.add_argument("input_folder", help="folder of WARC files, local or remote, or an https base URL")
    parser.add_argument("output_folder", help="where filtered text, removed documents, logs and output go")
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--glob", dest="glob_pattern", help="only read WARC files matching this pattern")
    parser.add_argument("--paths-from", help="read the WARC files listed in this warc.paths.gz, relative to the input")
    parser.add_argument("--max-files", type=int, help="only the first N files of --paths-from (for trial runs)")
    parser.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    parser.add_argument("--workers", type=int, default=-1, help="local worker processes")
    parser.add_argument("--jobs", action="store_true", help="run each stage on Hugging Face Jobs")
    parser.add_argument("--flavor", default="cpu-upgrade")
    parser.add_argument("--tasks-per-job", type=int, default=10)
    parser.add_argument("--workers-per-job", type=int, default=8)
    parser.add_argument("--max-jobs", type=int, default=-1, help="Jobs running at once (-1: all)")
    parser.add_argument("--timeout", default="2h", help="per Job")
    parser.add_argument("--no-removed", action="store_true", help="don't keep removed pages (for whole dumps)")
    parser.add_argument("--driver", metavar="COMMIT", help="run this command itself as a Job, from this commit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.driver:
        arguments = [argument for argument in sys.argv[1:] if argument not in ("--driver", args.driver)]
        print(launch_driver("dataflow.recipes.english_web", arguments, args.driver, name="english-web-driver"))
        return
    if args.jobs:
        executor = jobs(
            flavor=args.flavor, tasks_per_job=args.tasks_per_job, workers_per_job=args.workers_per_job,
            max_jobs=args.max_jobs, timeout=args.timeout,
        )  # fmt: skip
    else:
        executor = local(args.workers)
    paths = warc_paths(args.paths_from)[: args.max_files] if args.paths_from else None
    results = english_web(
        args.input_folder, args.output_folder, executor, tasks=args.tasks, glob_pattern=args.glob_pattern,
        paths=paths, stages=tuple(args.stages), keep_removed=not args.no_removed,
    )  # fmt: skip
    for name, stage_stats in results.items():
        print(f"== {name}\n{stage_stats}")


if __name__ == "__main__":
    main()
