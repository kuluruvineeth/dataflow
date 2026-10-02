import argparse
import logging
from collections.abc import Callable

from dataflow.executor.base import Pipeline, PipelineExecutor
from dataflow.executor.local import LocalPipelineExecutor
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
from dataflow.utils.stats import PipelineStats

ExecutorFactory = Callable[[Pipeline, str, int], PipelineExecutor]
STAGES = ("filter", "signatures", "buckets", "clusters", "dedup")


def local(workers: int = -1) -> ExecutorFactory:
    def factory(pipeline: Pipeline, logging_dir: str, tasks: int) -> PipelineExecutor:
        return LocalPipelineExecutor(pipeline, logging_dir, tasks=tasks, workers=workers)

    return factory


def jobs(tasks_per_job: int = 10, **kwargs) -> ExecutorFactory:
    from dataflow.executor.jobs import JobsPipelineExecutor

    def factory(pipeline: Pipeline, logging_dir: str, tasks: int) -> PipelineExecutor:
        per_job = min(tasks, tasks_per_job)
        return JobsPipelineExecutor(pipeline, logging_dir, tasks=tasks, tasks_per_job=per_job, **kwargs)

    return factory


def english_web(
    input_folder: str,
    output_folder: str,
    executor: ExecutorFactory | None = None,
    tasks: int = 1,
    glob_pattern: str | None = None,
    url_filter: URLFilter | None = None,
    language_filter: LanguageFilter | None = None,
    stages: tuple[str, ...] = STAGES,
) -> dict[str, PipelineStats]:
    """The FineWeb recipe for English web text: filter each WARC, MinHash-deduplicate the run, mask PII."""
    executor = executor or local()
    config = MinhashConfig()
    stats: dict[str, PipelineStats] = {}

    def removed(step: str) -> JsonlWriter:
        return JsonlWriter(f"{output_folder}/removed/{step}", "${filter_reason}/${rank}.jsonl")

    def run(name: str, pipeline: Pipeline, stage_tasks: int = tasks) -> None:
        if name in stages:
            stats[name] = executor(pipeline, f"{output_folder}/logs/{name}", stage_tasks).run()

    filtered, signatures = f"{output_folder}/filtered", f"{output_folder}/minhash/signatures"
    buckets, clusters = f"{output_folder}/minhash/buckets", f"{output_folder}/minhash/clusters"
    run(
        "filter",
        [
            WarcReader(input_folder, glob_pattern=glob_pattern),
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
    run("clusters", [MinhashDedupCluster(buckets, clusters)], 1)
    run(
        "dedup",
        [
            JsonlReader(filtered),
            MinhashDedupFilter(clusters, exclusion_writer=JsonlWriter(f"{output_folder}/removed/7_minhash")),
            PIIFormatter(),
            ParquetWriter(f"{output_folder}/output"),
        ],
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="English web text from Common Crawl WARC files (FineWeb recipe).")
    parser.add_argument("input_folder", help="folder of WARC files, local or remote")
    parser.add_argument("output_folder", help="where filtered text, removed documents, logs and output go")
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--glob", dest="glob_pattern", help="only read WARC files matching this pattern")
    parser.add_argument("--workers", type=int, default=-1, help="local worker processes")
    parser.add_argument("--jobs", action="store_true", help="run each stage on Hugging Face Jobs")
    parser.add_argument("--flavor", default="cpu-upgrade")
    parser.add_argument("--tasks-per-job", type=int, default=10)
    parser.add_argument("--workers-per-job", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.jobs:
        executor = jobs(flavor=args.flavor, tasks_per_job=args.tasks_per_job, workers_per_job=args.workers_per_job)
    else:
        executor = local(args.workers)
    for name, stage_stats in english_web(
        args.input_folder, args.output_folder, executor, tasks=args.tasks, glob_pattern=args.glob_pattern
    ).items():
        print(f"== {name}\n{stage_stats}")


if __name__ == "__main__":
    main()
