import argparse
import json
import logging
import re

import httpx

from dataflow.executor.local import LocalPipelineExecutor
from dataflow.executor.remote import schedule_daily
from dataflow.io import get_datafolder
from dataflow.pipeline.readers.hplt import HPLT_POOL, HpltPoolReader
from dataflow.pipeline.writers import ParquetWriter

SKIPPED = ("archivebot",)


def pool_batches(skip: tuple[str, ...] = SKIPPED) -> list[str]:
    """Every batch of the pool (`<crawl>/<n>`), from each crawl's `.map` file, sorted, minus the skipped crawls."""
    listing = httpx.get(HPLT_POOL, timeout=60).text
    batches = []
    for crawl in re.findall(r'href="([^"/?]+)\.map"', listing):
        if crawl in skip:
            continue
        lines = httpx.get(f"{HPLT_POOL}{crawl}.map", timeout=60).text.split()
        batches += sorted({line.removeprefix(HPLT_POOL).rsplit("/", 1)[0] for line in lines})
    return sorted(batches)


def frozen_batches(output: str) -> list[str]:
    """The batch list of this extraction, written on the first run and reused, so task numbers never move."""
    folder = get_datafolder(output)
    if folder.exists("batches.json"):
        with folder.open("batches.json", "r") as file:
            return json.load(file)
    batches = pool_batches()
    with folder.open("batches.json", "w") as file:
        json.dump(batches, file)
    return batches


def extract(output: str, language: str = "tel_Telu", workers: int = 8) -> None:
    """One task per batch; completed batches are skipped, so a stopped run continues where it was."""
    batches = frozen_batches(output)
    pipeline = [HpltPoolReader(batches, language), ParquetWriter(f"{output}/documents")]
    LocalPipelineExecutor(pipeline, f"{output}/logs", tasks=len(batches), workers=workers).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one language's documents from the HPLT 4.0 pool")
    parser.add_argument("output", help="folder for the batch list, documents and logs")
    parser.add_argument("--language", default="tel_Telu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--schedule", metavar="COMMIT", help="create a daily Job that continues the extraction")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.schedule:
        arguments = [args.output, "--language", args.language, "--workers", str(args.workers)]
        print(
            schedule_daily("dataflow.recipes.hplt_language", arguments, args.schedule, "hplt-language", "cpu-upgrade")
        )
        return
    extract(args.output, args.language, args.workers)


if __name__ == "__main__":
    main()
