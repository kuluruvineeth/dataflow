from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
import pyarrow.parquet as pq
from huggingface_hub.utils import build_hf_headers

from dataflow.data import DocumentsPipeline
from dataflow.io import DataFolderLike, get_datafolder
from dataflow.pipeline.base import PipelineStep
from dataflow.sources.ccindex import COMMON_CRAWL
from dataflow.sources.http import RateLimiter, get_range, http_client

GZIP_MAGIC = b"\x1f\x8b"
HF_HTTPS = "https://huggingface.co/"


@dataclass(frozen=True)
class Span:
    """One request: bytes `start` to `end` of a WARC file, covering `records` (offset, length)."""

    start: int
    end: int
    records: tuple[tuple[int, int], ...]


def coalesce(records: Iterable[tuple[int, int]], max_gap: int = 16_384, max_bytes: int = 8 << 20) -> list[Span]:
    """Merge records of one WARC file into as few requests as possible: a record joins the previous request when the
    gap between them is at most `max_gap` and the request stays under `max_bytes`."""
    spans: list[Span] = []
    for offset, length in sorted(records):
        last = spans[-1] if spans else None
        if last and offset - last.end <= max_gap and offset + length - last.start <= max_bytes:
            spans[-1] = Span(last.start, max(last.end, offset + length), (*last.records, (offset, length)))
        else:
            spans.append(Span(offset, offset + length, ((offset, length),)))
    return spans


class RangeFetcher(PipelineStep):
    """Fetch the records listed in selection tables by byte range and write them, unchanged, into new WARC files.

    Each WARC record in Common Crawl is its own gzip member, so concatenated records are a valid `.warc.gz`. One output
    file per selection table, skipped when it already exists, so a stopped run continues where it was. Every record's
    length is checked against the index.
    """

    type = "Fetcher"
    name = "warc ranges"

    def __init__(
        self,
        selection_folder: DataFolderLike,
        output_folder: DataFolderLike,
        crawls: list[str] | None = None,
        base: str = COMMON_CRAWL,
        requests_per_second: float = 10.0,
        threads: int = 8,
        max_gap: int = 16_384,
    ):
        self.selection_folder = get_datafolder(selection_folder)
        self.output_folder = get_datafolder(output_folder)
        self.crawls = crawls
        self.base = base
        self.requests_per_second = requests_per_second
        self.threads = threads
        self.max_gap = max_gap

    def selection_files(self) -> list[str]:
        paths = self.selection_folder.list_files(glob_pattern="*/*.parquet")
        return [path for path in paths if self.crawls is None or path.split("/", 1)[0] in self.crawls]

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        limiter = RateLimiter(self.requests_per_second)
        client = http_client()
        for path in self.selection_files()[rank::world_size]:
            target = path.removesuffix(".parquet") + ".warc.gz"
            if self.output_folder.exists(target):
                self.stat_update("skipped_files")
                continue
            with self.selection_folder.open(path, "rb", cache_type="none") as file:
                table = pq.read_table(file, columns=["warc_filename", "warc_record_offset", "warc_record_length"])
            by_file: dict[str, list[tuple[int, int]]] = {}
            for name, offset, length in zip(*(table[c].to_pylist() for c in table.column_names), strict=True):
                by_file.setdefault(name, []).append((offset, length))
            jobs = [(name, span) for name, records in by_file.items() for span in coalesce(records, self.max_gap)]
            with self.track_time(), ThreadPoolExecutor(self.threads) as pool:
                payloads = list(pool.map(lambda job: self.fetch(job[0], job[1], client, limiter), jobs))
            records = [
                record
                for (_, span), payload in zip(jobs, payloads, strict=True)
                for record in self.split(span, payload)
            ]
            with self.output_folder.open(target, "wb") as out:
                out.write(b"".join(records))
            self.stat_update("files")
            self.stat_update("requests", value=len(jobs))
        yield from ()

    def fetch(self, name: str, span: Span, client: httpx.Client, limiter: RateLimiter) -> bytes:
        if self.base.startswith("https://"):
            headers = build_hf_headers() if self.base.startswith(HF_HTTPS) else None
            return get_range(self.base + name, span.start, span.end, client, limiter, headers)
        limiter.wait()
        with get_datafolder(self.base).open(name, "rb", cache_type="none") as file:
            file.seek(span.start)
            return file.read(span.end - span.start)

    def split(self, span: Span, payload: bytes) -> list[bytes]:
        records = []
        for offset, length in span.records:
            record = payload[offset - span.start : offset - span.start + length]
            if len(record) != length or not record.startswith(GZIP_MAGIC):
                self.stat_update("length_mismatches")
                continue
            records.append(record)
            self.stat_update("records")
            self.stat_update("bytes", value=length, unit="byte")
        return records
