import gzip
import json
from typing import IO

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dataflow.data import DocumentsPipeline
from dataflow.io import DataFolderLike, get_datafolder
from dataflow.pipeline.base import PipelineStep
from dataflow.sources.http import HttpRangeFile, RateLimiter

COMMON_CRAWL = "https://data.commoncrawl.org/"
SELECTED = (
    "url_host_name",
    "content_languages",
    "content_mime_detected",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
)


def open_remote(base: str, path: str, limiter: RateLimiter | None) -> IO:
    """Range-read a file under `base`: Common Crawl over HTTPS, or any folder dataflow can open (the HF bucket)."""
    if base.startswith("https://"):
        return HttpRangeFile(base + path, limiter)
    return get_datafolder(base).open(path, "rb", cache_type="none")


def index_files(crawl: str, base: str = COMMON_CRAWL) -> list[str]:
    """The columnar index files (subset=warc) of one crawl, relative to `base`."""
    with open_remote(base, f"crawl-data/{crawl}/cc-index-table.paths.gz", None) as file:
        paths = gzip.decompress(file.read()).decode().split()
    return [path for path in paths if "/subset=warc/" in path]


def crawl_of(path: str) -> str:
    return path.split("crawl=", 1)[1].split("/", 1)[0]


class LanguageSelector(PipelineStep):
    """Select every record of one language from Common Crawl's columnar index, reading as little as possible.

    Per row group, only `content_languages` is read; the record locations (`SELECTED`) are read only for row groups
    that contain the language. One small Parquet table per index file, and stats for rows, matches (any position and
    primary) and the bytes actually read, taken from the Parquet metadata.
    """

    type = "Selector"
    name = "cc-index language"

    def __init__(
        self,
        index_paths: list[str],
        output_folder: DataFolderLike,
        language: str = "tel",
        base: str = COMMON_CRAWL,
        requests_per_second: float = 10.0,
    ):
        self.index_paths = index_paths
        self.output_folder = get_datafolder(output_folder)
        self.language = language
        self.base = base
        self.requests_per_second = requests_per_second

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        limiter = RateLimiter(self.requests_per_second)
        summary: dict[str, dict[str, int]] = {}
        for path in self.index_paths[rank::world_size]:
            crawl = crawl_of(path)
            with self.track_time(), open_remote(self.base, path, limiter) as file:
                table = self.select(pq.ParquetFile(file))
                self.stat_update("requests", value=getattr(file, "requests", 0))
            with self.output_folder.open(f"{crawl}/{path.rsplit('/', 1)[1]}", "wb") as out:
                pq.write_table(table, out)
            counts = summary.setdefault(crawl, {"files": 0, "matched": 0, "primary": 0})
            counts["files"] += 1
            counts["matched"] += table.num_rows
            counts["primary"] += int(pc.sum(table["primary"]).as_py() or 0)
            self.stat_update("files")
        with self.output_folder.open(f"summary/{rank:05d}.json", "w") as out:
            json.dump(summary, out)
        yield from ()

    def select(self, parquet: pq.ParquetFile) -> pa.Table:
        metadata = parquet.metadata
        names = [metadata.schema.column(i).name for i in range(metadata.num_columns)]
        language_column, selected_columns = names.index("content_languages"), [names.index(n) for n in SELECTED]
        parts = []
        for group in range(metadata.num_row_groups):
            row_group = metadata.row_group(group)
            self.stat_update("row_groups")
            self.stat_update("rows", value=row_group.num_rows)
            self.stat_update("bytes", value=row_group.column(language_column).total_compressed_size, unit="byte")
            languages = parquet.read_row_group(group, columns=["content_languages"])["content_languages"]
            match = pc.fill_null(pc.match_substring(languages, self.language), False)
            if not pc.any(match).as_py():
                continue
            self.stat_update("row_groups_read")
            other = [index for index in selected_columns if index != language_column]
            self.stat_update("bytes", value=sum(row_group.column(i).total_compressed_size for i in other), unit="byte")
            rows = parquet.read_row_group(group, columns=list(SELECTED)).filter(match)
            primary = pc.starts_with(rows["content_languages"], self.language)
            self.stat_update("matched", value=len(rows))
            self.stat_update("primary", value=pc.sum(primary).as_py() or 0)
            parts.append(rows.append_column("primary", primary))
        if parts:
            return pa.concat_tables(parts)
        fields = [parquet.schema_arrow.field(name) for name in SELECTED]
        return pa.schema([*fields, pa.field("primary", pa.bool_())]).empty_table()
