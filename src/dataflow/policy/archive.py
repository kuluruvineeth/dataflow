import json
from collections.abc import Iterable, Iterator
from urllib.parse import urlsplit

from warcio.archiveiterator import ArchiveIterator

from dataflow.io import DataFolderLike, get_datafolder
from dataflow.policy.robots import Robots


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


def read_robots_warc(folder: DataFolderLike, path: str) -> Iterator[tuple[str, int, bytes]]:
    """(origin, HTTP status, body) for every robots.txt response in a Common Crawl robotstxt WARC."""
    with get_datafolder(folder).open(path, "rb") as file:
        for record in ArchiveIterator(file):
            if record.rec_type != "response" or not record.http_headers:
                continue
            url = record.rec_headers.get_header("WARC-Target-URI") or ""
            if not url.endswith("/robots.txt"):
                continue
            status = int(record.http_headers.get_statuscode() or 0)
            yield origin_of(url), status, record.content_stream().read()


class RobotsArchive:
    """robots.txt answers as a crawl recorded them, so archived pages are judged by the rules of their own crawl."""

    def __init__(self, entries: dict[str, tuple[int, str]] | None = None):
        self.entries = entries or {}
        self._parsed: dict[str, Robots] = {}

    @classmethod
    def from_warcs(cls, folder: DataFolderLike, origins: Iterable[str] | None = None) -> "RobotsArchive":
        wanted = set(origins) if origins is not None else None
        folder = get_datafolder(folder)
        entries = {}
        for path in folder.list_files(glob_pattern="*.warc.gz"):
            for origin, status, body in read_robots_warc(folder, path):
                if wanted is None or origin in wanted:
                    entries[origin] = (status, body.decode("utf-8", errors="replace"))
        return cls(entries)

    def save(self, folder: DataFolderLike, path: str) -> None:
        with get_datafolder(folder).open(path, "w") as file:
            for origin, (status, body) in sorted(self.entries.items()):
                file.write(json.dumps({"origin": origin, "status": status, "body": body}) + "\n")

    @classmethod
    def load(cls, folder: DataFolderLike, path: str) -> "RobotsArchive":
        with get_datafolder(folder).open(path, "r") as file:
            rows = (json.loads(line) for line in file if line.strip())
            return cls({row["origin"]: (row["status"], row["body"]) for row in rows})

    def __call__(self, url: str) -> Robots | None:
        origin = origin_of(url)
        if origin not in self.entries:
            return None
        if origin not in self._parsed:
            status, body = self.entries[origin]
            self._parsed[origin] = Robots.parse(body) if 200 <= status < 300 else Robots.from_status(status)
        return self._parsed[origin]
