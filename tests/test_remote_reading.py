import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import pytest
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.base import PipelineStep
from dataflow.pipeline.readers import WarcReader
from dataflow.sources.http import open_url

PAGE = "<html><body><article><p>{}</p></article></body></html>"


def write_warc(path, urls: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        writer = WARCWriter(file, gzip=True)
        for url in urls:
            http = StatusAndHeaders("200 OK", [("Content-Type", "text/html")], protocol="HTTP/1.1")
            body = BytesIO(PAGE.format(url).encode())
            writer.write_record(writer.create_warc_record(url, "response", payload=body, http_headers=http))


@pytest.fixture
def server(tmp_path):
    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()


def test_explicit_paths_read_only_those_files(tmp_path):
    write_warc(tmp_path / "a/one.warc.gz", ["https://one.example/"])
    write_warc(tmp_path / "a/two.warc.gz", ["https://two.example/"])
    documents = list(WarcReader(tmp_path, paths=["a/two.warc.gz"]).run())
    assert [document.metadata["url"] for document in documents] == ["https://two.example/"]


def test_warc_files_stream_over_http_from_a_path_list(tmp_path, server):
    write_warc(tmp_path / "crawl-data/s1/x.warc.gz", ["https://a.example/", "https://b.example/"])
    write_warc(tmp_path / "crawl-data/s2/y.warc.gz", ["https://c.example/"])
    paths = ["crawl-data/s1/x.warc.gz", "crawl-data/s2/y.warc.gz"]
    reader = WarcReader(server, paths=paths)
    assert [document.metadata["url"] for document in reader.run()] == [
        "https://a.example/",
        "https://b.example/",
        "https://c.example/",
    ]


class DropsOnce(BaseHTTPRequestHandler):
    """Serves BODY with range support; the first full request sends only part of it, then closes the connection."""

    dropped = False

    def do_GET(self):
        start = int(self.headers["Range"].removeprefix("bytes=").rstrip("-")) if self.headers["Range"] else 0
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(BODY) - start))
        self.end_headers()
        if not start and not DropsOnce.dropped:
            DropsOnce.dropped = True
            self.wfile.write(BODY[: len(BODY) // 3])
            self.close_connection = True
            return
        self.wfile.write(BODY[start:])

    def log_message(self, *args):
        pass


BODY = bytes(range(256)) * 4096


def test_a_dropped_download_resumes_from_the_last_byte():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), DropsOnce)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with open_url(f"http://127.0.0.1:{httpd.server_address[1]}/file", backoff=0) as file:
            assert file.read() == BODY
            assert file.raw.resumes == 1
    finally:
        httpd.shutdown()


def test_a_remote_folder_needs_a_path_list():
    with pytest.raises(ValueError, match="pass paths"):
        WarcReader("https://data.commoncrawl.org/")


class Nothing(PipelineStep):
    def run(self, data=None, rank: int = 0, world_size: int = 1):
        yield from ()


def test_completed_ranks_come_from_one_listing(tmp_path):
    executor = LocalPipelineExecutor([Nothing()], str(tmp_path), tasks=5, workers=1)
    executor.run()
    assert executor.completed_ranks() == {0, 1, 2, 3, 4}
    (tmp_path / "completions/00003").unlink()
    assert executor.get_incomplete_ranks() == [3]


def test_a_driver_runs_the_module_from_the_pinned_source_tree():
    from dataflow.executor.remote import source_command

    command = source_command("dataflow.recipes.english_web", ["in", "out dir", "--jobs"], "abc123")
    assert command[:2] == ["sh", "-c"]
    assert "archive/abc123.tar.gz" in command[2] and "cd /src/dataflow-abc123" in command[2]
    assert command[2].endswith("uv run python -m dataflow.recipes.english_web in 'out dir' --jobs")
