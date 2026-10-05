import gzip
import json
import time

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataflow.sources import HttpRangeFile, LanguageSelector, RateLimiter, index_files

CRAWL = "CC-MAIN-2099-01"
INDEX = f"cc-index/table/cc-main/warc/crawl={CRAWL}/subset=warc/part-00000.parquet"


def serve(payload: bytes, failures: int = 0):
    calls = {"failures": failures, "count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["failures"]:
            calls["failures"] -= 1
            return httpx.Response(503)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(payload))})
        start, end = map(int, request.headers["range"].removeprefix("bytes=").split("-"))
        return httpx.Response(206, content=payload[start : end + 1])

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_range_file_reads_and_seeks_with_one_request_per_read():
    client, calls = serve(bytes(range(100)))
    file = HttpRangeFile("https://example.org/f", client=client)
    assert file.read(4) == bytes([0, 1, 2, 3])
    file.seek(-3, 2)
    assert file.read() == bytes([97, 98, 99])
    file.seek(50)
    assert file.read(2) == bytes([50, 51]) and file.tell() == 52
    assert calls["count"] == 4 and file.requests == 4


def test_range_file_retries_a_503_then_gives_up():
    client, _ = serve(b"abc", failures=2)
    assert HttpRangeFile("https://example.org/f", client=client, backoff=0).read() == b"abc"
    client, _ = serve(b"abc", failures=10)
    with pytest.raises(OSError, match="503"):
        HttpRangeFile("https://example.org/f", client=client, retries=2, backoff=0)


def test_a_401_from_an_expired_signed_url_is_retried():
    from dataflow.sources.http import get_range

    answers = [httpx.Response(401), httpx.Response(206, content=b"abc")]
    client = httpx.Client(transport=httpx.MockTransport(lambda request: answers.pop(0)))
    assert get_range("https://example.org/f", 0, 3, client, backoff=0) == b"abc"


def test_rate_limiter_spaces_calls():
    limiter = RateLimiter(per_second=50)
    started = time.monotonic()
    for _ in range(11):
        limiter.wait()
    assert time.monotonic() - started >= 0.19


def fake_index(root) -> None:
    rows = {
        "url_host_name": ["a.in", "b.in", "c.com", "d.com", "e.in", "f.in"],
        "content_languages": ["tel", "eng,tel", "eng", None, "tel,eng", "hin"],
        "content_mime_detected": ["text/html"] * 6,
        "warc_filename": [f"w{i}.warc.gz" for i in range(6)],
        "warc_record_offset": list(range(0, 600, 100)),
        "warc_record_length": [50] * 6,
        "url": [f"https://x/{i}" for i in range(6)],
    }
    path = root / INDEX
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table(rows), path, row_group_size=2)
    listing = root / f"crawl-data/{CRAWL}/cc-index-table.paths.gz"
    listing.parent.mkdir(parents=True)
    other = INDEX.replace("subset=warc", "subset=robotstxt")
    listing.write_bytes(gzip.compress(f"{INDEX}\n{other}\n".encode()))


def test_index_files_keeps_the_warc_subset(tmp_path):
    fake_index(tmp_path)
    assert index_files(CRAWL, base=str(tmp_path)) == [INDEX]


def test_selector_reads_locations_only_where_the_language_is(tmp_path):
    fake_index(tmp_path)
    selector = LanguageSelector([INDEX], tmp_path / "out", base=str(tmp_path))
    list(selector.run())
    table = pq.read_table(tmp_path / "out" / CRAWL / "part-00000.parquet")
    assert table["url_host_name"].to_pylist() == ["a.in", "b.in", "e.in"]
    assert table["primary"].to_pylist() == [True, False, True]
    assert "url" not in table.column_names
    metrics = selector.stats.metrics
    assert (metrics["rows"].total, metrics["matched"].total, metrics["primary"].total) == (6, 3, 2)
    assert (metrics["row_groups"].total, metrics["row_groups_read"].total) == (3, 2)
    summary = json.loads((tmp_path / "out" / "summary" / "00000.json").read_text())
    assert summary == {CRAWL: {"files": 1, "matched": 3, "primary": 2}}


def test_a_file_without_the_language_writes_an_empty_table(tmp_path):
    fake_index(tmp_path)
    list(LanguageSelector([INDEX], tmp_path / "out", language="kan", base=str(tmp_path)).run())
    table = pq.read_table(tmp_path / "out" / CRAWL / "part-00000.parquet")
    assert table.num_rows == 0 and "primary" in table.column_names
