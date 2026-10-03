from io import BytesIO

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from warcio.archiveiterator import ArchiveIterator
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from dataflow.pipeline.readers import WarcReader
from dataflow.sources.fetch import RangeFetcher, Span, coalesce
from dataflow.sources.http import get_range

PAGE = "<html><body><p>{}</p></body></html>"


def test_coalesce_merges_close_records_and_respects_the_size_cap():
    records = [(1000, 100), (0, 100), (5000, 50), (1150, 10)]
    assert coalesce(records, max_gap=100) == [
        Span(0, 100, ((0, 100),)),
        Span(1000, 1160, ((1000, 100), (1150, 10))),
        Span(5000, 5050, ((5000, 50),)),
    ]
    assert len(coalesce(records, max_gap=10_000)) == 1
    assert len(coalesce(records, max_gap=10_000, max_bytes=1_000)) == 3


def write_warc(path) -> list[tuple[str, int, int]]:
    """Five pages, each response preceded by its request record, as Common Crawl writes them; returns the responses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        writer = WARCWriter(file, gzip=True)
        for number in range(5):
            url = f"https://site{number}.in/"
            request = writer.create_warc_record(url, "request", payload=BytesIO(b"GET / HTTP/1.1\r\n\r\n"))
            writer.write_record(request)
            http = StatusAndHeaders("200 OK", [("Content-Type", "text/html")], protocol="HTTP/1.1")
            body = BytesIO(PAGE.format(f"page {number}").encode())
            writer.write_record(writer.create_warc_record(url, "response", payload=body, http_headers=http))
    responses = []
    with open(path, "rb") as file:
        iterator = ArchiveIterator(file)
        for record in iterator:
            if record.rec_type == "response":
                url = record.rec_headers.get_header("WARC-Target-URI")
                record.content_stream().read()
                responses.append((url, iterator.get_record_offset(), iterator.get_record_length()))
    return responses


def selection(tmp_path, rows: list[tuple[int, int]]) -> None:
    folder = tmp_path / "selection/CC-MAIN-2099-01"
    folder.mkdir(parents=True)
    table = pa.table(
        {
            "warc_filename": ["crawl-data/a.warc.gz"] * len(rows),
            "warc_record_offset": [offset for offset, _ in rows],
            "warc_record_length": [length for _, length in rows],
        }
    )
    pq.write_table(table, folder / "part-00000.parquet")


def test_fetched_records_form_a_warc_that_the_reader_reads(tmp_path):
    responses = write_warc(tmp_path / "base/crawl-data/a.warc.gz")
    chosen = [responses[0], responses[2], responses[3]]
    selection(tmp_path, [(offset, length) for _, offset, length in chosen])
    near = chosen[2][1] - (chosen[1][1] + chosen[1][2])
    assert chosen[1][1] - (chosen[0][1] + chosen[0][2]) > near
    fetcher = RangeFetcher(tmp_path / "selection", tmp_path / "out", base=str(tmp_path / "base"), max_gap=near)
    list(fetcher.run())
    documents = list(WarcReader(tmp_path / "out").run())
    assert [document.metadata["url"] for document in documents] == [url for url, _, _ in chosen]
    metrics = fetcher.stats.metrics
    assert metrics["records"].total == 3 and "length_mismatches" not in metrics
    assert metrics["requests"].total == 2

    again = RangeFetcher(tmp_path / "selection", tmp_path / "out", base=str(tmp_path / "base"))
    list(again.run())
    assert again.stats.metrics["skipped_files"].total == 1


def test_a_wrong_length_is_counted_and_left_out(tmp_path):
    responses = write_warc(tmp_path / "base/crawl-data/a.warc.gz")
    (_, offset, length), (_, other, other_length) = responses[0], responses[4]
    selection(tmp_path, [(offset, length), (other + 1, other_length)])
    fetcher = RangeFetcher(tmp_path / "selection", tmp_path / "out", base=str(tmp_path / "base"))
    list(fetcher.run())
    assert fetcher.stats.metrics["records"].total == 1
    assert fetcher.stats.metrics["length_mismatches"].total == 1


def test_get_range_asks_for_exactly_the_span():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["range"] = request.headers["range"]
        return httpx.Response(206, content=b"x" * 10)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert get_range("https://example.org/w.warc.gz", 100, 110, client) == b"x" * 10
    assert seen["range"] == "bytes=100-109"
