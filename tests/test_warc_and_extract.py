import time
from io import BytesIO

from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from dataflow.data import Document
from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.extractors import Trafilatura
from dataflow.pipeline.extractors.base import BaseExtractor
from dataflow.pipeline.readers import WarcReader

ARTICLE = (
    "<html><head><title>Rivers</title></head><body><nav><a href='/'>Home</a><a href='/shop'>Shop</a></nav>"
    "<article><h1>How rivers shape towns</h1>"
    "<p>The river runs through the old town, and the people who live there have built bridges with stone and wood. "
    "Every spring the water rises to the edge of the market, so the traders move their stalls to the hill.</p>"
    "<p>Children learn to swim in the shallow parts, and the fishermen teach them which currents to avoid. "
    "Visitors say that the town feels calm, even when the river is loud after a storm.</p></article>"
    "<footer>Copyright 2024. All rights reserved.</footer></body></html>"
)


def write_warc(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        writer = WARCWriter(file, gzip=True)
        for url, content_type, body, rec_type in records:
            http = StatusAndHeaders("200 OK", [("Content-Type", content_type)], protocol="HTTP/1.1")
            record = writer.create_warc_record(url, rec_type, payload=BytesIO(body), http_headers=http)
            writer.write_record(record)


def test_warc_reader_keeps_only_html_responses(tmp_path):
    write_warc(
        tmp_path / "in/a.warc.gz",
        [
            ("https://a.example/river", "text/html; charset=utf-8", ARTICLE.encode(), "response"),
            ("https://a.example/logo.png", "image/png", b"\x89PNG", "response"),
            ("https://b.example/latin1", "text/html", "<p>café</p>".encode("latin-1"), "response"),
        ],
    )
    docs = list(WarcReader(tmp_path / "in")())
    assert [doc.metadata["url"] for doc in docs] == ["https://a.example/river", "https://b.example/latin1"]
    assert docs[0].text.startswith("<html>")
    assert docs[0].id.startswith("<urn:uuid:")
    assert "date" in docs[0].metadata
    assert "caf" in docs[1].text


def test_truncated_archive_keeps_complete_records_and_is_counted(tmp_path):
    path = tmp_path / "in/a.warc.gz"
    records = [(f"https://a.example/{i}", "text/html", ARTICLE.encode(), "response") for i in range(2)]
    write_warc(path, records)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 200])
    reader = WarcReader(tmp_path / "in")
    assert [doc.metadata["url"] for doc in reader()] == ["https://a.example/0"]
    assert reader.stats.metrics["truncated_files"].total == 1


def test_trafilatura_keeps_the_article_and_drops_boilerplate():
    text = Trafilatura().extract(ARTICLE)
    assert "The river runs through the old town" in text
    assert "Shop" not in text
    assert "All rights reserved" not in text


class SlowOnDemand(BaseExtractor):
    name = "slow on demand"

    def extract(self, text: str) -> str:
        if text == "slow":
            time.sleep(10)
        if text == "boom":
            raise ValueError("bad html")
        return text.upper()


def test_sandbox_skips_slow_and_broken_documents_and_keeps_going():
    step = SlowOnDemand(timeout=0.5)
    docs = [Document("a", "1"), Document("slow", "2"), Document("boom", "3"), Document("b", "4")]
    assert [doc.text for doc in step(iter(docs))] == ["A", "B"]
    assert step.stats.metrics["timeout"].total == 1
    assert step.stats.metrics["extraction_error"].total == 1
    assert step.stats.metrics["forwarded"].total == 2


def test_extraction_runs_inside_parallel_tasks(tmp_path):
    for i in range(2):
        record = (f"https://x.example/{i}", "text/html", ARTICLE.encode(), "response")
        write_warc(tmp_path / f"in/{i}.warc.gz", [record])
    pipeline = [WarcReader(tmp_path / "in"), Trafilatura(timeout=5)]
    stats = LocalPipelineExecutor(pipeline, tmp_path / "logs", tasks=2, workers=2).run()
    assert stats.stats[1].metrics["forwarded"].total == 2


def test_cld2_languages_come_from_the_metadata_record(tmp_path):
    path = tmp_path / "warc/a.warc.gz"
    path.parent.mkdir(parents=True)
    with open(path, "wb") as file:
        writer = WARCWriter(file, gzip=True)
        for url, metadata in [
            ("https://a.com/", b'languages-cld2: {"languages":[{"code":"te"},{"code":"en"}]}\n'),
            ("https://b.com/", None),
        ]:
            http = StatusAndHeaders("200 OK", [("Content-Type", "text/html")], protocol="HTTP/1.1")
            response = writer.create_warc_record(url, "response", payload=BytesIO(ARTICLE.encode()), http_headers=http)
            writer.write_record(response)
            if metadata:
                concurrent = {"WARC-Concurrent-To": response.rec_headers.get_header("WARC-Record-ID")}
                record = writer.create_warc_record(
                    url, "metadata", payload=BytesIO(metadata), warc_headers_dict=concurrent
                )
                writer.write_record(record)
    documents = list(WarcReader(tmp_path / "warc", cld2_languages=True).run())
    assert {d.metadata["url"]: d.metadata["cld2_languages"] for d in documents} == {
        "https://a.com/": ["te", "en"],
        "https://b.com/": [],
    }
    assert "cld2_languages" not in next(WarcReader(tmp_path / "warc").run()).metadata
