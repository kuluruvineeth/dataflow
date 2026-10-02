from io import BytesIO

from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.filters import PolicyFilter
from dataflow.pipeline.readers import WarcReader
from dataflow.pipeline.writers import JsonlWriter
from dataflow.policy.archive import RobotsArchive
from dataflow.policy.gate import OptOutRegistry, PolicyGate

PAGE = b"<html><head></head><body><p>A page about rivers and bridges.</p></body></html>"


def write_warc(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        writer = WARCWriter(file, gzip=True)
        for url, status, headers, body in records:
            http = StatusAndHeaders(status, headers, protocol="HTTP/1.1")
            writer.write_record(writer.create_warc_record(url, "response", payload=BytesIO(body), http_headers=http))


def test_robots_archive_reads_common_crawl_robotstxt_warcs(tmp_path):
    write_warc(
        tmp_path / "robotstxt/r.warc.gz",
        [
            (
                "https://a.com/robots.txt",
                "200 OK",
                [("Content-Type", "text/plain")],
                b"User-agent: GPTBot\nDisallow: /\n",
            ),
            ("https://b.com/robots.txt", "404 Not Found", [("Content-Type", "text/html")], b"missing"),
            ("https://c.com/robots.txt", "503 Unavailable", [("Content-Type", "text/html")], b"busy"),
        ],
    )
    archive = RobotsArchive.from_warcs(tmp_path / "robotstxt")
    assert set(archive.entries) == {"https://a.com", "https://b.com", "https://c.com"}
    assert archive("https://b.com/page").allow_all and archive("https://c.com/page").disallow_all
    assert archive("https://unknown.com/page") is None
    archive.save(tmp_path, "robots.jsonl")
    assert RobotsArchive.load(tmp_path, "robots.jsonl").entries == archive.entries


def test_policy_filter_judges_raw_pages_with_their_crawl_robots(tmp_path):
    write_warc(
        tmp_path / "robotstxt/r.warc.gz",
        [("https://a.com/robots.txt", "200 OK", [], b"User-agent: GPTBot\nDisallow: /\n")],
    )
    html = [("Content-Type", "text/html")]
    write_warc(
        tmp_path / "warc/w.warc.gz",
        [
            ("https://a.com/story", "200 OK", html, PAGE),
            ("https://b.com/story", "200 OK", [*html, ("X-Robots-Tag", "noai")], PAGE),
            (
                "https://c.com/story",
                "200 OK",
                html,
                PAGE.replace(b"<head>", b'<head><meta name="robots" content="noai">'),
            ),
            ("https://d.com/story", "200 OK", html, PAGE),
            ("https://e.com/story", "200 OK", html, PAGE),
        ],
    )
    gate = PolicyGate(registry=OptOutRegistry.from_lines(["d.com"]))
    step = PolicyFilter(
        RobotsArchive.from_warcs(tmp_path / "robotstxt"),
        gate,
        exclusion_writer=JsonlWriter(tmp_path / "removed", "${filter_reason}/${rank}.jsonl"),
    )
    stats = LocalPipelineExecutor(
        [WarcReader(tmp_path / "warc"), step, JsonlWriter(tmp_path / "kept")], tmp_path / "logs"
    ).run()
    reasons = sorted(path.name for path in (tmp_path / "removed").iterdir())
    assert reasons == ["ai_agent_blocked", "meta_noai", "opt_out_registry", "x_robots_noai"]
    policy = next(step_stats for step_stats in stats.stats if step_stats.name == "Filter: policy")
    assert policy.metrics["forwarded"].total == 1
    assert policy.metrics["robots_unknown"].total == 4
