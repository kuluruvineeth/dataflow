import json

import pyarrow.parquet as pq

from dataflow.browse import describe, preview, read_documents, sample, select
from dataflow.data import Document
from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.filters.base import BaseFilter
from dataflow.pipeline.readers import JsonlReader
from dataflow.pipeline.writers import JsonlWriter, ParquetWriter
from dataflow.publish import build_preview
from dataflow.report import build_report, render_report, save_report

TELUGU = "తెలుగు భాష చాలా అందమైనది"


class KeepEnglish(BaseFilter):
    name = "keep english"

    def filter(self, document):
        return True if document.metadata["language"] == "en" else (False, "not_english")


def corpus(data, rank=0, world_size=1):
    rows = [
        ("a", "en", "https://www.example.com/a", "hello world"),
        ("b", "te", "https://telugu.example.in/b", TELUGU),
        ("c", "en", "https://news.example.com/c", "breaking news today"),
        ("d", "fr", "https://exemple.fr/d", "bonjour le monde"),
    ]
    for doc_id, language, url, text in rows[rank::world_size]:
        yield Document(text, doc_id, {"language": language, "url": url})


def run(tmp_path, exclusion_writer):
    pipeline = [corpus, KeepEnglish(exclusion_writer=exclusion_writer), JsonlWriter(tmp_path / "kept")]
    LocalPipelineExecutor(pipeline, tmp_path / "logs", tasks=2, workers=1).run()


def test_removed_documents_can_be_written_as_parquet_with_their_reason(tmp_path):
    run(tmp_path, ParquetWriter(tmp_path / "removed/keep", "${filter_reason}/${rank}.parquet"))
    table = pq.read_table(tmp_path / "removed/keep/not_english")
    assert sorted(table.column("id").to_pylist()) == ["b", "d"]
    assert set(table.column("filter_reason").to_pylist()) == {"not_english"}


def test_select_and_sample(tmp_path):
    run(tmp_path, JsonlWriter(tmp_path / "removed/keep", "${filter_reason}/${rank}.jsonl"))
    removed = list(read_documents(tmp_path / "removed"))
    assert sorted(d.id for d in select(removed, reason="not_english")) == ["b", "d"]
    assert [d.id for d in select(removed, domain="example.in")] == ["b"]
    assert [d.id for d in select(removed, grep=r"\p{Telugu}+")] == ["b"]
    kept = list(read_documents(tmp_path / "kept"))
    assert sample(kept, 1, seed=3) == sample(kept, 1, seed=3)
    assert len(sample(kept, 5)) == 2


def test_preview_never_cuts_a_telugu_character_apart():
    shown = preview(TELUGU, width=2)
    assert shown == "తె" + "లు" + "…"
    assert describe(Document(TELUGU, "x", {"url": "u", "language": "te", "language_score": 0.98})).startswith(
        "[te] [0.98] u"
    )


def test_report_has_the_funnel_languages_and_domains(tmp_path):
    run(tmp_path, None)
    rows = build_report(tmp_path / "logs", kept=tmp_path / "kept")
    funnel = {(r["group"], r["key"]): r for r in rows if r["section"] == "funnel"}
    assert funnel[("Filter: keep english", "dropped_not_english")]["value"] == 2
    assert funnel[("Filter: keep english", "forwarded")]["share"] == 0.5
    domains = {r["key"]: r["value"] for r in rows if r["section"] == "domain"}
    assert domains == {"example.com": 1, "news.example.com": 1}
    assert "dropped_not_english" in render_report(rows)
    save_report(rows, tmp_path / "logs")
    assert pq.read_table(tmp_path / "logs/report.parquet").num_rows == len(rows)


def test_preview_dataset_has_one_config_per_outcome(tmp_path):
    run(tmp_path, JsonlWriter(tmp_path / "removed/keep", "${filter_reason}/${rank}.jsonl"))
    counts = build_preview(tmp_path / "kept", tmp_path / "removed", tmp_path / "repo", title="Test run")
    assert counts == {"kept": 2, "removed_keep": 2}
    card = (tmp_path / "repo/README.md").read_text()
    assert "config_name: removed_keep" in card and "path: data/kept/*.parquet" in card
    removed = pq.read_table(tmp_path / "repo/data/removed_keep")
    assert removed.schema.names == [
        "id",
        "text",
        "url",
        "domain",
        "language",
        "language_score",
        "filter_reason",
        "metadata",
    ]
    assert set(removed.column("filter_reason").to_pylist()) == {"not_english"}
    assert sorted(removed.column("domain").to_pylist()) == ["exemple.fr", "telugu.example.in"]
    assert json.loads(removed.column("metadata")[0].as_py()) == {}
    assert len(list(JsonlReader(tmp_path / "kept").run())) == 2
