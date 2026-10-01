import gzip
from collections import deque

import orjson

from dataflow.data import Document
from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.readers import JsonlReader
from dataflow.pipeline.writers import JsonlWriter


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))


def read_gz(path):
    return [orjson.loads(line) for line in gzip.open(path)]


def test_reader_builds_documents_and_moves_extra_fields_to_metadata(tmp_path):
    write_jsonl(tmp_path / "in/a.jsonl", [{"text": "hello", "id": "x", "url": "u"}, {"text": "world"}, {"text": ""}])
    docs = list(JsonlReader(tmp_path / "in", default_metadata={"source": "test"})())
    assert [doc.text for doc in docs] == ["hello", "world"]
    assert docs[0].id == "x"
    assert docs[0].metadata == {"source": "test", "url": "u"}
    assert docs[1].id == "a.jsonl/1"


def test_reader_skips_malformed_lines(tmp_path):
    path = tmp_path / "in/a.jsonl"
    path.parent.mkdir()
    path.write_text('{"text": "ok"}\nnot json\n{"text": "fine"}\n')
    assert [doc.text for doc in JsonlReader(tmp_path / "in")()] == ["ok", "fine"]


def test_reader_respects_limit(tmp_path):
    write_jsonl(tmp_path / "in/a.jsonl", [{"text": str(i)} for i in range(10)])
    assert len(list(JsonlReader(tmp_path / "in", limit=3)())) == 3


def test_writer_fills_rank_and_metadata_into_filenames(tmp_path):
    docs = [Document("a", "1", {"lang": "en"}), Document("b", "2", {"lang": "te"})]
    writer = JsonlWriter(tmp_path, output_filename="${lang}/${rank}.jsonl")
    deque(writer(iter(docs), rank=7), maxlen=0)
    assert read_gz(tmp_path / "en/00007.jsonl.gz") == [{"text": "a", "id": "1", "metadata": {"lang": "en"}}]
    assert read_gz(tmp_path / "te/00007.jsonl.gz") == [{"text": "b", "id": "2", "metadata": {"lang": "te"}}]


def test_write_then_read_round_trip(tmp_path):
    docs = [Document("first", "1", {"n": 1}), Document("second", "2")]
    deque(JsonlWriter(tmp_path)(iter(docs)), maxlen=0)
    assert list(JsonlReader(tmp_path)()) == docs


def test_tasks_each_read_their_own_files(tmp_path):
    for i in range(4):
        write_jsonl(tmp_path / f"in/{i}.jsonl", [{"text": f"doc {i}-{j}"} for j in range(2)])
    pipeline = [JsonlReader(tmp_path / "in"), JsonlWriter(tmp_path / "out")]
    LocalPipelineExecutor(pipeline, tmp_path / "logs", tasks=2, workers=1).run()
    rank0 = [row["text"] for row in read_gz(tmp_path / "out/00000.jsonl.gz")]
    rank1 = [row["text"] for row in read_gz(tmp_path / "out/00001.jsonl.gz")]
    assert rank0 == ["doc 0-0", "doc 0-1", "doc 2-0", "doc 2-1"]
    assert rank1 == ["doc 1-0", "doc 1-1", "doc 3-0", "doc 3-1"]
