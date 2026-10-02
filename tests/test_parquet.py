import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataflow.data import Document
from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.readers import ParquetReader
from dataflow.pipeline.writers import ParquetWriter

DOCS = [
    Document("The river runs through the old town.", "a", {"url": "https://a.example", "score": 0.9}),
    Document("తెలుగు భాష చాలా అందమైనది.", "b", {"url": "https://b.example", "language": "te"}),
    Document("Visitors say the town feels calm.", "c", {"url": "https://c.example", "score": 0.4}),
]


def write(folder, docs, **kwargs):
    with ParquetWriter(folder, **kwargs) as writer:
        for doc in docs:
            writer.write(doc)


def test_documents_round_trip_with_metadata_as_columns(tmp_path):
    write(tmp_path, DOCS)
    table = pq.read_table(tmp_path / "00000.parquet")
    assert table.column_names == ["text", "id", "url", "score", "language"]
    docs = list(ParquetReader(tmp_path).run())
    assert [doc.text for doc in docs] == [doc.text for doc in DOCS]
    assert [doc.id for doc in docs] == ["a", "b", "c"]
    assert docs[1].metadata == {"url": "https://b.example", "score": None, "language": "te"}


def test_batches_become_row_groups(tmp_path):
    write(tmp_path, DOCS * 5, batch_size=4)
    metadata = pq.ParquetFile(tmp_path / "00000.parquet").metadata
    assert metadata.num_rows == 15
    assert [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)] == [4, 4, 4, 3]
    assert metadata.row_group(0).column(0).compression == "ZSTD"


def test_unknown_column_after_first_batch_is_an_error(tmp_path):
    late = Document("A page with a new field.", "d", {"url": "https://d.example", "license": "cc-by"})
    with pytest.raises(ValueError, match="license"):
        write(tmp_path, [*DOCS, late], batch_size=3)


def test_explicit_schema_accepts_any_known_columns(tmp_path):
    schema = pa.schema([("text", pa.string()), ("id", pa.string()), ("url", pa.string()), ("license", pa.string())])
    late = Document("A page with a new field.", "d", {"url": "https://d.example", "license": "cc-by"})
    first = Document("The river runs through the old town.", "a", {"url": "https://a.example"})
    write(tmp_path, [first, late], batch_size=1, schema=schema)
    assert pq.read_table(tmp_path / "00000.parquet").column("license").to_pylist() == [None, "cc-by"]


def test_reader_options(tmp_path):
    pq.write_table(pa.table({"content": ["one", "", "three"], "url": ["u1", "u2", "u3"]}), tmp_path / "x.parquet")
    docs = list(ParquetReader(tmp_path, text_key="content", read_metadata=False).run())
    assert [(doc.text, doc.id, doc.metadata) for doc in docs] == [
        ("one", "x.parquet/0", {}),
        ("three", "x.parquet/2", {}),
    ]
    assert len(list(ParquetReader(tmp_path, text_key="content", limit=1).run())) == 1


def test_parallel_tasks_write_one_file_each(tmp_path):
    (tmp_path / "in").mkdir()
    for i in range(2):
        pq.write_table(
            pa.Table.from_pylist([{"text": doc.text, "id": f"{i}{doc.id}"} for doc in DOCS]),
            tmp_path / f"in/{i}.parquet",
        )
    pipeline = [ParquetReader(tmp_path / "in"), ParquetWriter(tmp_path / "out")]
    LocalPipelineExecutor(pipeline, tmp_path / "logs", tasks=2, workers=2).run()
    ids = [pq.read_table(tmp_path / f"out/{rank:05d}.parquet").column("id").to_pylist() for rank in range(2)]
    assert ids == [["0a", "0b", "0c"], ["1a", "1b", "1c"]]
