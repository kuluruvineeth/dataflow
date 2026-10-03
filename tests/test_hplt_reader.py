import json

import zstandard

from dataflow.pipeline.readers.hplt import HpltPoolReader


def row(name: str, language: str, probability: float, allowed: bool = True) -> dict:
    return {"u": f"https://{name}.in/", "id": name, "openlid-v3": {"lang": [language], "prob": [probability]},
            "allowed": allowed}  # fmt: skip


ROWS = [
    (row("a", "tel_Telu", 0.98), "తెలుగు"),
    (row("b", "deu_Latn", 1.0), "Hallo"),
    (row("c", "tel_Telu", 0.3), "mixed"),
    (row("d", "tel_Telu", 0.9, allowed=False), "ఆపు"),
    (row("e", "tel_Telu", 0.7), "భాష"),
]


def write_batch(folder) -> None:
    folder.mkdir(parents=True)
    compressor = zstandard.ZstdCompressor()
    metadata = "\n".join(json.dumps(meta) for meta, _ in ROWS) + "\n"
    text = "\n".join(json.dumps({"text": body}, ensure_ascii=False) for _, body in ROWS) + "\n"
    (folder / "metadata.zst").write_bytes(compressor.compress(metadata.encode()))
    (folder / "text.zst").write_bytes(compressor.compress(text.encode()))


def test_keeps_confident_allowed_documents_of_the_language(tmp_path):
    write_batch(tmp_path / "wide00002/1")
    reader = HpltPoolReader(["wide00002/1"], base=str(tmp_path))
    documents = list(reader.run())
    assert [(d.id, d.text, d.metadata["url"]) for d in documents] == [
        ("a", "తెలుగు", "https://a.in/"),
        ("e", "భాష", "https://e.in/"),
    ]
    metrics = reader.stats.metrics
    assert (metrics["documents"].total, metrics["matched"].total, metrics["disallowed"].total) == (5, 3, 1)
    assert metrics["text_bytes"].total == len("తెలుగుభాష".encode())


def test_counting_needs_only_the_metadata(tmp_path):
    write_batch(tmp_path / "wide00002/1")
    (tmp_path / "wide00002/1/text.zst").unlink()
    reader = HpltPoolReader(["wide00002/1"], read_text=False, base=str(tmp_path))
    assert list(reader.run()) == []
    assert reader.stats.metrics["kept"].total == 2
