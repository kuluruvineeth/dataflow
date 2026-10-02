import random
from collections.abc import Iterable, Iterator
from urllib.parse import urlsplit

import regex

from dataflow.data import Document
from dataflow.io import DataFolderLike, get_datafolder
from dataflow.pipeline.readers import JsonlReader, ParquetReader

GRAPHEME = regex.compile(r"\X")


def read_documents(path: DataFolderLike) -> Iterator[Document]:
    folder = get_datafolder(path)
    parquet, jsonl = ParquetReader(folder), JsonlReader(folder)
    for file in folder.list_files():
        if file.endswith(".parquet"):
            yield from parquet.read_file(file)
        elif ".jsonl" in file:
            yield from jsonl.read_file(file)


def domain_of(url: str | None) -> str:
    host = urlsplit(url or "").hostname or ""
    return host.removeprefix("www.")


def select(
    documents: Iterable[Document],
    reason: str | None = None,
    domain: str | None = None,
    grep: str | None = None,
) -> Iterator[Document]:
    pattern = regex.compile(grep) if grep else None
    for document in documents:
        metadata = document.metadata
        if reason and metadata.get("filter_reason") != reason:
            continue
        if domain and not domain_of(metadata.get("url")).endswith(domain):
            continue
        if pattern and not pattern.search(document.text):
            continue
        yield document


def sample(documents: Iterable[Document], size: int, seed: int = 0) -> list[Document]:
    rng = random.Random(seed)
    chosen: list[Document] = []
    for seen, document in enumerate(documents):
        if len(chosen) < size:
            chosen.append(document)
        elif (slot := rng.randrange(seen + 1)) < size:
            chosen[slot] = document
    return chosen


def preview(text: str, width: int = 300) -> str:
    graphemes = GRAPHEME.findall(" ".join(text.split()))
    if len(graphemes) <= width:
        return "".join(graphemes)
    return "".join(graphemes[:width]) + "…"


def describe(document: Document, width: int = 300) -> str:
    metadata = document.metadata
    tags = [metadata.get("filter_reason"), metadata.get("language")]
    if "language_score" in metadata:
        tags.append(f"{metadata['language_score']:.2f}")
    header = " ".join(f"[{tag}]" for tag in tags if tag)
    url = metadata.get("url", document.id)
    return f"{header} {url}\n    {preview(document.text, width)}"
