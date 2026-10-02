import struct
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np

from dataflow.data import Document, DocumentsPipeline
from dataflow.io import DataFolderLike, get_datafolder
from dataflow.pipeline.base import PipelineStep
from dataflow.pipeline.filters.base import BaseFilter, FilterResult
from dataflow.pipeline.writers.base import DiskWriter
from dataflow.utils.binaryio import read_tuples
from dataflow.utils.hashing import hash64
from dataflow.utils.text import simplify_text

MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)
COUNT = "<QI"


def words_of(text: str) -> list[str]:
    return simplify_text(text).split()


def ngram_hashes(words: list[str], n: int) -> np.ndarray:
    """One 64-bit hash per n-gram, combined from per-word hashes so it costs one hash per word, not per n-gram."""
    if len(words) < n:
        return np.empty(0, dtype=np.uint64)
    word_hashes = np.fromiter((hash64(word) for word in words), dtype=np.uint64, count=len(words))
    count = len(words) - n + 1
    hashes = word_hashes[:count].copy()
    for offset in range(1, n):
        hashes = hashes * MULTIPLIER + word_hashes[offset : offset + count]
    return hashes


def informative(words: list[str], n: int) -> np.ndarray:
    """Per n-gram: are at least half its words longer than one character? Number runs, tables and alphabets are not."""
    if len(words) < n:
        return np.zeros(0, dtype=bool)
    long = np.fromiter((len(word) > 1 for word in words), dtype=np.int32, count=len(words))
    return 2 * np.convolve(long, np.ones(n, dtype=np.int32), "valid") >= n


def text_ngrams(text: str, n: int) -> np.ndarray:
    """The distinct informative n-grams of a training text."""
    words = words_of(text)
    return np.unique(ngram_hashes(words, n)[informative(words, n)])


@dataclass
class NgramIndex:
    """Sorted n-gram hashes of every eval item, each paired with the item it came from."""

    n: int
    hashes: np.ndarray
    items: np.ndarray

    @classmethod
    def build(cls, texts: Iterable[str], n: int) -> "NgramIndex":
        hashes, items = [np.empty(0, dtype=np.uint64)], [np.empty(0, dtype=np.uint32)]
        for item, text in enumerate(texts):
            unique = np.unique(ngram_hashes(words_of(text), n))
            hashes.append(unique)
            items.append(np.full(len(unique), item, dtype=np.uint32))
        all_hashes, all_items = np.concatenate(hashes), np.concatenate(items)
        order = np.argsort(all_hashes, kind="stable")
        return cls(n, all_hashes[order], all_items[order])

    def __len__(self) -> int:
        return len(self.hashes)

    def contains(self, hashes: np.ndarray) -> np.ndarray:
        if not len(self.hashes):
            return np.zeros(len(hashes), dtype=bool)
        positions = np.minimum(np.searchsorted(self.hashes, hashes), len(self.hashes) - 1)
        return self.hashes[positions] == hashes

    def matches(self, text: str) -> np.ndarray:
        """The distinct informative n-grams of `text` that occur in some eval item."""
        hashes = text_ngrams(text, self.n)
        return hashes[self.contains(hashes)]

    def items_of(self, ngram_hash: int) -> np.ndarray:
        value = np.uint64(ngram_hash)
        start, end = np.searchsorted(self.hashes, value, "left"), np.searchsorted(self.hashes, value, "right")
        return self.items[start:end]

    def items_with(self, hashes: np.ndarray) -> np.ndarray:
        """Every eval item that contains at least one of `hashes`."""
        return np.unique(self.items[np.isin(self.hashes, hashes)])

    def shared_ngrams(self, max_items: int) -> np.ndarray:
        """N-grams in more than `max_items` eval items: templates and number runs, not the content of a test."""
        values, counts = np.unique(self.hashes, return_counts=True)
        return values[counts > max_items]

    def save(self, path: str | Path | IO) -> None:
        np.savez(path, n=self.n, hashes=self.hashes, items=self.items)

    @classmethod
    def load(cls, path: str | Path | IO) -> "NgramIndex":
        with np.load(path) as data:
            return cls(int(data["n"]), data["hashes"], data["items"])


def load_index(folder: DataFolderLike, n: int) -> NgramIndex:
    with get_datafolder(folder).open(f"ngrams-{n}.npz", "rb") as file:
        return NgramIndex.load(file)


def document_frequencies(folder: DataFolderLike) -> dict[int, int]:
    """How many training documents contain each matched eval n-gram, summed over every task's counts."""
    folder = get_datafolder(folder)
    totals: Counter[int] = Counter()
    for path in folder.list_files(glob_pattern="*.counts"):
        with folder.open(path, "rb") as file:
            for ngram_hash, documents in read_tuples(file, COUNT):
                totals[ngram_hash] += documents
    return dict(totals)


def contaminating_ngrams(
    index: NgramIndex, frequencies: dict[int, int], max_documents: int = 10, max_items: int = 10
) -> np.ndarray:
    """Matched n-grams that count as leaked test text: in at most `max_documents` training documents (more is web
    boilerplate) and in at most `max_items` eval items (more is a template shared by the tests)."""
    rare = np.array(sorted(h for h, documents in frequencies.items() if documents <= max_documents), dtype=np.uint64)
    return np.setdiff1d(rare, index.shared_ngrams(max_items))


def covered_fraction(text: str, n: int, ngrams: np.ndarray) -> float:
    """Share of the words of `text` inside an n-gram that is in `ngrams` (sorted): Tulu 3's per-item measure."""
    words = words_of(text)
    hits = np.isin(ngram_hashes(words, n), ngrams)
    if not hits.any():
        return 0.0
    covered = np.zeros(len(words), dtype=bool)
    for offset in range(n):
        covered[offset : offset + len(hits)] |= hits
    return float(covered.mean())


class NgramDecontamCount(PipelineStep):
    """First pass: for every eval n-gram found in the data, count the documents that contain it."""

    type = "Decont"
    name = "n-gram counts"

    def __init__(self, index_folder: DataFolderLike, output_folder: DataFolderLike, n: int = 13):
        self.index_folder = get_datafolder(index_folder)
        self.output_folder = get_datafolder(output_folder)
        self.n = n

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        index = load_index(self.index_folder, self.n)
        counts: Counter[int] = Counter()
        for document in data:
            self.stat_update("total")
            with self.track_time():
                matched = index.matches(document.text)
            if len(matched):
                self.stat_update("with_matches")
                counts.update(matched.tolist())
        packer = struct.Struct(COUNT)
        with self.output_folder.open(f"{rank:05d}.counts", "wb") as file:
            file.write(b"".join(packer.pack(h, documents) for h, documents in sorted(counts.items())))
        yield from ()


class NgramDecontamFilter(BaseFilter):
    """Drops documents that contain an eval n-gram, ignoring n-grams in more than `max_documents` documents (GPT-3)
    and n-grams in more than `max_items` eval items.

    Without `counts_folder` the document limit is not applied.
    """

    name = "n-gram decontamination"

    def __init__(
        self,
        index_folder: DataFolderLike,
        counts_folder: DataFolderLike | None = None,
        n: int = 13,
        max_documents: int = 10,
        max_items: int = 10,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        self.index_folder = get_datafolder(index_folder)
        self.counts_folder = get_datafolder(counts_folder) if counts_folder is not None else None
        self.n = n
        self.max_documents = max_documents
        self.max_items = max_items

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        self.index = load_index(self.index_folder, self.n)
        self.ignored = self.index.shared_ngrams(self.max_items)
        if self.counts_folder is not None:
            frequencies = document_frequencies(self.counts_folder)
            common = [h for h, documents in frequencies.items() if documents > self.max_documents]
            self.ignored = np.union1d(self.ignored, np.array(common, dtype=np.uint64))
        yield from super().run(data, rank, world_size)

    def filter(self, document: Document) -> FilterResult:
        matched = self.index.matches(document.text)
        if len(matched) and not np.isin(matched, self.ignored).all():
            return False, "contaminated"
        return True
