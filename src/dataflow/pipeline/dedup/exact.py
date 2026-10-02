import heapq
import struct
from collections.abc import Callable, Iterator
from contextlib import ExitStack, nullcontext
from typing import BinaryIO

from dataflow.data import Document, DocumentsPipeline
from dataflow.io import DataFolderLike, get_datafolder
from dataflow.pipeline.base import PipelineStep
from dataflow.pipeline.writers.base import DiskWriter
from dataflow.utils.binaryio import read_tuples
from dataflow.utils.hashing import hash64

SIGNATURE = "<QI"
DUPLICATE = "<I"


def document_text(document: Document) -> str:
    return document.text


class ExactDedupSignature(PipelineStep):
    type = "Dedup"
    name = "exact signatures"

    def __init__(
        self,
        output_folder: DataFolderLike,
        finder_workers: int = 1,
        content_getter: Callable[[Document], str] = document_text,
    ):
        self.output_folder = get_datafolder(output_folder)
        self.finder_workers = finder_workers
        self.content_getter = content_getter

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        signatures = []
        for index, document in enumerate(data):
            with self.track_time():
                signatures.append((hash64(self.content_getter(document)), index))
            self.stat_update("total")
        signatures.sort()
        buckets: list[list[tuple[int, int]]] = [[] for _ in range(self.finder_workers)]
        for hash_value, index in signatures:
            buckets[hash_value % self.finder_workers].append((hash_value, index))
        packer = struct.Struct(SIGNATURE)
        for bucket, records in enumerate(buckets):
            with self.output_folder.open(f"{bucket:04d}/{rank:05d}.sig", "wb") as file:
                file.write(b"".join(packer.pack(hash_value, index) for hash_value, index in records))
        yield from ()


def source_rank(path: str) -> int:
    return int(path.rsplit("/", 1)[-1].removesuffix(".sig"))


def read_signatures(file: BinaryIO, origin: int) -> Iterator[tuple[int, int, int]]:
    for hash_value, index in read_tuples(file, SIGNATURE):
        yield hash_value, origin, index


class ExactFindDedups(PipelineStep):
    type = "Dedup"
    name = "exact find duplicates"

    def __init__(self, data_folder: DataFolderLike, output_folder: DataFolderLike):
        self.data_folder = get_datafolder(data_folder)
        self.output_folder = get_datafolder(output_folder)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        buckets = {path.split("/")[0] for path in self.data_folder.list_files(glob_pattern="*/*.sig")}
        if len(buckets) != world_size:
            raise ValueError(f"found {len(buckets)} signature buckets; run this step with tasks={len(buckets)}")
        paths = self.data_folder.list_files(subdirectory=f"{rank:04d}", glob_pattern="*.sig")
        packer = struct.Struct(DUPLICATE)
        last_hash = None
        with ExitStack() as stack, self.output_folder.get_output_file_manager(mode="wb") as output:
            streams = [
                read_signatures(stack.enter_context(self.data_folder.open(path, "rb")), source_rank(path))
                for path in paths
            ]
            for hash_value, origin, index in heapq.merge(*streams):
                if hash_value == last_hash:
                    output.get_file(f"{origin:05d}/{rank:04d}.dups").write(packer.pack(index))
                    self.stat_update("duplicates")
                last_hash = hash_value
        yield from ()


class ExactDedupFilter(PipelineStep):
    type = "Dedup"
    name = "exact filter"

    def __init__(self, data_folder: DataFolderLike, exclusion_writer: DiskWriter | None = None):
        self.data_folder = get_datafolder(data_folder)
        self.exclusion_writer = exclusion_writer

    def load_duplicates(self, rank: int) -> list[int]:
        folder = f"{rank:05d}"
        if not self.data_folder.exists(folder):
            return []
        duplicates = []
        for path in self.data_folder.list_files(subdirectory=folder, glob_pattern="*.dups"):
            with self.data_folder.open(path, "rb") as file:
                duplicates.extend(index for (index,) in read_tuples(file, DUPLICATE))
        return sorted(duplicates)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        duplicates = self.load_duplicates(rank)
        next_duplicate = 0
        with self.exclusion_writer or nullcontext():
            for index, document in enumerate(data):
                self.stat_update("total")
                if next_duplicate < len(duplicates) and duplicates[next_duplicate] == index:
                    next_duplicate += 1
                    self.stat_update("dropped")
                    if self.exclusion_writer:
                        self.exclusion_writer.write(document, rank)
                    continue
                self.stat_update("forwarded")
                yield document
        if next_duplicate != len(duplicates):
            raise RuntimeError("input changed between the signature and filter stages")
