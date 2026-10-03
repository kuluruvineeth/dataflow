import logging
from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import IO

from dataflow.data import Document, DocumentsPipeline
from dataflow.io import DataFolderLike, get_datafolder
from dataflow.pipeline.base import PipelineStep
from dataflow.sources.http import open_url

logger = logging.getLogger(__name__)


class BaseDiskReader(PipelineStep):
    type = "Reader"

    def __init__(
        self,
        data_folder: DataFolderLike,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict | None = None,
        limit: int = -1,
        recursive: bool = True,
        glob_pattern: str | None = None,
        paths: list[str] | None = None,
    ):
        """`paths` lists the files explicitly instead of listing `data_folder`; with an http(s) `data_folder` it is
        required, and each file is streamed in a single request."""
        remote = isinstance(data_folder, str) and data_folder.startswith(("http://", "https://"))
        if remote and paths is None:
            raise ValueError("an http(s) data_folder cannot be listed; pass paths")
        self.base_url = data_folder.rstrip("/") + "/" if remote else None
        self.data_folder = None if remote else get_datafolder(data_folder)
        self.paths = paths
        self.text_key = text_key
        self.id_key = id_key
        self.default_metadata = default_metadata or {}
        self.limit = limit
        self.recursive = recursive
        self.glob_pattern = glob_pattern

    @abstractmethod
    def read_file(self, filepath: str) -> DocumentsPipeline: ...

    def open_input(self, filepath: str) -> AbstractContextManager[IO]:
        if self.base_url:
            return open_url(self.base_url + filepath)
        return self.data_folder.open(filepath, "rb")

    def adapt(self, data: dict, path: str, id_in_file: int) -> dict:
        metadata = data.pop("metadata", None) or {}
        return {
            "text": data.pop(self.text_key, ""),
            "id": str(data.pop(self.id_key, f"{path}/{id_in_file}")),
            "metadata": self.default_metadata | metadata | data,
        }

    def get_document_from_dict(self, data: dict, path: str, id_in_file: int) -> Document | None:
        fields = self.adapt(data, path, id_in_file)
        if not fields["text"]:
            return None
        return Document(**fields)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data
        if self.paths is not None:
            shard = self.paths[rank::world_size]
        else:
            shard = self.data_folder.get_shard(
                rank, world_size, recursive=self.recursive, glob_pattern=self.glob_pattern
            )
        if shard is None:
            raise RuntimeError(f"no files found in {self.data_folder.path}")
        read = 0
        for filepath in shard:
            logger.info("rank %d reading %s", rank, filepath)
            self.stat_update("input_files")
            for document in self.read_file(filepath):
                if self.limit != -1 and read >= self.limit:
                    return
                self.stat_update("documents")
                self.update_doc_stats(document)
                yield document
                read += 1
