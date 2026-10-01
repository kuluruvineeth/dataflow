import logging
from abc import abstractmethod

from dataflow.data import Document, DocumentsPipeline
from dataflow.io import DataFolderLike, get_datafolder
from dataflow.pipeline.base import PipelineStep

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
    ):
        self.data_folder = get_datafolder(data_folder)
        self.text_key = text_key
        self.id_key = id_key
        self.default_metadata = default_metadata or {}
        self.limit = limit
        self.recursive = recursive
        self.glob_pattern = glob_pattern

    @abstractmethod
    def read_file(self, filepath: str) -> DocumentsPipeline: ...

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
        shard = self.data_folder.get_shard(rank, world_size, recursive=self.recursive, glob_pattern=self.glob_pattern)
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
