import dataclasses
import logging
from abc import abstractmethod
from string import Template
from typing import IO

from dataflow.data import Document, DocumentsPipeline
from dataflow.io import DataFolderLike, get_datafolder
from dataflow.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)

EXTENSIONS = {"gzip": ".gz", "zstd": ".zst"}


class DiskWriter(PipelineStep):
    type = "Writer"
    default_output_filename: str = "${rank}"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str | None = None,
        compression: str | None = "gzip",
        mode: str = "wt",
    ):
        self.output_folder = get_datafolder(output_folder)
        filename = output_filename or self.default_output_filename
        extension = EXTENSIONS.get(compression, "")
        if extension and not filename.endswith(extension):
            filename += extension
        if "${rank}" not in filename:
            logger.warning("output filename %r has no ${rank}; parallel tasks may overwrite each other", filename)
        self.output_filename = Template(filename)
        self.output_mg = self.output_folder.get_output_file_manager(mode=mode, compression=compression)

    def adapt(self, document: Document) -> dict:
        return {key: value for key, value in dataclasses.asdict(document).items() if value}

    def get_output_filename(self, document: Document, rank: int) -> str:
        return self.output_filename.substitute({"rank": f"{rank:05d}", "id": document.id, **document.metadata})

    @abstractmethod
    def write_record(self, record: dict, file: IO) -> None: ...

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self.output_mg:
            for document in data:
                file = self.output_mg.get_file(self.get_output_filename(document, rank))
                self.write_record(self.adapt(document), file)
                yield document
