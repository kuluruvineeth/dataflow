import logging

import orjson

from dataflow.data import DocumentsPipeline
from dataflow.io import DataFolderLike
from dataflow.pipeline.readers.base import BaseDiskReader

logger = logging.getLogger(__name__)


class JsonlReader(BaseDiskReader):
    name = "jsonl"

    def __init__(self, data_folder: DataFolderLike, compression: str | None = "infer", **kwargs):
        super().__init__(data_folder, **kwargs)
        self.compression = compression

    def read_file(self, filepath: str) -> DocumentsPipeline:
        with self.data_folder.open(filepath, "rt", compression=self.compression) as file:
            for line_number, line in enumerate(file):
                try:
                    data = orjson.loads(line)
                except orjson.JSONDecodeError as error:
                    logger.warning("skipping bad line %d in %s: %s", line_number, filepath, error)
                    continue
                document = self.get_document_from_dict(data, filepath, line_number)
                if document:
                    yield document
