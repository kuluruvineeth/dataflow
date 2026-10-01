from typing import IO

import orjson

from dataflow.io import DataFolderLike
from dataflow.pipeline.writers.base import DiskWriter


class JsonlWriter(DiskWriter):
    name = "jsonl"
    default_output_filename = "${rank}.jsonl"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str | None = None,
        compression: str | None = "gzip",
    ):
        super().__init__(output_folder, output_filename, compression, mode="wb")

    def write_record(self, record: dict, file: IO) -> None:
        file.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
