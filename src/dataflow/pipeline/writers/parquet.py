from collections import defaultdict
from typing import IO

import pyarrow as pa
import pyarrow.parquet as pq

from dataflow.data import Document
from dataflow.io import DataFolderLike
from dataflow.pipeline.writers.base import DiskWriter


def infer_schema(rows: list[dict]) -> pa.Schema:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    return pa.Table.from_pydict({key: [row.get(key) for row in rows] for key in keys}).schema


class ParquetWriter(DiskWriter):
    name = "parquet"
    default_output_filename = "${rank}.parquet"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str | None = None,
        compression: str | None = "zstd",
        compression_level: int | None = 9,
        batch_size: int = 1000,
        schema: pa.Schema | None = None,
        content_defined_chunking: bool = True,
        write_page_index: bool = True,
    ):
        super().__init__(output_folder, output_filename, compression=None, mode="wb")
        self.compression = compression
        self.compression_level = compression_level
        self.batch_size = batch_size
        self.schema = schema
        self.content_defined_chunking = content_defined_chunking
        self.write_page_index = write_page_index
        self._buffers: dict[IO, list[dict]] = defaultdict(list)
        self._writers: dict[IO, pq.ParquetWriter] = {}

    def adapt(self, document: Document) -> dict:
        return {"text": document.text, "id": document.id, **document.metadata}

    def write_record(self, record: dict, file: IO) -> None:
        self._buffers[file].append(record)
        if len(self._buffers[file]) >= self.batch_size:
            self.flush(file)

    def flush(self, file: IO) -> None:
        rows = self._buffers.pop(file, [])
        if not rows:
            return
        if file not in self._writers:
            self._writers[file] = pq.ParquetWriter(
                file,
                self.schema or infer_schema(rows),
                compression=self.compression,
                compression_level=self.compression_level,
                use_content_defined_chunking=self.content_defined_chunking,
                write_page_index=self.write_page_index,
            )
        schema = self._writers[file].schema
        unknown = sorted({key for row in rows for key in row} - set(schema.names))
        if unknown:
            raise ValueError(f"columns {unknown} are not in the schema {schema.names}; pass an explicit schema")
        self._writers[file].write_table(pa.Table.from_pylist(rows, schema=schema))

    def __exit__(self, *exc) -> None:
        for file in list(self._buffers):
            self.flush(file)
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
        super().__exit__(*exc)
