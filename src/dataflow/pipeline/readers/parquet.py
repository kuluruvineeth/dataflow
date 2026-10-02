import pyarrow.parquet as pq

from dataflow.data import DocumentsPipeline
from dataflow.io import DataFolderLike
from dataflow.pipeline.readers.base import BaseDiskReader


class ParquetReader(BaseDiskReader):
    name = "parquet"

    def __init__(self, data_folder: DataFolderLike, batch_size: int = 1000, read_metadata: bool = True, **kwargs):
        super().__init__(data_folder, **kwargs)
        self.batch_size = batch_size
        self.read_metadata = read_metadata

    def read_file(self, filepath: str) -> DocumentsPipeline:
        with self.data_folder.open(filepath, "rb") as file, pq.ParquetFile(file) as parquet:
            columns = None
            if not self.read_metadata:
                columns = [key for key in (self.text_key, self.id_key) if key in parquet.schema_arrow.names]
            row = 0
            for batch in parquet.iter_batches(batch_size=self.batch_size, columns=columns):
                for data in batch.to_pylist():
                    document = self.get_document_from_dict(data, filepath, row)
                    row += 1
                    if document:
                        yield document
