from charset_normalizer import from_bytes
from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord

from dataflow.data import DocumentsPipeline
from dataflow.io import DataFolderLike
from dataflow.pipeline.readers.base import BaseDiskReader

HTML_TYPES = {"text/html", "application/xhtml+xml"}


class WarcReader(BaseDiskReader):
    name = "warc"

    def __init__(self, data_folder: DataFolderLike, compression: str | None = "infer", **kwargs):
        super().__init__(data_folder, **kwargs)
        self.compression = compression

    def read_file(self, filepath: str) -> DocumentsPipeline:
        with self.data_folder.open(filepath, "rb", compression=self.compression) as file:
            for record_number, record in enumerate(ArchiveIterator(file)):
                data = process_record(record)
                if data is None:
                    continue
                document = self.get_document_from_dict(data, filepath, record_number)
                if document:
                    yield document


def content_type(record: ArcWarcRecord) -> str | None:
    payload_type = record.rec_headers.get_header("WARC-Identified-Payload-Type")
    if payload_type:
        return payload_type.split(";")[0].strip().lower()
    if record.http_headers and (header := record.http_headers.get_header("Content-Type")):
        return header.split(";")[0].strip().lower()
    return None


def decode(content: bytes) -> str | None:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        match = from_bytes(content).best()
        return str(match) if match else None


def process_record(record: ArcWarcRecord) -> dict | None:
    if record.rec_type != "response" or content_type(record) not in HTML_TYPES:
        return None
    html = decode(record.content_stream().read())
    if not html:
        return None
    return {
        "text": html,
        "id": record.rec_headers.get_header("WARC-Record-ID"),
        "url": record.rec_headers.get_header("WARC-Target-URI"),
        "date": record.rec_headers.get_header("WARC-Date"),
    }
