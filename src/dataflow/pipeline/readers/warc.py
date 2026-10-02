import logging
import zlib

from charset_normalizer import from_bytes
from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord

from dataflow.data import DocumentsPipeline
from dataflow.pipeline.readers.base import BaseDiskReader

logger = logging.getLogger(__name__)

HTML_TYPES = {"text/html", "application/xhtml+xml"}


class TruncatedRecordError(Exception):
    pass


class WarcReader(BaseDiskReader):
    name = "warc"

    def read_file(self, filepath: str) -> DocumentsPipeline:
        with self.data_folder.open(filepath, "rb") as file:
            try:
                for record_number, record in enumerate(ArchiveIterator(file)):
                    data = process_record(record)
                    if data is None:
                        continue
                    document = self.get_document_from_dict(data, filepath, record_number)
                    if document:
                        yield document
            except (TruncatedRecordError, EOFError, zlib.error) as error:
                logger.warning("stopped reading truncated archive %s: %s", filepath, error)
                self.stat_update("truncated_files")


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
    content = record.content_stream().read()
    if record.raw_stream.limit > 0:
        raise TruncatedRecordError(f"{record.raw_stream.limit} bytes missing")
    html = decode(content)
    if not html:
        return None
    return {
        "text": html,
        "id": record.rec_headers.get_header("WARC-Record-ID"),
        "url": record.rec_headers.get_header("WARC-Target-URI"),
        "date": record.rec_headers.get_header("WARC-Date"),
    }
