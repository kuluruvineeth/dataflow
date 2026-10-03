import json
import logging
import zlib

from charset_normalizer import from_bytes
from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord

from dataflow.data import DocumentsPipeline
from dataflow.pipeline.readers.base import BaseDiskReader

logger = logging.getLogger(__name__)

HTML_TYPES = {"text/html", "application/xhtml+xml"}
POLICY_HEADERS = ("X-Robots-Tag", "TDM-Reservation", "Content-Usage")


class TruncatedRecordError(Exception):
    pass


class WarcReader(BaseDiskReader):
    """HTML responses from WARC files.

    With `cld2_languages=True`, each page also gets `cld2_languages`: the language codes Common Crawl's CLD2 found,
    read from the metadata record that follows the response (the same source as the index's `content_languages`).
    """

    name = "warc"

    def __init__(self, data_folder, cld2_languages: bool = False, **kwargs):
        super().__init__(data_folder, **kwargs)
        self.cld2_languages = cld2_languages

    def read_file(self, filepath: str) -> DocumentsPipeline:
        pending: dict[str, dict] = {}
        with self.open_input(filepath) as file:
            try:
                for record_number, record in enumerate(ArchiveIterator(file)):
                    if self.cld2_languages and record.rec_type == "metadata":
                        data = pending.pop(record.rec_headers.get_header("WARC-Concurrent-To"), None)
                        if data is not None:
                            data["cld2_languages"] = cld2_codes(record.content_stream().read())
                            yield from self.emit(data, filepath, data.pop("_record_number"))
                        continue
                    data = process_record(record)
                    if data is None:
                        continue
                    if self.cld2_languages:
                        data["_record_number"] = record_number
                        pending[data["id"]] = data
                        continue
                    yield from self.emit(data, filepath, record_number)
            except (TruncatedRecordError, EOFError, zlib.error) as error:
                logger.warning("stopped reading truncated archive %s: %s", filepath, error)
                self.stat_update("truncated_files")
        for data in pending.values():
            data["cld2_languages"] = []
            yield from self.emit(data, filepath, data.pop("_record_number"))

    def emit(self, data: dict, filepath: str, record_number: int) -> DocumentsPipeline:
        document = self.get_document_from_dict(data, filepath, record_number)
        if document:
            yield document


def cld2_codes(metadata: bytes) -> list[str]:
    for line in metadata.decode("utf-8", errors="replace").splitlines():
        if line.startswith("languages-cld2:"):
            try:
                return [language["code"] for language in json.loads(line.split(":", 1)[1])["languages"]]
            except (ValueError, KeyError, TypeError):
                return []
    return []


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
    data = {
        "text": html,
        "id": record.rec_headers.get_header("WARC-Record-ID"),
        "url": record.rec_headers.get_header("WARC-Target-URI"),
        "date": record.rec_headers.get_header("WARC-Date"),
    }
    headers = {name: value for name in POLICY_HEADERS if (value := record.http_headers.get_header(name))}
    if headers:
        data["http_headers"] = headers
    return data
