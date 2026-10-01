import logging
import multiprocessing
from abc import abstractmethod
from collections.abc import Callable
from multiprocessing.connection import Connection

from dataflow.data import DocumentsPipeline
from dataflow.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)


def _serve(conn: Connection, extract: Callable[[str], str | None]) -> None:
    conn.send("ready")
    while True:
        try:
            text = conn.recv()
        except EOFError:
            return
        try:
            conn.send(extract(text))
        except Exception as error:
            conn.send(error)


class ExtractorSandbox:
    def __init__(self, extract: Callable[[str], str | None], timeout: float, startup_timeout: float = 60):
        self.extract = extract
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.process = None
        self.conn = None

    def _start(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.conn, child_conn = ctx.Pipe()
        self.process = ctx.Process(target=_serve, args=(child_conn, self.extract), daemon=True)
        self.process.start()
        child_conn.close()
        if not self.conn.poll(self.startup_timeout):
            self._stop()
            raise TimeoutError("extraction worker did not start")
        self.conn.recv()

    def _stop(self) -> None:
        if self.process is not None:
            self.conn.close()
            self.process.kill()
            self.process.join()
        self.process = None
        self.conn = None

    def process_document(self, text: str) -> str | None:
        if self.process is None or not self.process.is_alive():
            self._stop()
            self._start()
        self.conn.send(text)
        if not self.conn.poll(self.timeout):
            self._stop()
            raise TimeoutError("extraction timed out")
        try:
            result = self.conn.recv()
        except EOFError:
            self._stop()
            raise
        if isinstance(result, Exception):
            raise result
        return result

    def __enter__(self) -> "ExtractorSandbox":
        return self

    def __exit__(self, *exc) -> None:
        self._stop()


class BaseExtractor(PipelineStep):
    type = "Extractor"

    def __init__(self, timeout: float = 1):
        self.timeout = timeout

    @abstractmethod
    def extract(self, text: str) -> str | None: ...

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with ExtractorSandbox(self.extract, self.timeout) as sandbox:
            for document in data:
                self.stat_update("total")
                with self.track_time():
                    try:
                        document.text = sandbox.process_document(document.text)
                    except TimeoutError:
                        self.stat_update("timeout")
                        logger.warning("extraction timed out for %s", document.id)
                        continue
                    except EOFError:
                        self.stat_update("broken_process")
                        continue
                    except Exception as error:
                        self.stat_update("extraction_error")
                        logger.warning("extraction failed for %s: %s", document.id, error)
                        continue
                if document.text:
                    self.stat_update("forwarded")
                    self.update_doc_stats(document)
                    yield document
                else:
                    self.stat_update("dropped")
