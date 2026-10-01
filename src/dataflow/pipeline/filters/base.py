from abc import abstractmethod
from contextlib import nullcontext

from dataflow.data import Document, DocumentsPipeline
from dataflow.pipeline.base import PipelineStep
from dataflow.pipeline.writers.base import DiskWriter

FilterResult = bool | tuple[bool, str]


class BaseFilter(PipelineStep):
    type = "Filter"

    def __init__(self, exclusion_writer: DiskWriter | None = None):
        self.exclusion_writer = exclusion_writer

    @abstractmethod
    def filter(self, document: Document) -> FilterResult: ...

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self.exclusion_writer or nullcontext():
            for document in data:
                result = self.filter(document)
                keep, reason = result if isinstance(result, tuple) else (result, None)
                if keep:
                    yield document
                elif self.exclusion_writer:
                    if reason:
                        document.metadata["filter_reason"] = reason
                    self.exclusion_writer.write(document, rank)
