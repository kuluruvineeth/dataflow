from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from dataflow.data import Document, DocumentsPipeline
from dataflow.utils.stats import Stats


class PipelineStep(ABC):
    name: str = None
    type: str = None

    @property
    def stats(self) -> Stats:
        if "_stats" not in self.__dict__:
            self._stats = Stats(repr(self))
        return self._stats

    def stat_update(self, *labels: str, value: float = 1, unit: str | None = None) -> None:
        for label in labels:
            self.stats.update(label, value, unit)

    def update_doc_stats(self, document: Document) -> None:
        self.stat_update("doc_len", value=len(document.text))

    def track_time(self) -> AbstractContextManager[None]:
        return self.stats.track_time()

    @abstractmethod
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data

    def __call__(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        return self.run(data, rank, world_size)

    def __repr__(self):
        return f"{self.type}: {self.name}"
