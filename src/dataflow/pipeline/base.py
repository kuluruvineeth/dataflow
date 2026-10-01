from abc import ABC, abstractmethod

from dataflow.data import DocumentsPipeline


class PipelineStep(ABC):
    name: str = None
    type: str = None

    @abstractmethod
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            yield from data

    def __call__(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        return self.run(data, rank, world_size)

    def __repr__(self):
        return f"{self.type}: {self.name}"
