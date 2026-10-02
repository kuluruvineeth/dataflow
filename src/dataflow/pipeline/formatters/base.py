from abc import abstractmethod

from dataflow.data import DocumentsPipeline
from dataflow.pipeline.base import PipelineStep


class BaseFormatter(PipelineStep):
    type = "Formatter"

    @abstractmethod
    def format(self, text: str) -> str: ...

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for document in data:
            with self.track_time():
                text = self.format(document.text)
            self.stat_update("total")
            if text != document.text:
                self.stat_update("changed")
                document.text = text
            if not text.strip():
                self.stat_update("dropped_empty")
                continue
            yield document
