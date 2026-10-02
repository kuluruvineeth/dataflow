from dataflow.data import Document, DocumentsPipeline
from dataflow.pipeline.base import PipelineStep
from dataflow.pipeline.writers.base import DiskWriter
from dataflow.utils.hashing import hash64


class HoldoutSplit(PipelineStep):
    type = "Split"
    name = "holdout"

    def __init__(self, holdout_writer: DiskWriter, fraction: float = 0.001, salt: str = "holdout"):
        if not 0 <= fraction <= 1:
            raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
        self.holdout_writer = holdout_writer
        self.fraction = fraction
        self.salt = salt
        self.threshold = int(fraction * 2**64)

    def in_holdout(self, document: Document) -> bool:
        return hash64(f"{self.salt}\0{document.text}") < self.threshold

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self.holdout_writer:
            for document in data:
                self.stat_update("total")
                if self.in_holdout(document):
                    self.stat_update("held_out")
                    self.holdout_writer.write(document, rank)
                    continue
                self.stat_update("forwarded")
                yield document
