from trafilatura import extract

from dataflow.pipeline.extractors.base import BaseExtractor


class Trafilatura(BaseExtractor):
    name = "trafilatura"

    def __init__(self, favour_precision: bool = True, deduplicate: bool = True, timeout: float = 1, **kwargs):
        super().__init__(timeout)
        self.favour_precision = favour_precision
        self.deduplicate = deduplicate
        self.kwargs = kwargs

    def extract(self, text: str) -> str | None:
        return extract(
            text,
            favor_precision=self.favour_precision,
            include_comments=False,
            deduplicate=self.deduplicate,
            **self.kwargs,
        )
