from collections.abc import Callable

from dataflow.data import Document
from dataflow.pipeline.filters.base import BaseFilter, FilterResult
from dataflow.pipeline.writers.base import DiskWriter


class LambdaFilter(BaseFilter):
    name = "lambda"

    def __init__(
        self,
        filter_function: Callable[[Document], FilterResult],
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        self.filter_function = filter_function

    def filter(self, document: Document) -> FilterResult:
        return self.filter_function(document)
