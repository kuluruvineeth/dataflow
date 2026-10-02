from collections.abc import Callable

from dataflow.data import Document
from dataflow.pipeline.filters.base import BaseFilter, FilterResult
from dataflow.pipeline.filters.gopher_repetition import find_duplicates
from dataflow.pipeline.writers.base import DiskWriter
from dataflow.utils.text import ends_sentence, split_words


class FineWebQualityFilter(BaseFilter):
    name = "fineweb quality"

    def __init__(
        self,
        line_punct_thr: float | None = 0.12,
        line_punct_exclude_zero: bool = False,
        short_line_thr: float | None = 0.67,
        short_line_length: int = 30,
        char_duplicates_ratio: float | None = 0.01,
        new_line_ratio: float | None = 0.3,
        word_splitter: Callable[[str], list[str]] = split_words,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        self.line_punct_thr = line_punct_thr
        self.line_punct_exclude_zero = line_punct_exclude_zero
        self.short_line_thr = short_line_thr
        self.short_line_length = short_line_length
        self.char_duplicates_ratio = char_duplicates_ratio
        self.new_line_ratio = new_line_ratio
        self.word_splitter = word_splitter

    def filter(self, document: Document) -> FilterResult:
        text = document.text
        lines = [line for line in text.split("\n") if line.strip()]
        if not lines:
            return False, "empty"

        if self.line_punct_thr is not None:
            ratio = sum(map(ends_sentence, lines)) / len(lines)
            if ratio < self.line_punct_thr and not (ratio == 0 and self.line_punct_exclude_zero):
                return False, "line_punct_ratio"

        if self.short_line_thr is not None:
            ratio = sum(len(line) <= self.short_line_length for line in lines) / len(lines)
            if ratio > self.short_line_thr:
                return False, "short_line_ratio"

        if self.char_duplicates_ratio is not None:
            if find_duplicates(lines)[1] / len(text.replace("\n", "")) > self.char_duplicates_ratio:
                return False, "char_dup_ratio"

        if self.new_line_ratio is not None:
            if text.count("\n") / max(len(self.word_splitter(text)), 1) > self.new_line_ratio:
                return False, "list_ratio"
        return True
