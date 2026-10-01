from dataflow.data import Document
from dataflow.pipeline.filters.base import BaseFilter, FilterResult
from dataflow.pipeline.writers.base import DiskWriter
from dataflow.utils.text import PUNCTUATION_SET, split_words

STOP_WORDS = ["the", "be", "to", "of", "and", "that", "have", "with"]


class GopherQualityFilter(BaseFilter):
    name = "gopher quality"

    def __init__(
        self,
        min_doc_words: int | None = 50,
        max_doc_words: int | None = 100_000,
        min_avg_word_length: float | None = 3,
        max_avg_word_length: float | None = 10,
        max_symbol_word_ratio: float | None = 0.1,
        max_bullet_lines_ratio: float | None = 0.9,
        max_ellipsis_lines_ratio: float | None = 0.3,
        min_alpha_words_ratio: float | None = 0.8,
        min_stop_words: int | None = 2,
        stop_words: list[str] | None = None,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        self.min_doc_words = min_doc_words
        self.max_doc_words = max_doc_words
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.max_symbol_word_ratio = max_symbol_word_ratio
        self.max_bullet_lines_ratio = max_bullet_lines_ratio
        self.max_ellipsis_lines_ratio = max_ellipsis_lines_ratio
        self.min_alpha_words_ratio = min_alpha_words_ratio
        self.min_stop_words = min_stop_words
        self.stop_words = set(STOP_WORDS if stop_words is None else stop_words)

    def filter(self, document: Document) -> FilterResult:
        text = document.text
        words = split_words(text)
        if not words:
            return False, "gopher_short_doc"
        content_words = [word for word in words if any(char not in PUNCTUATION_SET for char in word)]
        n_content = len(content_words)

        if self.min_doc_words and n_content < self.min_doc_words:
            return False, "gopher_short_doc"
        if self.max_doc_words and n_content > self.max_doc_words:
            return False, "gopher_long_doc"

        avg_word_length = sum(map(len, content_words)) / n_content
        if self.min_avg_word_length and avg_word_length < self.min_avg_word_length:
            return False, "gopher_below_avg_threshold"
        if self.max_avg_word_length and avg_word_length > self.max_avg_word_length:
            return False, "gopher_above_avg_threshold"

        if self.max_symbol_word_ratio:
            if text.count("#") / len(words) > self.max_symbol_word_ratio:
                return False, "gopher_too_many_hashes"
            if (text.count("...") + text.count("…")) / len(words) > self.max_symbol_word_ratio:
                return False, "gopher_too_many_ellipsis"

        lines = text.splitlines() or [text]
        if self.max_bullet_lines_ratio:
            bullets = sum(line.lstrip().startswith(("•", "-")) for line in lines)
            if bullets / len(lines) > self.max_bullet_lines_ratio:
                return False, "gopher_too_many_bullets"
        if self.max_ellipsis_lines_ratio:
            ellipsis_ends = sum(line.rstrip().endswith(("...", "…")) for line in lines)
            if ellipsis_ends / len(lines) > self.max_ellipsis_lines_ratio:
                return False, "gopher_too_many_end_ellipsis"

        if self.min_alpha_words_ratio:
            alpha_words = sum(any(char.isalpha() for char in word) for word in words)
            if alpha_words / len(words) < self.min_alpha_words_ratio:
                return False, "gopher_below_alpha_threshold"

        if self.min_stop_words and len(self.stop_words & set(words)) < self.min_stop_words:
            return False, "gopher_not_enough_stop_words"

        return True
