import re
from collections import Counter
from collections.abc import Callable

from dataflow.data import Document
from dataflow.pipeline.filters.base import BaseFilter, FilterResult
from dataflow.pipeline.writers.base import DiskWriter
from dataflow.utils.text import ngrams, split_words

PARAGRAPH_PATTERN = re.compile(r"\n{2,}")
TOP_N_GRAMS = ((2, 0.2), (3, 0.18), (4, 0.16))
DUP_N_GRAMS = ((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.1))


def find_duplicates(items: list[str]) -> tuple[int, int]:
    seen = set()
    count = chars = 0
    for item in items:
        if item in seen:
            count += 1
            chars += len(item)
        else:
            seen.add(item)
    return count, chars


def top_ngram_chars(words: list[str], n: int) -> int:
    counts = Counter(ngrams(words, n))
    if not counts:
        return 0
    gram, count = counts.most_common(1)[0]
    return (sum(map(len, gram)) + n - 1) * count


def duplicated_ngram_chars(words: list[str], n: int) -> int:
    seen = set()
    chars = i = 0
    while i <= len(words) - n:
        gram = tuple(words[i : i + n])
        if gram in seen:
            chars += sum(map(len, gram))
            i += n
        else:
            seen.add(gram)
            i += 1
    return chars


class GopherRepetitionFilter(BaseFilter):
    name = "gopher repetition"

    def __init__(
        self,
        dup_line_frac: float | None = 0.3,
        dup_para_frac: float | None = 0.3,
        dup_line_char_frac: float | None = 0.2,
        dup_para_char_frac: float | None = 0.2,
        top_n_grams: tuple[tuple[int, float], ...] = TOP_N_GRAMS,
        dup_n_grams: tuple[tuple[int, float], ...] = DUP_N_GRAMS,
        word_splitter: Callable[[str], list[str]] = split_words,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        self.dup_line_frac = dup_line_frac
        self.dup_para_frac = dup_para_frac
        self.dup_line_char_frac = dup_line_char_frac
        self.dup_para_char_frac = dup_para_char_frac
        self.top_n_grams = top_n_grams
        self.dup_n_grams = dup_n_grams
        self.word_splitter = word_splitter

    def filter(self, document: Document) -> FilterResult:
        text = document.text
        if not text.strip():
            return False, "empty"

        paragraphs = PARAGRAPH_PATTERN.split(text.strip())
        duplicates, chars = find_duplicates(paragraphs)
        if self.dup_para_frac and duplicates / len(paragraphs) > self.dup_para_frac:
            return False, "dup_para_frac"
        if self.dup_para_char_frac and chars / len(text) > self.dup_para_char_frac:
            return False, "dup_para_char_frac"

        lines = [line for line in text.splitlines() if line.strip()]
        duplicates, chars = find_duplicates(lines)
        if self.dup_line_frac and duplicates / len(lines) > self.dup_line_frac:
            return False, "dup_line_frac"
        if self.dup_line_char_frac and chars / len(text) > self.dup_line_char_frac:
            return False, "dup_line_char_frac"

        words = self.word_splitter(text)
        for n, max_frac in self.top_n_grams:
            if top_ngram_chars(words, n) / len(text) > max_frac:
                return False, f"top_{n}_gram"
        for n, max_frac in self.dup_n_grams:
            if duplicated_ngram_chars(words, n) / len(text) > max_frac:
                return False, f"duplicated_{n}_n_grams"
        return True
