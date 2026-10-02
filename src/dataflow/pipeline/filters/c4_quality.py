import regex

from dataflow.data import Document
from dataflow.pipeline.filters.base import BaseFilter, FilterResult
from dataflow.pipeline.writers.base import DiskWriter
from dataflow.utils.text import count_sentences

CITATION_PATTERN = regex.compile(r"\[\d*]|\[edit]|\[citation needed]")
END_PUNCTUATION = (".", "?", "!", '"', "'")
ELLIPSIS = "..."
POLICY_PHRASES = ("terms of use", "privacy policy", "cookie policy", "uses cookies", "use of cookies", "use cookies")


class C4QualityFilter(BaseFilter):
    """Line and document rules from C4 (Raffel et al. 2020, section 2.2, and the reference c4_utils.py).

    Lines are dropped for: a word over `max_word_length` characters, no terminal punctuation (or ending in "..."),
    fewer than `min_words_per_line` words, mentioning "javascript", or a cookie/terms-of-use phrase. Whole documents
    are dropped for "lorem ipsum", a curly bracket, or fewer than `min_sentences` sentences in the lines that remain.
    Kept documents keep only the surviving lines. FineWeb uses this with `filter_no_terminal_punct=False`.
    Sentences are counted with spaCy's rule-based sentencizer, as in the reference; other splitters change decisions.
    """

    name = "c4 quality"

    def __init__(
        self,
        remove_citations: bool = True,
        filter_no_terminal_punct: bool = True,
        min_sentences: int | None = 5,
        min_words_per_line: int | None = 3,
        max_word_length: int | None = 1000,
        filter_lorem_ipsum: bool = True,
        filter_javascript: bool = True,
        filter_curly_bracket: bool = True,
        filter_policy: bool = True,
        language: str = "en",
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        self.remove_citations = remove_citations
        self.filter_no_terminal_punct = filter_no_terminal_punct
        self.min_sentences = min_sentences
        self.min_words_per_line = min_words_per_line
        self.max_word_length = max_word_length
        self.filter_lorem_ipsum = filter_lorem_ipsum
        self.filter_javascript = filter_javascript
        self.filter_curly_bracket = filter_curly_bracket
        self.filter_policy = filter_policy
        self.language = language

    def filter(self, document: Document) -> FilterResult:
        sentences, kept = 0, []
        for line in document.text.splitlines():
            line = line.strip()
            words = line.split()
            self.stat_update("lines")
            if self.max_word_length is not None and any(len(word) > self.max_word_length for word in words):
                self.stat_update("lines_dropped_too_long_word")
                continue
            if self.remove_citations:
                line = CITATION_PATTERN.sub("", line)
            if self.filter_no_terminal_punct and (not line.endswith(END_PUNCTUATION) or line.endswith(ELLIPSIS)):
                self.stat_update("lines_dropped_no_terminal_punct")
                continue
            if self.min_words_per_line is not None and len(words) < self.min_words_per_line:
                self.stat_update("lines_dropped_too_few_words")
                continue
            lower = line.lower()
            if self.filter_lorem_ipsum and "lorem ipsum" in lower:
                return False, "lorem_ipsum"
            if self.filter_javascript and "javascript" in lower:
                self.stat_update("lines_dropped_javascript")
                continue
            if self.filter_curly_bracket and "{" in line:
                return False, "curly_bracket"
            if self.filter_policy and any(phrase in lower for phrase in POLICY_PHRASES):
                self.stat_update("lines_dropped_policy")
                continue
            if self.min_sentences is not None:
                sentences += count_sentences(line, self.language)
            kept.append(line)
            self.stat_update("lines_kept")
        if self.min_sentences is not None and sentences < self.min_sentences:
            return False, "too_few_sentences"
        document.text = "\n".join(kept).strip()
        return True
