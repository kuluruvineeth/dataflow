from collections.abc import Callable

from dataflow.pipeline.filters.base import BaseFilter
from dataflow.pipeline.filters.fineweb_quality import FineWebQualityFilter
from dataflow.pipeline.filters.gopher_quality import GopherQualityFilter
from dataflow.pipeline.filters.gopher_repetition import GopherRepetitionFilter
from dataflow.pipeline.writers.jsonl import JsonlWriter
from dataflow.utils.text import split_indic_words

TELUGU_STOP_WORDS = ["నుండి", "కి", "ఉన్నాయి", "మీ", "గ్రామం", "ఉంది", "దూరంలో", "ఈ", "కేంద్రం"]


def telugu_quality_filters(
    removed_folder: str | None = None, word_splitter: Callable[[str], list[str]] = split_indic_words
) -> list[BaseFilter]:
    def removed(name: str) -> JsonlWriter | None:
        return JsonlWriter(f"{removed_folder}/{name}") if removed_folder else None

    return [
        GopherRepetitionFilter(
            dup_line_frac=0.256,
            dup_para_frac=None,
            dup_line_char_frac=None,
            dup_para_char_frac=None,
            top_n_grams=((2, 0.21), (3, 0.18), (4, 0.162)),
            dup_n_grams=((5, 0.142), (6, 0.133), (7, 0.122), (8, 0.114), (9, 0.105), (10, 0.096)),
            word_splitter=word_splitter,
            exclusion_writer=removed("gopher_repetition"),
        ),
        FineWebQualityFilter(
            line_punct_thr=0.08,
            short_line_thr=None,
            char_duplicates_ratio=0.1,
            new_line_ratio=0.18,
            word_splitter=word_splitter,
            exclusion_writer=removed("fineweb_quality"),
        ),
        GopherQualityFilter(
            min_avg_word_length=4,
            max_avg_word_length=68,
            min_alpha_words_ratio=0.739,
            stop_words=TELUGU_STOP_WORDS,
            word_splitter=word_splitter,
            exclusion_writer=removed("gopher_quality"),
        ),
    ]
