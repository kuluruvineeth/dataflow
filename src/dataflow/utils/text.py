import string
from collections.abc import Iterator, Sequence

import regex

PUNCTUATION_SET = set(string.punctuation) | set("…—–“”‘’«»„、。，．！？：；（）【】《》「」")

WORD_PATTERN = regex.compile(r"[\w\p{M}]+(?:['’][\w\p{M}]+)*|[^\w\p{M}\s]")
NUMBER_PATTERN = regex.compile(r"\p{Nd}+(?:[.,]\p{Nd}+)*")
PUNCTUATION_PATTERN = regex.compile(r"\p{P}+")
WHITESPACE_PATTERN = regex.compile(r"\s+")


def split_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)


def simplify_text(text: str) -> str:
    text = NUMBER_PATTERN.sub("0", text.lower())
    text = PUNCTUATION_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def ngrams(words: Sequence[str], n: int) -> Iterator[tuple[str, ...]]:
    return zip(*(words[i:] for i in range(n)), strict=False)
