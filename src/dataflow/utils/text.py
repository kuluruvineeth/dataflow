import string
from collections.abc import Iterator, Sequence
from functools import cache

import regex

PUNCTUATION_SET = set(string.punctuation) | set("…—–“”‘’«»„、。，．！？：；（）【】《》「」")

WORD_PATTERN = regex.compile(r"[\w\p{M}]+(?:['’][\w\p{M}]+)*|[^\w\p{M}\s]")
NUMBER_PATTERN = regex.compile(r"\p{Nd}+(?:[.,]\p{Nd}+)*")
PUNCTUATION_PATTERN = regex.compile(r"\p{P}+")
WHITESPACE_PATTERN = regex.compile(r"\s+")
TERMINAL_PUNCTUATION_PATTERN = regex.compile(r"\p{Sentence_Terminal}$")
INDIC_PUNCTUATION_PATTERN = regex.compile("([" + regex.escape(string.punctuation) + "\u0964\u0965])")
SPACES_PATTERN = regex.compile(r"[ \t]+")
NUMBER_SEQUENCE_PATTERN = regex.compile(r"(?:[0-9]+ [,.:/] )+[0-9]+")


def split_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)


def split_indic_words(text: str) -> list[str]:
    spaced = SPACES_PATTERN.sub(" ", INDIC_PUNCTUATION_PATTERN.sub(r" \1 ", text))
    spaced = NUMBER_SEQUENCE_PATTERN.sub(lambda match: match.group().replace(" ", ""), spaced)
    # spaces only, not newlines: the Telugu filter thresholds were tuned on exactly this splitting
    return [word.strip() for word in spaced.split(" ") if word.strip()]


def ends_sentence(line: str) -> bool:
    return TERMINAL_PUNCTUATION_PATTERN.search(line.rstrip()) is not None


def simplify_text(text: str) -> str:
    text = NUMBER_PATTERN.sub("0", text.lower())
    text = PUNCTUATION_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def ngrams(words: Sequence[str], n: int) -> Iterator[tuple[str, ...]]:
    return zip(*(words[i:] for i in range(n)), strict=False)


@cache
def sentence_splitter(language: str = "en"):
    import spacy

    nlp = spacy.blank(language)
    nlp.add_pipe("sentencizer")
    return nlp


def count_sentences(text: str, language: str = "en") -> int:
    nlp = sentence_splitter(language)
    nlp.max_length = len(text) + 10
    with nlp.memory_zone():
        try:
            return sum(1 for _ in nlp(text).sents)
        except Exception:
            # spaCy fails on text containing its own attribute name "IS_ALPHA"
            if "IS_ALPHA" not in text:
                raise
            return sum(1 for _ in nlp(text.replace("IS_ALPHA", "")).sents)
