import re
import string

PUNCTUATION_SET = set(string.punctuation) | set("…—–“”‘’«»„、。，．！？：；（）【】《》「」")

WORD_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")


def split_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)
