import pytest

from dataflow.data import Document
from dataflow.pipeline.filters import C4QualityFilter

GOOD = [
    "The river runs through the old town and past the mill.",
    "Children play on its banks every summer afternoon.",
    "Fishermen sit there at dawn with their long rods.",
    "In winter the water freezes near the stone bridge.",
    "Visitors often stop to take photographs of the view.",
]


def keep(text, **kwargs):
    document = Document(text, "0")
    return C4QualityFilter(**kwargs).filter(document), document.text


def test_a_clean_page_is_kept_unchanged():
    assert keep("\n".join(GOOD)) == (True, "\n".join(GOOD))


@pytest.mark.parametrize(
    ("line", "label"),
    [
        ("Menu Home About", "lines_dropped_no_terminal_punct"),
        ("Read more...", "lines_dropped_no_terminal_punct"),
        ("Hi there.", "lines_dropped_too_few_words"),
        ("Please enable JavaScript to view this page.", "lines_dropped_javascript"),
        ("By continuing you accept our privacy policy.", "lines_dropped_policy"),
        ("x" * 1001 + " is a very long word.", "lines_dropped_too_long_word"),
    ],
)
def test_bad_lines_are_removed_and_counted(line, label):
    step = C4QualityFilter()
    document = Document("\n".join([line, *GOOD]), "0")
    assert step.filter(document) is True
    assert document.text == "\n".join(GOOD)
    assert step.stats.metrics[label].total == 1


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ("Lorem ipsum dolor sit amet, consectetur.", "lorem_ipsum"),
        ("Use the { brace } in code samples here.", "curly_bracket"),
    ],
)
def test_some_lines_drop_the_whole_page(extra, reason):
    assert keep("\n".join([*GOOD, extra]))[0] == (False, reason)


def test_pages_with_too_few_sentences_are_dropped():
    assert keep("\n".join(GOOD[:4]))[0] == (False, "too_few_sentences")
    assert keep("\n".join(GOOD[:4]), min_sentences=None)[0] is True


def test_citations_are_removed_before_the_punctuation_check():
    text = "\n".join([*GOOD, "The bridge was built in 1820.[3]"])
    assert keep(text)[1].endswith("The bridge was built in 1820.")


def test_fineweb_setting_keeps_lines_without_terminal_punctuation():
    text = "\n".join(["Menu Home About Contact", *GOOD])
    assert keep(text, filter_no_terminal_punct=False) == (True, text)
