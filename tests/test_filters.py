import gzip

import orjson
import pytest

from dataflow.data import Document
from dataflow.pipeline.filters import GopherQualityFilter, LambdaFilter
from dataflow.pipeline.writers import JsonlWriter
from dataflow.utils.text import split_words

GOOD = (
    "The river runs through the old town, and the people who live there have built bridges with stone and wood. "
    "Every spring the water rises to the edge of the market, so the traders move their stalls to the hill. "
    "Children learn to swim in the shallow parts, and the fishermen teach them which currents to avoid. "
    "Visitors say that the town feels calm, even when the river is loud after a storm."
)


def run(step, docs):
    return list(step(iter(docs)))


def test_split_words_separates_punctuation():
    assert split_words("Hello, world... it's #1!") == ["Hello", ",", "world", ".", ".", ".", "it's", "#", "1", "!"]


def test_lambda_filter_keeps_only_matching_documents():
    docs = [Document("keep me", "1"), Document("drop me", "2")]
    assert [doc.id for doc in run(LambdaFilter(lambda doc: "keep" in doc.text), docs)] == ["1"]


def test_dropped_documents_go_to_exclusion_writer_with_reason(tmp_path):
    writer = JsonlWriter(tmp_path)
    step = LambdaFilter(lambda doc: (False, "too_short") if len(doc.text) < 5 else True, exclusion_writer=writer)
    kept = run(step, [Document("long enough", "1"), Document("tiny", "2")])
    assert [doc.id for doc in kept] == ["1"]
    dropped = [orjson.loads(line) for line in gzip.open(tmp_path / "00000.jsonl.gz")]
    assert dropped == [{"text": "tiny", "id": "2", "metadata": {"filter_reason": "too_short"}}]


def test_gopher_keeps_natural_prose():
    assert GopherQualityFilter().filter(Document(GOOD, "1")) is True


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("the and of to " * 5, "gopher_short_doc"),
        (GOOD + " supercalifragilisticexpialidocious" * 200, "gopher_above_avg_threshold"),
        (GOOD + " #tag" * 30, "gopher_too_many_hashes"),
        ("\n".join(f"- {line}" for line in GOOD.split(". ")), "gopher_too_many_bullets"),
        (GOOD + " 1234 5678" * 100, "gopher_below_alpha_threshold"),
        ("Rivers mountains forests valleys deserts oceans islands plains. " * 10, "gopher_not_enough_stop_words"),
    ],
)
def test_gopher_rules(text, reason):
    assert GopherQualityFilter().filter(Document(text, "1")) == (False, reason)
