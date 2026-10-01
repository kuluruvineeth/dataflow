import pytest

from dataflow.data import Document
from dataflow.pipeline.base import PipelineStep


class CountingStep(PipelineStep):
    def __init__(self):
        self.seen = 0

    def run(self, data, rank=0, world_size=1):
        for doc in data:
            self.seen += 1
            yield doc


def test_steps_are_lazy_until_drained():
    step = CountingStep()
    out = step(iter([Document(text="a", id="0"), Document(text="b", id="1")]))
    assert step.seen == 0
    assert [doc.id for doc in out] == ["0", "1"]
    assert step.seen == 2


def test_document_rejects_misspelled_fields():
    doc = Document(text="a", id="0")
    with pytest.raises(AttributeError):
        doc.metdata = {}
