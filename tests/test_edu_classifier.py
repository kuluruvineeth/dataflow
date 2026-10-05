import json

import numpy as np
import pytest

from dataflow.data import Document
from dataflow.pipeline.classifiers import RegressionClassifier
from dataflow.recipes.edu_classifier import (
    as_classes,
    binary_f1,
    cls_states,
    macro_f1,
    read_labels,
    stratified_split,
    train_head,
)

WORDS = ["plants", "light", "energy", "water", "boils", "click", "buy", "now", "cheap", "deals"]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A one-layer BERT with a regression head and a ten-word vocabulary, saved like a Hub model."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    folder = tmp_path_factory.mktemp("tiny-bert")
    (folder / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(folder / "vocab.txt"))
    config = BertConfig(vocab_size=len(WORDS) + 5, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, num_labels=1, problem_type="regression")  # fmt: skip
    BertForSequenceClassification(config).save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return str(folder)


def test_scores_and_integer_scores_go_into_metadata_in_batches(tiny_model):
    step = RegressionClassifier(tiny_model, key="edu", batch_size=2)
    documents = [Document(text, str(i)) for i, text in enumerate(["plants light", "buy now", "water boils"])]
    scored = list(step.run(documents))
    assert [d.id for d in scored] == ["0", "1", "2"]
    for document in scored:
        assert document.metadata["edu_int"] == int(round(max(0.0, min(document.metadata["edu"], 5.0))))
    assert step.stats.metrics["scored"].total == 3


def test_scores_all_matches_one_batch(tiny_model):
    step = RegressionClassifier(tiny_model, batch_size=2)
    texts = ["plants light energy", "cheap deals", "buy now", "water"]
    one_batch = RegressionClassifier(tiny_model, batch_size=8).scores(texts)
    # padding to a different length moves GPU arithmetic in the last digits, nothing more
    assert step.scores_all(texts) == pytest.approx(one_batch, abs=1e-4)


def test_classes_round_and_clip():
    assert as_classes(np.array([-0.7, 0.5, 2.49, 2.51, 7.0])).tolist() == [0, 0, 2, 3, 5]


def test_macro_f1_averages_over_classes_present():
    predicted, actual = np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2])
    # class 0: 1.0; class 1: 2*1/(2+1); class 2: 2*1/(1+2)
    assert macro_f1(predicted, actual) == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)


def test_binary_f1_at_the_fineweb_edu_cut():
    assert binary_f1(np.array([3.4, 2.4, 4.0, 1.0]), np.array([3, 3, 4, 0])) == pytest.approx(2 * 2 / (2 + 3))


def test_stratified_split_keeps_each_score_share_and_is_deterministic():
    scores = np.array([0] * 50 + [1] * 30 + [3] * 20)
    train, test = stratified_split(scores)
    assert len(test) == 10 and sorted(scores[test].tolist()) == [0] * 5 + [1] * 3 + [3] * 2
    assert set(train) | set(test) == set(range(100)) and not set(train) & set(test)
    assert (stratified_split(scores)[1] == test).all()


def test_read_labels_keeps_missing_reference_scores_as_minus_one(tmp_path):
    path = tmp_path / "labels.jsonl"
    rows = [{"text": "a", "score": 2, "reference_score": 3}, {"text": "b", "score": 0, "reference_score": None}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    texts, scores, reference = read_labels(path)
    assert texts == ["a", "b"] and scores.tolist() == [2, 0] and reference.tolist() == [3, -1]


def test_cls_states_have_one_row_per_text(tiny_model):
    assert cls_states(["plants light", "buy now", "water"], tiny_model, batch_size=2).shape == (3, 16)


def test_the_head_learns_a_score_that_is_a_function_of_the_state(tiny_model):
    rng = np.random.default_rng(0)
    states = rng.normal(size=(600, 16)).astype(np.float32)
    scores = np.clip(np.round(2.5 + 1.5 * states[:, 0]), 0, 5).astype(int)
    model, history = train_head(states, scores, stratified_split(scores), tiny_model, epochs=30, learning_rate=1e-2,
                                batch_size=64)  # fmt: skip
    assert len(history) == 30 and max(history) > history[0] and max(history) > 0.6
    assert model.classifier.weight.shape == (1, 16)
