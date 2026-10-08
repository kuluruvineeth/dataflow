import json
import pickle

import numpy as np
import pytest

from dataflow.data import Document
from dataflow.pipeline.classifiers import FastTextClassifier
from dataflow.pipeline.classifiers.fasttext import flatten
from dataflow.recipes.dclm_classifier import NEGATIVE, POSITIVE, eli5_texts, openhermes_texts, train, training_lines
from dataflow.utils.agreement import agreement, ranks, spread

GOOD = [
    "Plants make sugar from light because chlorophyll absorbs red and blue light and passes the energy on.",
    "The bridge stays up because the arch pushes the load sideways into the ground on both banks.",
    "Water boils at a lower temperature on a mountain because the air pressure there is lower.",
]
SPAM = [
    "click here buy now best price free shipping limited offer click here",
    "casino bonus win big today free spins sign up now best casino bonus",
    "cheap deals cheap deals buy now limited offer best price click",
]


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    folder = tmp_path_factory.mktemp("fasttext")
    lines = training_lines(GOOD * 30, SPAM * 30)
    (folder / "train.txt").write_text("\n".join(lines) + "\n")
    return train(folder / "train.txt", folder / "model.bin", epoch=25)


def test_flatten_joins_lines_like_dclm():
    assert flatten("  first line\nsecond line\n\nthird  \n") == "first line second line  third"


def test_scores_go_into_metadata_and_nothing_is_dropped(model_path):
    step = FastTextClassifier(model_path, key="dclm")
    documents = [Document(GOOD[0], "good"), Document(SPAM[0], "spam")]
    scored = {document.id: document.metadata["dclm"] for document in step.run(documents)}
    assert set(scored) == {"good", "spam"}
    assert scored["good"] > 0.5 > scored["spam"]
    assert step.stats.metrics["scored"].total == 2


def test_batches_give_the_same_scores_as_single_texts(model_path):
    step = FastTextClassifier(model_path, key="dclm", batch_size=2)
    texts = GOOD + SPAM
    assert step.scores(texts) == [step.score(text) for text in texts]
    documents = [Document(text, str(i)) for i, text in enumerate(texts)]
    assert [document.metadata["dclm"] for document in step.run(documents)] == [round(step.score(t), 6) for t in texts]


def test_a_step_with_a_scorer_does_not_load_its_model():
    step = FastTextClassifier("/no/such/model.bin", key="dclm")
    step.scorer = lambda texts: [0.25] * len(texts)
    scored = [document.metadata["dclm"] for document in step.run([Document(GOOD[0], "a"), Document(SPAM[0], "b")])]
    assert scored == [0.25, 0.25]
    assert step._model is None


def test_the_loaded_model_is_not_pickled(model_path):
    step = FastTextClassifier(model_path)
    step.score(GOOD[1])
    clone = pickle.loads(pickle.dumps(step))
    assert clone._model is None
    assert clone.score(GOOD[1]) == pytest.approx(step.score(GOOD[1]))


def eli5_row(q_id, scores, subreddit="explainlikeimfive", texts=None):
    texts = texts or [f"answer {i}" for i in range(len(scores))]
    return {"q_id": q_id, "title": f"why {q_id}", "selftext": "", "subreddit": subreddit,
            "answers": {"score": scores, "text": texts}}  # fmt: skip


def test_eli5_keeps_dclm_filters_and_the_longest_of_tied_best_answers():
    rows = [
        eli5_row("kept", [9, 2, 1]),
        eli5_row("few_answers", [9, 8]),
        eli5_row("low_best", [4, 3, 2]),
        eli5_row("other_sub", [9, 8, 7], subreddit="askscience"),
        eli5_row("kept", [9, 2, 1]),
        eli5_row("tie", [6, 6, 1], texts=["short", "the longer answer", "x"]),
    ]
    assert eli5_texts(rows) == ["why kept answer 0", "why tie the longer answer"]


def test_openhermes_conversations_become_one_text(tmp_path):
    path = tmp_path / "oh.json"
    conversation = [{"from": "human", "value": "Q?"}, {"from": "gpt", "value": "A."}]
    path.write_text(json.dumps([{"conversations": conversation}]))
    assert openhermes_texts(str(path)) == ["Q? A."]


def test_training_lines_are_labelled_flat_and_shuffled():
    lines = training_lines(["a\nb"], ["c"], seed=1)
    assert sorted(lines) == sorted([f"{POSITIVE} a b", f"{NEGATIVE} c"])


def test_spread_samples_across_the_whole_run():
    assert spread([f"{i:05d}.jsonl.gz" for i in range(100)], 4) == [
        "00000.jsonl.gz", "00025.jsonl.gz", "00050.jsonl.gz", "00075.jsonl.gz"
    ]  # fmt: skip


def test_ranks_share_ties():
    assert ranks(np.array([0.3, 0.1, 0.3, 0.9])).tolist() == [2.5, 1.0, 2.5, 4.0]


def test_agreement_is_perfect_for_a_monotone_transform_and_reversed_for_its_opposite():
    scores = np.linspace(0, 1, 50)
    same = agreement(scores**3, scores)
    assert same["spearman"] == pytest.approx(1.0) and same["top_0.1_overlap"] == 1.0
    opposite = agreement(1 - scores, scores)
    assert opposite["spearman"] == pytest.approx(-1.0) and opposite["top_0.1_overlap"] == 0.0
