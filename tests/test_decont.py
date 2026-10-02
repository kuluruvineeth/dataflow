import numpy as np

from dataflow.data import Document
from dataflow.pipeline.decont import (
    NgramDecontamCount,
    NgramDecontamFilter,
    NgramIndex,
    contaminating_ngrams,
    covered_fraction,
    document_frequencies,
    informative,
    ngram_hashes,
    text_ngrams,
    words_of,
)

QUESTION = "Which planet in our solar system has the largest number of confirmed moons as of this year"
BOILERPLATE = "click here to subscribe to our newsletter and never miss a single story from our team again"


def test_ngram_hashes_ignore_case_digits_and_punctuation():
    first = ngram_hashes(words_of("The capital of France is Paris, built in 1200."), 8)
    second = ngram_hashes(words_of("the CAPITAL of france is paris built in 3456"), 8)
    assert len(first) == 2
    assert np.array_equal(first, second)


def test_ngram_hashes_depend_on_word_order():
    assert not np.array_equal(ngram_hashes("a b c".split(), 3), ngram_hashes("c b a".split(), 3))
    assert len(ngram_hashes("too short".split(), 3)) == 0


def test_telugu_words_keep_their_vowel_signs():
    assert words_of("తెలుగు భాష, చాలా అందమైనది!") == ["తెలుగు", "భాష", "చాలా", "అందమైనది"]


def test_index_finds_which_items_an_ngram_came_from():
    items = ["one two three four five", "nothing in common here at all", "zero one two three four"]
    index = NgramIndex.build(items, 3)
    probe = ngram_hashes(words_of("one two three"), 3)
    assert index.contains(probe).tolist() == [True]
    assert sorted(index.items_of(int(probe[0])).tolist()) == [0, 2]
    assert index.items_with(probe).tolist() == [0, 2]
    assert not index.contains(ngram_hashes("five six seven".split(), 3)).any()


def test_index_round_trips_through_disk(tmp_path):
    index = NgramIndex.build(["a b c d", "b c d e"], 2)
    index.save(tmp_path / "index.npz")
    loaded = NgramIndex.load(tmp_path / "index.npz")
    assert loaded.n == 2
    assert np.array_equal(loaded.hashes, index.hashes)
    assert np.array_equal(loaded.items, index.items)


def documents() -> list[Document]:
    leaked = Document(f"Quiz night. {QUESTION}? Answer below.", "leaked")
    clean = Document("A long walk along the river at dawn, with coffee and no quiz at all in sight today.", "clean")
    footers = [Document(f"Story number {i}. {BOILERPLATE}.", f"footer-{i}") for i in range(11)]
    return [leaked, clean, *footers]


def test_counts_then_filter_drops_leaked_tests_but_not_boilerplate(tmp_path):
    NgramIndex.build([QUESTION, f"Write a footer. {BOILERPLATE}"], 13).save(tmp_path / "ngrams-13.npz")
    counter = NgramDecontamCount(tmp_path, tmp_path / "counts")
    list(counter.run(iter(documents())))
    assert counter.stats.metrics["with_matches"].total == 12
    frequencies = document_frequencies(tmp_path / "counts")
    assert sorted(set(frequencies.values())) == [1, 11]

    gpt3 = NgramDecontamFilter(tmp_path, tmp_path / "counts", max_documents=10)
    kept = [document.id for document in gpt3.run(iter(documents()))]
    assert "leaked" not in kept and "clean" in kept and len(kept) == 12
    assert gpt3.stats.metrics["dropped_contaminated"].total == 1

    strict = NgramDecontamFilter(tmp_path)
    assert [document.id for document in strict.run(iter(documents()))] == ["clean"]


def test_contaminating_ngrams_point_at_the_leaked_item(tmp_path):
    index = NgramIndex.build([QUESTION, f"Write a footer. {BOILERPLATE}"], 13)
    index.save(tmp_path / "ngrams-13.npz")
    list(NgramDecontamCount(tmp_path, tmp_path / "counts").run(iter(documents())))
    found = contaminating_ngrams(index, document_frequencies(tmp_path / "counts"), max_documents=10)
    assert index.items_with(found).tolist() == [0]


def test_ngrams_shared_by_many_eval_items_are_templates_not_leaks(tmp_path):
    template = "Answer with one of the following options and explain your reasoning in detail please"
    index = NgramIndex.build([QUESTION, *[f"Question {i}: pick a colour. {template}" for i in range(12)]], 13)
    index.save(tmp_path / "ngrams-13.npz")
    assert len(index.shared_ngrams(10)) > 0
    page = Document(f"Quiz rules. {template}.", "template")
    kept = NgramDecontamFilter(tmp_path, max_items=10).run(iter([page, Document(QUESTION, "leaked")]))
    assert [document.id for document in kept] == ["template"]


def test_number_runs_and_alphabets_are_not_informative():
    assert text_ngrams("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15", 13).size == 0
    assert text_ngrams("a b c d e f g h i j k l m n o p", 13).size == 0
    assert text_ngrams(QUESTION, 13).size == len(words_of(QUESTION)) - 12
    assert informative("the 1 cat 2 sat 3".split(), 4).tolist() == [True, True, True]


def test_covered_fraction_measures_how_much_of_an_item_the_ngrams_cover():
    prompt_ngrams = np.unique(ngram_hashes(words_of("one two three four five six seven eight nine"), 8))
    assert covered_fraction("one two three four five six seven eight nine ten", 8, prompt_ngrams) == 0.9
    assert covered_fraction("zero one two three", 8, prompt_ngrams) == 0.0
    assert covered_fraction("entirely different words here and there with no overlap", 8, prompt_ngrams) == 0.0
