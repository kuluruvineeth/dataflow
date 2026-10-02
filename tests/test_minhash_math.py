import pytest

from dataflow.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashSignature,
    candidate_probability,
    estimated_similarity,
    jaccard,
    similarity_threshold,
)
from dataflow.utils.text import ngrams, simplify_text, split_words

BASE = (
    "the river runs through the old town and the people who live there have built bridges with stone and wood "
    "every spring the water rises to the edge of the market so the traders move their stalls to the hill "
    "children learn to swim in the shallow parts and the fishermen teach them which currents to avoid "
    "visitors say that the town feels calm even when the river is loud after a storm"
)


def test_split_words_keeps_telugu_words_whole():
    assert split_words("తెలుగు భాష, చాలా అందమైనది!") == ["తెలుగు", "భాష", ",", "చాలా", "అందమైనది", "!"]


def test_simplify_text_normalises_case_numbers_and_punctuation():
    assert simplify_text("Hello, World!  Price: 1,299.50 in 2024.") == "hello world price 0 in 0"
    assert simplify_text("తెలుగు, భాష!") == "తెలుగు భాష"


def test_ngrams_slide_over_words():
    assert list(ngrams(["a", "b", "c", "d"], 3)) == [("a", "b", "c"), ("b", "c", "d")]


def test_identical_text_gives_identical_signatures_across_instances():
    sig_a = MinhashSignature().signature(BASE)
    sig_b = MinhashSignature().signature(BASE.upper() + "!!!")
    assert sig_a.shape == (112,)
    assert estimated_similarity(sig_a, sig_b) == 1.0


def test_too_short_text_has_no_signature():
    assert MinhashSignature().signature("only four words here") is None


@pytest.mark.parametrize(
    "variant",
    [
        BASE.replace("old town", "new city"),
        BASE.replace("storm", "long and heavy storm in the night"),
        " ".join(BASE.split()[: len(BASE.split()) // 2]),
        "completely different text about mountains and snow and climbing in the cold winter months of the year",
    ],
)
def test_signature_agreement_estimates_jaccard(variant):
    minhash = MinhashSignature()
    true = jaccard(set(minhash.shingles(BASE).ravel()), set(minhash.shingles(variant).ravel()))
    estimate = estimated_similarity(minhash.signature(BASE), minhash.signature(variant))
    assert estimate == pytest.approx(true, abs=0.15)


def test_buckets_split_the_signature_into_bands():
    minhash = MinhashSignature()
    bands = minhash.buckets(minhash.signature(BASE))
    assert len(bands) == 14
    assert all(len(band) == 8 for band in bands)


def test_band_probability_matches_known_values():
    config = MinhashConfig()
    assert similarity_threshold(config.num_buckets, config.hashes_per_bucket) == pytest.approx(0.719, abs=0.001)
    assert candidate_probability(0.8, 14, 8) == pytest.approx(0.924, abs=0.001)
    assert candidate_probability(0.5, 14, 8) < 0.06
