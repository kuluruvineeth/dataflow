from dataclasses import dataclass

import numpy as np

from dataflow.utils.hashing import hash64
from dataflow.utils.text import ngrams, simplify_text

MERSENNE_PRIME = np.uint64((1 << 61) - 1)


@dataclass(frozen=True)
class MinhashConfig:
    n_grams: int = 5
    num_buckets: int = 14
    hashes_per_bucket: int = 8
    seed: int = 1

    @property
    def num_hashes(self) -> int:
        return self.num_buckets * self.hashes_per_bucket


def candidate_probability(similarity: float, num_buckets: int, hashes_per_bucket: int) -> float:
    return 1 - (1 - similarity**hashes_per_bucket) ** num_buckets


def similarity_threshold(num_buckets: int, hashes_per_bucket: int) -> float:
    return (1 / num_buckets) ** (1 / hashes_per_bucket)


class MinhashSignature:
    def __init__(self, config: MinhashConfig | None = None):
        self.config = config or MinhashConfig()
        rng = np.random.RandomState(self.config.seed)
        self.a = rng.randint(1, MERSENNE_PRIME, size=(1, self.config.num_hashes), dtype=np.uint64)
        self.b = rng.randint(0, MERSENNE_PRIME, size=(1, self.config.num_hashes), dtype=np.uint64)

    def shingles(self, text: str) -> np.ndarray:
        words = simplify_text(text).split()
        hashes = [hash64(" ".join(gram)) for gram in ngrams(words, self.config.n_grams)]
        return np.array(hashes, dtype=np.uint64).reshape(-1, 1)

    def signature(self, text: str) -> np.ndarray | None:
        shingles = self.shingles(text)
        if shingles.size == 0:
            return None
        with np.errstate(over="ignore"):
            permuted = (shingles * self.a + self.b) % MERSENNE_PRIME
        return permuted.min(axis=0)

    def buckets(self, signature: np.ndarray) -> list[tuple[int, ...]]:
        return [tuple(band.tolist()) for band in np.split(signature, self.config.num_buckets)]


def estimated_similarity(signature_a: np.ndarray, signature_b: np.ndarray) -> float:
    return float(np.mean(signature_a == signature_b))


def jaccard(set_a: set, set_b: set) -> float:
    return len(set_a & set_b) / len(set_a | set_b) if set_a or set_b else 1.0
