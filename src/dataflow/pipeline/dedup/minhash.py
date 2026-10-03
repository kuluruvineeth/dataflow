import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from dataflow.data import DocumentsPipeline
from dataflow.io import DataFolderLike, get_datafolder
from dataflow.pipeline.base import PipelineStep
from dataflow.pipeline.dedup.exact import ExactDedupFilter
from dataflow.utils.binaryio import read_tuples
from dataflow.utils.hashing import hash64
from dataflow.utils.text import ngrams, simplify_text

PAIR = "<4I"
REMOVAL = "<I"

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


def band_format(config: MinhashConfig) -> str:
    return f"<{config.hashes_per_bucket}QI"


class MinhashDedupSignature(PipelineStep):
    type = "Dedup"
    name = "minhash signatures"

    def __init__(self, output_folder: DataFolderLike, config: MinhashConfig | None = None):
        self.output_folder = get_datafolder(output_folder)
        self.config = config or MinhashConfig()
        self.minhash = MinhashSignature(self.config)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        bands: list[list[tuple]] = [[] for _ in range(self.config.num_buckets)]
        for index, document in enumerate(data):
            self.stat_update("total")
            with self.track_time():
                signature = self.minhash.signature(document.text)
            if signature is None:
                self.stat_update("too_short")
                continue
            for bucket, band in enumerate(self.minhash.buckets(signature)):
                bands[bucket].append((*band, index))
        packer = struct.Struct(band_format(self.config))
        for bucket, records in enumerate(bands):
            records.sort()
            with self.output_folder.open(f"bucket_{bucket:03d}/{rank:05d}.sig", "wb") as file:
                file.write(b"".join(packer.pack(*record) for record in records))
        yield from ()


BAND_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


def band_keys(data: bytes, config: MinhashConfig) -> tuple[np.ndarray, np.ndarray]:
    """One 64-bit key per band (its hashes combined) and the document index, from a signature file's bytes."""
    records = np.frombuffer(data, dtype=np.dtype([("band", "<u8", (config.hashes_per_bucket,)), ("index", "<u4")]))
    keys = records["band"][:, 0].copy()
    for column in range(1, config.hashes_per_bucket):
        keys = keys * BAND_MULTIPLIER + records["band"][:, column]
    return keys, records["index"].copy()


class MinhashDedupBuckets(PipelineStep):
    """Pairs of documents that share a band, for one bucket.

    Every task's signatures for this bucket are read (in parallel), each band reduced to one 64-bit key, and the keys
    sorted together with their origin; equal neighbours are candidate duplicates. Plain arrays of 16 bytes per
    document, so a whole dump fits in memory, instead of one open stream per task.
    """

    type = "Dedup"
    name = "minhash buckets"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        config: MinhashConfig | None = None,
        threads: int = 16,
    ):
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.config = config or MinhashConfig()
        self.threads = threads

    def read(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self.input_folder.open(path, "rb") as file:
            keys, indexes = band_keys(file.read(), self.config)
        return keys, np.full(len(keys), source_rank(path), dtype=np.uint32), indexes

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if world_size != self.config.num_buckets:
            raise ValueError(f"run this step with tasks={self.config.num_buckets} (one per bucket)")
        paths = self.input_folder.list_files(subdirectory=f"bucket_{rank:03d}", glob_pattern="*.sig")
        with ThreadPoolExecutor(self.threads) as pool:
            parts = list(pool.map(self.read, paths))
        keys, origins, indexes = (
            np.concatenate([part[i] for part in parts]) if parts else np.empty(0) for i in range(3)
        )
        order = np.lexsort((indexes, origins, keys))
        keys, origins, indexes = keys[order], origins[order], indexes[order]
        same = np.nonzero(keys[1:] == keys[:-1])[0]
        pairs = np.empty(
            len(same), dtype=np.dtype([(name, "<u4") for name in ("rank_a", "index_a", "rank_b", "index_b")])
        )
        pairs["rank_a"], pairs["index_a"] = origins[same], indexes[same]
        pairs["rank_b"], pairs["index_b"] = origins[same + 1], indexes[same + 1]
        with self.output_folder.open(f"{rank:03d}.pairs", "wb") as out:
            out.write(pairs.tobytes())
        self.stat_update("signatures", value=len(keys))
        self.stat_update("pairs", value=len(pairs))
        yield from ()


def source_rank(path: str) -> int:
    return int(path.rsplit("/", 1)[-1].removesuffix(".sig"))


@dataclass
class UnionFind:
    parent: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        root = node
        while self.parent.get(root, root) != root:
            root = self.parent[root]
        while node != root:
            self.parent[node], node = root, self.parent.get(node, node)
        return root

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


class MinhashDedupCluster(PipelineStep):
    type = "Dedup"
    name = "minhash clusters"

    def __init__(self, input_folder: DataFolderLike, output_folder: DataFolderLike):
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if world_size != 1:
            raise ValueError("clustering needs every pair at once; run this step with tasks=1")
        clusters = UnionFind()
        for path in self.input_folder.list_files(glob_pattern="*.pairs"):
            with self.input_folder.open(path, "rb") as file:
                for rank_a, index_a, rank_b, index_b in read_tuples(file, PAIR):
                    clusters.union((rank_a, index_a), (rank_b, index_b))
        removals = sorted(clusters.parent)
        for _root in {clusters.find(node) for node in removals}:
            self.stat_update("clusters")
        packer = struct.Struct(REMOVAL)
        with self.output_folder.get_output_file_manager(mode="wb") as output:
            for origin, index in removals:
                output.get_file(f"{origin:05d}/clusters.dups").write(packer.pack(index))
                self.stat_update("to_remove")
        yield from ()


class MinhashDedupFilter(ExactDedupFilter):
    name = "minhash filter"
