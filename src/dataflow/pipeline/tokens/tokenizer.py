import json
from itertools import batched
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from dataflow.data import DocumentsPipeline
from dataflow.io import DataFolder, DataFolderLike, get_datafolder
from dataflow.pipeline.base import PipelineStep


def load_tokenizer(name_or_path: str) -> Tokenizer:
    if Path(name_or_path).is_file():
        return Tokenizer.from_file(name_or_path)
    return Tokenizer.from_pretrained(name_or_path)


class DocumentTokenizer(PipelineStep):
    type = "Tokenizer"
    name = "tokenize"

    def __init__(
        self,
        output_folder: DataFolderLike,
        tokenizer_name_or_path: str,
        eos_token: str = "<|endoftext|>",
        batch_size: int = 512,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.output_folder = get_datafolder(output_folder)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.eos_token = eos_token
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._tokenizer = None

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.tokenizer_name_or_path)
        return self._tokenizer

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_tokenizer": None}

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        eos_id = self.tokenizer.token_to_id(self.eos_token)
        if eos_id is None:
            raise ValueError(f"tokenizer has no {self.eos_token!r} token")
        dtype = np.dtype("<u2") if self.tokenizer.get_vocab_size() <= 2**16 else np.dtype("<u4")
        unshuffled = f"{rank:05d}_unshuffled.ds"
        doc_ends = []
        total = 0
        with self.output_folder.open(unshuffled, "wb") as file:
            for batch in batched(data, self.batch_size):
                with self.track_time():
                    encodings = self.tokenizer.encode_batch([doc.text for doc in batch], add_special_tokens=False)
                for encoding in encodings:
                    tokens = np.array([*encoding.ids, eos_id], dtype=dtype)
                    file.write(tokens.tobytes())
                    total += len(tokens)
                    doc_ends.append(total)
                    self.stat_update("tokens", value=len(tokens))
        order = np.random.default_rng(self.seed + rank).permutation(len(doc_ends)) if self.shuffle else None
        self.write_final(unshuffled, f"{rank:05d}.ds", doc_ends, order, dtype)
        yield from ()

    def write_final(
        self, source: str, target: str, doc_ends: list[int], order: np.ndarray | None, dtype: np.dtype
    ) -> None:
        starts = [0, *doc_ends[:-1]]
        new_ends = []
        total = 0
        with self.output_folder.open(source, "rb") as src, self.output_folder.open(target, "wb") as out:
            for doc in order if order is not None else range(len(doc_ends)):
                src.seek(starts[doc] * dtype.itemsize)
                length = doc_ends[doc] - starts[doc]
                out.write(src.read(length * dtype.itemsize))
                total += length
                new_ends.append(total)
        self.output_folder.rm(source)
        meta = {
            "tokenizer": self.tokenizer_name_or_path,
            "token_bytes": dtype.itemsize,
            "eos_token_id": self.tokenizer.token_to_id(self.eos_token),
        }
        write_index_and_meta(self.output_folder, target, new_ends, meta)


def write_index_and_meta(folder: DataFolder, filename: str, ends: list[int], meta: dict) -> None:
    with folder.open(f"{filename}.index", "wb") as index:
        index.write(np.array(ends, dtype="<u8").tobytes())
    with folder.open(f"{filename}.meta", "w") as file:
        json.dump({**meta, "documents": len(ends), "tokens": int(ends[-1]) if len(ends) else 0}, file)


def read_meta(folder: DataFolder, filename: str) -> dict:
    with folder.open(f"{filename}.meta", "r") as file:
        return json.load(file)


def read_ends(folder: DataFolder, filename: str) -> np.ndarray:
    with folder.open(f"{filename}.index", "rb") as file:
        return np.frombuffer(file.read(), dtype="<u8").astype(np.int64)


def read_tokenized(folder: DataFolderLike, filename: str) -> list[np.ndarray]:
    folder: DataFolder = get_datafolder(folder)
    dtype = np.dtype(f"<u{read_meta(folder, filename)['token_bytes']}")
    with folder.open(filename, "rb") as file:
        tokens = np.frombuffer(file.read(), dtype=dtype)
    ends = read_ends(folder, filename)
    if not len(ends):
        return []
    return np.split(tokens, ends[:-1])
