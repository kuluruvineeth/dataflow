from contextlib import ExitStack

import numpy as np

from dataflow.data import DocumentsPipeline
from dataflow.io import DataFolderLike, get_datafolder
from dataflow.pipeline.base import PipelineStep
from dataflow.pipeline.tokens.tokenizer import read_ends, read_meta, write_index_and_meta

SHARED_META = ("tokenizer", "token_bytes", "eos_token_id")


def split_into_files(lengths: list[int], max_tokens: int) -> list[list[int]]:
    files, current, total = [], [], 0
    for position, length in enumerate(lengths):
        if current and total + length > max_tokens:
            files.append(current)
            current, total = [], 0
        current.append(position)
        total += length
    if current:
        files.append(current)
    return files


class TokenMerger(PipelineStep):
    type = "Tokenizer"
    name = "merge tokens"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        name: str = "merged",
        max_tokens_per_file: int = 1_000_000_000,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.name = name
        self.max_tokens_per_file = max_tokens_per_file
        self.shuffle = shuffle
        self.seed = seed

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if world_size != 1:
            raise ValueError("merging needs every file at once; run this step with tasks=1")
        inputs = self.input_folder.list_files(glob_pattern="*.ds")
        if not inputs:
            raise RuntimeError(f"no .ds files in {self.input_folder.path}")
        metas = [read_meta(self.input_folder, path) for path in inputs]
        shared = {tuple(meta.get(key) for key in SHARED_META) for meta in metas}
        if len(shared) != 1:
            raise ValueError(f"input files come from different tokenizers: {sorted(map(str, shared))}")
        meta = {key: metas[0].get(key) for key in SHARED_META}
        token_bytes = meta["token_bytes"]

        lengths = [np.diff(read_ends(self.input_folder, path), prepend=0) for path in inputs]
        sources = np.repeat(np.arange(len(inputs)), [len(file_lengths) for file_lengths in lengths])
        if self.shuffle:
            sources = np.random.default_rng(self.seed).permutation(sources)
        next_document = [0] * len(inputs)
        order = []
        for source in sources:
            order.append((source, int(lengths[source][next_document[source]])))
            next_document[source] += 1

        with ExitStack() as stack:
            readers = [stack.enter_context(self.input_folder.open(path, "rb")) for path in inputs]
            for part, positions in enumerate(
                split_into_files([length for _, length in order], self.max_tokens_per_file)
            ):
                filename = f"{self.name}_{part:03d}.ds"
                with self.output_folder.open(filename, "wb") as out:
                    for source, length in (order[position] for position in positions):
                        out.write(readers[source].read(length * token_bytes))
                        self.stat_update("documents")
                        self.stat_update("tokens", value=length)
                ends = np.cumsum([order[position][1] for position in positions]).tolist()
                write_index_and_meta(self.output_folder, filename, ends, meta)
                self.stat_update("files")
        yield from ()
