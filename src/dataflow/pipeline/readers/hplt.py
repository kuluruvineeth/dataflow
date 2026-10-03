import io
import json
from collections.abc import Iterator
from contextlib import ExitStack
from itertools import repeat

import httpx
import zstandard

from dataflow.data import Document, DocumentsPipeline
from dataflow.io import get_datafolder
from dataflow.pipeline.base import PipelineStep

HPLT_POOL = "https://data.hplt-project.org/four/pool/"


class ResponseStream(io.RawIOBase):
    """A streamed HTTP body as a file, so a decompressor can read it as it arrives."""

    def __init__(self, response: httpx.Response):
        self.chunks = response.iter_raw(1 << 20)
        self.pending = b""

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while not self.pending:
            self.pending = next(self.chunks, b"")
            if not self.pending:
                return 0
        size = min(len(buffer), len(self.pending))
        buffer[:size], self.pending = self.pending[:size], self.pending[size:]
        return size


class HpltPoolReader(PipelineStep):
    """Documents of one language from batches of the HPLT 4.0 pool.

    A batch is `metadata.zst` and `text.zst`, line-aligned JSON. Both are streamed and decompressed on the fly; a line
    is kept when its language ID's best guess is `language` with at least `min_probability`, and robots.txt allowed
    it. With `read_text=False` only the metadata is streamed: enough to count, at a tenth of the bytes.
    """

    type = "Reader"
    name = "hplt pool"

    def __init__(
        self,
        batches: list[str],
        language: str = "tel_Telu",
        identifier: str = "openlid-v3",
        min_probability: float = 0.5,
        read_text: bool = True,
        base: str = HPLT_POOL,
    ):
        self.batches = batches
        self.language = language
        self.identifier = identifier
        self.min_probability = min_probability
        self.read_text = read_text
        self.base = base

    def lines(self, stack: ExitStack, path: str) -> Iterator[str]:
        if self.base.startswith("https://"):
            client = stack.enter_context(httpx.Client(timeout=300, follow_redirects=True))
            response = stack.enter_context(client.stream("GET", self.base + path))
            response.raise_for_status()
            raw = ResponseStream(response)
        else:
            raw = stack.enter_context(get_datafolder(self.base).open(path, "rb"))
        reader = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
        return io.TextIOWrapper(io.BufferedReader(reader), encoding="utf-8")

    def matches(self, metadata: dict) -> bool:
        guess = metadata.get(self.identifier) or {}
        languages, probabilities = guess.get("lang") or [None], guess.get("prob") or [0]
        return languages[0] == self.language and probabilities[0] >= self.min_probability

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for batch in self.batches[rank::world_size]:
            with ExitStack() as stack:
                metadata_lines = self.lines(stack, f"{batch}/metadata.zst")
                if self.read_text:
                    pairs = zip(metadata_lines, self.lines(stack, f"{batch}/text.zst"), strict=True)
                else:
                    pairs = zip(metadata_lines, repeat(None))
                for number, (metadata_line, text_line) in enumerate(pairs):
                    self.stat_update("documents")
                    if self.language not in metadata_line:
                        continue
                    metadata = json.loads(metadata_line)
                    if not self.matches(metadata):
                        continue
                    self.stat_update("matched")
                    if not metadata.get("allowed", True):
                        self.stat_update("disallowed")
                        continue
                    self.stat_update("kept")
                    if text_line is None:
                        continue
                    text = json.loads(text_line)["text"]
                    self.stat_update("text_bytes", value=len(text.encode("utf-8")), unit="byte")
                    yield Document(
                        text=text,
                        id=metadata.get("id") or f"{batch}/{number}",
                        metadata={
                            "url": metadata.get("u"),
                            "date": metadata.get("ts"),
                            "batch": batch,
                            "language_score": metadata[self.identifier]["prob"][0],
                        },
                    )
            self.stat_update("batches")
