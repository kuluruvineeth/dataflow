from itertools import islice
from pathlib import Path

from dataflow.data import DocumentsPipeline
from dataflow.pipeline.classifiers.base import ScoringStep

DCLM_OH_ELI5 = ("mlfoundations/fasttext-oh-eli5", "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin")


def flatten(text: str) -> str:
    """DCLM's preprocessing before predicting: all lines of the stripped text joined by single spaces."""
    return " ".join(text.strip().splitlines())


class FastTextClassifier(ScoringStep):
    """Scores every document with a fastText classifier and stores the probability of `label` in its metadata under
    `key`. Nothing is dropped: selection on the score happens later, so thresholds can change without rescoring.

    `model` is a local path or a (Hugging Face repo, file) pair such as `DCLM_OH_ELI5`.
    """

    type = "Classifier"
    name = "fasttext"

    def __init__(
        self,
        model: str | Path | tuple[str, str],
        label: str = "__label__hq",
        key: str = "fasttext_score",
        batch_size: int = 64,
    ):
        self.model_source = model
        self.label = label
        self.key = key
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fasttext.FastText import _FastText
            from huggingface_hub import hf_hub_download

            source = self.model_source
            path = hf_hub_download(*source) if isinstance(source, tuple) else Path(source)
            self._model = _FastText(str(path))
        return self._model

    def load(self):
        return self.model

    def score(self, text: str) -> float:
        labels, probabilities = self.model.predict(flatten(text), k=-1)
        return float(dict(zip(labels, probabilities, strict=True)).get(self.label, 0.0))

    def model_scores(self, texts: list[str]) -> list[float]:
        return [self.score(text) for text in texts]

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        documents = iter(data)
        while batch := list(islice(documents, self.batch_size)):
            with self.track_time():
                scores = self.scores([document.text for document in batch])
            for document, score in zip(batch, scores, strict=True):
                document.metadata[self.key] = round(score, 6)
                self.stat_update("scored")
                self.stat_update(self.key, value=document.metadata[self.key], unit="doc")
                yield document

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_model": None}
