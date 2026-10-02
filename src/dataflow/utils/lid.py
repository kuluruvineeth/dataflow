from typing import Protocol

from dataflow.utils.cache import cached_download

MODELS = {
    "ft176": ("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", "lid/lid.176.bin"),
    "glotlid": ("https://huggingface.co/cis-lmu/glotlid/resolve/main/model_v3.bin", "lid/glotlid_v3.bin"),
}


class LanguageIdentifier(Protocol):
    def predict(self, text: str) -> dict[str, float]: ...


class FastTextLID:
    def __init__(self, backend: str = "ft176", top_k: int = 5):
        if backend not in MODELS:
            raise ValueError(f"unknown backend {backend!r}, expected one of {sorted(MODELS)}")
        self.backend = backend
        self.top_k = top_k
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fasttext.FastText import _FastText

            self._model = _FastText(str(cached_download(*MODELS[self.backend])))
        return self._model

    def predict(self, text: str) -> dict[str, float]:
        labels, scores = self.model.predict(text.replace("\n", " "), k=self.top_k)
        return {label.removeprefix("__label__"): float(score) for label, score in zip(labels, scores, strict=True)}

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_model": None}
