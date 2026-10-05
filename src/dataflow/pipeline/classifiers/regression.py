from itertools import islice

from dataflow.data import DocumentsPipeline
from dataflow.pipeline.base import PipelineStep

FINEWEB_EDU = "HuggingFaceFW/fineweb-edu-classifier"


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class RegressionClassifier(PipelineStep):
    """Scores documents in batches with a sequence-classification model that has one regression output, such as
    FineWeb-Edu's educational-value classifier or one trained the same way. Stores the score under `key` and, as
    FineWeb-Edu publishes it, the score rounded and clipped to 0-5 under `{key}_int`. Nothing is dropped.

    Texts are cut to the model's first `max_length` tokens. Needs the `classifiers` extra (torch, transformers).
    """

    type = "Classifier"
    name = "regression"

    def __init__(self, model: str = FINEWEB_EDU, key: str = "edu_score", batch_size: int = 64, max_length: int = 512):
        self.model_name = model
        self.key = key
        self.batch_size = batch_size
        self.max_length = max_length
        self._loaded = None

    def load(self):
        if self._loaded is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = best_device()
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, dtype=dtype).to(device).eval()
            self._loaded = (AutoTokenizer.from_pretrained(self.model_name), model, device)
        return self._loaded

    def scores(self, texts: list[str]) -> list[float]:
        import torch

        tokenizer, model, device = self.load()
        inputs = tokenizer(texts, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_length)
        with torch.no_grad():
            logits = model(**inputs.to(device)).logits.squeeze(-1).float().cpu()
        return logits.tolist()

    def scores_all(self, texts: list[str]) -> list[float]:
        return [
            s
            for start in range(0, len(texts), self.batch_size)
            for s in self.scores(texts[start : start + self.batch_size])
        ]

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        documents = iter(data)
        while batch := list(islice(documents, self.batch_size)):
            with self.track_time():
                scores = self.scores([document.text for document in batch])
            for document, score in zip(batch, scores, strict=True):
                document.metadata[self.key] = round(score, 4)
                document.metadata[f"{self.key}_int"] = int(round(max(0.0, min(score, 5.0))))
                self.stat_update("scored")
                yield document

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_loaded": None}
