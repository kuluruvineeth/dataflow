from dataflow.data import Document
from dataflow.pipeline.filters.base import BaseFilter, FilterResult
from dataflow.pipeline.writers.base import DiskWriter
from dataflow.utils.lid import FastTextLID, LanguageIdentifier


class LanguageFilter(BaseFilter):
    name = "language"

    def __init__(
        self,
        languages: list[str],
        threshold: float | dict[str, float] = 0.65,
        backend: str = "ft176",
        lid: LanguageIdentifier | None = None,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        self.languages = languages
        self.thresholds = threshold if isinstance(threshold, dict) else dict.fromkeys(languages, threshold)
        self.lid = lid or FastTextLID(backend)

    def filter(self, document: Document) -> FilterResult:
        scores = self.lid.predict(document.text)
        language, score = max(scores.items(), key=lambda item: item[1])
        document.metadata["language"] = language
        document.metadata["language_score"] = round(score, 4)
        if language not in self.languages:
            return False, "other_language"
        if score < self.thresholds[language]:
            return False, "low_confidence"
        return True
