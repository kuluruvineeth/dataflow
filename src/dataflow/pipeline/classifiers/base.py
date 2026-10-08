from abc import abstractmethod
from collections.abc import Callable

from dataflow.pipeline.base import PipelineStep


class ScoringStep(PipelineStep):
    """A step that scores texts with a model. `scores` uses the step's own model, or `scorer` when an executor sets
    one, so that many ranks can share one loaded model (`RayPipelineExecutor` puts it in one actor)."""

    scorer: Callable[[list[str]], list[float]] | None = None

    @abstractmethod
    def load(self):
        """Loads the model, once, and returns it."""

    @abstractmethod
    def model_scores(self, texts: list[str]) -> list[float]:
        """Scores with the step's own model, whether or not `scorer` is set."""

    def scores(self, texts: list[str]) -> list[float]:
        if self.scorer is not None:
            return self.scorer(texts)
        return self.model_scores(texts)
