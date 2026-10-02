from collections.abc import Callable

from dataflow.data import Document
from dataflow.pipeline.filters.base import BaseFilter, FilterResult
from dataflow.pipeline.writers.base import DiskWriter
from dataflow.policy.gate import PolicyGate
from dataflow.policy.robots import Robots


class PolicyFilter(BaseFilter):
    """Drops pages our sourcing rule excludes: robots.txt (ours and AI-training tokens), opt-out signals, the registry.

    Place it right after the WARC reader, while the text is still the raw HTML and `http_headers` is in the metadata.
    `robots(url)` returns the robots.txt that applied when the page was crawled, or None when it is unknown; unknown
    robots.txt is not treated as a block, and the documents that had none are counted under `robots_unknown`.
    """

    name = "policy"

    def __init__(
        self,
        robots: Callable[[str], Robots | None] | None = None,
        gate: PolicyGate | None = None,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        self.robots = robots
        self.gate = gate or PolicyGate()

    def filter(self, document: Document) -> FilterResult:
        url = document.metadata.get("url", "")
        robots = self.robots(url) if self.robots else None
        if self.robots and robots is None:
            self.stat_update("robots_unknown")
        decision = self.gate.check(url, robots, document.metadata.get("http_headers", {}), document.text)
        return True if decision.allowed else (False, decision.reason)
