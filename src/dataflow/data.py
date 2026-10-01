from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    text: str
    id: str
    metadata: dict[str, Any] = field(default_factory=dict)


DocumentsPipeline = Generator[Document, None, None] | None
