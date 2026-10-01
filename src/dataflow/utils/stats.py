import json
import math
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass


@dataclass
class MetricStats:
    total: float = 0.0
    n: int = 0
    mean: float = 0.0
    min: float = math.inf
    max: float = -math.inf
    m2: float = 0.0
    unit: str = "doc"

    def update(self, value: float) -> None:
        self.total += value
        self.n += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def __add__(self, other: "MetricStats") -> "MetricStats":
        n = self.n + other.n
        if n == 0:
            return MetricStats(unit=self.unit)
        delta = other.mean - self.mean
        return MetricStats(
            total=self.total + other.total,
            n=n,
            mean=self.mean + delta * other.n / n,
            min=min(self.min, other.min),
            max=max(self.max, other.max),
            m2=self.m2 + other.m2 + delta * delta * self.n * other.n / n,
            unit=self.unit,
        )

    def __str__(self) -> str:
        if self.min == self.max == 1:
            return f"{self.total:g}"
        return f"{self.total:g} [{self.mean:.2f}±{self.std:.2f}, min={self.min:g}, max={self.max:g} /{self.unit}]"


class Stats:
    def __init__(self, name: str):
        self.name = name
        self.metrics: defaultdict[str, MetricStats] = defaultdict(MetricStats)
        self.time = MetricStats(unit="s")

    def update(self, label: str, value: float = 1, unit: str | None = None) -> None:
        metric = self.metrics[label]
        if unit:
            metric.unit = unit
        metric.update(value)

    @contextmanager
    def track_time(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.time.update(time.perf_counter() - start)

    def __add__(self, other: "Stats") -> "Stats":
        merged = Stats(self.name)
        for label in self.metrics.keys() | other.metrics.keys():
            merged.metrics[label] = self.metrics.get(label, MetricStats()) + other.metrics.get(label, MetricStats())
        merged.time = self.time + other.time
        return merged

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "time": asdict(self.time),
            "metrics": {label: asdict(metric) for label, metric in sorted(self.metrics.items())},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Stats":
        stats = cls(data["name"])
        stats.time = MetricStats(**data["time"])
        for label, metric in data["metrics"].items():
            stats.metrics[label] = MetricStats(**metric)
        return stats

    def __str__(self) -> str:
        lines = [self.name]
        lines += [f"    {label}: {metric}" for label, metric in sorted(self.metrics.items())]
        if self.time.n:
            lines.append(f"    time: {self.time.total:.3f}s total, {self.time.mean * 1000:.3f}ms/doc")
        return "\n".join(lines)


class PipelineStats:
    def __init__(self, stats: list[Stats] | None = None):
        self.stats = stats or []

    def __add__(self, other: "PipelineStats") -> "PipelineStats":
        if not self.stats:
            return other
        if not other.stats:
            return self
        return PipelineStats([a + b for a, b in zip(self.stats, other.stats, strict=True)])

    def to_json(self) -> str:
        return json.dumps([stats.to_dict() for stats in self.stats], indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PipelineStats":
        return cls([Stats.from_dict(data) for data in json.loads(text)])

    def __str__(self) -> str:
        return "\n".join(str(stats) for stats in self.stats)
