from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from dataflow.browse import domain_of, read_documents
from dataflow.io import DataFolderLike, get_datafolder
from dataflow.status import collect
from dataflow.utils.stats import PipelineStats

FUNNEL_LABELS = ("documents", "total", "forwarded", "dropped", "timeout")


def funnel_rows(stats: PipelineStats) -> list[dict]:
    rows = []
    for step in stats.stats:
        metrics = {label: metric.total for label, metric in step.metrics.items()}
        base = metrics.get("total") or metrics.get("documents") or 0
        labels = [label for label in FUNNEL_LABELS if label in metrics]
        labels += sorted(label for label in metrics if label.startswith("dropped_"))
        for label in labels:
            share = metrics[label] / base if base else None
            rows.append(
                {"section": "funnel", "group": step.name, "key": label, "value": metrics[label], "share": share}
            )
    return rows


def count_rows(path: DataFolderLike, top: int = 20) -> list[dict]:
    domains, languages, total = Counter(), Counter(), 0
    for document in read_documents(path):
        total += 1
        domains[domain_of(document.metadata.get("url"))] += 1
        languages[document.metadata.get("language", "unknown")] += 1
    rows = [{"section": "documents", "group": "kept", "key": "total", "value": total, "share": 1.0 if total else None}]
    for section, counter in (("language", languages), ("domain", domains)):
        rows += [
            {"section": section, "group": "kept", "key": key, "value": value, "share": value / total}
            for key, value in counter.most_common(top)
        ]
    return rows


def build_report(logging_dir: DataFolderLike, kept: DataFolderLike | None = None, top: int = 20) -> list[dict]:
    rows = funnel_rows(collect(logging_dir).stats)
    if kept is not None:
        rows += count_rows(kept, top)
    return rows


def save_report(rows: list[dict], logging_dir: DataFolderLike, filename: str = "report.parquet") -> str:
    folder = get_datafolder(logging_dir)
    with folder.open(filename, "wb") as file:
        pq.write_table(pa.Table.from_pylist(rows), file, compression="zstd")
    return filename


def render_report(rows: list[dict]) -> str:
    width = max((len(str(row["key"])) for row in rows), default=10) + 2
    lines, current = [], None
    for row in rows:
        heading = row["group"] if row["section"] == "funnel" else f"{row['section']} ({row['group']})"
        if heading != current:
            lines.append(f"\n{heading}")
            current = heading
        share = f"{row['share']:7.2%}" if row["share"] is not None else ""
        lines.append(f"  {row['key']:<{width}} {row['value']:>14,.0f}  {share}")
    return "\n".join(lines).lstrip("\n")
