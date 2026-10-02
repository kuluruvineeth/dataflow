import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dataflow.browse import domain_of, read_documents
from dataflow.data import Document
from dataflow.io import DataFolder, DataFolderLike, get_datafolder

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("url", pa.string()),
        ("domain", pa.string()),
        ("language", pa.string()),
        ("language_score", pa.float64()),
        ("filter_reason", pa.string()),
        ("metadata", pa.string()),
    ]
)
COLUMNS = {"url", "language", "language_score", "filter_reason"}


def row(document: Document) -> dict:
    metadata = document.metadata
    score = metadata.get("language_score")
    return {
        "id": document.id,
        "text": document.text,
        "url": metadata.get("url"),
        "domain": domain_of(metadata.get("url")) or None,
        "language": metadata.get("language"),
        "language_score": float(score) if score is not None else None,
        "filter_reason": metadata.get("filter_reason"),
        "metadata": json.dumps(
            {k: v for k, v in metadata.items() if k not in COLUMNS}, default=str, ensure_ascii=False
        ),
    }


def write_config(source: DataFolderLike, target: Path, rows_per_group: int, rows_per_file: int) -> int:
    target.mkdir(parents=True, exist_ok=True)
    written, writer, buffer = 0, None, []

    def flush() -> None:
        nonlocal writer
        if not buffer:
            return
        if writer is None:
            path = target / f"part-{written // rows_per_file:05d}.parquet"
            writer = pq.ParquetWriter(path, SCHEMA, compression="zstd", write_page_index=True)
        writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
        buffer.clear()

    for document in read_documents(source):
        buffer.append(row(document))
        written += 1
        if len(buffer) >= rows_per_group:
            flush()
        if written % rows_per_file == 0:
            flush()
            writer.close()
            writer = None
    flush()
    if writer is not None:
        writer.close()
    return written


def build_preview(
    kept: DataFolderLike,
    removed: DataFolderLike | None,
    out_dir: str | Path,
    title: str,
    rows_per_group: int = 20_000,
    rows_per_file: int = 200_000,
) -> dict[str, int]:
    out_dir = Path(out_dir)
    counts = {"kept": write_config(kept, out_dir / "data/kept", rows_per_group, rows_per_file)}
    if removed is not None:
        folder = get_datafolder(removed)
        steps = sorted({path.split("/")[0] for path in folder.list_files() if "/" in path})
        for step in steps:
            source = DataFolder(f"{folder.path.rstrip('/')}/{step}", fs=folder.fs)
            counts[f"removed_{step}"] = write_config(
                source, out_dir / f"data/removed_{step}", rows_per_group, rows_per_file
            )
    (out_dir / "README.md").write_text(card(title, counts))
    return counts


def card(title: str, counts: dict[str, int]) -> str:
    configs = "\n".join(
        f"- config_name: {name}\n  data_files:\n  - split: train\n    path: data/{name}/*.parquet" for name in counts
    )
    table = "\n".join(f"| `{name}` | {count:,} |" for name, count in counts.items())
    return f"""---
configs:
{configs}
---

# {title}

Documents kept and removed by a [dataflow](https://github.com/kuluruvineeth/dataflow) pipeline, one config per
outcome. Removed configs carry the reason in `filter_reason`; browse them in the Data Studio or query them in the SQL
console, for example `SELECT filter_reason, count(*) FROM removed_language GROUP BY 1`.

| Config | Documents |
|---|---|
{table}

Columns: `id`, `text`, `url`, `domain`, `language`, `language_score`, `filter_reason`, and `metadata` (the remaining
fields as JSON).
"""


def upload(out_dir: str | Path, repo_id: str, private: bool = True) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(out_dir),
        commit_message="Publish kept and removed documents",
    )
    return commit.commit_url
