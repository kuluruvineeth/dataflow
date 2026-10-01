# dataflow

Streaming, sharded, resumable pipelines for preparing language-model training data.

## Idea

A pipeline is a list of steps. Each step takes a stream of documents and yields a stream of documents, so data flows
through one document at a time and memory stays flat no matter how large the input is.

```
reader → extractor → filters → dedup → tokenizer → writer
```

Work is split into independent tasks. Each task processes its own shard of files, so the same pipeline runs on one
laptop or across many machines, and a crashed job resumes from the tasks that did not finish.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
