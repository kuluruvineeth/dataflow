"""DCLM's quality classifier, trained from scratch and compared with the released one.

DCLM (Li et al., 2024, appendix) trains fastText on 400K examples: 200K positives (100K OpenHermes-2.5, 100K
r/ExplainLikeImFive posts joined with their top answer) and 200K negatives from a trafilatura-extracted RefinedWeb
reproduction, with unigrams and bigrams and fastText's other defaults. Documents are then ranked by the probability
of `__label__hq`. Here the negatives are Falcon RefinedWeb pages (also trafilatura), and the ELI5 posts come from a
public copy without post scores; if fewer than 100K pass the other filters, all are used and the negatives match the
positives in number.

    python -m dataflow.recipes.dclm_classifier train out/
    python -m dataflow.recipes.dclm_classifier compare out/ hf://buckets/.../filtered --limit 20000
"""

import argparse
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dataflow.pipeline.classifiers import DCLM_OH_ELI5, FastTextClassifier
from dataflow.pipeline.classifiers.fasttext import flatten
from dataflow.pipeline.readers import JsonlReader

POSITIVE, NEGATIVE = "__label__hq", "__label__cc"
FASTTEXT = {"lr": 0.1, "dim": 100, "ws": 5, "epoch": 5, "minCount": 1, "wordNgrams": 2}
ELI5_FILES = [f"data/{split}/train-0000{i}-of-00002.parquet" for split in ("finetune", "reward", "rl") for i in (0, 1)]
REFINEDWEB_FILES = [
    "data/train-00000-of-05534-b8fc5348cbe605a5.parquet",
    "data/train-00001-of-05534-9bca3ce859516338.parquet",
]


def openhermes_texts(path: str) -> list[str]:
    """Every OpenHermes-2.5 conversation as one text, its turns joined by spaces."""
    with open(path) as file:
        return [" ".join(turn["value"] for turn in row["conversations"]) for row in json.load(file)]


def eli5_texts(rows: Iterable[dict], min_answer_score: int = 5, min_answers: int = 3) -> list[str]:
    """r/ExplainLikeImFive posts joined with their best answer (ties: the longest), DCLM's filters applied: the best
    answer scores at least 5 and the post has at least 3 answers. The post's own score (DCLM: at least 3) is not in
    the public copy we read."""
    texts, seen = [], set()
    for row in rows:
        if row["subreddit"] != "explainlikeimfive" or row["q_id"] in seen:
            continue
        answers = list(zip(row["answers"]["score"], row["answers"]["text"], strict=True))
        if len(answers) < min_answers:
            continue
        score, answer = max(answers, key=lambda pair: (pair[0], len(pair[1])))
        if score < min_answer_score:
            continue
        seen.add(row["q_id"])
        texts.append(" ".join(part for part in (row["title"], row["selftext"], answer) if part))
    return texts


def parquet_rows(paths: Iterable[str], columns: list[str]) -> Iterator[dict]:
    for path in paths:
        yield from pq.read_table(path, columns=columns).to_pylist()


def training_lines(positives: list[str], negatives: list[str], seed: int = 0) -> list[str]:
    lines = [f"{POSITIVE} {flatten(text)}" for text in positives]
    lines += [f"{NEGATIVE} {flatten(text)}" for text in negatives]
    random.Random(seed).shuffle(lines)
    return lines


def train(training_file: str | Path, output: str | Path, **overrides) -> Path:
    import fasttext

    model = fasttext.train_supervised(input=str(training_file), **{**FASTTEXT, **overrides})
    model.save_model(str(output))
    return Path(output)


def ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, ties sharing their average rank."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    result = np.empty(len(values))
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            result[order[start:end]] = (start + end + 1) / 2
            start = end
    return result


def agreement(ours: np.ndarray, reference: np.ndarray, top: float = 0.1) -> dict:
    """How closely two scorers agree on the same documents: rank and linear correlation, and the share of the
    reference's top fraction that ours also puts in its top fraction (DCLM keeps the top 10%)."""
    k = max(1, int(len(ours) * top))
    top_ours, top_reference = set(np.argsort(-ours)[:k]), set(np.argsort(-reference)[:k])
    return {
        "documents": len(ours),
        "spearman": float(np.corrcoef(ranks(ours), ranks(reference))[0, 1]),
        "pearson": float(np.corrcoef(ours, reference)[0, 1]),
        f"top_{top:g}_overlap": len(top_ours & top_reference) / k,
    }


def build_and_train(output: Path, per_source: int = 100_000, seed: int = 0) -> Path:
    from huggingface_hub import hf_hub_download

    rng = random.Random(seed)
    openhermes = openhermes_texts(hf_hub_download("teknium/OpenHermes-2.5", "openhermes2_5.json", repo_type="dataset"))
    eli5_rows = parquet_rows(
        [hf_hub_download("vincentmin/eli5_rlhf", f, repo_type="dataset") for f in ELI5_FILES],
        ["q_id", "title", "selftext", "subreddit", "answers"],
    )
    eli5 = eli5_texts(eli5_rows)
    web = [row["content"] for row in parquet_rows(
        [hf_hub_download("tiiuae/falcon-refinedweb", f, repo_type="dataset") for f in REFINEDWEB_FILES], ["content"]
    )]  # fmt: skip
    # the public ELI5 copy can hold fewer filtered posts than DCLM's 100K: use them all and keep the classes balanced
    positives = rng.sample(openhermes, per_source) + rng.sample(eli5, min(per_source, len(eli5)))
    if len(web) < len(positives):
        raise ValueError(f"{len(web)} RefinedWeb pages for {len(positives)} positives")
    negatives = rng.sample(web, len(positives))
    output.mkdir(parents=True, exist_ok=True)
    training_file = output / "train.txt"
    training_file.write_text("\n".join(training_lines(positives, negatives, seed)) + "\n")
    (output / "sources.json").write_text(json.dumps(
        {"openhermes": len(openhermes), "eli5_after_filters": len(eli5), "refinedweb": len(web),
         "positives": len(positives), "negatives": len(negatives), "seed": seed, "fasttext": FASTTEXT}, indent=2,
    ))  # fmt: skip
    return train(training_file, output / "model.bin")


def spread(paths: list[str], count: int) -> list[str]:
    """`count` paths evenly spaced through the sorted list, so a sample covers the whole run, not its first files."""
    paths = sorted(paths)
    step = max(1, len(paths) // count)
    return paths[::step][:count]


def compare(output: Path, documents_folder: str, limit: int, files: int = 20) -> dict:
    from dataflow.io import get_datafolder

    paths = spread(get_datafolder(documents_folder).list_files(), files)
    per_file = limit // len(paths)
    texts = [d.text for p in paths for d in JsonlReader(documents_folder, paths=[p], limit=per_file).run()]
    ours, official = FastTextClassifier(output / "model.bin"), FastTextClassifier(DCLM_OH_ELI5)
    result = agreement(np.array([ours.score(t) for t in texts]), np.array([official.score(t) for t in texts]))
    record = {"documents_folder": documents_folder, "files": paths, **result}
    (output / "agreement.json").write_text(json.dumps(record, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DCLM's fastText quality classifier, compare with theirs.")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="build the 400K-example training set and train")
    train_parser.add_argument("output", type=Path)
    train_parser.add_argument("--per-source", type=int, default=100_000)
    train_parser.add_argument("--seed", type=int, default=0)
    compare_parser = commands.add_parser("compare", help="score the same documents with ours and the released model")
    compare_parser.add_argument("output", type=Path)
    compare_parser.add_argument("documents", help="folder of JSONL documents")
    compare_parser.add_argument("--limit", type=int, default=20_000)
    compare_parser.add_argument("--files", type=int, default=20, help="files read, evenly spaced through the run")
    args = parser.parse_args()
    if args.command == "train":
        print(build_and_train(args.output, args.per_source, args.seed))
    else:
        print(json.dumps(compare(args.output, args.documents, args.limit, args.files), indent=2))


if __name__ == "__main__":
    main()
