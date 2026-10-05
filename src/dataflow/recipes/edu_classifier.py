"""FineWeb-Edu's educational-value classifier, retrained on scores from an open model and compared with the released
one.

FineWeb-Edu (HuggingFaceFW/fineweb-edu-classifier, src/train_edu_bert.py) puts a one-output regression head on
Snowflake/snowflake-arctic-embed-m, freezes the embeddings and the encoder, and trains for 20 epochs (AdamW,
learning rate 3e-4 decaying linearly, batch 256, no dropout) on 0-5 scores, split 90/10 by score with seed 42,
keeping the checkpoint with the best macro F1 of the rounded scores. With the encoder frozen, each page's [CLS] state
never changes, so it is computed once and only the pooler and the head train on it: the same function for a fraction
of the compute. Checkpoints are compared after every epoch rather than every 1,000 steps. The result is saved as a
BertForSequenceClassification and scored exactly like the released model.

FineWeb-Edu's 20 epochs ran over about 420K pages, roughly 33K optimizer steps. On fewer pages, 20 epochs leave the
head undertrained: on 45K training pages, `--epochs 180` gives a similar number of steps.

Labels are JSONL rows `{"text": page, "score": 0-5, "reference_score": 0-5 or null}`.

    python -m dataflow.recipes.edu_classifier train labels.jsonl out/
    python -m dataflow.recipes.edu_classifier compare out/ hf://buckets/.../filtered --limit 20000
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from dataflow.pipeline.classifiers.regression import FINEWEB_EDU, RegressionClassifier, best_device
from dataflow.pipeline.readers import JsonlReader
from dataflow.utils.agreement import agreement, spread

BASE = "Snowflake/snowflake-arctic-embed-m"


def read_labels(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    reference = [-1 if row.get("reference_score") is None else row["reference_score"] for row in rows]
    return [row["text"] for row in rows], np.array([row["score"] for row in rows]), np.array(reference)


def stratified_split(scores: np.ndarray, test: float = 0.1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Indices of a train and a test set holding about the same share of every score."""
    rng = random.Random(seed)
    train, held_out = [], []
    for score in sorted(set(scores.tolist())):
        members = [int(i) for i in np.flatnonzero(scores == score)]
        rng.shuffle(members)
        cut = round(len(members) * test)
        held_out += members[:cut]
        train += members[cut:]
    return np.array(sorted(train)), np.array(sorted(held_out))


def as_classes(scores: np.ndarray) -> np.ndarray:
    return np.clip(np.round(scores), 0, 5).astype(int)


def macro_f1(predicted: np.ndarray, actual: np.ndarray) -> float:
    """F1 averaged over every class present in either array, on scores rounded and clipped to 0-5."""
    predicted, actual = as_classes(predicted), as_classes(actual)
    f1s = []
    for label in sorted(set(predicted.tolist()) | set(actual.tolist())):
        true_positive = np.sum((predicted == label) & (actual == label))
        denominator = np.sum(predicted == label) + np.sum(actual == label)
        f1s.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(f1s))


def binary_f1(predicted: np.ndarray, actual: np.ndarray, threshold: int = 3) -> float:
    """F1 of "educational" (rounded score at least `threshold`), the cut FineWeb-Edu keeps."""
    p, a = as_classes(predicted) >= threshold, as_classes(actual) >= threshold
    true_positive = np.sum(p & a)
    return float(2 * true_positive / (np.sum(p) + np.sum(a))) if np.sum(p) + np.sum(a) else 0.0


def cls_states(texts: list[str], base: str = BASE, batch_size: int = 64, max_length: int = 512) -> np.ndarray:
    """The frozen encoder's last-layer [CLS] state for every text."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = best_device()
    tokenizer, encoder = AutoTokenizer.from_pretrained(base), AutoModel.from_pretrained(base).to(device).eval()
    states = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start : start + batch_size], return_tensors="pt", padding="longest",
                              truncation=True, max_length=max_length)  # fmt: skip
            states.append(encoder(**batch.to(device)).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(states)


def train_head(states: np.ndarray, scores: np.ndarray, split: tuple, base: str = BASE, epochs: int = 20,
               learning_rate: float = 3e-4, batch_size: int = 256, seed: int = 0):  # fmt: skip
    """Trains the pooler and regression head of `base` on fixed [CLS] states; returns the full model with the best
    epoch's head and the per-epoch held-out macro F1."""
    import torch
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=1, problem_type="regression", classifier_dropout=0.0, hidden_dropout_prob=0.0
    )
    pooler, head = model.bert.pooler, model.classifier
    parameters = [*pooler.parameters(), *head.parameters()]

    def predict(x: torch.Tensor) -> torch.Tensor:
        return head(pooler.activation(pooler.dense(x))).squeeze(-1)

    train_index, test_index = split
    x, y = torch.from_numpy(states), torch.from_numpy(scores.astype(np.float32))
    steps = epochs * -(-len(train_index) // batch_size)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / steps)
    generator = torch.Generator().manual_seed(seed)
    best, history = (-1.0, None), []
    for _ in range(epochs):
        order = torch.from_numpy(train_index)[torch.randperm(len(train_index), generator=generator)]
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            loss = torch.nn.functional.mse_loss(predict(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
        with torch.no_grad():
            f1 = macro_f1(predict(x[test_index]).numpy(), scores[test_index])
        history.append(round(f1, 4))
        if f1 > best[0]:
            best = (f1, {k: v.clone() for k, v in [*pooler.state_dict().items(), *head.state_dict().items()]})
    names = list(pooler.state_dict())
    pooler.load_state_dict({k: best[1][k] for k in names})
    head.load_state_dict({k: v for k, v in best[1].items() if k not in names})
    return model, history


def train(labels: str | Path, output: Path, base: str = BASE, epochs: int = 20, seed: int = 0) -> dict:
    from transformers import AutoTokenizer

    texts, scores, reference = read_labels(labels)
    split = stratified_split(scores)
    model, history = train_head(cls_states(texts, base), scores, split, base, epochs, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "model")
    AutoTokenizer.from_pretrained(base).save_pretrained(output / "model")
    held_out = split[1]
    test_texts = [texts[i] for i in held_out]
    ours = np.array(RegressionClassifier(str(output / "model")).scores_all(test_texts))
    official = np.array(RegressionClassifier(FINEWEB_EDU).scores_all(test_texts))
    metrics = {"labels": str(labels), "pages": len(texts), "train": len(split[0]), "test": len(held_out),
               "epoch_macro_f1": history, "ours_vs_labels": evaluate(ours, scores[held_out]),
               "official_vs_labels": evaluate(official, scores[held_out])}  # fmt: skip
    with_reference = held_out[reference[held_out] >= 0]
    if len(with_reference):
        keep = np.isin(held_out, with_reference)
        metrics["ours_vs_reference"] = evaluate(ours[keep], reference[with_reference])
        metrics["official_vs_reference"] = evaluate(official[keep], reference[with_reference])
        metrics["labels_vs_reference"] = evaluate(scores[with_reference], reference[with_reference])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def evaluate(predicted: np.ndarray, actual: np.ndarray) -> dict:
    return {
        "macro_f1": round(macro_f1(predicted, actual), 4),
        "binary_f1_at_3": round(binary_f1(predicted, actual), 4),
        "exact": round(float(np.mean(as_classes(predicted) == as_classes(actual))), 4),
    }


def compare(output: Path, documents_folder: str, limit: int, files: int = 20) -> dict:
    from dataflow.io import get_datafolder

    paths = spread(get_datafolder(documents_folder).list_files(), files)
    per_file = limit // len(paths)
    texts = [d.text for p in paths for d in JsonlReader(documents_folder, paths=[p], limit=per_file).run()]
    ours = np.array(RegressionClassifier(str(output / "model")).scores_all(texts))
    official = np.array(RegressionClassifier(FINEWEB_EDU).scores_all(texts))
    kept_ours, kept_official = as_classes(ours) >= 3, as_classes(official) >= 3
    result = {**agreement(ours, official), "kept_at_3_ours": float(kept_ours.mean()),
              "kept_at_3_official": float(kept_official.mean()),
              "kept_at_3_overlap": float((kept_ours & kept_official).sum() / max(1, kept_official.sum()))}  # fmt: skip
    (output / "agreement.json").write_text(json.dumps({"documents_folder": documents_folder, "files": paths,
                                                       **result}, indent=2))  # fmt: skip
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FineWeb-Edu's classifier on open-model labels, compare.")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="train the head on frozen embeddings and evaluate it")
    train_parser.add_argument("labels", type=Path)
    train_parser.add_argument("output", type=Path)
    train_parser.add_argument("--base", default=BASE)
    train_parser.add_argument("--epochs", type=int, default=20, help="match ~33K steps on small label sets")
    train_parser.add_argument("--seed", type=int, default=0)
    compare_parser = commands.add_parser("compare", help="score the same documents with ours and the released model")
    compare_parser.add_argument("output", type=Path)
    compare_parser.add_argument("documents", help="folder of JSONL documents")
    compare_parser.add_argument("--limit", type=int, default=20_000)
    compare_parser.add_argument("--files", type=int, default=20, help="files read, evenly spaced through the run")
    args = parser.parse_args()
    if args.command == "train":
        result = train(args.labels, args.output, args.base, args.epochs, args.seed)
    else:
        result = compare(args.output, args.documents, args.limit, args.files)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
