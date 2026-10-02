import json

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from dataflow.data import Document
from dataflow.pipeline.tokens import DocumentTokenizer, TokenMerger, check_tokenized, read_tokenized
from dataflow.pipeline.tokens.merger import split_into_files

TEXTS = [f"document {i}: the river runs through town number {i}" for i in range(30)]


@pytest.fixture(scope="module")
def tokenizer_file(tmp_path_factory):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(TEXTS * 5, trainer)
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    tokenizer.save(str(path))
    return str(path)


def tokenize_in_tasks(folder, tokenizer_file, tasks=3):
    for rank in range(tasks):
        docs = (Document(text, str(i)) for i, text in enumerate(TEXTS[rank::tasks]))
        list(DocumentTokenizer(folder, tokenizer_file, seed=1)(docs, rank=rank, world_size=tasks))


def merged_texts(folder, tokenizer_file):
    tokenizer = Tokenizer.from_file(tokenizer_file)
    files = sorted(path.name for path in folder.glob("*.ds"))
    return [tokenizer.decode(doc[:-1].tolist()) for name in files for doc in read_tokenized(folder, name)]


def merge(tmp_path, name, **kwargs):
    out = tmp_path / name
    step = TokenMerger(tmp_path / "tasks", out, **kwargs)
    list(step())
    return out, step


def test_split_into_files_cuts_only_between_documents():
    assert split_into_files([4, 4, 4, 9, 1], max_tokens=8) == [[0, 1], [2], [3], [4]]
    assert split_into_files([], max_tokens=8) == []


def test_merge_keeps_every_document_and_respects_the_file_size(tmp_path, tokenizer_file):
    tokenize_in_tasks(tmp_path / "tasks", tokenizer_file)
    out, step = merge(tmp_path, "out", max_tokens_per_file=100)
    assert sorted(merged_texts(out, tokenizer_file)) == sorted(TEXTS)
    files = sorted(out.glob("*.ds"))
    assert len(files) > 1 and step.stats.metrics["files"].total == len(files)
    for path in files:
        assert check_tokenized(out, path.name) == []
        meta = json.loads((out / f"{path.name}.meta").read_text())
        assert meta["tokens"] <= 100 or meta["documents"] == 1


def test_same_seed_same_order_and_shuffle_off_keeps_file_order(tmp_path, tokenizer_file):
    tokenize_in_tasks(tmp_path / "tasks", tokenizer_file)
    first = merged_texts(merge(tmp_path, "a", seed=3)[0], tokenizer_file)
    again = merged_texts(merge(tmp_path, "b", seed=3)[0], tokenizer_file)
    other = merged_texts(merge(tmp_path, "c", seed=4)[0], tokenizer_file)
    assert first == again and first != other
    in_order = merged_texts(merge(tmp_path, "d", shuffle=False)[0], tokenizer_file)
    assert in_order == merged_texts(tmp_path / "tasks", tokenizer_file)


def test_merge_refuses_files_from_different_tokenizers(tmp_path, tokenizer_file):
    tokenize_in_tasks(tmp_path / "tasks", tokenizer_file)
    meta_path = tmp_path / "tasks/00001.ds.meta"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "tokenizer": "another-tokenizer"}))
    with pytest.raises(ValueError, match="different tokenizers"):
        merge(tmp_path, "out")
    with pytest.raises(ValueError, match="tasks=1"):
        list(TokenMerger(tmp_path / "tasks", tmp_path / "x")(rank=0, world_size=2))


def test_check_tokenized_finds_corruption(tmp_path, tokenizer_file):
    tokenize_in_tasks(tmp_path / "tasks", tokenizer_file, tasks=1)
    path = tmp_path / "tasks/00000.ds"
    assert check_tokenized(tmp_path / "tasks", "00000.ds") == []
    data = bytearray(path.read_bytes())
    data[-2:] = b"\x05\x00"
    path.write_bytes(bytes(data))
    assert check_tokenized(tmp_path / "tasks", "00000.ds") == ["1 documents do not end with the end-of-text token"]
    path.write_bytes(bytes(data[:-2]))
    assert "bytes, the index expects" in check_tokenized(tmp_path / "tasks", "00000.ds")[0]
