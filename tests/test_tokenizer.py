import json
import pickle

import numpy as np
import orjson
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from dataflow.data import Document
from dataflow.executor.local import LocalPipelineExecutor
from dataflow.pipeline.readers import JsonlReader
from dataflow.pipeline.tokens import DocumentTokenizer, read_tokenized

TEXTS = [
    "The river runs through the old town.",
    "Children learn to swim in the shallow parts.",
    "తెలుగు భాష చాలా అందమైనది.",
    "Visitors say the town feels calm.",
]


@pytest.fixture(scope="module")
def tokenizer_file(tmp_path_factory):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(TEXTS * 20, trainer)
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    tokenizer.save(str(path))
    return str(path)


def tokenize(tmp_path, tokenizer_file, shuffle):
    step = DocumentTokenizer(tmp_path, tokenizer_file, shuffle=shuffle, seed=7)
    list(step(iter(Document(text, str(i)) for i, text in enumerate(TEXTS))))
    return step


def decode(tokenizer_file, docs):
    tokenizer = Tokenizer.from_file(tokenizer_file)
    eos = tokenizer.token_to_id("<|endoftext|>")
    assert all(doc[-1] == eos for doc in docs)
    return [tokenizer.decode(doc[:-1].tolist()) for doc in docs]


def test_documents_round_trip_in_order_without_shuffle(tmp_path, tokenizer_file):
    step = tokenize(tmp_path, tokenizer_file, shuffle=False)
    docs = read_tokenized(tmp_path, "00000.ds")
    assert decode(tokenizer_file, docs) == TEXTS
    meta = json.loads((tmp_path / "00000.ds.meta").read_text())
    assert meta == {"tokenizer": tokenizer_file, "token_bytes": 2, "documents": 4, "tokens": sum(map(len, docs))}
    assert step.stats.metrics["tokens"].total == meta["tokens"]
    assert not (tmp_path / "00000_unshuffled.ds").exists()


def test_shuffle_reorders_whole_documents_reproducibly(tmp_path, tokenizer_file):
    tokenize(tmp_path / "a", tokenizer_file, shuffle=True)
    tokenize(tmp_path / "b", tokenizer_file, shuffle=True)
    first = decode(tokenizer_file, read_tokenized(tmp_path / "a", "00000.ds"))
    second = decode(tokenizer_file, read_tokenized(tmp_path / "b", "00000.ds"))
    assert sorted(first) == sorted(TEXTS)
    assert first == second
    assert first != TEXTS


def test_index_holds_cumulative_document_ends(tmp_path, tokenizer_file):
    tokenize(tmp_path, tokenizer_file, shuffle=False)
    ends = np.frombuffer((tmp_path / "00000.ds.index").read_bytes(), dtype="<u8")
    lengths = [len(doc) for doc in read_tokenized(tmp_path, "00000.ds")]
    assert ends.tolist() == np.cumsum(lengths).tolist()
    assert (tmp_path / "00000.ds").stat().st_size == ends[-1] * 2


def test_missing_eos_token_is_an_error(tmp_path, tokenizer_file):
    with pytest.raises(ValueError, match="<eos>"):
        list(DocumentTokenizer(tmp_path, tokenizer_file, eos_token="<eos>")(iter([Document("x", "0")])))


def test_tokenizer_is_not_pickled_and_tasks_run_in_parallel(tmp_path, tokenizer_file):
    step = DocumentTokenizer(tmp_path / "out", tokenizer_file)
    step.tokenizer.get_vocab_size()
    assert pickle.loads(pickle.dumps(step))._tokenizer is None
    (tmp_path / "in").mkdir()
    for i in range(2):
        (tmp_path / f"in/{i}.jsonl").write_bytes(b"".join(orjson.dumps({"text": t}) + b"\n" for t in TEXTS))
    LocalPipelineExecutor([JsonlReader(tmp_path / "in"), step], tmp_path / "logs", tasks=2, workers=2).run()
    for rank in range(2):
        assert sorted(decode(tokenizer_file, read_tokenized(tmp_path / "out", f"{rank:05d}.ds"))) == sorted(TEXTS)
