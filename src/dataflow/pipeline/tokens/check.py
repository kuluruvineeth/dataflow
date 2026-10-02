import numpy as np

from dataflow.io import DataFolderLike, get_datafolder
from dataflow.pipeline.tokens.tokenizer import read_ends, read_meta


def check_tokenized(folder: DataFolderLike, filename: str) -> list[str]:
    folder = get_datafolder(folder)
    meta = read_meta(folder, filename)
    ends = read_ends(folder, filename)
    size = folder.info(filename)["size"]
    problems = []
    if len(ends) and np.any(np.diff(ends, prepend=0) <= 0):
        problems.append("document ends are not strictly increasing")
    expected = int(ends[-1]) if len(ends) else 0
    if size != expected * meta["token_bytes"]:
        problems.append(f"{filename} has {size} bytes, the index expects {expected * meta['token_bytes']}")
    if (meta["documents"], meta["tokens"]) != (len(ends), expected):
        problems.append(
            f"meta says {meta['documents']} documents / {meta['tokens']} tokens, index has {len(ends)} / {expected}"
        )
    if not problems and meta.get("eos_token_id") is not None and len(ends):
        with folder.open(filename, "rb") as file:
            tokens = np.frombuffer(file.read(), dtype=f"<u{meta['token_bytes']}")
        missing = int(np.sum(tokens[ends - 1] != meta["eos_token_id"]))
        if missing:
            problems.append(f"{missing} documents do not end with the end-of-text token")
    return problems
