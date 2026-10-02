from hashlib import blake2b


def hash64(text: str) -> int:
    return int.from_bytes(blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")
