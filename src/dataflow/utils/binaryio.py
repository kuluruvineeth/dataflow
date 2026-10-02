import struct
from collections.abc import Iterator
from typing import BinaryIO


def read_tuples(file: BinaryIO, fmt: str, records_per_read: int = 4096) -> Iterator[tuple]:
    record = struct.Struct(fmt)
    while chunk := file.read(record.size * records_per_read):
        yield from record.iter_unpack(chunk)
