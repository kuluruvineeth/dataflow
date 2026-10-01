from glob import has_magic
from pathlib import Path
from typing import IO

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs
from fsspec.implementations.dirfs import DirFileSystem


class OutputFileManager:
    def __init__(self, fs: AbstractFileSystem, mode: str = "wt", compression: str | None = "infer"):
        self.fs = fs
        self.mode = mode
        self.compression = compression
        self._files: dict[str, IO] = {}

    def get_file(self, filename: str) -> IO:
        if filename not in self._files:
            self._files[filename] = self.fs.open(filename, mode=self.mode, compression=self.compression)
        return self._files[filename]

    def close(self) -> None:
        for file in self._files.values():
            file.close()
        self._files.clear()

    def __enter__(self) -> "OutputFileManager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DataFolder(DirFileSystem):
    def __init__(self, path: str, fs: AbstractFileSystem | None = None, auto_mkdir: bool = True, **storage_options):
        if fs is None:
            fs, path = url_to_fs(path, **storage_options)
        super().__init__(path=path, fs=fs)
        self.auto_mkdir = auto_mkdir

    def list_files(self, subdirectory: str = "", recursive: bool = True, glob_pattern: str | None = None) -> list[str]:
        if glob_pattern and not has_magic(glob_pattern):
            glob_pattern = f"*{glob_pattern}"
        maxdepth = None if recursive else 1
        if glob_pattern:
            pattern = f"{subdirectory}/{glob_pattern}" if subdirectory else glob_pattern
            entries = self.glob(pattern, maxdepth=maxdepth, detail=True)
        else:
            entries = self.find(subdirectory, maxdepth=maxdepth, detail=True)
        return sorted(path for path, info in entries.items() if info["type"] != "directory")

    def get_shard(self, rank: int, world_size: int, **kwargs) -> list[str] | None:
        files = self.list_files(**kwargs)
        if not files:
            return None
        return files[rank::world_size]

    def open(self, path: str, mode: str = "rb", **kwargs) -> IO:
        if self.auto_mkdir and ("w" in mode or "a" in mode):
            self.fs.makedirs(self.fs._parent(self._join(path)), exist_ok=True)
        return super().open(path, mode=mode, **kwargs)

    def get_output_file_manager(self, **kwargs) -> OutputFileManager:
        return OutputFileManager(self, **kwargs)


type DataFolderLike = str | Path | tuple[str, dict] | tuple[str, AbstractFileSystem] | DataFolder


def get_datafolder(data: DataFolderLike) -> DataFolder:
    if isinstance(data, DataFolder):
        return data
    if isinstance(data, str | Path):
        return DataFolder(str(data))
    if isinstance(data, tuple) and isinstance(data[0], str) and isinstance(data[1], dict):
        return DataFolder(data[0], **data[1])
    if isinstance(data, tuple) and isinstance(data[0], str) and isinstance(data[1], AbstractFileSystem):
        return DataFolder(data[0], fs=data[1])
    raise ValueError("expected a DataFolder, a path, (path, storage_options) or (path, filesystem)")
