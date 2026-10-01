import pytest
from fsspec.implementations.local import LocalFileSystem

from dataflow.io import DataFolder, get_datafolder


@pytest.fixture
def folder(tmp_path):
    for name in ["b.jsonl", "a.jsonl", "notes.txt", "sub/c.jsonl", "sub/deep/d.jsonl"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return DataFolder(str(tmp_path))


def test_list_files_is_sorted_and_recursive(folder):
    assert folder.list_files() == ["a.jsonl", "b.jsonl", "notes.txt", "sub/c.jsonl", "sub/deep/d.jsonl"]


def test_list_files_without_recursion(folder):
    assert folder.list_files(recursive=False) == ["a.jsonl", "b.jsonl", "notes.txt"]


def test_glob_extension_shorthand(folder):
    assert folder.list_files(glob_pattern=".txt") == ["notes.txt"]
    assert folder.list_files(glob_pattern="**/*.jsonl") == ["a.jsonl", "b.jsonl", "sub/c.jsonl", "sub/deep/d.jsonl"]


def test_shards_are_disjoint_and_cover_everything(folder):
    files = folder.list_files()
    shards = [folder.get_shard(rank, 3) for rank in range(3)]
    assert shards == [files[0::3], files[1::3], files[2::3]]
    assert sorted(sum(shards, [])) == files


def test_empty_folder_has_no_shard(tmp_path):
    assert DataFolder(str(tmp_path)).get_shard(0, 1) is None


def test_open_for_writing_creates_parent_folders(tmp_path):
    folder = DataFolder(str(tmp_path))
    with folder.open("x/y/z.txt", "w") as file:
        file.write("hi")
    assert (tmp_path / "x/y/z.txt").read_text() == "hi"


def test_get_datafolder_accepts_paths_and_filesystems(tmp_path):
    assert get_datafolder(tmp_path).path == str(tmp_path)
    assert get_datafolder((str(tmp_path), {})).path == str(tmp_path)
    assert get_datafolder((str(tmp_path), LocalFileSystem())).path == str(tmp_path)
    folder = DataFolder(str(tmp_path))
    assert get_datafolder(folder) is folder
    with pytest.raises(ValueError):
        get_datafolder(42)
