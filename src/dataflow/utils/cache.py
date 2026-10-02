import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def cache_dir() -> Path:
    return Path(os.environ.get("DATAFLOW_CACHE", Path.home() / ".cache" / "dataflow"))


def cached_download(url: str, relative_path: str) -> Path:
    target = cache_dir() / relative_path
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s", url)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, tmp)
    os.replace(tmp.name, target)
    return target
