import json
import logging
import os
import random
import socket
import threading
import time
import traceback

from dataflow.io import DataFolder
from dataflow.utils.stats import PipelineStats

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Writes `progress/<rank>.json` every `interval` seconds (jittered) while a rank runs, and once at the end."""

    def __init__(self, logging_dir: DataFolder, rank: int, stats: list, interval: float = 30):
        self.logging_dir = logging_dir
        self.rank = rank
        self.stats = stats
        self.interval = interval
        self.started = time.time()
        self.documents = 0
        self.status = "running"
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> "ProgressReporter":
        self.write()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join()
        if exc is not None:
            self.status = "failed"
            self.error = f"{exc_type.__name__}: {exc}"
            with self.logging_dir.open(f"errors/{self.rank:05d}.json", "w") as file:
                json.dump({"rank": self.rank, "time": time.time(), "traceback": traceback.format_exc()}, file)
        else:
            self.status = "done"
        self.write()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval * random.uniform(0.8, 1.2)):
            try:
                self.write()
            except Exception as error:  # noqa: BLE001
                logger.warning("rank %d: could not write progress: %s", self.rank, error)

    def snapshot(self) -> dict:
        for _ in range(3):
            try:
                stats = json.loads(PipelineStats(list(self.stats)).to_json())
                break
            except RuntimeError:
                time.sleep(0.01)
        else:
            stats = []
        now = time.time()
        return {
            "rank": self.rank,
            "status": self.status,
            "host": socket.gethostname(),
            "job_id": os.environ.get("JOB_ID"),
            "started": self.started,
            "updated": now,
            "documents": self.documents,
            "documents_per_second": self.documents / max(now - self.started, 1e-9),
            "error": self.error,
            "stats": stats,
        }

    def write(self) -> None:
        snapshot = self.snapshot()
        with self.logging_dir.open(f"progress/{self.rank:05d}.json", "w") as file:
            json.dump(snapshot, file)
