import io
import logging
import threading
import time

import httpx

logger = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}


class RateLimiter:
    """At most `per_second` calls per second in this process, shared by every thread."""

    def __init__(self, per_second: float):
        self.interval = 1 / per_second
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next - now
            self.next = max(now, self.next) + self.interval
        if delay > 0:
            time.sleep(delay)


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests: one request per read, nothing cached, every request
    rate-limited, and transient failures retried with back-off of at least `backoff` seconds (Common Crawl asks for
    10 s after a 503)."""

    def __init__(
        self,
        url: str,
        limiter: RateLimiter | None = None,
        client: httpx.Client | None = None,
        retries: int = 6,
        backoff: float = 10.0,
    ):
        self.url = url
        self.limiter = limiter
        self.client = client or httpx.Client(timeout=120, follow_redirects=True)
        self.retries = retries
        self.backoff = backoff
        self.position = 0
        self.requests = 0
        self.size = int(self.request("HEAD").headers["content-length"])

    def request(self, method: str, headers: dict | None = None) -> httpx.Response:
        for attempt in range(self.retries + 1):
            if self.limiter:
                self.limiter.wait()
            self.requests += 1
            try:
                response = self.client.request(method, self.url, headers=headers)
                if response.status_code not in RETRY_STATUSES:
                    response.raise_for_status()
                    return response
                error = f"HTTP {response.status_code}"
            except httpx.TransportError as exception:
                error = type(exception).__name__
            if attempt == self.retries:
                raise OSError(f"{self.url}: {error} after {self.retries} retries")
            logger.warning("%s: %s, retrying", self.url, error)
            time.sleep(self.backoff * 2**attempt)
        raise AssertionError("unreachable")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        end = self.size if size is None or size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        headers = {"Range": f"bytes={self.position}-{end - 1}"}
        data = self.request("GET", headers).content
        self.position += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
