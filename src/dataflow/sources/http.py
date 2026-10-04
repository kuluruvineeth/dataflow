import io
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO

import httpx

logger = logging.getLogger(__name__)

RETRY_STATUSES = {403, 429, 500, 502, 503, 504}
USER_AGENT = "dataflow/0.1 (+https://github.com/kuluruvineeth/dataflow)"


def http_client() -> httpx.Client:
    return httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": USER_AGENT})


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


def request(
    client: httpx.Client,
    method: str,
    url: str,
    headers: dict | None = None,
    limiter: RateLimiter | None = None,
    retries: int = 6,
    backoff: float = 10.0,
) -> httpx.Response:
    """One HTTP request, rate-limited, retried on 403, 429, 5xx and network errors with back-off of at least `backoff`
    seconds. Common Crawl's CDN answers 403 or 503 when it is asked too fast, and asks for 10 s after either."""
    for attempt in range(retries + 1):
        if limiter:
            limiter.wait()
        try:
            response = client.request(method, url, headers=headers)
            if response.status_code not in RETRY_STATUSES:
                response.raise_for_status()
                return response
            error = f"HTTP {response.status_code}"
        except httpx.TransportError as exception:
            error = type(exception).__name__
        if attempt == retries:
            raise OSError(f"{url}: {error} after {retries} retries")
        logger.warning("%s: %s, retrying", url, error)
        time.sleep(backoff * 2**attempt)
    raise AssertionError("unreachable")


def get_range(
    url: str,
    start: int,
    end: int,
    client: httpx.Client,
    limiter: RateLimiter | None = None,
    headers: dict | None = None,
    **kwargs,
) -> bytes:
    """Bytes `start` to `end` (exclusive) of `url`."""
    headers = {**(headers or {}), "Range": f"bytes={start}-{end - 1}"}
    return request(client, "GET", url, headers, limiter, **kwargs).content


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests: one request per read, nothing cached, every request
    rate-limited and retried."""

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
        self.client = client or http_client()
        self.retries = retries
        self.backoff = backoff
        self.position = 0
        self.requests = 0
        self.size = int(self.request("HEAD").headers["content-length"])

    def request(self, method: str, headers: dict | None = None) -> httpx.Response:
        self.requests += 1
        return request(self.client, method, self.url, headers, self.limiter, self.retries, self.backoff)

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


class ResponseStream(io.RawIOBase):
    """A remote file streamed with one GET and read as a file as it arrives. When the connection drops mid-body, the
    rest is requested from the last byte received, so a download of tens of gigabytes survives a reset."""

    def __init__(
        self,
        client: httpx.Client,
        url: str,
        limiter: RateLimiter | None = None,
        retries: int = 6,
        backoff: float = 10.0,
    ):
        self.client = client
        self.url = url
        self.limiter = limiter
        self.retries = retries
        self.backoff = backoff
        self.received = 0
        self.resumes = 0
        self.response = None
        self.pending = b""
        self.connect()

    def connect(self) -> None:
        headers = {"Range": f"bytes={self.received}-"} if self.received else None
        for attempt in range(self.retries + 1):
            if self.limiter:
                self.limiter.wait()
            try:
                response = self.client.send(self.client.build_request("GET", self.url, headers=headers), stream=True)
                if response.status_code not in RETRY_STATUSES:
                    if self.received and response.status_code != 206:
                        response.close()
                        raise OSError(f"{self.url}: HTTP {response.status_code} to a range request, cannot resume")
                    response.raise_for_status()
                    self.response = response
                    self.chunks = response.iter_raw(1 << 20)
                    return
                response.close()
                error = f"HTTP {response.status_code}"
            except httpx.TransportError as exception:
                error = type(exception).__name__
            if attempt == self.retries:
                raise OSError(f"{self.url}: {error} after {self.retries} retries")
            logger.warning("%s: %s, retrying", self.url, error)
            time.sleep(self.backoff * 2**attempt)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while not self.pending:
            try:
                self.pending = next(self.chunks, b"")
            except httpx.TransportError as error:
                logger.warning("%s: %s after %d bytes, resuming", self.url, type(error).__name__, self.received)
                self.response.close()
                self.resumes += 1
                self.connect()
                continue
            if not self.pending:
                return 0
            self.received += len(self.pending)
        size = min(len(buffer), len(self.pending))
        buffer[:size], self.pending = self.pending[:size], self.pending[size:]
        return size

    def close(self) -> None:
        if self.response is not None:
            self.response.close()
        super().close()


@contextmanager
def open_url(url: str, limiter: RateLimiter | None = None, retries: int = 6, backoff: float = 10.0) -> Iterator[IO]:
    """A whole remote file as one streamed GET, retried until it starts and resumed if the connection drops."""
    with http_client() as client, ResponseStream(client, url, limiter, retries, backoff) as stream:
        yield io.BufferedReader(stream, 1 << 20)
