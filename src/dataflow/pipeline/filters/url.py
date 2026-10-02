import re
import tarfile
from collections.abc import Iterable

from tldextract import TLDExtract

from dataflow.data import Document
from dataflow.pipeline.filters.base import BaseFilter, FilterResult
from dataflow.pipeline.writers.base import DiskWriter
from dataflow.utils.cache import cached_download

UT1_URL = "https://dsi.ut-capitole.fr/blacklists/download/{category}.tar.gz"
FINEWEB_LISTS_URL = (
    "https://raw.githubusercontent.com/huggingface/datatrove/1977fbb0f3c163d43cace334b073dda17fa3ef26"
    "/src/datatrove/assets/{name}"
)
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def load_ut1(category: str) -> tuple[set[str], set[str]]:
    archive = cached_download(UT1_URL.format(category=category), f"ut1/{category}.tar.gz")
    domains: set[str] = set()
    urls: set[str] = set()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = {"domains": domains, "urls": urls}.get(member.name.rsplit("/", 1)[-1])
            if target is None or not member.isfile():
                continue
            for line in tar.extractfile(member).read().decode("utf-8", "ignore").splitlines():
                if line and not line.startswith("#"):
                    target.add(line.strip())
    return domains, urls


def load_word_list(name: str) -> set[str]:
    path = cached_download(FINEWEB_LISTS_URL.format(name=name), f"url-lists/{name}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return {NON_ALNUM.sub("", line.lower()) for line in lines if line and not line.startswith("#")}


class URLFilter(BaseFilter):
    name = "url"

    def __init__(
        self,
        ut1_categories: Iterable[str] = ("adult",),
        blocked_domains: Iterable[str] = (),
        blocked_urls: Iterable[str] = (),
        banned_words: Iterable[str] = (),
        soft_banned_words: Iterable[str] = (),
        soft_word_threshold: int = 2,
        banned_subwords: Iterable[str] = (),
        fineweb_word_lists: bool = False,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        self.ut1_categories = tuple(ut1_categories)
        self.blocked_domains = set(blocked_domains)
        self.blocked_urls = set(blocked_urls)
        self.banned_words = set(banned_words)
        self.soft_banned_words = set(soft_banned_words)
        self.soft_word_threshold = soft_word_threshold
        self.banned_subwords = set(banned_subwords)
        self.fineweb_word_lists = fineweb_word_lists
        self._lists_loaded = not (self.ut1_categories or fineweb_word_lists)
        self._extract = TLDExtract(suffix_list_urls=())

    def _load_lists(self) -> None:
        for category in self.ut1_categories:
            domains, urls = load_ut1(category)
            self.blocked_domains |= domains
            self.blocked_urls |= urls
        if self.fineweb_word_lists:
            self.banned_words |= load_word_list("banned_words.txt")
            self.soft_banned_words |= load_word_list("soft_banned_words.txt")
            self.banned_subwords |= load_word_list("banned_subwords.txt")
        self._lists_loaded = True

    def filter(self, document: Document) -> FilterResult:
        if not self._lists_loaded:
            self._load_lists()
        url = document.metadata.get("url")
        if not url:
            return False, "missing_url"
        parts = self._extract(url)
        if parts.top_domain_under_public_suffix in self.blocked_domains:
            return False, "domain"
        if parts.fqdn in self.blocked_domains:
            return False, "subdomain"
        if url.split("://", 1)[-1] in self.blocked_urls:
            return False, "url"
        words = set(NON_ALNUM.split(url.lower()))
        if words & self.banned_words:
            return False, "banned_word"
        if len(words & self.soft_banned_words) >= self.soft_word_threshold:
            return False, "soft_banned_words"
        compact = NON_ALNUM.sub("", url.lower())
        if any(subword in compact for subword in self.banned_subwords):
            return False, "banned_subword"
        return True
