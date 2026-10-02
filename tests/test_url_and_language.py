import io
import tarfile

import pytest

from dataflow.data import Document
from dataflow.pipeline.filters import LanguageFilter, URLFilter
from dataflow.pipeline.filters import url as url_module


class FakeLID:
    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text

    def predict(self, text):
        return self.scores_by_text[text]


def doc_with_url(url):
    return Document("some text", "1", {"url": url})


@pytest.fixture
def url_filter():
    return URLFilter(
        ut1_categories=(),
        blocked_domains={"bad.com", "spam.example.org"},
        blocked_urls={"good.com/secret-page"},
        banned_words={"casino"},
        soft_banned_words={"free", "win", "prize"},
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://news.bad.com/article", (False, "domain")),
        ("https://spam.example.org/x", (False, "subdomain")),
        ("https://good.com/secret-page", (False, "url")),
        ("https://good.com/best-casino-guide", (False, "banned_word")),
        ("https://good.com/win-a-free-phone", (False, "soft_banned_words")),
        ("https://good.com/free-recipes", True),
        ("https://example.org/rivers", True),
    ],
)
def test_url_filter_rules(url_filter, url, expected):
    assert url_filter.filter(doc_with_url(url)) == expected


def test_url_filter_requires_a_url(url_filter):
    assert url_filter.filter(Document("text", "1")) == (False, "missing_url")


def test_ut1_lists_are_downloaded_once_and_parsed(tmp_path, monkeypatch):
    archive = tmp_path / "ut1/adult.tar.gz"
    archive.parent.mkdir()
    with tarfile.open(archive, "w:gz") as tar:
        for name, content in [("adult/domains", b"blocked.net\n# comment\n"), ("adult/urls", b"ok.com/bad\n")]:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    monkeypatch.setattr(url_module, "cached_download", lambda url, path: tmp_path / path)
    step = URLFilter(ut1_categories=("adult",))
    assert step.filter(doc_with_url("https://www.blocked.net/")) == (False, "domain")
    assert step.filter(doc_with_url("http://ok.com/bad")) == (False, "url")
    assert step.filter(doc_with_url("http://ok.com/good")) is True


def test_language_filter_keeps_wanted_languages_above_threshold():
    lid = FakeLID(
        {
            "english": {"en": 0.93, "de": 0.02},
            "unsure": {"en": 0.41, "fr": 0.30},
            "telugu": {"te": 0.88},
            "chinese": {"zh": 0.99},
        }
    )
    step = LanguageFilter(["en", "te"], threshold={"en": 0.65, "te": 0.7}, lid=lid)
    docs = {text: Document(text, text) for text in lid.scores_by_text}
    assert step.filter(docs["english"]) is True
    assert step.filter(docs["telugu"]) is True
    assert step.filter(docs["unsure"]) == (False, "low_confidence")
    assert step.filter(docs["chinese"]) == (False, "other_language")
    assert docs["chinese"].metadata == {"language": "zh", "language_score": 0.99}


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        LanguageFilter(["en"], backend="nope")
