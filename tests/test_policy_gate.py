import pytest

from dataflow.policy.gate import OptOutRegistry, PolicyGate, RobotsCache, page_opt_out, says_no_training
from dataflow.policy.robots import Robots

URL = "https://news.example.com/2026/story.html"


def test_our_token_must_be_allowed():
    robots = Robots.parse("User-agent: *\nDisallow: /2026/\n")
    assert PolicyGate().check(URL, robots).reason == "robots_disallow"


@pytest.mark.parametrize("agent", ["GPTBot", "ClaudeBot", "CCBot", "Google-Extended", "Applebot-Extended", "Ai2Bot"])
def test_blocking_any_ai_training_token_is_an_opt_out(agent):
    robots = Robots.parse(f"User-agent: {agent}\nDisallow: /\n\nUser-agent: *\nAllow: /\n")
    assert PolicyGate().check(URL, robots).reason == "ai_agent_blocked"


def test_ai_tokens_only_count_when_named_and_a_partial_block_counts_only_where_it_applies():
    robots = Robots.parse("User-agent: GPTBot\nDisallow: /private/\n")
    assert PolicyGate().check(URL, robots).allowed
    assert PolicyGate().check("https://news.example.com/private/a", robots).reason == "ai_agent_blocked"


def test_content_signal_in_robots_txt():
    robots = Robots.parse("User-Agent: *\nContent-Signal: search=yes, ai-train=no\nAllow: /\n")
    assert PolicyGate().check(URL, robots).reason == "robots_content_signal"


@pytest.mark.parametrize(
    ("value", "no"),
    [
        ("ai-train=no", True),
        ("search=yes, AI-Train=No", True),
        ("train-ai=n", True),
        ("ai-train=yes", False),
        ("", False),
    ],
)
def test_says_no_training(value, no):
    assert says_no_training(value) is no


@pytest.mark.parametrize(
    ("headers", "html", "reason"),
    [
        ({"X-Robots-Tag": "noai, noimageai"}, "", "x_robots_noai"),
        ({"TDM-Reservation": "1"}, "", "tdm_reservation"),
        ({"Content-Usage": "train-ai=n"}, "", "content_usage"),
        ({}, '<meta name="robots" content="noai, noimageai">', "meta_noai"),
        ({}, "<META CONTENT='1' NAME='tdm-reservation'>", "tdm_reservation"),
        ({}, '<meta name="robots" content="noindex">', None),
        ({"X-Robots-Tag": "noindex"}, "", None),
    ],
)
def test_page_level_signals(headers, html, reason):
    assert page_opt_out(headers, html) == reason


def test_opt_out_registry_matches_hosts_subdomains_and_prefixes():
    registry = OptOutRegistry.from_lines(["# owners who asked", "example.org", "https://blog.example.com/private/"])
    assert "https://www.example.org/a" in registry
    assert "https://shop.example.org/b" in registry
    assert "https://blog.example.com/private/post" in registry
    assert "https://blog.example.com/public/post" not in registry
    assert PolicyGate(registry=registry).check("https://example.org/", None).reason == "opt_out_registry"


def test_robots_cache_refetches_after_24_hours_and_survives_outages():
    now, answers = [0.0], [(200, b"User-agent: *\nDisallow: /a\n")]

    def fetch(url):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    cache = RobotsCache(fetch, clock=lambda: now[0])
    assert not cache.get("https://x.com/a/1").allowed("https://x.com/a/1", "bot")
    now[0] = 23 * 3600
    assert not cache.get("https://x.com/a/2").allowed("https://x.com/a/2", "bot")  # cached: no fetch
    now[0] = 25 * 3600
    answers += [OSError("unreachable")]
    assert not cache.get("https://x.com/a/3").allowed("https://x.com/a/3", "bot")  # keeps the last good copy
    answers += [(404, b"")]
    assert cache.get("https://x.com/a/4").allowed("https://x.com/a/4", "bot")  # 4xx: everything allowed
    assert RobotsCache(lambda url: (503, b"")).get("https://y.com/").disallow_all
