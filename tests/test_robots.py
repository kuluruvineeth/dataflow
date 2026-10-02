import pytest

from dataflow.policy.robots import Robots

# RFC 9309, section 5.1
EXAMPLE = """
User-Agent: *
Disallow: *.gif$
Disallow: /example/
Allow: /publications/

User-Agent: foobot
Disallow:/
Allow:/example/page.html
Allow:/example/allowed.gif

User-Agent: barbot
User-Agent: bazbot
Disallow: /example/page.html

User-Agent: quxbot

EOF
"""
SITE = "https://www.example.com"


@pytest.mark.parametrize(
    ("agent", "path", "allowed"),
    [
        ("foobot", "/example/page.html", True),
        ("foobot", "/example/allowed.gif", True),
        ("foobot", "/", False),
        ("foobot", "/other", False),
        ("barbot", "/example/page.html", False),
        ("bazbot", "/example/page.html", False),
        ("barbot", "/example/other.gif", True),
        ("quxbot", "/example/page.html", True),
        ("otherbot", "/publications/paper.html", True),
        ("otherbot", "/example/x", False),
        ("otherbot", "/images/cat.gif", False),
        ("otherbot", "/images/cat.gif?size=2", True),
        ("FooBot/2.1", "/example/page.html", True),
    ],
)
def test_rfc_9309_example(agent, path, allowed):
    assert Robots.parse(EXAMPLE).allowed(SITE + path, agent) is allowed


def test_longest_match_wins_and_allow_wins_a_tie():
    robots = Robots.parse("User-Agent: foobot\nAllow: /example/page/\nDisallow: /example/page/disallowed.gif\n")
    assert robots.allowed(SITE + "/example/page/", "foobot")
    assert not robots.allowed(SITE + "/example/page/disallowed.gif", "foobot")
    tie = Robots.parse("User-Agent: *\nDisallow: /folder\nAllow: /folder\n")
    assert tie.allowed(SITE + "/folder/page", "anybot")


def test_special_characters_and_percent_encoding():
    robots = Robots.parse(
        "User-Agent: *\nDisallow: /path/file-with-a-%2A.html\nDisallow: /path/foo-%24\nDisallow: /foo/bar/ツ\n"
        "Disallow: /fish*.php\n"
    )
    assert not robots.allowed(SITE + "/path/file-with-a-*.html", "bot")
    assert not robots.allowed(SITE + "/path/foo-$", "bot")
    assert not robots.allowed(SITE + "/foo/bar/%E3%83%84", "bot")
    assert not robots.allowed(SITE + "/fishheads/catfish.php?parameters", "bot")
    assert robots.allowed(SITE + "/Fish.PHP", "bot")


def test_robots_txt_itself_is_always_allowed_and_empty_files_allow_everything():
    assert Robots.parse("User-Agent: *\nDisallow: /\n").allowed(SITE + "/robots.txt", "bot")
    assert Robots.parse("").allowed(SITE + "/anything", "bot")


@pytest.mark.parametrize(("status", "allowed"), [(404, True), (403, True), (500, False), (503, False)])
def test_status_codes(status, allowed):
    assert Robots.from_status(status).allowed(SITE + "/page", "bot") is allowed


def test_groups_for_the_same_agent_are_combined_and_comments_ignored():
    robots = Robots.parse("User-agent: a # first\nDisallow: /x\n\nUser-agent: a\nDisallow: /y # second\n")
    assert not robots.allowed(SITE + "/x", "a") and not robots.allowed(SITE + "/y", "a")


def test_content_signals_are_kept_per_group():
    robots = Robots.parse("User-Agent: *\nContent-Signal: search=yes, ai-train=no\nAllow: /\n")
    assert robots.signals_for("anybot") == [("content-signal", "search=yes, ai-train=no")]


def test_many_wildcards_do_not_blow_up():
    robots = Robots.parse("User-agent: *\nDisallow: /" + "*" * 40 + "x$\n")
    path = "/" + "a" * 5000
    assert robots.allowed(SITE + path, "bot")
    assert not robots.allowed(SITE + path + "x", "bot")


def test_wildcards_match_in_order_and_dollar_anchors_the_end():
    robots = Robots.parse("User-agent: *\nDisallow: /*?*sort=*&page=$\nDisallow: /a*b*c\n")
    assert not robots.allowed(SITE + "/list?x=1&sort=asc&page=", "bot")
    assert robots.allowed(SITE + "/list?x=1&sort=asc&page=2", "bot")
    assert not robots.allowed(SITE + "/a-b-c-d", "bot")
    assert robots.allowed(SITE + "/a-c-b", "bot")
