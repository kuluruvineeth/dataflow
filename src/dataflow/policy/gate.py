import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from dataflow.policy.robots import Robots

AI_TRAINING_AGENTS = ("GPTBot", "ClaudeBot", "CCBot", "Google-Extended", "Applebot-Extended", "Ai2Bot")
OUR_AGENT = "AksharaBot"
DAY = 24 * 3600
META = re.compile(r"<meta\s[^>]*>", re.IGNORECASE)
ATTRIBUTE = re.compile(r"""(name|content|http-equiv)\s*=\s*["']([^"']*)["']""", re.IGNORECASE)


def says_no_training(value: str) -> bool:
    """True for Content-Signal "ai-train=no" and aipref Content-Usage "train-ai=n" (any spacing or case)."""
    terms = dict(
        (key.strip().lower(), val.strip().lower())
        for key, _, val in (part.partition("=") for part in re.split(r"[,;\s]+", value))
        if key
    )
    return terms.get("ai-train") == "no" or terms.get("train-ai") in ("n", "no")


def page_opt_out(headers: dict[str, str], html: str = "") -> str | None:
    """Opt-out signals carried by a page itself: HTTP headers and <meta> tags."""
    lowered = {key.lower(): value for key, value in headers.items()}
    if "noai" in lowered.get("x-robots-tag", "").lower():
        return "x_robots_noai"
    if lowered.get("tdm-reservation", "").strip() == "1":
        return "tdm_reservation"
    if says_no_training(lowered.get("content-usage", "")):
        return "content_usage"
    for tag in META.findall(html[:200_000]):
        attributes = {key.lower(): value for key, value in ATTRIBUTE.findall(tag)}
        name = attributes.get("name", attributes.get("http-equiv", "")).lower()
        content = attributes.get("content", "").lower()
        if name in ("robots", "googlebot") and re.search(r"\bnoai\b", content):
            return "meta_noai"
        if name == "tdm-reservation" and content.strip() == "1":
            return "tdm_reservation"
    return None


@dataclass
class OptOutRegistry:
    """Hosts and URL prefixes whose owners asked to be left out."""

    hosts: set[str] = field(default_factory=set)
    prefixes: tuple[str, ...] = ()

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> "OptOutRegistry":
        hosts, prefixes = set(), []
        for line in lines:
            entry = line.split("#", 1)[0].strip().lower()
            if not entry:
                continue
            if "/" in entry.split("://", 1)[-1]:
                prefixes.append(entry.split("://", 1)[-1])
            else:
                hosts.add(entry.removeprefix("www."))
        return cls(hosts, tuple(prefixes))

    def __contains__(self, url: str) -> bool:
        host = (urlsplit(url).hostname or "").removeprefix("www.")
        if any(host == entry or host.endswith("." + entry) for entry in self.hosts):
            return True
        bare = url.lower().split("://", 1)[-1].removeprefix("www.")
        return any(bare.startswith(prefix.removeprefix("www.")) for prefix in self.prefixes)


@dataclass
class Decision:
    allowed: bool
    reason: str | None = None


class PolicyGate:
    """The sourcing rule: our own token must be allowed, no AI-training token may be blocked, no opt-out signal."""

    def __init__(
        self,
        agent: str = OUR_AGENT,
        ai_agents: Iterable[str] = AI_TRAINING_AGENTS,
        registry: OptOutRegistry | None = None,
    ):
        self.agent = agent
        self.ai_agents = tuple(ai_agents)
        self.registry = registry or OptOutRegistry()

    def check(
        self, url: str, robots: Robots | None, headers: dict[str, str] | None = None, html: str = ""
    ) -> Decision:
        if url in self.registry:
            return Decision(False, "opt_out_registry")
        if robots is not None:
            if not robots.allowed(url, self.agent):
                return Decision(False, "robots_disallow")
            for agent in self.ai_agents:
                if robots.groups_for(agent, fallback=False) and not robots.allowed(url, agent, fallback=False):
                    return Decision(False, "ai_agent_blocked")
            for _, value in robots.signals_for(self.agent):
                if says_no_training(value):
                    return Decision(False, "robots_content_signal")
        if reason := page_opt_out(headers or {}, html):
            return Decision(False, reason)
        return Decision(True)


class RobotsCache:
    """robots.txt per host, refetched after 24 hours (RFC 9309 section 2.4).

    `fetch(url)` returns (status, body) after following redirects; it raises OSError when the host is unreachable.
    A host that stays unreachable keeps its last good copy, or a complete disallow if there never was one.
    """

    def __init__(
        self, fetch: Callable[[str], tuple[int, bytes]], ttl: float = DAY, clock: Callable[[], float] = time.time
    ):
        self.fetch = fetch
        self.ttl = ttl
        self.clock = clock
        self._entries: dict[str, tuple[float, Robots]] = {}

    def get(self, url: str) -> Robots:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        cached = self._entries.get(origin)
        if cached and self.clock() - cached[0] < self.ttl:
            return cached[1]
        try:
            status, body = self.fetch(origin + "/robots.txt")
        except OSError:
            return cached[1] if cached else Robots(disallow_all=True)
        if 200 <= status < 300:
            robots = Robots.parse(body)
        elif status >= 500 and cached:
            return cached[1]
        else:
            robots = Robots.from_status(status)
        self._entries[origin] = (self.clock(), robots)
        return robots
