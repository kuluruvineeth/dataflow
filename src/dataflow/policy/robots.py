from dataclasses import dataclass, field
from urllib.parse import quote, unquote, urlsplit

MAX_BYTES = 500 * 1024
SAFE = "/:@!$&'()*+,;=?-._~"
KEYS = {"user-agent": "user-agent", "useragent": "user-agent", "user agent": "user-agent", "allow": "allow",
        "disallow": "disallow", "content-signal": "content-signal", "content-usage": "content-usage"}  # fmt: skip


def canonical(text: str) -> str:
    """Percent-encoding normalised so that "%7E", "~" and "ツ" vs "%E3%83%84" compare equal."""
    return quote(unquote(text), safe=SAFE)


@dataclass(frozen=True)
class Rule:
    allow: bool
    path: str
    parts: tuple[str, ...]
    anchored: bool

    @classmethod
    def parse(cls, allow: bool, path: str) -> "Rule":
        anchored = path.endswith("$")
        body = path[:-1] if anchored else path
        return cls(allow, canonical(path), tuple(canonical(part) for part in body.split("*")), anchored)

    def matches(self, target: str) -> bool:
        # greedy wildcard matching: linear time, unlike a regex of ".*" pieces, which some real rules make explode
        first, *middle, last = self.parts if len(self.parts) > 1 else (self.parts[0], self.parts[0])
        if len(self.parts) == 1:
            return target == first if self.anchored else target.startswith(first)
        if not target.startswith(first):
            return False
        position = len(first)
        for part in middle:
            found = target.find(part, position)
            if found < 0:
                return False
            position = found + len(part)
        if self.anchored:
            return target.endswith(last) and len(target) - len(last) >= position
        return target.find(last, position) >= 0


@dataclass
class Group:
    agents: list[str] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    signals: list[tuple[str, str]] = field(default_factory=list)


def product_token(agent: str) -> str:
    return agent.split("/")[0].strip().lower()


@dataclass
class Robots:
    """A parsed robots.txt (RFC 9309), plus the AI-use signals some sites put in it."""

    groups: list[Group] = field(default_factory=list)
    allow_all: bool = False
    disallow_all: bool = False

    @classmethod
    def parse(cls, text: str | bytes) -> "Robots":
        if isinstance(text, bytes):
            text = text[:MAX_BYTES].decode("utf-8", errors="replace")
        groups: list[Group] = []
        current: Group | None = None
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = KEYS.get(key.lower())
            if key == "user-agent":
                if current is None or current.rules or current.signals:
                    current = Group()
                    groups.append(current)
                current.agents.append(product_token(value))
            elif current is None:
                continue
            elif key in ("allow", "disallow"):
                if value:
                    current.rules.append(Rule.parse(key == "allow", value))
            elif key:
                current.signals.append((key, value))
        return cls(groups)

    @classmethod
    def from_status(cls, status: int) -> "Robots":
        """RFC 9309 section 2.3.1: 4xx means no restrictions; 5xx or no answer means a complete disallow."""
        if 400 <= status < 500:
            return cls(allow_all=True)
        return cls(disallow_all=True)

    def groups_for(self, agent: str, fallback: bool = True) -> list[Group]:
        token = product_token(agent)
        matched = [group for group in self.groups if token in group.agents]
        if not matched and fallback:
            matched = [group for group in self.groups if "*" in group.agents]
        return matched

    def allowed(self, url: str, agent: str, fallback: bool = True) -> bool:
        if self.allow_all:
            return True
        if self.disallow_all:
            return False
        parts = urlsplit(url)
        target = canonical(parts.path or "/") + (f"?{canonical(parts.query)}" if parts.query else "")
        if target == "/robots.txt":
            return True
        rules = [rule for group in self.groups_for(agent, fallback) for rule in group.rules if rule.matches(target)]
        if not rules:
            return True
        best = max(rules, key=lambda rule: (len(rule.path), rule.allow))
        return best.allow

    def signals_for(self, agent: str) -> list[tuple[str, str]]:
        return [signal for group in self.groups_for(agent) for signal in group.signals]
