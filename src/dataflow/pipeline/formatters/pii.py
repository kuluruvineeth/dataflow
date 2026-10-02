import ipaddress
import re

from dataflow.pipeline.formatters.base import BaseFormatter
from dataflow.utils.hashing import hash64

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
)
OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_PATTERN = re.compile(rf"(?<!\d)(?<!\d\.){OCTET}(?:\.{OCTET}){{3}}(?!\.?\d)")
EMAIL_REPLACEMENTS = ("email@example.com", "firstname.lastname@example.org")
IP_REPLACEMENTS = ("192.0.2.1", "198.51.100.23", "203.0.113.42")


def pick(original: str, replacements: tuple[str, ...]) -> str:
    return replacements[hash64(original) % len(replacements)]


class PIIFormatter(BaseFormatter):
    name = "pii"

    def __init__(self, emails: bool = True, ips: bool = True, public_ips_only: bool = True):
        self.emails = emails
        self.ips = ips
        self.public_ips_only = public_ips_only

    def replace_ip(self, match: re.Match) -> str:
        address = match.group()
        if self.public_ips_only and not ipaddress.ip_address(address).is_global:
            return address
        return pick(address, IP_REPLACEMENTS)

    def format(self, text: str) -> str:
        if self.emails:
            text = EMAIL_PATTERN.sub(lambda match: pick(match.group(), EMAIL_REPLACEMENTS), text)
        if self.ips:
            text = IPV4_PATTERN.sub(self.replace_ip, text)
        return text
