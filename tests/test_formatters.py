from dataflow.data import Document
from dataflow.pipeline.formatters import BaseFormatter, FTFYFormatter, PIIFormatter, SymbolLinesFormatter
from dataflow.pipeline.formatters.pii import EMAIL_REPLACEMENTS, IP_REPLACEMENTS

TELUGU = "తెలుగు భాష “చాలా” అందమైనది."


class Upper(BaseFormatter):
    def format(self, text: str) -> str:
        return text.upper().strip("X")


def test_base_formatter_counts_changes_and_drops_emptied_documents():
    step = Upper()
    docs = list(step.run(iter([Document("abc", "0"), Document("ABC", "1"), Document("xx", "2")])))
    assert [(doc.id, doc.text) for doc in docs] == [("0", "ABC"), ("1", "ABC")]
    metrics = step.stats.metrics
    assert (metrics["total"].total, metrics["changed"].total, metrics["dropped_empty"].total) == (3, 2, 1)


def test_ftfy_repairs_telugu_mojibake_and_keeps_typography():
    broken = TELUGU.encode("utf-8").decode("latin-1")
    assert broken.startswith("à°")
    assert FTFYFormatter().format(broken) == TELUGU
    assert FTFYFormatter().format("“curly” ’quotes’ ﬁ") == "“curly” ’quotes’ ﬁ"


def test_ftfy_composes_telugu_vowel_signs_with_nfc():
    decomposed = "క\u0c46\u0c56"
    assert FTFYFormatter().format(decomposed) == "క\u0c48"
    assert FTFYFormatter(normalization=None).format(decomposed) == decomposed


def test_pii_replaces_emails_and_public_ips_consistently():
    step = PIIFormatter()
    text = step.format("రాయండి ravi.kumar@gmail.com, సర్వర్ 49.204.10.3.")
    email, ip = text.split()[1].rstrip(","), text.split()[3].rstrip(".")
    assert email in EMAIL_REPLACEMENTS and ip in IP_REPLACEMENTS
    assert text.endswith(".")
    assert step.format("again ravi.kumar@gmail.com") == f"again {email}"


def test_pii_keeps_private_ips_and_numbers_that_are_not_ips():
    step = PIIFormatter()
    for text in ["home 192.168.1.10", "loop 127.0.0.1", "version 1.2.3.4.5", "serial 1234.5.6.78", "x 10.0.0.256"]:
        assert step.format(text) == text
    assert PIIFormatter(public_ips_only=False).format("home 192.168.1.10") != "home 192.168.1.10"


def test_symbol_lines_removed_or_turned_into_one_paragraph_break():
    assert SymbolLinesFormatter().format("Title\n* * *\n---\nText") == "Title\nText"
    pipes = SymbolLinesFormatter("|", paragraph_break=True)
    assert pipes.format("a\n|\n| |\nb\n||\nc") == "a\n\nb\n\nc"
    assert pipes.format("a | b\n   \nc") == "a | b\n   \nc"
