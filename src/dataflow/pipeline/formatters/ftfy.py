import ftfy

from dataflow.pipeline.formatters.base import BaseFormatter


class FTFYFormatter(BaseFormatter):
    name = "ftfy"

    def __init__(self, normalization: str | None = "NFC", unescape_html: str | bool = "auto"):
        self.config = ftfy.TextFixerConfig(
            unescape_html=unescape_html,
            fix_latin_ligatures=False,
            fix_character_width=False,
            uncurl_quotes=False,
            fix_line_breaks=False,
            normalization=normalization,
        )

    def format(self, text: str) -> str:
        return ftfy.fix_text(text, config=self.config)
