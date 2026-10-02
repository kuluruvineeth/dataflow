from collections.abc import Iterable

from dataflow.pipeline.formatters.base import BaseFormatter
from dataflow.utils.text import PUNCTUATION_SET


class SymbolLinesFormatter(BaseFormatter):
    name = "symbol lines"

    def __init__(self, symbols: Iterable[str] | None = None, paragraph_break: bool = False):
        self.symbols = set(symbols) if symbols is not None else PUNCTUATION_SET
        self.paragraph_break = paragraph_break

    def is_symbol_line(self, line: str) -> bool:
        return line.strip() != "" and all(char in self.symbols or char.isspace() for char in line)

    def format(self, text: str) -> str:
        lines, in_run = [], False
        for line in text.split("\n"):
            if not self.is_symbol_line(line):
                lines.append(line)
                in_run = False
            elif not in_run:
                if self.paragraph_break:
                    lines.append("")
                in_run = True
        return "\n".join(lines)
