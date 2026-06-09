"""Pre-egress empirical-privacy guard for FoT reasoning traces.

FoT shares abstracted text, not raw problems — but an LLM can still echo verbatim chunks of the
local input. The LeakageScanner measures how much of an insight's n-gram content appears verbatim
in the client's LOCAL raw corpus (the task prompts/answers that must not leave the device); the
TraceRedactor drops insights over a threshold BEFORE upload.

This is an EMPIRICAL control (the paper's prompt-reconstruction concern), NOT a structural privacy
guarantee. It must never be described with DeComFL's "raw data never leaves" / structural-privacy
language. Torch-free.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def _ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass
class RedactionResult:
    kept: Dict[str, str] = field(default_factory=dict)
    dropped: Dict[str, float] = field(default_factory=dict)  # insight_key -> overlap that tripped


class LeakageScanner:
    """Measures verbatim n-gram overlap of a candidate string against a local raw corpus."""

    def __init__(self, raw_corpus: Sequence[str], *, n: int = 4) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self._corpus_docs: List[List[str]] = [_tokens(doc) for doc in raw_corpus]
        self._corpus_ngrams: Set[Tuple[str, ...]] = set()
        for toks in self._corpus_docs:
            self._corpus_ngrams |= _ngrams(toks, n)

    def _contained(self, toks: List[str]) -> bool:
        """Is the token sequence a verbatim contiguous run inside any corpus doc?"""
        m = len(toks)
        for doc in self._corpus_docs:
            for i in range(len(doc) - m + 1):
                if doc[i : i + m] == toks:
                    return True
        return False

    def overlap(self, text: str) -> float:
        """Fraction of the text's n-grams that appear verbatim in the local corpus (0.0..1.0)."""
        toks = _tokens(text)
        if not toks:
            return 0.0
        grams = _ngrams(toks, self.n)
        if not grams:
            # Candidate is shorter than n tokens: n-gram overlap is undefined. Fall back to a
            # verbatim-containment check so a short secret echoed verbatim is NOT treated as 0.0
            # overlap and waved through (fail-closed for the small-but-sensitive case).
            return 1.0 if self._contained(toks) else 0.0
        hits = sum(1 for g in grams if g in self._corpus_ngrams)
        return hits / len(grams)


class TraceRedactor:
    """Drops insights whose verbatim overlap with the local raw corpus exceeds max_overlap."""

    def __init__(self, scanner: LeakageScanner, *, max_overlap: float = 0.5) -> None:
        if not 0.0 <= max_overlap <= 1.0:
            raise ValueError("max_overlap must be in [0, 1]")
        self.scanner = scanner
        self.max_overlap = max_overlap

    def redact(self, insights: Dict[str, str]) -> RedactionResult:
        result = RedactionResult()
        for key, text in insights.items():
            ov = self.scanner.overlap(text)
            if ov > self.max_overlap:
                result.dropped[key] = ov
            else:
                result.kept[key] = text
        return result
