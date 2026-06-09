"""Ingest-side guard for FoT reasoning traces (adversarial-trace-injection control).

The server validates each incoming trace's shape and screens for prompt-injection markers before
it is fed to the distiller, so a malicious client cannot poison the shared insight library with
oversized, malformed, or instruction-injecting "insights". Torch-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from fedlearn.fot.model import ReasoningTrace

# Lowercase substrings that have no place in an abstracted reasoning insight and strongly suggest
# an attempt to steer a downstream LLM (the distiller) or exfiltrate.
INJECTION_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "disregard the above",
    "disregard previous",
    "system prompt",
    "you are now",
    "reveal your",
    "exfiltrate",
    "delete all",
    "drop table",
    "rm -rf",
)


@dataclass
class TraceValidator:
    max_insights: int = 64
    max_chars_per_insight: int = 8000

    def problems(self, trace: ReasoningTrace) -> List[str]:
        """Return all problems with a trace (empty == safe to ingest)."""
        probs = list(trace.validate())
        if len(trace.insights) > self.max_insights:
            probs.append(f"too many insights ({len(trace.insights)} > {self.max_insights})")
        for key, text in trace.insights.items():
            if isinstance(text, str):
                if len(text) > self.max_chars_per_insight:
                    probs.append(f"insight '{key}' exceeds {self.max_chars_per_insight} chars")
                low = text.lower()
                for marker in INJECTION_MARKERS:
                    if marker in low:
                        probs.append(f"insight '{key}' contains injection marker '{marker}'")
                        break
        return probs

    def is_safe(self, trace: ReasoningTrace) -> bool:
        return not self.problems(trace)
