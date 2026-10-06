"""Memory store and retrieval.

Deliberately dumb: BM25-ish lexical scoring, no embeddings, no vector index.
Layer 1 is an instrument, not a retrieval contribution -- swapping in a better
retriever later changes nothing about the ledger or the credit estimator.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

TOKEN_RE = re.compile(r"[a-z0-9_\-\.]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def count_tokens(text: str) -> int:
    """Cheap proxy for model tokens. Good enough for budgeting a bundle."""
    return max(1, len(text) // 4)


@dataclass
class Entry:
    entry_id: str
    text: str
    # ground truth, used ONLY to validate the instrument; never read by the
    # retriever, the masker, the model, or the detector.
    planted: str = "none"        # "useful" | "distractor" | "irrelevant" | "none"
    # a literal artifact (flag, function, path) the entry recommends. The
    # behavioural signal checks whether it shows up in the agent's output.
    artifact: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


@dataclass
class Scored:
    entry: Entry
    score: float
    rank: int


class MemoryStore:
    def __init__(self, entries: list[Entry]):
        self.entries = {e.entry_id: e for e in entries}
        self._docs = {e.entry_id: tokenize(e.text) for e in entries}
        self._df = Counter()
        for toks in self._docs.values():
            for t in set(toks):
                self._df[t] += 1
        self._n = max(1, len(self._docs))
        self._avg_len = sum(len(d) for d in self._docs.values()) / self._n

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, entry_id: str) -> Entry:
        return self.entries[entry_id]

    def _bm25(self, q_tokens: list[str], entry_id: str, k1=1.5, b=0.75) -> float:
        doc = self._docs[entry_id]
        if not doc:
            return 0.0
        tf = Counter(doc)
        score = 0.0
        for t in set(q_tokens):
            if t not in tf:
                continue
            idf = math.log(1 + (self._n - self._df[t] + 0.5) / (self._df[t] + 0.5))
            denom = tf[t] + k1 * (1 - b + b * len(doc) / self._avg_len)
            score += idf * (tf[t] * (k1 + 1)) / denom
        return score

    def retrieve(self, query: str, top_k: int = 4) -> list[Scored]:
        q = tokenize(query)
        scored = [(eid, self._bm25(q, eid)) for eid in self._docs]
        scored.sort(key=lambda x: (-x[1], x[0]))
        out = []
        for rank, (eid, s) in enumerate(scored[:top_k]):
            out.append(Scored(entry=self.entries[eid], score=float(s), rank=rank))
        return out
