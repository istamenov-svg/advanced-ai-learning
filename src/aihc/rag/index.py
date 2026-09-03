"""BM25 lexical index, pure Python, no dependencies.

Why BM25 first and not embeddings: it is deterministic, inspectable, and for a corpus of
internal policy documents it is competitive. Exercise 3.4 in the module README adds a dense
(embedding) index and a hybrid merge so you can measure whether it helps on the golden set.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .chunk import Chunk, load_chunks

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or of to in for on with is are be by at as this that it from not no".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.text) for c in chunks]
        self.avgdl = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        self.df: Counter = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.n = len(self.docs)

    def _idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        return math.log(1 + (self.n - n_t + 0.5) / (n_t + 0.5))

    def score(self, query: str, i: int) -> float:
        q = tokenize(query)
        d = self.docs[i]
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            num = tf[t] * (self.k1 + 1)
            den = tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            s += self._idf(t) * num / den
        return s

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        scored = [(self.chunks[i], self.score(query, i)) for i in range(self.n)]
        scored = [x for x in scored if x[1] > 0]
        # tie-break on chunk id so ranking is deterministic
        scored.sort(key=lambda x: (-x[1], x[0].id))
        return scored[:k]


_index: BM25Index | None = None


def get_index() -> BM25Index:
    global _index
    if _index is None:
        _index = BM25Index(load_chunks())
    return _index
