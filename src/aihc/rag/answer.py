"""Grounded answering with citations and a repeatability guarantee.

Flow:
  question -> router
    numbers question -> metrics.query_metrics -> fact block -> model narrates -> cached
    documents question -> BM25 top-k -> prompt with chunk IDs -> model answers with
                          [chunk-id] citations -> citations verified -> cached

Run: python -m aihc.rag.answer "What was cost per show in Q2 2026 for the Tampa region?"
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict

from .. import config
from ..llm import complete
from ..metrics import query_metrics, dataset_fingerprint
from .index import get_index
from .router import parse_metric_question

SYSTEM_DOCS = """You answer questions about a dental group's marketing operations using ONLY the provided document chunks.
Rules:
- Cite every claim with the chunk id in square brackets, e.g. [paid_social_policy#1a2b3c4d].
- If the chunks do not contain the answer, reply exactly: "Not in the provided documents." and nothing else.
- Do not add background knowledge. Do not compute numbers. Keep it under 120 words."""

SYSTEM_METRICS = """You write one or two plain sentences reporting metrics from a FACT BLOCK.
Rules:
- Use only the numbers in the fact block, verbatim, with the units shown. Never compute or round further.
- State the period and filters exactly as given.
- No advice, no speculation."""


@dataclass
class Answer:
    question: str
    route: str               # "metrics" | "docs"
    text: str
    citations: list[str]
    citations_valid: bool
    facts: dict | None
    cache_hit: bool
    model: str
    fingerprint: str


def _fmt(metric: str, v):
    if v is None:
        return "n/a"
    if metric in ("cpl", "cps", "production_per_show"):
        return f"${v:,.2f}"
    if metric in ("show_rate", "acceptance_rate"):
        return f"{v * 100:.1f}%"
    if metric == "roi":
        return f"{v:.2f}x"
    if metric in ("spend", "production"):
        return f"${v:,.2f}"
    return f"{v:,}"


def answer_metrics(question: str, use_cache: bool = True) -> Answer:
    mq = parse_metric_question(question)
    facts = query_metrics(mq.period, mq.region, mq.channel, mq.location_id, mq.group_by)
    # Fact block: only the requested metric, pre-formatted, so the model has nothing to compute.
    if mq.group_by:
        values = {k: _fmt(mq.metric, r[mq.metric]) for k, r in facts["rows"].items()}
    else:
        values = {"total": _fmt(mq.metric, facts["totals"][mq.metric])}
    block = {"metric": mq.metric, "definition": facts["definitions"][mq.metric],
             "period": mq.period.label, "filters": {k: v for k, v in facts["filters"].items() if v},
             "values": values}
    user = f"QUESTION: {question}\n\nFACT BLOCK:\n{json.dumps(block, indent=2)}"

    def stub(_s, _u):
        filt = ", ".join(f"{k}={v}" for k, v in block["filters"].items()) or "all locations and channels"
        if mq.group_by:
            parts = "; ".join(f"{k}: {v}" for k, v in values.items())
            return f"{mq.metric} for {block['period']} ({filt}), by {mq.group_by}: {parts}."
        return f"{mq.metric} for {block['period']} ({filt}) was {values['total']}."

    res = complete(SYSTEM_METRICS, user, use_cache=use_cache, stub_fn=stub)
    return Answer(question, "metrics", res.text, [], True, block, res.cache_hit, res.model,
                  dataset_fingerprint())


MIN_SCORE = 3.0   # BM25 relevance floor; below this we do not even show chunks to the model.
                  # Exercise 3.2: tune this against the golden set and watch the abstain/recall trade-off.


def answer_docs(question: str, k: int = 4, use_cache: bool = True) -> Answer:
    hits = [(c, s) for c, s in get_index().search(question, k=k) if s >= MIN_SCORE]
    ids = [c.id for c, _ in hits]
    ctx = "\n\n".join(f"[{c.id}] (from {c.source})\n{c.text}" for c, _ in hits) or "(no chunks matched)"
    user = f"DOCUMENT CHUNKS:\n{ctx}\n\nQUESTION: {question}"

    def stub(_s, _u):
        if not hits:
            return "Not in the provided documents."
        # Offline stand-in for the model: pick the paragraph in the top chunk with the most
        # query-term overlap and cite the chunk. A real model does better; the citation
        # contract is the same.
        from .index import tokenize
        c = hits[0][0]
        qt = set(tokenize(question))
        paras = [p for p in c.text.split("\n\n") if not p.startswith("#")] or [c.text]
        best = max(paras, key=lambda p: len(qt & set(tokenize(p))))
        return f"{best[:400]} [{c.id}]"

    res = complete(SYSTEM_DOCS, user, use_cache=use_cache, stub_fn=stub)
    cited = re.findall(r"\[([a-z0-9_\-]+#[0-9a-f]{8})\]", res.text)
    valid = all(c in ids for c in cited) and (bool(cited) or res.text.strip().startswith("Not in the provided"))
    return Answer(question, "docs", res.text, cited, valid, None, res.cache_hit, res.model, dataset_fingerprint())


def answer(question: str, use_cache: bool = True) -> Answer:
    if parse_metric_question(question):
        return answer_metrics(question, use_cache=use_cache)
    return answer_docs(question, use_cache=use_cache)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What was cost per show in Q2 2026 for the Tampa region?"
    a = answer(q)
    print(json.dumps(asdict(a), indent=2, default=str))
