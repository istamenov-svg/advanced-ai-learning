"""Eval harness (the testing sense of "harness").

Runs every golden question through the answer pipeline and scores:
  metrics questions : exact match on the formatted value(s) in the fact block
  docs questions    : (a) expected source retrieved, (b) answer cites a valid chunk,
                      (c) required phrases present, (d) abstains when it should
  all questions     : repeatability = asking twice yields byte-identical text

Run:  python -m aihc.evals.run_evals            (uses cache; fast)
      python -m aihc.evals.run_evals --no-cache  (forces model calls; what CI should do on prompt changes)

Exit code is non-zero if any gate fails, so this can be a CI step (see .github/workflows).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..rag.answer import answer

GOLDEN = Path(__file__).with_name("golden.jsonl")
GATES = {"metrics_exact_match": 0.95, "citation_validity": 1.0, "retrieval_hit": 0.9, "repeatability": 1.0,
         "abstain_ok": 1.0}


def load_golden() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


def score_one(item: dict, use_cache: bool) -> dict:
    a1 = answer(item["question"], use_cache=use_cache)
    a2 = answer(item["question"], use_cache=True)   # second ask should hit cache
    r = {"id": item["id"], "route": a1.route, "repeatable": a1.text == a2.text, "cache_hit_second": a2.cache_hit}
    if item["type"] == "metrics":
        vals = a1.facts["values"] if a1.facts else {}
        if "expect_values" in item:
            r["exact"] = vals == item["expect_values"]
        else:
            r["exact"] = vals.get("total") == item["expect_value"]
        r["route_ok"] = a1.route == "metrics"
        r["got"] = vals
    else:
        r["route_ok"] = a1.route == "docs"
        if item.get("expect_abstain"):
            r["abstained"] = a1.text.strip().startswith("Not in the provided documents")
            r["citation_valid"] = a1.citations_valid
            r["retrieval_hit"] = True   # nothing to retrieve
        else:
            r["retrieval_hit"] = any(c.split("#")[0] == item["expect_source"] for c in a1.citations)
            r["citation_valid"] = a1.citations_valid and bool(a1.citations)
            r["phrases"] = all(p.lower() in a1.text.lower() for p in item.get("must_contain", []))
        r["got"] = a1.text[:160]
    r["model"] = a1.model
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)
    rows = [score_one(it, use_cache=not args.no_cache) for it in load_golden()]

    m = [r for r in rows if r["route"] == "metrics"]
    d = [r for r in rows if r["route"] == "docs"]
    summary = {
        "n": len(rows),
        "route_accuracy": sum(r["route_ok"] for r in rows) / len(rows),
        "metrics_exact_match": sum(r.get("exact", False) for r in m) / max(1, len(m)),
        "retrieval_hit": sum(r.get("retrieval_hit", False) for r in d) / max(1, len(d)),
        "citation_validity": sum(r.get("citation_valid", False) for r in d) / max(1, len(d)),
        "phrase_match": sum(r.get("phrases", True) for r in d) / max(1, len(d)),
        "abstain_ok": float(all(r.get("abstained", True) for r in d)),
        "repeatability": sum(r["repeatable"] for r in rows) / len(rows),
        "models": sorted({r["model"] for r in rows}),
    }
    failed = [k for k, thr in GATES.items() if summary[k] < thr]
    if args.verbose:
        for r in rows:
            print(json.dumps(r, default=str))
    print(json.dumps(summary, indent=2))
    print("GATES:", "PASS" if not failed else f"FAIL {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
