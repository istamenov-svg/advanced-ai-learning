"""Scheduled insights agent.

Run:
  python -m aihc.agent.insights_agent --dry-run          # no model; deterministic report from the tools alone
  python -m aihc.agent.insights_agent                    # model-driven: agent loop with tools
  python -m aihc.agent.insights_agent --weeks 1 --as-of 2026-06-29

What makes it "permanent":
  - state.json holds the last run's key figures; the report leads with what changed
  - runs.jsonl logs every run: tokens, tool calls, duration, cost ceiling hit or not
  - it is idempotent per (as_of, weeks): re-running the same period overwrites the same report file
  - a scheduler (cron, Windows Task Scheduler, GitHub Actions) invokes it; see modules/04-permanent-agent
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

from .. import config
from ..llm import tool_loop
from ..metrics import dataset_fingerprint
from .tools import TOOLS, dispatch, _periods

SYSTEM = """You are the weekly marketing analytics agent for a multi-location dental group.
Produce a Markdown report for the regional marketing managers. Rules:
- Every number must come from a tool result. Never compute, extrapolate, or estimate.
- Lead with what changed versus the prior period. Skip metrics that moved less than 5%.
- For each flagged anomaly, check search_docs for a policy that applies (e.g. budget hold rules) and cite the chunk id.
- End with at most three recommended actions, each tied to a specific number and, where relevant, a cited policy.
- Under 400 words. No preamble."""


def load_state() -> dict:
    return json.loads(config.STATE_FILE.read_text()) if config.STATE_FILE.exists() else {}


def save_state(state: dict) -> None:
    config.STATE_FILE.write_text(json.dumps(state, indent=2))


def log_run(rec: dict) -> None:
    with config.RUN_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _pct(v):
    return "n/a" if v is None else f"{v:+.1f}%"


def dry_run_report(weeks: int, as_of: str | None) -> str:
    """Deterministic report built directly from the tools. No model involved.

    This is also the fallback if the model call fails on a scheduled run: a correct,
    less eloquent report beats no report.
    """
    cur, prev = _periods(weeks, as_of)
    cmp_total = dispatch("compare_periods", {"weeks": weeks}, as_of)["totals"]
    by_region = dispatch("compare_periods", {"weeks": weeks, "group_by": "region"}, as_of)["rows"]
    flags = dispatch("anomalies", {"weeks": weeks, "threshold_pct": 15}, as_of)
    policy = dispatch("search_docs", {"query": "paid social CPL budget hold four consecutive weeks"}, as_of)

    lines = [f"# Marketing insights: {cur.label}", "",
             f"Compared with {prev.label}. Data fingerprint `{dataset_fingerprint()}`. Generated without a model (dry run).", "",
             "## Totals", "", "| Metric | Current | Previous | Change |", "|---|---|---|---|"]
    for k in ("spend", "leads", "shows", "cpl", "cps", "show_rate", "roi"):
        d = cmp_total[k]
        lines.append(f"| {k} | {d['current']} | {d['previous']} | {_pct(d['pct_change'])} |")
    lines += ["", "## By region (CPL / CPS change)", ""]
    for r, d in by_region.items():
        lines.append(f"- {r}: CPL {_pct(d['cpl']['pct_change'])}, CPS {_pct(d['cps']['pct_change'])}, leads {_pct(d['leads']['pct_change'])}")
    lines += ["", "## Flags (>=15% move in CPL or CPS)", ""]
    if not flags:
        lines.append("- none")
    for f in flags:
        lines.append(f"- {f['cell']}: {f['metric']} {f['previous']} -> {f['current']} ({_pct(f['pct_change'])})")
    if policy:
        lines += ["", "## Applicable policy", "", f"[{policy[0]['id']}] {policy[0]['text'][:300]}..."]
    lines += ["", "## Recommended actions", "",
              "- (dry run) Review each flagged cell against the policy above; the model-driven run writes these."]
    return "\n".join(lines)


def run(weeks: int, as_of: str | None, dry_run: bool) -> str:
    t0 = time.time()
    cur, prev = _periods(weeks, as_of)
    state = load_state()
    usage = {"input_tokens": 0, "output_tokens": 0}

    def log(u):
        usage["input_tokens"] += u["input_tokens"]
        usage["output_tokens"] += u["output_tokens"]

    trace = []
    if dry_run or config.STUB:
        report = dry_run_report(weeks, as_of)
        mode = "dry_run"
    else:
        prior = state.get("last_totals")
        user = (f"Report period: {cur.label}. Prior period: {prev.label}. Use weeks={weeks} in every tool call.\n"
                f"Previous run's totals for reference (already reported, do not repeat unless changed): {json.dumps(prior)}")
        try:
            report, trace = tool_loop(SYSTEM, user, TOOLS, lambda n, a: dispatch(n, a, as_of), log=log)
            mode = "model"
        except Exception as e:   # scheduled runs must not die silently
            report = dry_run_report(weeks, as_of) + f"\n\n> Model run failed ({type(e).__name__}: {e}); dry-run report substituted."
            mode = "fallback"

    config.REPORTS_DIR.mkdir(exist_ok=True)
    out = config.REPORTS_DIR / f"{cur.end}_w{weeks}.md"
    out.write_text(report, encoding="utf-8")

    totals = dispatch("query_metrics", {"weeks": weeks}, as_of)["totals"]
    save_state({"last_run": dt.datetime.now(dt.timezone.utc).isoformat(), "last_period": cur.label,
                "last_totals": totals, "fingerprint": dataset_fingerprint()})
    log_run({"ts": time.time(), "period": cur.label, "mode": mode, "seconds": round(time.time() - t0, 2),
             "tool_calls": len(trace), **usage, "report": str(out)})
    print(f"[{mode}] wrote {out}  tool_calls={len(trace)}  tokens={usage}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=4)
    ap.add_argument("--as-of", default=None, help="ISO date; use the latest week on or before this")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(a.weeks, a.as_of, a.dry_run)
