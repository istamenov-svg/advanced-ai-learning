"""Tool definitions for the insights agent.

Each tool is (a) a JSON schema the model sees and (b) a Python function the harness runs.
All of them wrap the deterministic metrics layer or the retriever. None of them let the
model do arithmetic.
"""
from __future__ import annotations

from ..metrics import Period, query_metrics, compare_periods, anomalies, last_n_weeks
from ..rag.index import get_index

TOOLS = [
    {
        "name": "query_metrics",
        "description": "Aggregate funnel metrics (spend, leads, shows, CPL, CPS, show rate, acceptance rate, production, ROI) for a period, with optional region/channel filters and grouping.",
        "input_schema": {
            "type": "object",
            "properties": {
                "weeks": {"type": "integer", "description": "How many trailing weeks to aggregate", "default": 4},
                "region": {"type": "string"},
                "channel": {"type": "string"},
                "group_by": {"type": "string", "enum": ["region", "channel", "location_id"]},
            },
        },
    },
    {
        "name": "compare_periods",
        "description": "Compare the trailing N weeks with the N weeks before that. Returns current, previous and percent change per metric.",
        "input_schema": {
            "type": "object",
            "properties": {
                "weeks": {"type": "integer", "default": 4},
                "region": {"type": "string"},
                "channel": {"type": "string"},
                "group_by": {"type": "string", "enum": ["region", "channel", "location_id"]},
            },
        },
    },
    {
        "name": "anomalies",
        "description": "Region x channel cells whose CPL or CPS moved more than a threshold percent versus the prior period.",
        "input_schema": {
            "type": "object",
            "properties": {"weeks": {"type": "integer", "default": 4},
                           "threshold_pct": {"type": "number", "default": 15}},
        },
    },
    {
        "name": "search_docs",
        "description": "Search internal policy and playbook documents. Returns chunk ids and text; cite the ids.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
]


def _periods(weeks: int, as_of: str | None = None) -> tuple[Period, Period]:
    cur = last_n_weeks(weeks, as_of)
    prev = last_n_weeks(weeks * 2, as_of)
    prev = Period(prev.start, _week_before(cur.start), f"previous {weeks} weeks ending {_week_before(cur.start)}")
    return cur, prev


def _week_before(iso: str) -> str:
    import datetime as dt
    return (dt.date.fromisoformat(iso) - dt.timedelta(days=7)).isoformat()


def dispatch(name: str, args: dict, as_of: str | None = None):
    weeks = int(args.get("weeks", 4))
    if name == "query_metrics":
        cur, _ = _periods(weeks, as_of)
        return query_metrics(cur, args.get("region"), args.get("channel"), group_by=args.get("group_by"))
    if name == "compare_periods":
        cur, prev = _periods(weeks, as_of)
        return compare_periods(cur, prev, args.get("region"), args.get("channel"), group_by=args.get("group_by"))
    if name == "anomalies":
        cur, prev = _periods(weeks, as_of)
        return anomalies(cur, prev, float(args.get("threshold_pct", 15)))
    if name == "search_docs":
        return [{"id": c.id, "source": c.source, "text": c.text} for c, _ in get_index().search(args["query"], 3)]
    raise ValueError(f"unknown tool {name}")
