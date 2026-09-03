"""Question routing: is this a numbers question (metrics layer) or a documents question (retrieval)?

This version is rule-based so it runs offline and is fully deterministic. Exercise 3.3
replaces `parse_metric_question` with a model call that returns the same `MetricQuery`
dataclass via structured output. Notice what changes and what does not: the model may
parse the question, but the computation still happens in metrics.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..metrics import Period, quarter, last_n_weeks, load_locations, load_funnel

METRIC_ALIASES = {
    "cpl": ["cost per lead", "cpl"],
    "cps": ["cost per show", "cps"],
    "show_rate": ["show rate"],
    "acceptance_rate": ["acceptance rate", "treatment acceptance", "case acceptance"],
    "production_per_show": ["production per show"],
    "roi": ["roi", "return on"],
    "spend": ["spend", "budget spent"],
    "leads": ["leads", "lead volume", "how many leads"],
    "shows": ["shows", "how many shows"],
    "production": ["production"],
}


@dataclass
class MetricQuery:
    metric: str
    period: Period
    region: str | None = None
    channel: str | None = None
    location_id: str | None = None
    group_by: str | None = None


def _period(q: str) -> Period | None:
    m = re.search(r"\bq([1-4])\s*(20\d\d)\b", q)
    if m:
        return quarter(int(m.group(2)), int(m.group(1)))
    m = re.search(r"\blast\s+(\d+)\s+weeks?\b", q)
    if m:
        return last_n_weeks(int(m.group(1)))
    if "last week" in q:
        return last_n_weeks(1)
    return None


def parse_metric_question(question: str) -> MetricQuery | None:
    q = question.lower()
    metric = None
    for key, names in METRIC_ALIASES.items():
        if any(n in q for n in names):
            metric = key
            break
    period = _period(q)
    if metric is None or period is None:
        return None
    regions = sorted(load_locations()["region"].unique(), key=len, reverse=True)
    region = next((r for r in regions if r.lower() in q), None)
    channels = sorted(load_funnel()["channel"].unique(), key=len, reverse=True)
    channel = next((c for c in channels if c.lower().split(" /")[0] in q), None)
    loc = re.search(r"\b(l\d\d)\b", q)
    group_by = None
    if re.search(r"\bby (region)\b", q):
        group_by = "region"
    elif re.search(r"\bby (channel)\b", q):
        group_by = "channel"
    elif re.search(r"\bby (location)\b", q):
        group_by = "location_id"
    return MetricQuery(metric, period, region, channel, loc.group(1).upper() if loc else None, group_by)
