"""Deterministic metrics layer.

Rule of the repo: numbers come from here, never from the model. The model receives the
output of these functions as a "fact block" and writes prose around it.

Everything is plain pandas so the same question against the same files gives the same
answer to the cent. `dataset_fingerprint()` is the hash that keys the answer cache.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from functools import lru_cache

import pandas as pd

from .config import DATA_DIR, DOCS_DIR

METRIC_DEFS = {
    "spend": "sum(spend)",
    "leads": "sum(leads)",
    "shows": "sum(shows)",
    "production": "sum(production_usd)",
    "cpl": "spend / leads",
    "cps": "spend / shows",
    "show_rate": "shows / appointments",
    "acceptance_rate": "plans_accepted / plans_presented",
    "production_per_show": "production_usd / shows",
    "roi": "production_usd / spend",
}


@dataclass(frozen=True)
class Period:
    start: str  # inclusive, ISO date
    end: str    # inclusive, ISO date
    label: str


def quarter(year: int, q: int) -> Period:
    starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    return Period(f"{year}-{starts[q]}", f"{year}-{ends[q]}", f"Q{q} {year}")


def last_n_weeks(n: int, as_of: str | None = None) -> Period:
    f = load_funnel()
    weeks = sorted(f["week_start"].unique())
    if as_of:
        weeks = [w for w in weeks if w <= as_of]
    sel = weeks[-n:]
    return Period(sel[0], sel[-1], f"last {n} weeks ending {sel[-1]}")


def dataset_fingerprint() -> str:
    """Hash of every data file. Changes iff the data changes."""
    h = hashlib.sha256()
    for p in sorted(list(DATA_DIR.glob("*.csv")) + list(DOCS_DIR.glob("*.md"))):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def _load(fp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    loc = pd.read_csv(DATA_DIR / "locations.csv")
    fun = pd.read_csv(DATA_DIR / "funnel_weekly.csv")
    return loc, fun.merge(loc, on="location_id", how="left")


def load_funnel() -> pd.DataFrame:
    return _load(dataset_fingerprint())[1]


def load_locations() -> pd.DataFrame:
    return _load(dataset_fingerprint())[0]


def _filter(df: pd.DataFrame, period: Period, region: str | None, channel: str | None,
            location_id: str | None) -> pd.DataFrame:
    m = (df["week_start"] >= period.start) & (df["week_start"] <= period.end)
    if region:
        m &= df["region"].str.lower() == region.lower()
    if channel:
        m &= df["channel"].str.lower() == channel.lower()
    if location_id:
        m &= df["location_id"] == location_id
    return df[m]


def _compute(g: pd.DataFrame) -> dict:
    s = g[["spend", "leads", "appointments", "shows", "plans_presented", "plans_accepted", "production_usd"]].sum()
    def div(a, b):
        return round(float(a) / float(b), 4) if b else None
    return {
        "spend": round(float(s.spend), 2),
        "leads": int(s.leads),
        "appointments": int(s.appointments),
        "shows": int(s.shows),
        "production": round(float(s.production_usd), 2),
        "cpl": div(s.spend, s.leads),
        "cps": div(s.spend, s.shows),
        "show_rate": div(s.shows, s.appointments),
        "acceptance_rate": div(s.plans_accepted, s.plans_presented),
        "production_per_show": div(s.production_usd, s.shows),
        "roi": div(s.production_usd, s.spend),
        "weeks": int(g["week_start"].nunique()),
    }


def query_metrics(period: Period, region: str | None = None, channel: str | None = None,
                  location_id: str | None = None, group_by: str | None = None) -> dict:
    """Aggregate funnel metrics for a period with optional filters.

    group_by: None | 'region' | 'channel' | 'location_id'
    Returns a dict that is JSON-serializable and stable across runs.
    """
    df = _filter(load_funnel(), period, region, channel, location_id)
    out = {"period": asdict(period), "filters": {"region": region, "channel": channel, "location_id": location_id},
           "definitions": METRIC_DEFS}
    if group_by:
        out["rows"] = {str(k): _compute(g) for k, g in sorted(df.groupby(group_by), key=lambda kv: str(kv[0]))}
    else:
        out["totals"] = _compute(df)
    return out


def compare_periods(current: Period, previous: Period, region: str | None = None,
                    channel: str | None = None, group_by: str | None = None) -> dict:
    """Period-over-period deltas. The agent uses this to flag what changed."""
    cur = query_metrics(current, region, channel, group_by=group_by)
    prev = query_metrics(previous, region, channel, group_by=group_by)

    def delta(a: dict, b: dict) -> dict:
        d = {}
        for k in ("spend", "leads", "shows", "production", "cpl", "cps", "show_rate", "acceptance_rate", "roi"):
            x, y = a.get(k), b.get(k)
            if x is None or y in (None, 0):
                d[k] = {"current": x, "previous": y, "pct_change": None}
            else:
                d[k] = {"current": x, "previous": y, "pct_change": round((x - y) / y * 100, 1)}
        return d

    if group_by:
        keys = sorted(set(cur["rows"]) | set(prev["rows"]))
        rows = {k: delta(cur["rows"].get(k, {}), prev["rows"].get(k, {})) for k in keys}
        return {"current": asdict(current), "previous": asdict(previous), "group_by": group_by, "rows": rows}
    return {"current": asdict(current), "previous": asdict(previous), "totals": delta(cur["totals"], prev["totals"])}


def anomalies(current: Period, previous: Period, threshold_pct: float = 15.0) -> list[dict]:
    """Region x channel cells whose CPL or CPS moved more than threshold. Deterministic ordering."""
    df = load_funnel()
    df["cell"] = df["region"] + " | " + df["channel"]
    cur = query_metrics(current, group_by="cell")["rows"]
    prev = query_metrics(previous, group_by="cell")["rows"]
    flags = []
    for cell in sorted(cur):
        c, p = cur[cell], prev.get(cell)
        if not p:
            continue
        for metric in ("cpl", "cps"):
            if c[metric] and p[metric]:
                pct = (c[metric] - p[metric]) / p[metric] * 100
                if abs(pct) >= threshold_pct:
                    flags.append({"cell": cell, "metric": metric, "current": c[metric],
                                  "previous": p[metric], "pct_change": round(pct, 1)})
    return sorted(flags, key=lambda r: -abs(r["pct_change"]))
