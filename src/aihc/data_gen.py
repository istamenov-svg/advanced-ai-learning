"""Synthetic healthcare-marketing dataset: a 12-location DSO's weekly acquisition funnel
plus a handful of internal documents for retrieval.

Deterministic (fixed seed) so the eval golden set stays valid. Regenerate with:
    python -m aihc.data_gen
Change SEED or the parameters below to simulate "the dataset changed" and watch the
answer cache invalidate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR, DOCS_DIR

SEED = 20260903

LOCATIONS = [
    ("L01", "Tampa Bay Smiles", "Tampa", "FL"),
    ("L02", "Brandon Family Dental", "Tampa", "FL"),
    ("L03", "Clearwater Dental Group", "Tampa", "FL"),
    ("L04", "Orlando Dental Partners", "Orlando", "FL"),
    ("L05", "Lake Nona Smiles", "Orlando", "FL"),
    ("L06", "Kissimmee Dental Care", "Orlando", "FL"),
    ("L07", "Coral Ridge Dental", "South Florida", "FL"),
    ("L08", "Boca Smile Studio", "South Florida", "FL"),
    ("L09", "Pembroke Pines Dental", "South Florida", "FL"),
    ("L10", "Buckhead Dental Arts", "Atlanta", "GA"),
    ("L11", "Marietta Family Dentistry", "Atlanta", "GA"),
    ("L12", "Decatur Dental Group", "Atlanta", "GA"),
]

# channel: (weekly spend mean, cost per lead, lead->appt, appt->show, show->plan, plan->accept, avg case value)
CHANNELS = {
    "Google Search":   (2400, 95,  0.58, 0.72, 0.80, 0.52, 1450),
    "Paid Social":     (1500, 55,  0.38, 0.55, 0.75, 0.40, 1100),
    "Organic / SEO":   (400,  30,  0.55, 0.78, 0.82, 0.58, 1350),
    "Referral":        (150,  20,  0.72, 0.85, 0.88, 0.66, 1900),
    "Direct Mail":     (900,  140, 0.45, 0.62, 0.78, 0.48, 1250),
}


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2025-09-01", periods=52, freq="W-MON")
    rows = []
    for loc_id, name, region, state in LOCATIONS:
        loc_scale = rng.uniform(0.7, 1.4)
        for ch, (spend_mu, cpl, r1, r2, r3, r4, case) in CHANNELS.items():
            ch_drift = rng.normal(0, 0.03)
            for i, wk in enumerate(weeks):
                season = 1.0 + 0.12 * np.sin(2 * np.pi * (i / 52.0))       # mild seasonality
                trend = 1.0 + ch_drift * (i / 52.0)
                # A deliberate story in the data: Paid Social CPL degrades in South Florida from May 2026
                shock = 1.0
                if ch == "Paid Social" and region == "South Florida" and wk >= pd.Timestamp("2026-05-04"):
                    shock = 1.35
                spend = max(0.0, rng.normal(spend_mu * loc_scale * season, spend_mu * 0.12))
                eff_cpl = cpl * shock * rng.uniform(0.9, 1.1)
                leads = int(rng.poisson(spend / eff_cpl)) if eff_cpl > 0 else 0
                appts = int(rng.binomial(leads, min(1, r1 * trend)))
                shows = int(rng.binomial(appts, r2))
                plans = int(rng.binomial(shows, r3))
                accepted = int(rng.binomial(plans, r4))
                production = float(np.round(accepted * case * rng.uniform(0.85, 1.15), 2))
                rows.append(dict(
                    week_start=wk.date().isoformat(), location_id=loc_id, channel=ch,
                    spend=round(spend, 2), leads=leads, appointments=appts, shows=shows,
                    plans_presented=plans, plans_accepted=accepted, production_usd=production,
                ))
    funnel = pd.DataFrame(rows)
    locations = pd.DataFrame(LOCATIONS, columns=["location_id", "name", "region", "state"])
    return locations, funnel


DOCS = {
    "kpi_definitions.md": """# KPI definitions (marketing analytics standard, v3)

Cost per lead (CPL) = spend / leads. A lead is any new-patient inquiry with a valid phone or email, deduplicated by phone within 30 days.

Cost per show (CPS) = spend / shows. A show is a booked appointment where the patient arrived; no-shows and same-day cancellations are excluded.

Show rate = shows / appointments booked.

Treatment acceptance rate = plans accepted / plans presented, measured on the presentation date, not on the payment date.

Production per show = production_usd / shows. Production is scheduled treatment value at the time of acceptance, not collections.

Marketing ROI = production_usd / spend. Board reporting uses this definition; do not substitute collections.

Reporting periods are ISO weeks starting Monday. Quarters follow the calendar year.
""",
    "paid_social_policy.md": """# Paid social policy for dental and aesthetics brands (effective 2026-01-15)

1. Paid social is a top-of-funnel channel. Target blended CPL under $70 and a show rate above 50%. A location whose paid social CPL exceeds $90 for four consecutive weeks moves to a 50% budget hold pending creative refresh.

2. Creative must not make outcome claims (no "pain-free", no before/after implying guaranteed results). Testimonials require written patient consent on file.

3. Retargeting audiences built from website visitors are permitted. Audiences built from lead lists containing treatment interest are not, because treatment interest tied to an identifiable person is treated as sensitive health information under our data policy, regardless of HIPAA applicability.

4. Budget reallocation between channels within a region requires regional marketing manager approval when the move exceeds 20% of the region's monthly budget.
""",
    "lead_followup_playbook.md": """# Lead follow-up playbook (front office)

Speed to lead: first call attempt within 5 minutes of form submission during business hours; within 15 minutes of open the next business day otherwise. Locations meeting the 5-minute standard book 55-65% of leads; locations averaging over 30 minutes book 30-40%.

Cadence: 3 call attempts and 2 texts over 72 hours, then move to the nurture email sequence.

Do not quote prices for treatment over the phone beyond the published new-patient exam offer.

Referral leads bypass the call queue and are scheduled directly by the office manager.
""",
    "oncology_marketing_compliance_note.md": """# Note: marketing to oncology patients (legal review, 2025-11)

Condition-specific targeting (e.g. audiences defined by a cancer diagnosis) is prohibited across all paid channels. Awareness campaigns must target geography and demographics only.

Any landing page collecting symptoms or diagnosis information is in scope for HIPAA; the form vendor must be under a BAA. Marketing analytics pipelines may only receive de-identified aggregates from those forms.

This note applies to oncology service lines only; the dental and aesthetics policies are separate.
""",
    "q2_2026_board_memo.md": """# Q2 2026 marketing summary for the board (excerpt)

Google Search remained the largest paid channel and the most efficient on a cost-per-show basis. Referral continues to produce the highest production per show but cannot be scaled with budget.

South Florida paid social efficiency deteriorated starting in May; the working hypothesis is creative fatigue combined with a competitor's promotional push in Broward County. A creative refresh was commissioned in June with results expected in Q3.

Atlanta locations are in their first full year of paid media and are being evaluated on lead volume rather than ROI until Q4.

Decision requested: approve a 15% shift of South Florida paid social budget into Google Search for Q3.
""",
}


def write_all() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    locations, funnel = generate()
    locations.to_csv(DATA_DIR / "locations.csv", index=False)
    funnel.to_csv(DATA_DIR / "funnel_weekly.csv", index=False)
    for name, text in DOCS.items():
        (DOCS_DIR / name).write_text(text, encoding="utf-8")
    print(f"wrote {len(funnel):,} funnel rows, {len(locations)} locations, {len(DOCS)} docs to {DATA_DIR}")


if __name__ == "__main__":
    write_all()
