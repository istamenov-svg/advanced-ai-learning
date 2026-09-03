"""Cost and margin model for the packaged-tool offer. Replace the ASSUMPTIONS with your measured numbers.

Run: python modules/06-productize/pricing_model.py
"""
from __future__ import annotations

# ---- ASSUMPTIONS: replace from runs.jsonl and the provider's current price list ----
PRICE_IN_PER_M = 3.00       # $ per million input tokens  (check current pricing; this is a placeholder)
PRICE_OUT_PER_M = 15.00     # $ per million output tokens
REPORT_IN_TOKENS = 25_000   # per agent run, from runs.jsonl (tool results dominate)
REPORT_OUT_TOKENS = 1_200
ASK_IN_TOKENS = 3_000       # per /ask question
ASK_OUT_TOKENS = 200
HOSTING_PER_CLIENT = 15.0   # $ / month share of a small VM or serverless
YOUR_TIME_HOURS = 1.0       # review + client call per client per month
YOUR_HOURLY = 250.0


def cost_per_report() -> float:
    return REPORT_IN_TOKENS / 1e6 * PRICE_IN_PER_M + REPORT_OUT_TOKENS / 1e6 * PRICE_OUT_PER_M


def cost_per_ask() -> float:
    return ASK_IN_TOKENS / 1e6 * PRICE_IN_PER_M + ASK_OUT_TOKENS / 1e6 * PRICE_OUT_PER_M


def monthly_cost(reports: int, asks: int) -> float:
    return reports * cost_per_report() + asks * cost_per_ask() + HOSTING_PER_CLIENT + YOUR_TIME_HOURS * YOUR_HOURLY


if __name__ == "__main__":
    print(f"cost per report ${cost_per_report():.3f}, per question ${cost_per_ask():.4f}")
    scenarios = {"light (4 reports, 20 asks)": (4, 20), "medium (4 + 8 regional, 100 asks)": (12, 100),
                 "heavy (weekly x 20 locations, 500 asks)": (80, 500)}
    prices = [750, 1500, 3000]
    print(f"\n{'scenario':45} {'cost/mo':>10} " + " ".join(f"{'@$'+str(p):>12}" for p in prices))
    for name, (r, a) in scenarios.items():
        c = monthly_cost(r, a)
        margins = " ".join(f"{(p - c) / p:>11.0%} " for p in prices)
        print(f"{name:45} {c:>10.2f} {margins}")
    print("\nNote: model cost is a rounding error next to your time. Price on outcome, not tokens.")
