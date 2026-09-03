# Module 4: A permanent, scheduled analytical agent (Week 4, ~10 hours)

## Hour 1: run it
```bash
python -m aihc.agent.insights_agent --dry-run --as-of 2026-06-29 --weeks 8
cat reports/2026-06-29_w8.md
cat state.json
cat runs.jsonl
```
The dry run uses only the deterministic tools. It already finds the planted story in the data (South Florida paid social CPL up ~40%) and pulls the applicable policy chunk. Then, with a key:
```bash
python -m aihc.agent.insights_agent --as-of 2026-06-29 --weeks 8
```
Read the report. Check every number against the dry-run report. Any number in the model's report that is not in a tool result is a bug in the system prompt; fix and re-run.

## Hour 2: read `agent/tools.py`, `agent/insights_agent.py`, and `llm.tool_loop`
The loop in `tool_loop` is the whole idea of an agent: about 30 lines. Everything else is tools, state, logging, and guardrails.

## Exercise 4.1: state and deltas (2 hours)
Run three times with `--as-of` 2026-05-04, 2026-06-01, 2026-06-29. Confirm `state.json` carries forward and that each report leads with changes. Then modify the system prompt so the agent explicitly says "unchanged since last report" for metrics within 5% of `last_totals`.

## Exercise 4.2: schedule it (2 hours)
Pick one; all three are provided.

**GitHub Actions** (no server, free, secrets handled): push the repo to a private GitHub repo, add `ANTHROPIC_API_KEY` as a repository secret, and the workflow in `.github/workflows/weekly_insights.yml` runs every Monday 07:00 ET, commits the report to `reports/`. Trigger it manually once from the Actions tab to test.

**Windows Task Scheduler** (your laptop): 
```
schtasks /Create /SC WEEKLY /D MON /ST 07:00 /TN "aihc-weekly" /TR "\"C:\path\to\.venv\Scripts\python.exe\" -m aihc.agent.insights_agent" /RU "%USERNAME%"
```
Set "Start in" to the repo folder in the task's properties. Downside: laptop must be on.

**cron** (Mac/Linux/any VM): `0 7 * * 1 cd /path/to/repo && .venv/bin/python -m aihc.agent.insights_agent >> cron.log 2>&1`

## Exercise 4.3: delivery (1 hour)
Add a `deliver()` step: email the report via SMTP (or the Gmail API), or post to Slack via an incoming webhook. Keep the Markdown file as the durable record either way.

## Exercise 4.4: guardrails and cost (1 hour)
`runs.jsonl` has tokens per run. Compute cost per run from the model's published prices. Set `MAX_TOOL_CALLS_PER_RUN` to 3 and watch the agent get cut off; set it back. Add a cost ceiling that aborts the loop if input tokens exceed a number. Decide what the fallback should be (the dry-run report is the default).

## Exercise 4.5: add a tool (1 hour)
Add `location_leaderboard` (top and bottom 3 locations by CPS for the period) to `tools.py` and `TOOLS`. Notice the agent starts using it without any prompt change. That is what tool descriptions are for.

## Done when
The agent has run on a schedule at least twice without you, each report's numbers reconcile to the dry-run, cost per run is written down, and delivery works.
