# Module 5: The harness (Week 5, first half, ~5 hours)

## What a harness is
The harness is the software around the model. The model turns a prompt into text. The harness decides what goes into the prompt (your files, tool results, past turns, rules), what to do with the text that comes out (parse a tool call, apply an edit, run a command, show it to you), and when to stop. Two people with the same model and different harnesses get very different results; the harness is most of the product.

Concretely, a coding harness like Cursor does the following before and after each model call: indexes your repo and picks which files to include; prepends its own system prompt plus your `.cursor/rules`; exposes tools (read file, edit file, run terminal, search); runs the tool the model asks for and feeds back the result; applies diffs and asks you to accept; keeps a running summary when the conversation exceeds the context window.

The second meaning, an **eval harness**, is the fixture that runs a model against a test set and scores it. You built one in Module 3 (`run_evals.py`).

## Is Cursor a harness? Yes. What else is there?

| Harness | Runs where | Unattended? | Configuration file | Best for |
|---|---|---|---|---|
| Cursor | editor | no | `.cursor/rules/*.mdc` | building and editing this repo |
| Claude Code | terminal, CI | yes (`-p` headless mode) | `CLAUDE.md`, hooks | scripted or scheduled coding tasks |
| Codex CLI | terminal | yes | `AGENTS.md` | same category as Claude Code |
| GitHub Copilot agent | editor + GitHub | partially (issues to PRs) | `.github/copilot-instructions.md` | teams already on GitHub |
| Aider | terminal, open source | yes | `.aider.conf.yml` | seeing how a harness works; readable source |
| Anthropic Agent SDK / OpenAI Agents SDK | your code | yes | code | your own agent (Module 4) with the loop managed for you |
| LangGraph / Pydantic AI | your code | yes | code | multi-step workflows with explicit state graphs |
| n8n / Zapier agents | hosted | yes | UI | non-coders wiring SaaS tools together |
| `mini_harness.py` | your code | yes | code | understanding all of the above |

How to compare any two: tools exposed, context strategy (index vs. explicit files vs. summaries), where it runs, can it run without a human, permission model, scriptability, cost model.

## Hour 1: the rules file
Open `.cursor/rules/project.mdc`. It is the persistent instruction Cursor injects for this repo. Edit it: add a rule you care about (e.g. "never modify golden.jsonl without asking"). Then ask Cursor to do something that violates it and see if it complies.

## Hour 2-3: `mini_harness.py`
```bash
python modules/05-harness/mini_harness.py "What changed in South Florida paid social in the last 8 weeks and does any policy apply?"
```
About 60 lines: a loop, two tools, a stop condition. Tasks: add a third tool (`query_metrics` from `agent/tools.py`), add a max-iterations guard, log every tool call to stdout. Then re-read Cursor's or Claude Code's documentation and map each feature to a line in this file.

## Hour 4: same task, three harnesses
Give the same request ("add a `location_leaderboard` tool to the agent") to Cursor, to the mini harness (it will fail; note why), and, if you have it, Claude Code. Write half a page: what each one saw, what each one did, where each stopped and asked.

## Hour 5: pick your stack
Write the one-page comparison for your use case: Cursor for building; something headless (Claude Code, Agent SDK, or your own loop) for the scheduled agent. The Module 4 agent is already harness-independent because the tools and the loop are yours.

## Done when
The rules file has your edits, the mini harness has three tools and a guard, and the comparison page exists.
