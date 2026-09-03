"""A harness in ~60 lines. This is what Cursor, Claude Code, and the Agent SDK all do,
minus the file editing, the UI, and the context management.

Run: python modules/05-harness/mini_harness.py "your question"
Needs ANTHROPIC_API_KEY. Without one it prints the loop it would run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aihc import config                     # noqa: E402
from aihc.agent.tools import dispatch       # noqa: E402  (reuse the deterministic tools)

SYSTEM = "You are an analyst. Use tools for any number or policy. Cite chunk ids. Stop when you have answered."

TOOLS = [
    {"name": "compare_periods",
     "description": "Trailing N weeks vs the N before. Args: weeks, region, channel, group_by.",
     "input_schema": {"type": "object", "properties": {
         "weeks": {"type": "integer"}, "region": {"type": "string"}, "channel": {"type": "string"},
         "group_by": {"type": "string"}}}},
    {"name": "search_docs",
     "description": "Search internal policy documents.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    # Exercise: add query_metrics here and in dispatch (it already exists in aihc.agent.tools).
]

MAX_ITERATIONS = 8   # Exercise: this is the guard. Remove it and watch what an unbounded loop can do to a bill.


def run(question: str) -> str:
    if config.STUB:
        return ("No API key. The loop would be: send SYSTEM + question + TOOLS -> if the model returns tool_use, "
                "run dispatch(name, args), append the result, repeat -> else return the text.")
    import anthropic
    client = anthropic.Anthropic(api_key=config.API_KEY)
    messages = [{"role": "user", "content": question}]
    for i in range(MAX_ITERATIONS):
        resp = client.messages.create(model=config.MODEL, max_tokens=800, temperature=0,
                                      system=SYSTEM, tools=TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":                       # stop condition: the model is done
            return "".join(b.text for b in resp.content if b.type == "text")
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                print(f"[{i}] tool {b.name} {b.input}", file=sys.stderr)   # observability
                out = dispatch(b.name, b.input)                             # the harness runs the tool
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, default=str)})
        messages.append({"role": "user", "content": results})             # feed results back
    return "[stopped: hit MAX_ITERATIONS]"


if __name__ == "__main__":
    print(run(" ".join(sys.argv[1:]) or "What changed in South Florida paid social in the last 8 weeks?"))
