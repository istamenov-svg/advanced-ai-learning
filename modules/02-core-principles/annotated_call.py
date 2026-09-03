"""Module 2 exercise: one API call, every concept labeled.

Run with no key: prints the request body.  Run with a key: makes the call and prints the response.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aihc import config  # noqa: E402  (loads .env)

# --- SYSTEM PROMPT: persistent instructions; sets role, rules, output contract. Cached across calls if marked.
SYSTEM = [{
    "type": "text",
    "text": ("You are an analyst for a dental group. Answer in JSON with keys 'answer' and 'confidence'. "
             "If a tool is available for the question, call it instead of guessing."),
    "cache_control": {"type": "ephemeral"},   # PROMPT CACHING: reuse this prefix; cheaper and faster on repeat calls
}]

# --- TOOL DEFINITION: a JSON schema the model can "call". The model emits arguments; YOUR code runs the function.
TOOLS = [{
    "name": "query_metrics",
    "description": "Return cost per lead for a region and quarter from the data warehouse.",
    "input_schema": {
        "type": "object",
        "properties": {"region": {"type": "string"}, "quarter": {"type": "string", "description": "e.g. Q2 2026"}},
        "required": ["region", "quarter"],
    },
}]

# --- FEW-SHOT: an example turn showing the format you want. One good example beats a paragraph of rules.
FEW_SHOT = [
    {"role": "user", "content": "What is show rate?"},
    {"role": "assistant", "content": json.dumps({"answer": "shows divided by booked appointments", "confidence": "high"})},
]

# --- USER MESSAGE: the actual question. Everything the model knows about your data must be here or come via tools.
QUESTION = "What was cost per lead in Tampa in Q2 2026?"

request = dict(
    model=config.MODEL,           # MODEL PINNING: an exact ID, so behavior does not change under you
    max_tokens=300,               # OUTPUT BUDGET: hard cap on generated tokens; cost and latency scale with this
    temperature=0,                # SAMPLING: 0 = pick the most likely token each step; reduces (not removes) variance
    system=SYSTEM,
    tools=TOOLS,
    messages=FEW_SHOT + [{"role": "user", "content": QUESTION}],  # CONTEXT WINDOW: all of this is tokenized and counted
)

if config.STUB:
    print("No API key; here is the request that would be sent:\n")
    print(json.dumps(request, indent=2))
    print("\nExpected: stop_reason == 'tool_use' with a tool_use block {region: 'Tampa', quarter: 'Q2 2026'}.")
    sys.exit(0)

import anthropic  # noqa: E402

client = anthropic.Anthropic(api_key=config.API_KEY)
resp = client.messages.create(**request)
print("stop_reason:", resp.stop_reason)      # 'tool_use' means the model wants your code to run something
print("usage:", resp.usage)                  # TOKENS in/out; cache_read_input_tokens shows caching working on call 2
for block in resp.content:
    if block.type == "tool_use":
        print("TOOL CALL ->", block.name, block.input)   # the model's arguments; the harness would now run the tool
    elif block.type == "text":
        print("TEXT ->", block.text)         # STRUCTURED OUTPUT: should parse as JSON per the system prompt
