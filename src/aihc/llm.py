"""The only file that talks to a model.

Three things live here on purpose:
1. Pinned model, temperature 0. Reduces variance; does not guarantee identical output.
2. An answer cache keyed on (prompt, dataset fingerprint, prompt version). This is what
   actually guarantees "same answer unless the data changes": the second ask never
   reaches the model.
3. Stub mode when no API key is present, so the whole pipeline runs offline and in CI.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from . import config
from .metrics import dataset_fingerprint


@dataclass
class LLMResult:
    text: str
    cache_hit: bool
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stub: bool = False


def _cache_key(system: str, user: str, extra: str = "") -> str:
    raw = json.dumps({"s": system, "u": user, "fp": dataset_fingerprint(),
                      "pv": config.PROMPT_VERSION, "m": config.MODEL, "x": extra}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def complete(system: str, user: str, *, use_cache: bool = True, max_tokens: int | None = None,
             stub_fn=None) -> LLMResult:
    """Single-turn completion with caching.

    stub_fn: callable(system, user) -> str used when no API key is available, so callers
    can supply a template answer built from the fact block.
    """
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(system, user)
    path = config.CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        d = json.loads(path.read_text())
        return LLMResult(text=d["text"], cache_hit=True, model=d["model"], stub=d.get("stub", False))

    if config.STUB:
        text = stub_fn(system, user) if stub_fn else "[stub mode: no API key; set ANTHROPIC_API_KEY]"
        res = LLMResult(text=text, cache_hit=False, model="stub", stub=True)
    else:
        import anthropic  # imported lazily so tests run without the package configured
        client = anthropic.Anthropic(api_key=config.API_KEY)
        msg = client.messages.create(
            model=config.MODEL,
            max_tokens=max_tokens or config.MAX_OUTPUT_TOKENS,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        res = LLMResult(text="".join(b.text for b in msg.content if getattr(b, "type", "") == "text"),
                        cache_hit=False, model=config.MODEL,
                        input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)
    if use_cache:
        path.write_text(json.dumps({"text": res.text, "model": res.model, "stub": res.stub,
                                    "ts": time.time()}))
    return res


def tool_loop(system: str, user: str, tools: list[dict], dispatch, *, max_calls: int | None = None,
              log=None) -> tuple[str, list[dict]]:
    """Minimal agent loop: model picks a tool, we run it, feed the result back, until the
    model stops calling tools. Returns (final_text, trace).

    dispatch: callable(name, args) -> JSON-serializable result
    This is the same loop every harness runs; see modules/05-harness/mini_harness.py.
    """
    max_calls = max_calls or config.MAX_TOOL_CALLS_PER_RUN
    trace: list[dict] = []
    if config.STUB:
        raise RuntimeError("tool_loop needs a real model; the agent has a --dry-run path that does not.")
    import anthropic
    client = anthropic.Anthropic(api_key=config.API_KEY)
    messages = [{"role": "user", "content": user}]
    calls = 0
    while True:
        msg = client.messages.create(model=config.MODEL, max_tokens=config.MAX_OUTPUT_TOKENS,
                                     temperature=0, system=system, tools=tools, messages=messages)
        if log:
            log({"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens})
        tool_uses = [b for b in msg.content if b.type == "tool_use"]
        if msg.stop_reason != "tool_use" or not tool_uses:
            return "".join(b.text for b in msg.content if b.type == "text"), trace
        messages.append({"role": "assistant", "content": msg.content})
        results = []
        for tu in tool_uses:
            calls += 1
            if calls > max_calls:
                return f"[stopped: exceeded {max_calls} tool calls]", trace
            out = dispatch(tu.name, tu.input)
            trace.append({"tool": tu.name, "args": tu.input, "result_chars": len(json.dumps(out, default=str))})
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(out, default=str)})
        messages.append({"role": "user", "content": results})
