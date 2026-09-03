"""Level-2 productization scaffold: the same pipeline behind an HTTP API.

Run: python modules/06-productize/api.py   (then open http://localhost:8000/docs)
Exercise: add an X-API-Key header check and a /health endpoint.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fastapi import FastAPI            # noqa: E402
from pydantic import BaseModel         # noqa: E402

from aihc.agent.insights_agent import run as run_agent   # noqa: E402
from aihc.metrics import dataset_fingerprint             # noqa: E402
from aihc.rag.answer import answer                       # noqa: E402

app = FastAPI(title="aihc", version="0.1.0")


class Ask(BaseModel):
    question: str


class ReportReq(BaseModel):
    weeks: int = 4
    as_of: str | None = None
    dry_run: bool = False


@app.post("/ask")
def ask(req: Ask):
    a = answer(req.question)
    return asdict(a)


@app.post("/report")
def report(req: ReportReq):
    md = run_agent(req.weeks, req.as_of, req.dry_run)
    return {"fingerprint": dataset_fingerprint(), "markdown": md}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
