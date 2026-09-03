# advanced-ai-learning

A hands-on learning project: how models are built, the vocabulary, grounding an LLM on your own data with repeatable answers, a scheduled analytical agent, what a harness is, and how to productize the result. Domain data is a synthetic multi-location dental (DSO) marketing funnel.

Start with `CURRICULUM.md`. Then:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # add ANTHROPIC_API_KEY
python -m aihc.data_gen                              # creates data/
pytest                                               # everything that does not need an API key
python -m aihc.rag.answer "What was cost per show in Q2 2026 for the Tampa region?"
python -m aihc.evals.run_evals
python -m aihc.agent.insights_agent --dry-run
```

Layout:

```
CURRICULUM.md            the 5-week plan and the six modules
modules/0N-*/README.md   per-module reading, exercise, done-when
src/aihc/                the code you build on
  data_gen.py            synthetic dataset + policy documents
  metrics.py             deterministic metrics layer (numbers come from here, never from the model)
  llm.py                 one place for API calls: pinned model, temperature 0, answer cache
  rag/                   chunk -> index (BM25) -> retrieve -> answer with citations
  evals/                 golden set + eval runner (the "harness" in the testing sense)
  agent/                 scheduled insights agent + state + run log
tests/                   pytest, no API key needed
.cursor/rules/           what Cursor reads before it touches this repo
.github/workflows/       the schedule that runs the agent without a server
```

Everything that calls a model goes through `src/aihc/llm.py`. With no API key set, `llm.py` runs in stub mode so the pipeline can be exercised end to end offline; answers are then template-generated from the fact block and clearly marked.
