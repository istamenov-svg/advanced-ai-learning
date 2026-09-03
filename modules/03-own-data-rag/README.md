# Module 3: Your own data, grounded and repeatable (Week 3, ~10 hours)

The code for this module is already in `src/aihc/` and passes its evals in stub mode. Your job is to understand it, run it against a real model, break it, fix it, and then point it at real data.

## Hour 1: run it
```bash
python -m aihc.data_gen
python -m aihc.rag.answer "What was cost per show in Q2 2026 for the Tampa region?"
python -m aihc.rag.answer "What paid social CPL level triggers a budget hold?"
python -m aihc.rag.answer "What paid social CPL level triggers a budget hold?"   # cache_hit: true
python -m aihc.evals.run_evals -v
```
Read the output of each. Then read the files in this order: `metrics.py`, `llm.py`, `rag/chunk.py`, `rag/index.py`, `rag/router.py`, `rag/answer.py`, `evals/run_evals.py`. Total under 700 lines.

## Hour 2-3: add your API key and re-run with a real model
```bash
rm -rf data/.cache
python -m aihc.evals.run_evals --no-cache -v
```
Compare with the stub run. The metrics questions should be identical (the numbers never came from the model). The docs answers will read better. Check `citation_validity` is still 1.0 and `abstain_ok` is still 1.0. If not, that is your first prompt-engineering task: adjust `SYSTEM_DOCS` in `answer.py`, bump `PROMPT_VERSION` in `config.py`, re-run.

## Exercise 3.1: prove the repeatability claim (1 hour)
1. Ask the same docs question five times with `--no-cache` semantics (call `answer(q, use_cache=False)` in a loop). Count distinct outputs. This is temperature 0's real-world variance.
2. Now with the cache on. Five identical outputs.
3. Edit one sentence in `data/docs/paid_social_policy.md`. Ask again. Observe the fingerprint change and the cache miss.
Write the three observations in `notes.md`.

## Exercise 3.2: tune retrieval (1 hour)
Change `MIN_SCORE` in `answer.py` to 1.0 and to 6.0. Run evals each time. Record `retrieval_hit` and `abstain_ok`. Then change chunk size in `chunk.py` (`max_chars` 300 and 1200). Same. You now have a table showing that chunking and thresholds matter more than prompt wording.

## Exercise 3.3: let the model parse, keep the math in code (2 hours)
`router.parse_metric_question` is rule-based and brittle ("what did we spend on paid social last quarter" fails). Replace it with a model call that returns the `MetricQuery` fields via tool use / structured output. Keep `query_metrics` untouched. Add five harder phrasings to `golden.jsonl`. Evals must stay at 100% exact match on values; only routing accuracy should change.

## Exercise 3.4: hybrid retrieval (2 hours, optional)
Add an embedding index (Voyage or any embedding API; or a local sentence-transformers model) alongside BM25 and merge with reciprocal rank fusion. Measure recall@k on the golden set before and after. Keep whichever wins; delete the other. Most people are surprised by the result on a small policy corpus.

## Exercise 3.5: real data (2 hours)
Export one real dataset shaped like `funnel_weekly.csv` (anonymized: location codes, no patient data). Drop it in `data/`, adjust column names in `metrics.py` if needed, regenerate golden values with `query_metrics`, re-run evals. Add two of your own real policy documents to `data/docs/`.

## Done when
Evals pass with a real model; `notes.md` has the repeatability and retrieval tables; at least one real export runs end to end.

## What you should now be able to say to a client
"Numbers in our reports are computed, not generated. The model writes the sentence around a figure that code produced from your data; it cannot change the figure. Narrative answers cite the internal document they came from, and we verify every citation exists. Same question, same data, same answer; we hash the data and the prompt to guarantee it."
