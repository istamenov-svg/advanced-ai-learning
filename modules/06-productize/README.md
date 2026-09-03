# Module 6: Productizing (Week 5, second half, ~5 hours)

## The three levels
1. **Repeatable service.** You run the repo for a client; they get the weekly report and can ask questions through you. Deliverable: the report. This is a CubedHealth engagement with better margins because the analysis is automated.
2. **Packaged tool.** The client (or their agency) sends exports; a hosted API returns reports and answers. `api.py` in this folder is the scaffold. Price per location per month.
3. **Product.** Self-serve data connectors, multi-tenant, dashboards, billing. Needs an engineering team. Do not start here.

The right first move is level 1 with two or three clients, using their real data to harden the evals, then level 2 when two clients ask for the same thing.

## Hour 1: the API scaffold
```bash
uvicorn modules.06-productize.api:app --reload    # or: python modules/06-productize/api.py
curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"What was cost per show in Q2 2026 for the Tampa region?"}'
curl -X POST localhost:8000/report -H 'content-type: application/json' -d '{"weeks":4,"dry_run":true}'
```
Tasks: add an API key header check; add a `/health` endpoint that returns the dataset fingerprint and the last eval summary. That is your "is it working" answer for a client.

## Hour 2: the cost model
```bash
python modules/06-productize/pricing_model.py
```
Fill in real token counts from `runs.jsonl` and current model prices. It prints cost per client per month at three usage levels and the gross margin at three price points. Decide a price. Write it down.

## Hour 3: data handling
Work through `PHI_CHECKLIST.md`. For most marketing funnel data (spend, leads, shows by location and channel) there is no PHI. The moment a lead export with names, phone numbers, or treatment interest enters the pipeline, the checklist applies. Decide your policy: strip identifiers before ingestion (recommended for level 1 and 2), or run under a BAA.

## Hour 4: the eval gate as a sales argument
Put `python -m aihc.evals.run_evals` in CI (the GitHub Actions workflow already does it). Save the summary JSON from each run. That history is the answer to "how do you know it's accurate": numeric exact-match rate, citation validity, abstention on out-of-scope questions, over time, against a growing golden set that includes the client's own questions.

## Hour 5: the offer
Fill in `OFFER_TEMPLATE.md` for a 20-location DSO. One page. What they get every Monday, what they can ask, what it costs, what data you need, what you do not do with it, and how accuracy is guaranteed. Send it to someone who would tell you the truth about it.

## Done when
API runs with a key check, price decided with a margin number behind it, PHI policy written, offer drafted.
