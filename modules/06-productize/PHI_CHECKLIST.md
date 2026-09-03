# Data handling checklist for healthcare marketing analytics

Not legal advice. Use it to decide what questions to bring to counsel.

## 1. Classify each data source before ingestion
| Source | Typical fields | PHI risk | Default handling |
|---|---|---|---|
| Ad platform exports (Google, Meta) | spend, impressions, clicks by campaign/location | none | ingest |
| Call tracking summaries | calls, duration, source by location | low unless transcripts | ingest aggregates; exclude transcripts |
| CRM / lead exports | name, phone, email, treatment interest, source, status | high when tied to a covered entity and treatment | strip identifiers, keep source/status/dates/location |
| PMS / practice management | patient IDs, appointments, procedures, production | PHI | aggregates only, produced inside the client's environment |
| Web analytics | sessions, conversions | low; check for URL parameters carrying symptoms/conditions | ingest; scrub query strings |
| Internal policies, playbooks, memos | text | none, but may be confidential | ingest |

## 2. Rules the pipeline enforces
- [ ] Row-level identifiers (name, phone, email, DOB, address, MRN) are dropped or hashed before any file lands in `data/`.
- [ ] Treatment interest is kept only as a category and never joined back to an identifier.
- [ ] Nothing row-level is logged (`runs.jsonl` records counts and tokens, not content).
- [ ] Document chunks sent to the model are from policy/playbook sources, not from lead records.
- [ ] Model provider: confirm API data is not used for training under your plan; obtain a BAA if any PHI could reach the API.
- [ ] Access: API key per client, reports stored per client, no shared cache across clients (cache key must include a tenant id at level 2).

## 3. Marketing-specific HIPAA points to review with counsel
- Marketing communications to patients generally need authorization unless they fall under treatment/face-to-face/nominal-value exceptions; analytics on aggregates is a different question from outreach.
- Tracking technologies on patient-facing pages (pixels on appointment forms) have been the subject of HHS guidance and enforcement; know what the client's site does before pulling web analytics.
- Oncology and behavioral health carry additional state-level sensitivities; the synthetic `oncology_marketing_compliance_note.md` in this repo is a model of the kind of internal note you want to have.

## 4. Contract language to have ready
- Data processing description: what you receive, what you retain, for how long, where it is stored.
- Sub-processor list: model provider, hosting, email/Slack delivery.
- Deletion on termination and on request.
