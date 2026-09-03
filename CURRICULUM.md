# Advanced AI: a 5-week learning project

Audience: one person (Ivan), 25+ years in marketing and commercial operations, comfortable with Python at the read-and-modify level, working in Cursor, ~10 hours per week. Domain data: healthcare marketing (DSO / aesthetics / oncology patient acquisition). Everything below is built to be run, not just read.

The six questions you asked map to six modules. Each module has a README in `modules/`, code in `src/aihc/`, a build exercise, and a "done when" check. Weeks 1 and 2 cover Modules 1 and 2 (theory, one exercise each). Weeks 3 and 4 cover Modules 3 and 4 (the two large builds). Week 5 covers Modules 5 and 6 (harness and productization), plus the capstone. Stretch to 6 weeks if Module 3 takes longer; it is the one where most people underestimate the evaluation work.

## Corrections to the starting definitions

Three things in your framing need adjusting before you start, because they change what you build.

**Hallucination and non-determinism are two separate problems.** Hallucination is the model asserting something not supported by its inputs. Non-determinism is the model giving different outputs for the same input. You can have a perfectly grounded answer that is worded differently every run, and you can have an identically-worded answer that is wrong every run. They have different fixes. Grounding (Module 3) fixes the first. Temperature 0, output caching keyed on the dataset fingerprint, and moving arithmetic out of the model fix the second. The strongest version of "same answer unless the data changes" is not a model setting at all: it is computing the numbers deterministically in code, and having the model only narrate them.

**"Integrating own data into an LLM" is three different techniques, and fine-tuning is usually the wrong one.** Retrieval (RAG) puts your documents into the prompt at question time. Tool use lets the model call your database and get exact figures. Fine-tuning changes the model's weights and is for teaching style or format, not facts; it does not reduce hallucination about your data and it goes stale the moment the data changes. Module 3 builds the first two. Fine-tuning is covered in Module 1 so you know what it is and when to reject it.

**A "harness" has two meanings and you will hear both.** In the agent-coding world (Cursor, Claude Code, Codex CLI), the harness is the software around the model: the loop that sends prompts, runs tools, reads files, applies edits, and decides when to stop. In the evaluation world, a harness is the fixture that runs a model against a test set and scores it. You need both. Module 5 covers the first; Module 3 builds the second.

## Module map

| Week | Module | Question it answers | Build |
|---|---|---|---|
| 1 | 1. How models are built | Software and hardware behind training and serving | `model_math.py`: parameter, memory, and compute calculator plus a working attention block in NumPy |
| 2 | 2. Core principles and vocabulary | What the terms mean and how the pieces relate | Glossary self-test; annotate a real API call end to end |
| 3 | 3. Own data, grounded and repeatable | RAG + tool use + eval + caching | Deterministic metrics layer, BM25 retriever, grounded answerer, golden-set eval |
| 4 | 4. Permanent analytical agent | Scheduled, stateful, tool-using agent | `insights_agent.py` producing a weekly report with period-over-period deltas |
| 5 | 5. The harness | What Cursor is, alternatives, how to configure it | `.cursor/rules`, a Claude Code / Agent SDK comparison, a hand-rolled 60-line harness |
| 5 | 6. Productizing | From repo to something someone pays for | Packaging, API, eval gate, pricing sheet, PHI/HIPAA checklist |
| 5-6 | Capstone | All of it | Agent runs on your real exports, on a schedule, with evals green |

## Prerequisites (do before Week 1, ~2 hours)

1. Python 3.11+, `pip install -r requirements.txt`.
2. An Anthropic API key in `.env` (copy `.env.example`). Everything that calls a model goes through `src/aihc/llm.py`, so one key location.
3. Open the repo folder in Cursor. Read `.cursor/rules/project.mdc`; that file is the first thing Module 5 will explain.
4. Run `python -m aihc.data_gen` to create the synthetic dataset in `data/`. Look at the CSVs. They are shaped like a multi-location DSO's marketing funnel (spend by channel, leads, appointments, shows, treatment acceptance, production), plus a few policy and playbook documents for retrieval.

---

## Module 1: How models are built (Week 1)

**What you asked:** what is needed from a software and hardware perspective.

**Core content**

Transformer architecture at the level of tensors: tokens become embeddings, embeddings pass through N identical blocks (attention + feed-forward), final layer projects to vocabulary logits. Attention is the part that lets each token look at earlier tokens; the feed-forward layers are where most of the parameters live.

Pretraining: next-token prediction on trillions of tokens. The cost is roughly 6 × parameters × tokens FLOPs. A 70B model on 15T tokens is ~6.3e24 FLOPs. That number is why hardware matters.

Post-training: supervised fine-tuning (SFT) on demonstrations, then preference optimization (RLHF, DPO, or RL against verifiable rewards). This is what turns a text predictor into an assistant. It is also where "refusals" and "style" come from.

Hardware: GPUs and TPUs are matrix-multiply engines with high-bandwidth memory (HBM). Training is limited by memory (weights + optimizer state + activations, about 16 bytes per parameter in mixed precision with Adam) and by interconnect bandwidth between chips. Inference is limited by memory bandwidth: every generated token requires streaming all weights through the chip once. Quantization (fp16 → int8 → int4) exists to shrink that.

Parallelism: data parallel (copy the model, split the batch), tensor parallel (split each matrix across chips), pipeline parallel (split layers across chips). Frontier training runs use all three across tens of thousands of chips.

Software stack, bottom to top: CUDA / ROCm kernels → PyTorch or JAX → distributed training frameworks (Megatron, DeepSpeed, FSDP) → data pipelines (dedup, filtering, tokenization) → eval suites → serving engines (vLLM, TensorRT-LLM, SGLang) → API.

Fine-tuning for you: LoRA and QLoRA train a small adapter instead of all weights, so a 7B model can be fine-tuned on one consumer GPU. Use case: teach a consistent output format or voice. Not a use case: teach facts about your data.

**Reading (pick 3)**
- "The Illustrated Transformer" (Jay Alammar). Diagrams, no math beyond matrices.
- Andrej Karpathy, "Let's build GPT: from scratch, in code" (video, 2h). The best single explanation of what a model is.
- Kaplan et al. 2020 and Hoffmann et al. 2022 ("Chinchilla"): scaling laws. Read the abstracts and figure 1 of each.
- Anthropic's docs on model overview and pricing; note the input/output token asymmetry.
- Any vLLM or TensorRT-LLM architecture overview for how serving works.

**Build exercise:** `modules/01-how-models-are-built/model_math.py`. Given a model config (layers, hidden size, heads, vocab, context), compute parameter count, training memory, inference memory at several precisions, training FLOPs for a token budget, and tokens/second ceiling from HBM bandwidth. Then run the NumPy single-head attention block and confirm the output shape and that attention weights sum to 1.

**Done when:** you can explain, without notes, why a 70B model does not fit on one 80GB GPU for training but does fit for int4 inference, and what the number 6ND means.

---

## Module 2: Core principles, terms, and technologies (Week 2)

**What you asked:** what are AI's core principles, terms, and technologies.

**Core content**, organized as five layers, with the terms that belong to each.

1. *Model layer:* parameters, weights, tokens, tokenizer, context window, embeddings, logits, temperature, top-p, sampling, stop sequences, system prompt, few-shot, chain of thought, reasoning/extended thinking, base vs instruct vs reasoning models, open-weights vs closed.
2. *Data layer:* pretraining corpus, SFT data, preference data, synthetic data, data contamination, deduplication, PII scrubbing.
3. *Application layer:* prompt, prompt template, structured output (JSON schema), function/tool calling, retrieval (RAG), vector database, embedding model, reranker, chunking, hybrid search (BM25 + dense), agent, agent loop, MCP (Model Context Protocol), memory, guardrails, prompt injection.
4. *Evaluation layer:* golden set, exact-match vs rubric vs LLM-as-judge, faithfulness, answer relevance, retrieval recall@k, regression test, A/B, observability/tracing.
5. *Operations layer:* latency, throughput, tokens per second, cost per query, prompt caching, batch API, rate limits, fallbacks, model routing, versioning (pin model IDs), determinism.

Principles that cut across layers: (a) the model only knows what is in the weights or in the prompt; (b) the model predicts, it does not look up; (c) everything the model emits is text, so structure must be enforced from outside; (d) evaluation is the product, the prompt is a detail; (e) cost and latency scale with tokens, not with "difficulty".

**Reading**
- Anthropic docs: Messages API, tool use, prompt caching, structured outputs. Read them with the API in another window.
- Simon Willison's blog tag "llms" for a running vocabulary of what is real vs hype.
- Lilian Weng, "LLM Powered Autonomous Agents" (2023). Still the cleanest definition of an agent.
- OWASP Top 10 for LLM Applications, for prompt injection and the rest.

**Build exercise:** `modules/02-core-principles/annotated_call.py`. One API call with a system prompt, a tool definition, structured output, temperature 0, and prompt caching. Every line has a comment naming the concept it demonstrates. Then complete `glossary_selftest.md` from memory before checking `GLOSSARY.md`.

**Done when:** you can read a vendor's architecture diagram and place every box in one of the five layers.

---

## Module 3: Your own data, grounded and repeatable (Week 3)

**What you asked:** integrate own data to minimize hallucination and get the same response every time unless the dataset changes.

**Core content**

Separate the two goals as described above, then build a pipeline that achieves both.

*Grounding architecture (reduces hallucination):*
1. Structured questions ("what was cost per show in Q2 for the Tampa region") go to a deterministic metrics layer, not to the model. Code computes the number. The model receives the number as a fact block and writes the sentence.
2. Unstructured questions ("what does our policy say about paid social for oncology") go to retrieval. Documents are chunked with stable IDs, indexed, and the top-k chunks are placed in the prompt with the instruction to cite chunk IDs and to say "not in the provided documents" when that is true.
3. Every answer carries citations. A post-check verifies that each cited ID exists in the retrieved set. Uncited claims fail the answer.

*Repeatability architecture (reduces variance):*
1. Temperature 0 and a pinned model ID. This reduces variance but does not eliminate it; providers do not guarantee bit-identical outputs.
2. Dataset fingerprint: a hash of the data files and the prompt template version. Answers are cached keyed on (question, fingerprint). Same question + same data = the cached answer, byte for byte. Data changes → fingerprint changes → recompute.
3. Numbers never come from the model. This is the single biggest source of "it said 14.2% yesterday and 14.6% today".

*Why not fine-tune:* the dataset changes; a fine-tuned model would need retraining on every change and still would not cite sources.

*Evaluation:* a golden set of 25-40 question/answer pairs with expected citations. Metrics: retrieval recall@k, exact match on numeric answers, citation validity, and a faithfulness judge for narrative answers. This is your regression suite; run it on every change to chunking, prompt, or model.

**Reading**
- Lewis et al. 2020, "Retrieval-Augmented Generation" (abstract + figure).
- Anthropic docs: "Contextual retrieval" and "Reduce hallucinations".
- Ragas or DeepEval docs for the faithfulness metric definitions (you will implement a simple version yourself first).

**Build exercise:** `src/aihc/metrics.py` (deterministic layer), `src/aihc/rag/` (chunk, index, retrieve, answer), `src/aihc/evals/run_evals.py` with `golden.jsonl`. Then replace the synthetic data with one real export (a CRM lead export or a spend report) and re-run the evals.

**Done when:** `python -m aihc.evals.run_evals` reports numeric exact-match ≥ 95% and citation validity 100%, and asking the same question twice returns an identical answer with `cache_hit=True`.

---

## Module 4: A permanent, scheduled analytical agent (Week 4)

**What you asked:** a structured and permanent agent that automates analysis and insights on any cadence.

**Core content**

An agent is a loop: the model receives a goal and a set of tools, chooses a tool, the harness runs it, the result goes back to the model, repeat until the model emits a final answer. "Permanent" means three additional things: it has state between runs (what it reported last time), it is triggered by a scheduler rather than a person, and its output lands somewhere durable.

Design that holds up:
- *Tools are the deterministic layer from Module 3.* The agent calls `query_metrics`, `compare_periods`, `search_docs`, `write_report`. It never computes numbers itself.
- *State store:* a JSON file (later a table) with the last run's key figures, so the agent can say "cost per lead is up 12% vs. last week" and flag only what changed.
- *Cadence options:* cron (Linux/macOS), Task Scheduler (Windows), GitHub Actions on a schedule (free, no server), a hosted worker (Modal, Railway, a small VM), or a scheduled task inside an assistant product. The repo ships a GitHub Actions workflow and a Windows Task Scheduler command.
- *Output:* a Markdown report to `reports/YYYY-MM-DD.md`, and optionally email or Slack.
- *Guardrails:* max tool calls per run, timeout, cost ceiling, and a dry-run flag. An agent on a schedule with no ceiling is a billing incident.
- *Observability:* log every tool call and token count per run to `runs.jsonl`.

**Reading**
- Anthropic, "Building effective agents". Read the section on when not to build an agent.
- Anthropic docs on the Agent SDK, for when you want the loop managed for you.
- Any article on idempotent scheduled jobs; the concept transfers directly.

**Build exercise:** `src/aihc/agent/insights_agent.py` plus `state.json`, the GitHub Actions workflow, and the Task Scheduler command. Run it three times with the data changed between runs and check that the report only flags real deltas.

**Done when:** the agent runs unattended on a schedule for a week, each report is correct against the metrics layer, and the total spend per run is known and bounded.

---

## Module 5: The harness (Week 5, first half)

**What you asked:** what is a harness, is Cursor one, what else is there.

**Core content**

Definition: the harness is everything around the model that turns text prediction into work. Concretely: the system prompt, the tool set, the loop, file/shell access, permission rules, context management (what goes into the prompt and what is summarized away), and stop conditions. The model is interchangeable; the harness is the product. Cursor, Claude Code, GitHub Copilot agent mode, Codex CLI, Aider, Windsurf, and Cline are all harnesses aimed at coding. Cowork, OpenClaw-style assistants, and n8n/Zapier agents are harnesses aimed at other tasks.

How to compare harnesses: what tools does it expose, how does it manage context (does it read your repo, index it, summarize it), does it run in your editor or your terminal or a server, can it run unattended, how are permissions handled, can you script it, and what does it cost (subscription vs. per-token).

Cursor specifically: an editor with an embedded harness. It indexes the repo, exposes file edit and terminal tools, and reads `.cursor/rules/*.mdc` as persistent instructions. Strengths: tight edit loop, good for building. Limits: it is interactive and editor-bound, so it is not what runs your Module 4 agent on a schedule.

Other ways to build a harness:
1. Use a terminal harness (Claude Code, Codex CLI). Scriptable, can run headless in CI. `CLAUDE.md` plays the role of `.cursor/rules`.
2. Use an SDK (Anthropic Agent SDK, OpenAI Agents SDK, LangGraph, Pydantic AI). You write the goal and tools; the loop is provided.
3. Write it yourself. The repo includes `modules/05-harness/mini_harness.py`, about 60 lines: a loop, two tools, a stop condition. Once you have written this, every commercial harness reads as a configuration of the same thing.

**Reading**
- Cursor docs: Rules, Agent mode, MCP.
- Claude Code docs: CLAUDE.md, hooks, headless mode.
- Anthropic Agent SDK overview.

**Build exercise:** (a) write `.cursor/rules/project.mdc` for this repo (a draft is provided), (b) run `mini_harness.py` and add a third tool, (c) run the same task through Cursor and through the mini harness and write down the differences.

**Done when:** you can explain what Cursor does with your prompt before the model sees it, and you have a one-page comparison of three harnesses for your use case.

---

## Module 6: Productizing (Week 5, second half)

**What you asked:** how do I productize my work on AI.

**Core content**

Productizing means someone other than you can get the outcome without you in the loop. Levels, in order of effort:

1. *Repeatable service:* the same repo, run by you, for a client. Deliverable is the report. This is where CubedHealth-style engagements start. Price per report or per month.
2. *Packaged tool:* CLI or hosted API (`FastAPI` scaffold in `modules/06-productize/api.py`), client provides data exports, gets reports and a Q&A endpoint. Price per seat or per location.
3. *Product:* multi-tenant, self-serve data connection, dashboards, billing. Requires an engineering team; out of scope for this project but the path is clear once level 2 works.

Non-negotiables at every level:
- *Evaluation gate:* the golden-set eval from Module 3 runs in CI; no change ships if it regresses. This is your quality argument to a buyer.
- *Data handling:* healthcare marketing data is often not PHI, but lead exports with names, phone numbers, and treatment interest can be. Decide up front: strip identifiers before ingestion, sign a BAA with your model provider if PHI is unavoidable, and document it. The repo has a checklist.
- *Cost model:* tokens per report × runs per month × clients. Know it before quoting.
- *Model pinning and fallbacks:* pin model IDs, test the upgrade path on the eval set before switching.
- *Positioning:* you are not selling "AI". You are selling a weekly answer to "where is patient acquisition leaking and what should we change", with numbers that reconcile to the client's own data. The AI is how the margin works.

**Reading**
- Anthropic's usage and privacy policies for API data handling.
- HHS guidance on marketing and HIPAA (the marketing-specific rules are short).
- One SaaS pricing primer (e.g. from a16z or Lenny's Newsletter) on usage-based vs. seat-based.

**Build exercise:** `modules/06-productize/`: FastAPI wrapper with `/ask` and `/report`, `pricing_model.py`, `PHI_CHECKLIST.md`, and a one-page offer document for a DSO with 20 locations.

**Done when:** you have a runnable API, a cost-per-client number, and a written offer you would actually send.

---

## Capstone (end of Week 5 or Week 6)

Replace the synthetic data with one real client-shaped export (anonymized). Regenerate the golden set against it. Run the scheduled agent for two consecutive periods. Ship the report to yourself by email or Slack. Write a half-page on what you would change before showing it to a paying client.

## How to use Cursor with this repo

Open the folder. The rules file tells Cursor about the module structure, the "numbers never come from the model" rule, and the eval gate. When you work on an exercise, open the module README and the target file, and ask Cursor to explain before asking it to change. Run tests with `pytest`. The repo is deliberately small enough that you can read every file; do that once before Week 3.
