# Glossary, by layer

## Model layer
- **Parameter / weight**: a learned number. "7B" = seven billion of them. Parameter count drives memory and cost.
- **Token**: the unit the model reads and writes; roughly 3/4 of an English word. Pricing and context limits are in tokens.
- **Tokenizer**: the fixed mapping from text to tokens. Same for every call to a given model.
- **Context window**: the maximum tokens (input + output) in one call. Everything the model "knows" about your situation must fit here or be in the weights.
- **Embedding**: a vector of numbers representing a token, a chunk, or a document, such that similar meanings are nearby. Used for retrieval.
- **Logits**: the model's raw scores over the vocabulary for the next token, before sampling.
- **Temperature / top-p**: sampling controls. Temperature 0 picks the most likely token; higher values add randomness.
- **System prompt**: standing instructions placed before the conversation.
- **Few-shot**: examples in the prompt demonstrating the desired behavior.
- **Chain of thought / extended thinking**: the model writing intermediate reasoning before the answer; improves accuracy on multi-step problems, costs tokens.
- **Base vs. instruct vs. reasoning model**: pretrained only vs. post-trained to follow instructions vs. post-trained to reason at length before answering.
- **Open-weights**: the parameters are downloadable (Llama, Mistral, Qwen). "Open source" is often a misnomer because training data and code are usually not released.
- **Fine-tuning / LoRA**: further training on your examples. LoRA trains a small adapter instead of all weights. Teaches format and style; does not reliably teach facts.
- **Distillation**: training a small model to imitate a large one.
- **Quantization**: storing weights in fewer bits (int8, int4) to shrink memory and speed inference at a small quality cost.

## Data layer
- **Pretraining corpus**: trillions of tokens of web, books, code. Where general knowledge comes from.
- **SFT / preference data**: human-written demonstrations and human rankings used in post-training.
- **Synthetic data**: model-generated training data.
- **Contamination**: test data leaking into training data, inflating benchmark scores.

## Application layer
- **Prompt template**: a prompt with slots filled at runtime. Version it.
- **Structured output**: forcing the model's reply into a schema (JSON) so code can consume it.
- **Tool / function calling**: the model emits a function name and arguments; your code executes it and returns the result.
- **RAG (retrieval-augmented generation)**: fetch relevant text at question time and put it in the prompt.
- **Chunking**: splitting documents into retrievable pieces. Chunk size and boundaries affect retrieval quality more than most prompt changes.
- **Vector database**: stores embeddings for nearest-neighbor search (Chroma, pgvector, Pinecone, etc.).
- **BM25**: classic keyword ranking. Deterministic, cheap, often good enough; combine with vectors for "hybrid search".
- **Reranker**: a second model that reorders retrieved chunks by relevance to the question.
- **Agent**: a model in a loop with tools, deciding which tool to call next until a goal is met.
- **MCP (Model Context Protocol)**: a standard for exposing tools and data sources to models and harnesses.
- **Memory**: state carried across calls or sessions (a file, a table, a summary), not something inside the model.
- **Guardrails**: input/output checks outside the model (schema validation, citation checks, PII filters, allow-lists).
- **Prompt injection**: text in the model's input (a web page, a document, a form submission) that tries to give it instructions. The main security problem of LLM applications.

## Evaluation layer
- **Golden set**: questions with known correct answers, used as a regression test.
- **Exact match / rubric / LLM-as-judge**: three ways to score an answer: string equality, human rubric, or another model scoring against criteria.
- **Faithfulness**: does the answer follow from the provided context? Distinct from correctness.
- **Recall@k**: was the right chunk among the top k retrieved?
- **Regression**: a change (prompt, model, chunking) that makes previously passing evals fail.
- **Tracing / observability**: logging every prompt, tool call, token count, and latency per request.

## Operations layer
- **Latency / throughput / tokens per second**: time to first token, requests per minute, generation speed.
- **Prompt caching**: the provider stores a prompt prefix so repeated calls are cheaper and faster.
- **Batch API**: asynchronous bulk processing at a discount.
- **Rate limits**: per-minute caps on requests and tokens.
- **Model routing / fallback**: send easy questions to a cheap model, hard ones to an expensive one; switch providers on failure.
- **Determinism**: same input, same output. Temperature 0 helps; caching keyed on inputs guarantees it; computing numbers in code removes the largest source of drift.
- **Fingerprint**: a hash of the inputs (data, prompt version, model ID) used to key caches and detect change.
