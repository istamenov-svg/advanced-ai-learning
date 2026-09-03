# Module 1: How models are built (Week 1, ~10 hours)

Full content and reading list are in `CURRICULUM.md`. This file is the work plan.

## Hours 1-4: read and watch
1. Karpathy, "Let's build GPT" (2h). Follow along in a scratch file; do not try to make it perfect.
2. The Illustrated Transformer (45 min).
3. Chinchilla abstract and Figure 1 (30 min). Write one sentence on what "compute-optimal" means.

## Hours 5-7: run and modify `model_math.py`
```bash
python modules/01-how-models-are-built/model_math.py
```
Tasks:
- Add a config for a model you actually use via API (look up published parameter counts where available; for closed models, estimate from pricing and latency and write down your reasoning).
- The 70B config over-counts attention parameters because real 70B models use grouped-query attention (fewer K/V heads). Add an `n_kv_heads` field and correct `param_count`. Compare with the published 70B figure.
- Change `mfu` from 0.4 to 0.25 and note how training days move. This is the gap between a well-run and a poorly-run cluster.

## Hours 8-9: attention in NumPy
- Complete the three exercises listed at the bottom of `part_b()`: multi-head, feed-forward + residual, parameter count reconciliation.

## Hour 10: write-up (half a page, in this folder as `notes.md`)
Answer without looking:
1. Why does a 70B model not fit on one 80GB GPU for training but does fit for int4 inference?
2. What does 6ND mean and where do N and D come from?
3. Why is single-stream decoding limited by memory bandwidth and not by FLOPs?
4. When would you fine-tune (LoRA) and when would you refuse to?

## Done when
All four questions answered correctly in your own words, and `model_math.py` has your added config and the GQA fix.
