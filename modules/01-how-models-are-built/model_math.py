"""Module 1 exercise: the arithmetic behind "how big, how much memory, how much compute".

Run: python modules/01-how-models-are-built/model_math.py
Then: change CONFIGS, or add a config for a model you use, and check the numbers against
the vendor's published figures.

Part A: parameter count, memory, FLOPs, serving ceiling for a dense transformer.
Part B: a single attention head in NumPy, so "attention" stops being a word.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelConfig:
    name: str
    layers: int
    d_model: int        # hidden size
    n_heads: int
    d_ff: int           # feed-forward inner size (often 4x d_model, or ~2.7x for SwiGLU)
    vocab: int
    context: int


CONFIGS = [
    ModelConfig("GPT-2 small (124M)", 12, 768, 12, 3072, 50257, 1024),
    ModelConfig("Llama-2-7B-like", 32, 4096, 32, 11008, 32000, 4096),
    ModelConfig("70B-class", 80, 8192, 64, 28672, 32000, 8192),
]

BYTES = {"fp32": 4, "bf16": 2, "int8": 1, "int4": 0.5}


def param_count(c: ModelConfig) -> dict:
    attn = 4 * c.d_model * c.d_model                    # Q, K, V, O projections
    ff = 3 * c.d_model * c.d_ff if c.d_ff < 4 * c.d_model else 2 * c.d_model * c.d_ff  # SwiGLU has 3 matrices
    norms = 2 * 2 * c.d_model
    per_layer = attn + ff + norms
    embed = c.vocab * c.d_model                         # tied output head assumed
    total = c.layers * per_layer + embed
    return {"attention": c.layers * attn, "feed_forward": c.layers * ff, "embeddings": embed, "total": total}


def training_memory_gb(params: int) -> float:
    """Mixed precision + Adam: bf16 weights (2) + fp32 master (4) + grads (2) + 2 optimizer moments (8) = 16 B/param.
    Activations are extra and depend on batch and context; this is the floor."""
    return params * 16 / 1e9


def inference_memory_gb(params: int, precision: str) -> float:
    return params * BYTES[precision] / 1e9


def kv_cache_gb(c: ModelConfig, tokens: int, precision: str = "bf16") -> float:
    """Per sequence: 2 (K and V) * layers * d_model * tokens * bytes. This is why long context costs memory."""
    return 2 * c.layers * c.d_model * tokens * BYTES[precision] / 1e9


def training_flops(params: int, tokens: int) -> float:
    return 6.0 * params * tokens                        # forward ~2ND, backward ~4ND


def train_days(flops: float, n_gpus: int, gpu_tflops: float = 989, mfu: float = 0.4) -> float:
    """H100 bf16 dense peak ~989 TFLOPs; real utilization (MFU) 30-45%."""
    return flops / (n_gpus * gpu_tflops * 1e12 * mfu) / 86400


def decode_tokens_per_sec_ceiling(params: int, precision: str, hbm_gb_per_s: float = 3350) -> float:
    """Single-stream decode is memory-bandwidth bound: every token streams all weights once.
    H100 HBM3 ~3.35 TB/s. Batching amortizes this, which is why serving engines batch."""
    return hbm_gb_per_s / (params * BYTES[precision] / 1e9)


def part_a() -> None:
    for c in CONFIGS:
        p = param_count(c)
        n = p["total"]
        print(f"\n== {c.name} ==")
        print(f"params: {n/1e9:.2f}B  (attn {p['attention']/n:.0%}, ff {p['feed_forward']/n:.0%}, embed {p['embeddings']/n:.0%})")
        print(f"training memory floor: {training_memory_gb(n):.0f} GB  -> {training_memory_gb(n)/80:.1f} x 80GB GPUs before activations")
        for prec in ("bf16", "int8", "int4"):
            print(f"inference weights {prec}: {inference_memory_gb(n, prec):.1f} GB; "
                  f"single-stream decode ceiling ~{decode_tokens_per_sec_ceiling(n, prec):.0f} tok/s on one H100")
        print(f"KV cache at full context ({c.context} tok, bf16): {kv_cache_gb(c, c.context):.2f} GB per sequence")
        for toks in (1e12, 15e12):
            f = training_flops(n, toks)
            print(f"train on {toks/1e12:.0f}T tokens: {f:.2e} FLOPs -> {train_days(f, 1024):.0f} days on 1,024 H100s at 40% MFU")


def attention_head(x: np.ndarray, d_k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Causal single-head self-attention. x: (seq, d_model). Returns (output, attention weights)."""
    rng = np.random.default_rng(seed)
    d_model = x.shape[1]
    Wq, Wk, Wv = (rng.normal(0, d_model ** -0.5, (d_model, d_k)) for _ in range(3))
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.T / np.sqrt(d_k)                      # (seq, seq): how much each token attends to each other
    mask = np.triu(np.ones_like(scores, dtype=bool), k=1)  # causal: no looking at the future
    scores = np.where(mask, -np.inf, scores)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)         # softmax, rows sum to 1
    return weights @ V, weights


def part_b() -> None:
    seq, d_model, d_k = 6, 16, 8
    x = np.random.default_rng(1).normal(size=(seq, d_model))
    out, w = attention_head(x, d_k)
    print("\n== attention head ==")
    print("output shape", out.shape, "(seq, d_k)")
    print("weights rows sum to 1:", np.allclose(w.sum(axis=1), 1.0))
    print("causal (upper triangle zero):", np.allclose(np.triu(w, k=1), 0))
    print(np.round(w, 2))
    # Exercise: (1) make it multi-head by splitting d_model across heads and concatenating;
    # (2) add the feed-forward block (two matrices + nonlinearity) and a residual connection;
    # (3) count the parameters you created and compare with param_count().


if __name__ == "__main__":
    part_a()
    part_b()
