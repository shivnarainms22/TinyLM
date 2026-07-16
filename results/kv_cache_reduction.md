# KV-cache reduction — MLA vs. MHA (closing the Phase 5 hypothesis clause)

The pinned hypothesis has two clauses. The benchmark-parity clause is answered by
`results/hpc_rerun_ablation.md`. This document answers the second:

> "...while demonstrating a **measurable KV-cache memory reduction at inference
> versus an equivalent MHA baseline (Run A)**."

**Verdict: confirmed. Multi-head Latent Attention gives a 3.56× smaller KV cache
(71.9% reduction) than an equivalent MHA at the same `d_model`, heads, and layers.**

## Where the number comes from

At inference the KV cache stores, per token per layer:

| Variant | What is cached | Floats / token / layer |
|---|---|---:|
| **MHA** (Run A) | full **K** and **V**: `2 · n_heads · head_dim` = `2 · d_model` | **2048** |
| **MLA** (Run D) | compressed **latent** + decoupled **RoPE key**: `d_latent + d_rope` | **576** |

With the locked config (`d_model 1024`, `n_heads 16`, `head_dim 64`, `d_latent 512`,
`d_rope 64`):

```
ratio      = 2048 / 576 = 3.56×
reduction  = 1 − 576/2048 = 71.9%
```

MLA never materializes per-head K/V in the cache — it stores the `d_latent`-wide
latent (plus a single shared `d_rope`-wide RoPE key) and re-expands to per-head K/V
on the fly via `k_up`/`v_up` at each step. That compression is the whole point of
the architecture and is independent of sequence length or batch.

## Concrete footprint

Full 2048-token context, batch 1, bf16 (2 bytes/element), all 18 layers:

| | MHA (Run A) | MLA (Run D) | Saved |
|---|---:|---:|---:|
| KV cache | **144.0 MiB** | **40.5 MiB** | **103.5 MiB (3.56×)** |

The gap scales linearly with batch × sequence length: at a 32-sequence,
2048-token serving load it is ~4.5 GiB (MHA) vs ~1.3 GiB (MLA).

Reproduce: `python scripts/kv_cache_footprint.py` (add `--seq-len`, `--batch`).

## How this is verified

- **Cache width** — that MLA's per-layer cached tensors are actually
  `(d_latent, d_rope)` wide and *not* `n_heads · head_dim`, when decoding one token
  at a time — is asserted end-to-end in
  `tests/test_mla.py::test_kv_cache_shape_incremental` (feeds tokens through
  `MLAttention.forward_with_cache` and checks the accumulated cache shapes).
- **The arithmetic** turning that width into the 3.56× ratio and the MiB footprint
  is pinned in `tests/test_kv_cache_footprint.py`, computed from `ModelConfig` (no
  hard-coded constants) by `scripts/kv_cache_footprint.py`.

So the headline is not an estimate: the cached shapes are exercised in code and the
ratio is derived directly from the locked architecture.

## Honest caveats

- **Memory, not compute.** MLA trades cache memory for a little extra work per step
  (the `k_up`/`v_up` re-expansion of the latent). The claim is strictly about
  inference *memory*, which is what the hypothesis asked for.
- **This implementation's cache form.** DeepSeek-V2's fully "absorbed" MLA can fold
  `k_up`/`v_up` into the query/output projections and cache only the latent, shaving
  the `d_rope` term. This repo caches `d_latent + d_rope = 576`/token/layer as
  written; the number above reflects the code that actually runs, not the theoretical
  floor.
- **Same-shape comparison.** Run A (MHA) and Run D (MLA) share `d_model`, `n_heads`,
  and `n_layers`, so this is an apples-to-apples "equivalent MHA baseline" exactly as
  the hypothesis specifies — not a comparison against a differently-sized model.
