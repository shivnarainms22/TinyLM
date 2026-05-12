---
date: 2026-05-12
topic: tinylm-phase1
status: approved
reference: 250M_SLM_Implementation_Plan_revised.pdf (Phase 1, Steps 1–4)
predecessor: 2026-05-12-tinylm-phase0-design.md
---

# TinyLM — Phase 1 Design Lock-in

## Project context

Phase 0 delivered the five lock-in artifacts (hypothesis, baseline eval JSON,
ablation plan, benchmark suite, per-project CLAUDE.md). Phase 1 lays the
architectural foundation: the model code (MHA + MLA variants), the Muon
optimizer, and a defensive test suite that has to be green before any
training run.

No GPU work in Phase 1. Everything runs on Windows CPU in PyTorch eager
mode. Training, data pipeline, evaluation wrappers, and inference cache code
are all out of scope here.

## Decisions locked in brainstorming (2026-05-12)

| Decision | Value | Reasoning |
|---|---|---|
| modded-nanogpt relationship | Vendor Muon into `src/tinylm/muon.py` with attribution comment; skip the FineWeb dataloader port (Phase 3 owns the data pipeline) | Full ownership of every file in `src/tinylm/` for interviews; submodules are fragile on Windows + Colab; a full fork carries dead weight (`train_gpt2.py`, `record.py`) we never touch. MIT licensed — vendoring with attribution is standard. |
| Model file layout | Single `src/tinylm/model.py` (RMSNorm, MHA, MLA, FFN, Block, LM head) | 275M model is small enough that one file reads cleanly top-to-bottom. Splitting adds navigation overhead without benefit. |
| MLA source | Port + simplify from DeepSeek-V2 HuggingFace `transformers` reference; attribution comment at top of class | Battle-tested math, especially the decoupled-RoPE detail the PDF flags as the #1 bug source. Paper-from-scratch is too risky; clean-room third-party impls are less validated. |
| Test scope | Maximal coverage — 8 tests in `test_mla.py`, 4 in `test_muon.py` (PDF requires 3 MLA tests minimum) | Every test catches a class of silent bug that would waste training compute. User explicitly chose maximal. |
| Inherited conventions (from TinyLlama baseline) | RMSNorm, SwiGLU FFN, no bias on linears, RoPE base 10000 | Not real decisions — they follow from baseline lock in Phase 0. |
| Model config | `dataclass` in `model.py` | Phase 1 only needs models instantiable for tests. YAML run configs land in Phase 2 with `train.py`. |
| Locked architecture dims (from `docs/ablation_plan.md`) | `n_layers=18, d_model=1024, n_heads=16, d_latent=512, d_rope=64, ffn_hidden=2816, ctx=2048, tie_weights=True, vocab_size=32000` | Already locked in Phase 0 ablation plan. Phase 1 implements to these. |
| CLAUDE.md typo fix | Change "four MLA unit tests" → "three PDF-mandatory MLA tests plus defensive coverage" | PDF Phase 1 Step 3 actually lists 3 tests (lines 175–196 of `docx_text.txt`), not 4. |

## Phase 1 deliverables (this spec → 6 artifacts)

| # | Artifact | Path | Notes |
|---|---|---|---|
| 1 | Project dependencies, pinned | `pyproject.toml` | torch, pytest. No transformers/datasets/lm-eval yet — those land per-phase. |
| 2 | Vendored Muon optimizer | `src/tinylm/muon.py` | Newton-Schulz + Muon class, ported from Keller Jordan's reference. Attribution at top. |
| 3 | Model code (MHA + MLA variants) | `src/tinylm/model.py` | RMSNorm, RoPE helpers, MHAttention, MLAttention, SwiGLU FFN, Block, TinyLM (full model), `ModelConfig` dataclass. |
| 4 | MLA test suite | `tests/test_mla.py` | 3 PDF + 5 defensive = 8 tests. |
| 5 | Muon test suite | `tests/test_muon.py` | 4 tests covering NS convergence, non-square transpose, Nesterov vs vanilla, param filtering. |
| 6 | CLAUDE.md correction | `CLAUDE.md` | "four MLA unit tests" → corrected phrasing. |

## MLA test list (canonical)

| # | Name | Asserts | Source |
|---|---|---|---|
| 1 | `test_kv_latent_compressed` | `kv_down(x).shape[-1] == d_latent (512)`, not `n_heads*d_head` | PDF Step 3 Test 1 |
| 2 | `test_output_shape_preserved` | `mla(x).shape == (B, T, d_model)` | PDF Step 3 Test 2 |
| 3 | `test_causal_masking` | Perturbing position `t` does not change outputs at positions `< t` | PDF Step 3 Test 3 |
| 4 | `test_rope_decoupling` | Positional info is carried only by the `d_rope` projection: zeroing the RoPE branch destroys position sensitivity; zeroing the latent branch keeps causal/positional structure intact within numerical tolerance | Defensive — flagged in PDF prose as #1 bug source |
| 5 | `test_total_param_count` | `sum(p.numel() for p in TinyLM(cfg).parameters())` falls in `[270e6, 285e6]` for the MLA variant | Defensive — PDF Phase 4 Step 0 explicitly tells us to verify this before training |
| 6 | `test_gradient_flow` | After a forward+backward on random input/labels, every learnable parameter has a non-zero `.grad` (no dead branches) | Defensive — catches frozen sub-modules |
| 7 | `test_mla_mha_equivalence` | At `d_latent=d_model, d_rope=0`, MLA's output should match a standard MHA module on the same inputs within `atol=1e-4`, when weights are aligned | Defensive — sanity that the MLA compression is the only source of difference |
| 8 | `test_kv_cache_shape_incremental` | Running `forward_with_cache(token_t)` produces a per-layer cache where the cached tensor's last dim is `d_latent + d_rope` (not `n_heads * d_head`) | Defensive — Phase 5 KV-reduction headline depends on this being correct |

## Muon test list (canonical)

| # | Name | Asserts |
|---|---|---|
| 1 | `test_newton_schulz_orthogonalizes` | For a random matrix `X`, `newton_schulz(X, steps=5)` returns `Y` with `Y @ Y.T ≈ I_k` (within tolerance) on the smaller dim, i.e. singular values cluster near 1 |
| 2 | `test_non_square_transpose_trick` | NS works for both tall (`m > n`) and wide (`m < n`) matrices and returns the original shape |
| 3 | `test_nesterov_momentum_applied` | One Muon step on a small matrix with a fixed gradient produces an update that differs from a vanilla-momentum baseline (proves Nesterov look-ahead is wired in) |
| 4 | `test_param_filter_excludes_embed_lm_head_norm_bias` | The helper that partitions `model.named_parameters()` puts embedding / LM head / norm / bias into the AdamW group, not the Muon group |

## Repo structure after Phase 1

```
D:\TinyLM\
├── pyproject.toml            # NEW: torch + pytest
├── src/tinylm/
│   ├── __init__.py           # NEW
│   ├── model.py              # NEW: RMSNorm, RoPE, MHA, MLA, FFN, Block, TinyLM, ModelConfig
│   └── muon.py               # NEW: newton_schulz, Muon, param-partition helper
├── tests/
│   ├── __init__.py           # NEW
│   ├── test_mla.py           # NEW: 8 tests
│   └── test_muon.py          # NEW: 4 tests
└── CLAUDE.md                 # MODIFIED: typo fix
```

## Acceptance gate for Phase 1

```bash
pytest tests/ -v
```

Must report **12 passed, 0 failed** on Windows CPU eager mode. No
`torch.compile`. No GPU required.

If any test fails the phase is not done — no exceptions, no skips.

## Out of scope for Phase 1 (deferred)

- `train.py` (Phase 2)
- `data.py` / FineWeb shard loader / annealing mix (Phase 3)
- YAML configs `configs/run_{A,B,C,D}_*.yaml` (Phase 2 — populated with `train.py`)
- `eval_wrapper.py` (HF PreTrainedModel adapter for lm-eval) (Phase 5)
- `inference.py` (KV-cache memory measurement script) (Phase 5)
- Any GPU run, including the Phase 2 toy run

## Risks / open issues

- **DeepSeek-V2 reference port complexity.** The HF DeepSeek-V2 modeling file
  is ~800 lines and supports features we don't need (e.g., expert routing,
  YARN scaling). The port must aggressively strip down. Mitigation: keep
  `MLAttention` under ~150 lines and use `test_mla_mha_equivalence` to
  validate the simplification.
- **`torch.compile` is disabled in Phase 1.** Training in Phase 2+ may
  surface compile-only issues. Mitigation: re-run the test suite under
  `torch.compile` mode at the start of Phase 2 as a smoke check before the
  toy run.
- **RoPE decoupling test is the hardest to write correctly.** Testing
  "positional info is carried *only* by the d_rope branch" needs a clever
  experimental setup. Plan to use a position-shift invariance check.
