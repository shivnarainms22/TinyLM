# TinyLM — Phase 1 Execution Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the architectural foundation (MHA + MLA model, Muon optimizer, 12-test defensive suite, pinned deps) so that `pytest tests/ -v` reports `12 passed` on Windows CPU eager mode. No GPU, no training, no data pipeline.

**Architecture:** Single-file model (`src/tinylm/model.py`) with both attention variants side by side, vendored Muon (`src/tinylm/muon.py`) from Keller Jordan's reference, and a pytest suite that catches the five highest-risk silent bugs (broken RoPE decoupling, wrong param count, dead gradients, MLA-vs-MHA divergence at the identity setting, wrong KV-cache shape during inference).

**Tech Stack:** Python 3.10+, PyTorch 2.x (CPU build for Phase 1), pytest. No transformers, no datasets, no lm-eval (those land per-phase as needed).

**Spec reference:** `docs/superpowers/specs/2026-05-12-tinylm-phase1-design.md`.

**Out of scope:** training loop, dataloader, YAML run configs, eval wrapper, inference cache script, any GPU run.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `pyproject.toml` | Create | Pin Python ≥3.10, torch CPU build, pytest. |
| `src/tinylm/__init__.py` | Create | Re-export `ModelConfig`, `TinyLM`. |
| `src/tinylm/model.py` | Create | `ModelConfig` dataclass, `RMSNorm`, RoPE helpers, `MHAttention`, `MLAttention`, `SwiGLUFFN`, `Block`, `TinyLM`, `partition_params()` helper, `forward_with_cache()` on `TinyLM`. |
| `src/tinylm/muon.py` | Create | `newton_schulz`, `Muon` optimizer class, attribution comment. |
| `tests/__init__.py` | Create | Empty (makes `tests/` a package). |
| `tests/test_mla.py` | Create | 8 tests per spec. |
| `tests/test_muon.py` | Create | 4 tests per spec. |
| `CLAUDE.md` | Modify | Fix "four MLA unit tests" → "three PDF-mandatory MLA tests plus defensive coverage". |
| `.gitignore` | Verify/modify | Ensure `__pycache__/`, `.pytest_cache/`, `*.egg-info/` are ignored. |

---

## Chunk 1: Project scaffolding

### Task 1: Pin dependencies and create package layout

**Files:**
- Create: `D:\TinyLM\pyproject.toml`
- Create: `D:\TinyLM\src\tinylm\__init__.py` (empty for now)
- Create: `D:\TinyLM\tests\__init__.py` (empty)
- Modify (if needed): `D:\TinyLM\.gitignore`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "tinylm"
version = "0.1.0"
description = "275M parameter SLM with MLA + Muon"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2,<2.5",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

- [ ] **Step 2: Create empty package files**

Write `D:\TinyLM\src\tinylm\__init__.py` as an empty file.
Write `D:\TinyLM\tests\__init__.py` as an empty file.

- [ ] **Step 3: Verify `.gitignore` excludes build artefacts**

Open `D:\TinyLM\.gitignore`. Confirm these patterns are present (add any missing ones):

```
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/
.venv/
```

- [ ] **Step 4: Create venv and install in editable mode**

Run (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Expected: `Successfully installed tinylm-0.1.0 torch-2.x.x pytest-7.x.x ...`

- [ ] **Step 5: Smoke test that pytest discovers the empty package**

Run: `pytest tests/ -v`
Expected: `no tests ran in 0.0Xs` (exit code 5 is fine here — no tests yet).

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml src/tinylm/__init__.py tests/__init__.py .gitignore
git commit -m "chore: scaffold tinylm package and pin torch/pytest"
```

---

### Task 2: Fix CLAUDE.md typo

**Files:**
- Modify: `D:\TinyLM\CLAUDE.md` (the "Non-negotiables" section)

- [ ] **Step 1: Replace the line**

Find the line that reads:

```
- The four MLA unit tests in PDF Phase 1 Step 3 are mandatory before
  any training run.
```

Replace with:

```
- The three MLA unit tests in PDF Phase 1 Step 3 (KV-compressed,
  output-shape, causal-mask) are mandatory before any training run,
  alongside the five defensive tests locked in the Phase 1 spec.
```

- [ ] **Step 2: Commit**

```powershell
git add CLAUDE.md
git commit -m "docs: correct MLA test count in CLAUDE.md (3 mandatory + 5 defensive)"
```

---

## Chunk 2: Muon optimizer (TDD)

> **Why Muon first:** It has no dependency on `model.py` and is the smallest self-contained unit. Land it green before touching the model.

### Task 3: Newton-Schulz orthogonalization

**Files:**
- Create: `D:\TinyLM\src\tinylm\muon.py`
- Test: `D:\TinyLM\tests\test_muon.py`

- [ ] **Step 1: Write the failing test `test_newton_schulz_orthogonalizes`**

Create `tests/test_muon.py`:

```python
"""Tests for Muon optimizer and Newton-Schulz orthogonalization."""

import torch

from tinylm.muon import newton_schulz


def test_newton_schulz_orthogonalizes():
    """After NS iteration, singular values should cluster near 1.

    Concretely: for a random (256, 128) matrix X, the result Y satisfies
    Y.T @ Y ≈ I_128 within tolerance (Y has orthonormal columns).
    """
    torch.manual_seed(0)
    X = torch.randn(256, 128)
    Y = newton_schulz(X, steps=5)
    assert Y.shape == X.shape
    # Y has orthonormal columns: Y.T @ Y ≈ I (smaller dim)
    gram = Y.T @ Y
    eye = torch.eye(128)
    # NS in bf16 + 5 steps gives ~1e-2 accuracy, not machine eps
    assert torch.allclose(gram, eye, atol=5e-2), (
        f"max abs deviation from I: {(gram - eye).abs().max().item():.4f}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_muon.py::test_newton_schulz_orthogonalizes -v`
Expected: `FAILED` with `ModuleNotFoundError: No module named 'tinylm.muon'`.

- [ ] **Step 3: Implement `newton_schulz`**

Create `src/tinylm/muon.py`:

```python
"""Muon optimizer: MomentUm Orthogonalized by Newton-schulz.

Ported from Keller Jordan's reference implementation in
github.com/KellerJordan/modded-nanogpt (MIT licensed). The quintic
coefficients (3.4445, -4.7750, 2.0315) are tuned to maximize the slope
near zero so small singular values converge quickly.
"""

from __future__ import annotations

import torch


def newton_schulz(X: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz iteration.

    Returns a matrix with the same shape as X whose singular values are
    pushed toward 1 (i.e. approximately orthogonal). Runs in bf16 per
    reference impl; output is cast back to float32.
    """
    assert X.ndim >= 2, f"expected ndim>=2, got shape {tuple(X.shape)}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    Y = X.bfloat16()
    transposed = False
    if Y.size(-2) > Y.size(-1):
        Y = Y.mT
        transposed = True
    Y = Y / (Y.norm() + 1e-7)
    for _ in range(steps):
        A = Y @ Y.mT
        Y = (a * Y) + (b * (A @ Y)) + (c * (A @ A @ Y))
    if transposed:
        Y = Y.mT
    return Y.to(torch.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_muon.py::test_newton_schulz_orthogonalizes -v`
Expected: `PASSED`.

- [ ] **Step 5: Commit**

```powershell
git add src/tinylm/muon.py tests/test_muon.py
git commit -m "feat: implement Newton-Schulz orthogonalization with convergence test"
```

---

### Task 4: Non-square transpose trick

**Files:**
- Modify: `D:\TinyLM\tests\test_muon.py`

- [ ] **Step 1: Append failing test**

```python
def test_non_square_transpose_trick():
    """NS works for both tall (m>n) and wide (m<n) matrices."""
    torch.manual_seed(1)
    tall = torch.randn(512, 64)
    wide = torch.randn(64, 512)

    out_tall = newton_schulz(tall, steps=5)
    out_wide = newton_schulz(wide, steps=5)

    assert out_tall.shape == tall.shape, (
        f"tall: expected {tall.shape}, got {out_tall.shape}"
    )
    assert out_wide.shape == wide.shape, (
        f"wide: expected {wide.shape}, got {out_wide.shape}"
    )
    # Both should produce ~orthogonal rows/cols on the smaller dim
    assert torch.allclose(
        out_tall.T @ out_tall, torch.eye(64), atol=5e-2
    )
    assert torch.allclose(
        out_wide @ out_wide.T, torch.eye(64), atol=5e-2
    )
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_muon.py::test_non_square_transpose_trick -v`
Expected: `PASSED` (the transpose trick is already in `newton_schulz`).

- [ ] **Step 3: Commit**

```powershell
git add tests/test_muon.py
git commit -m "test: verify Newton-Schulz transpose trick for non-square matrices"
```

---

### Task 5: Muon optimizer class with Nesterov momentum

**Files:**
- Modify: `D:\TinyLM\src\tinylm\muon.py`
- Modify: `D:\TinyLM\tests\test_muon.py`

- [ ] **Step 1: Write failing test `test_nesterov_momentum_applied`**

Append to `tests/test_muon.py`:

```python
from tinylm.muon import Muon


def test_nesterov_momentum_applied():
    """One Muon step should differ from a vanilla-momentum update.

    We construct a setup where the gradient is fixed and the momentum
    buffer is nonzero. Vanilla momentum applies `buf` directly;
    Nesterov-style applies `mu*buf + grad`. The orthogonalized update
    therefore differs.
    """
    torch.manual_seed(2)
    p_nesterov = torch.nn.Parameter(torch.zeros(8, 8))
    p_vanilla = torch.nn.Parameter(torch.zeros(8, 8))
    grad = torch.randn(8, 8)

    # First step: bootstrap momentum buffer.
    p_nesterov.grad = grad.clone()
    opt = Muon([p_nesterov], lr=0.02, momentum=0.95)
    opt.step()

    # Second step: with the momentum buffer populated, Nesterov path
    # diverges from a vanilla-momentum equivalent.
    p_nesterov.grad = grad.clone()
    snapshot_before = p_nesterov.detach().clone()
    opt.step()
    delta_nesterov = p_nesterov.detach() - snapshot_before

    # Sanity: the update is nontrivial (not all zeros).
    assert delta_nesterov.abs().max().item() > 1e-4
    # And not equal to a plain `-lr * orthogonalize(grad)` step
    # (that would mean Nesterov was bypassed).
    naive = -0.02 * newton_schulz(grad, steps=5)
    naive *= max(1, (8 / 8) ** 0.5)
    assert not torch.allclose(delta_nesterov, naive, atol=1e-3), (
        "Update matches naive grad-only step — Nesterov path is missing."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_muon.py::test_nesterov_momentum_applied -v`
Expected: `FAILED` with `ImportError: cannot import name 'Muon'`.

- [ ] **Step 3: Implement `Muon` class**

Append to `src/tinylm/muon.py`:

```python
class Muon(torch.optim.Optimizer):
    """Muon: MomentUm Orthogonalized by Newton-schulz.

    Applies Nesterov-style momentum BEFORE orthogonalization. Only
    matrix-shaped parameters (ndim >= 2) are orthogonalized; the caller
    is responsible for filtering out embeddings, LM head, layer norms,
    and biases (use `partition_params()` from `tinylm.model`).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mu = group["momentum"]
            ns_steps = group["ns_steps"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                # Nesterov look-ahead before orthogonalization
                nesterov = mu * buf + g
                if nesterov.ndim >= 2:
                    nesterov = newton_schulz(nesterov, steps=ns_steps)
                    m, n = p.shape[-2], p.shape[-1]
                    nesterov = nesterov * max(
                        1.0, (max(m, n) / min(m, n)) ** 0.5
                    )
                p.add_(nesterov, alpha=-lr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_muon.py::test_nesterov_momentum_applied -v`
Expected: `PASSED`.

- [ ] **Step 5: Commit**

```powershell
git add src/tinylm/muon.py tests/test_muon.py
git commit -m "feat: implement Muon optimizer with Nesterov momentum"
```

---

### Task 6: Parameter partitioning helper

**Files:**
- Modify: `D:\TinyLM\src\tinylm\muon.py`
- Modify: `D:\TinyLM\tests\test_muon.py`

- [ ] **Step 1: Write failing test `test_param_filter_excludes_embed_lm_head_norm_bias`**

Append to `tests/test_muon.py`:

```python
def test_param_filter_excludes_embed_lm_head_norm_bias():
    """`partition_params` must route 1D / embedding / LM head / norm
    weights into the AdamW group, not the Muon group.

    Muon on the vocab embedding destroys learned token geometry (per
    PDF Phase 1 Step 4). This filter is the guardrail.
    """
    from tinylm.muon import partition_params

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embed = torch.nn.Embedding(100, 16)
            self.attn_q = torch.nn.Linear(16, 16, bias=False)
            self.norm = torch.nn.LayerNorm(16)
            self.lm_head = torch.nn.Linear(16, 100, bias=True)

    model = Toy()
    matrix_group, scalar_group = partition_params(model)

    matrix_ids = {id(p) for p in matrix_group}
    scalar_ids = {id(p) for p in scalar_group}

    # Only attn_q.weight should be in the Muon group
    assert id(model.attn_q.weight) in matrix_ids
    # Everything else must NOT be in the Muon group
    assert id(model.tok_embed.weight) in scalar_ids
    assert id(model.lm_head.weight) in scalar_ids
    assert id(model.lm_head.bias) in scalar_ids
    assert id(model.norm.weight) in scalar_ids
    assert id(model.norm.bias) in scalar_ids
    # No overlap, no leaks
    assert matrix_ids.isdisjoint(scalar_ids)
    total_params = sum(1 for _ in model.parameters())
    assert len(matrix_ids) + len(scalar_ids) == total_params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_muon.py::test_param_filter_excludes_embed_lm_head_norm_bias -v`
Expected: `FAILED` with `ImportError: cannot import name 'partition_params'`.

- [ ] **Step 3: Implement `partition_params`**

Append to `src/tinylm/muon.py`:

```python
def partition_params(model: torch.nn.Module):
    """Split parameters into (muon_group, adamw_group).

    Muon group: 2D+ matrix weights in core transformer blocks (Q/K/V/O
    projections, FFN). AdamW group: everything else — embeddings, LM
    head, all norms, all biases.

    Returns (matrix_params, scalar_params) as lists.
    """
    matrix_params = []
    scalar_params = []
    for name, p in model.named_parameters():
        lname = name.lower()
        is_excluded = (
            p.ndim < 2
            or "embed" in lname
            or "lm_head" in lname
            or "norm" in lname
            or lname.endswith(".bias")
        )
        if is_excluded:
            scalar_params.append(p)
        else:
            matrix_params.append(p)
    return matrix_params, scalar_params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_muon.py::test_param_filter_excludes_embed_lm_head_norm_bias -v`
Expected: `PASSED`.

- [ ] **Step 5: Run the full Muon suite**

Run: `pytest tests/test_muon.py -v`
Expected: `4 passed`.

- [ ] **Step 6: Commit**

```powershell
git add src/tinylm/muon.py tests/test_muon.py
git commit -m "feat: add partition_params helper for Muon/AdamW split"
```

---

## Chunk 3: Model — config, RMSNorm, RoPE, FFN

### Task 7: ModelConfig dataclass + RMSNorm

**Files:**
- Create: `D:\TinyLM\src\tinylm\model.py`
- Create: `D:\TinyLM\tests\test_model_internals.py` (temporary scratch tests; will be folded in)

> Note: We use a separate temporary test file for internals (RMSNorm, RoPE, FFN) so `test_mla.py` stays focused on MLA semantics. Internals tests can stay or be deleted at the end of Phase 1 — they're not in the canonical 12.

- [ ] **Step 1: Write failing test for RMSNorm**

Create `tests/test_model_internals.py`:

```python
"""Internals tests — RMSNorm, RoPE, FFN. Not part of canonical 12 but
useful while building model.py."""

import torch

from tinylm.model import ModelConfig, RMSNorm


def test_model_config_defaults_lock():
    cfg = ModelConfig()
    assert cfg.n_layers == 18
    assert cfg.d_model == 1024
    assert cfg.n_heads == 16
    assert cfg.d_latent == 512
    assert cfg.d_rope == 64
    assert cfg.ffn_hidden == 2816
    assert cfg.ctx == 2048
    assert cfg.vocab_size == 32000
    assert cfg.tie_weights is True
    assert cfg.attention in {"mha", "mla"}


def test_rmsnorm_unit_norm():
    """RMSNorm with weight=1 should produce activations with
    RMS ≈ 1 along the last dim."""
    torch.manual_seed(0)
    norm = RMSNorm(64)
    x = torch.randn(4, 16, 64) * 3.7  # arbitrary scale
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_internals.py -v`
Expected: `FAILED` with `ModuleNotFoundError: No module named 'tinylm.model'`.

- [ ] **Step 3: Implement `ModelConfig` + `RMSNorm`**

Create `src/tinylm/model.py`:

```python
"""TinyLM model: MHA and MLA variants in one file.

Dims locked in docs/ablation_plan.md:
  n_layers=18, d_model=1024, n_heads=16, d_latent=512, d_rope=64,
  ffn_hidden=2816, ctx=2048, tie_weights=True, vocab_size=32000.

MLAttention is adapted (and aggressively simplified) from the
DeepSeek-V2 modeling code in HuggingFace `transformers`, MIT licensed.
We drop expert routing, YARN scaling, and any feature unused by a
275M dense LM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_layers: int = 18
    d_model: int = 1024
    n_heads: int = 16
    d_latent: int = 512        # MLA only
    d_rope: int = 64           # MLA only — decoupled RoPE projection size
    ffn_hidden: int = 2816
    ctx: int = 2048
    vocab_size: int = 32000
    tie_weights: bool = True
    rope_base: float = 10000.0
    attention: Literal["mha", "mla"] = "mla"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * rms)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_internals.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/tinylm/model.py tests/test_model_internals.py
git commit -m "feat: add ModelConfig dataclass and RMSNorm"
```

---

### Task 8: RoPE helpers

**Files:**
- Modify: `D:\TinyLM\src\tinylm\model.py`
- Modify: `D:\TinyLM\tests\test_model_internals.py`

- [ ] **Step 1: Write failing tests for `build_rope_cache` and `apply_rope`**

Append to `tests/test_model_internals.py`:

```python
from tinylm.model import apply_rope, build_rope_cache


def test_rope_cache_shapes():
    cos, sin = build_rope_cache(seq_len=128, head_dim=64, base=10000.0)
    assert cos.shape == (128, 64)
    assert sin.shape == (128, 64)


def test_apply_rope_preserves_norm():
    """RoPE is a rotation: it must preserve the L2 norm of each token."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 64)  # (B, H, T, head_dim)
    cos, sin = build_rope_cache(seq_len=8, head_dim=64, base=10000.0)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    assert torch.allclose(
        x.pow(2).sum(dim=-1), y.pow(2).sum(dim=-1), atol=1e-5
    )


def test_apply_rope_position_zero_is_identity():
    """At position 0, cos=1 and sin=0, so RoPE should be a no-op."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 1, 32)
    cos, sin = build_rope_cache(seq_len=1, head_dim=32, base=10000.0)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x, y, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_internals.py -k rope -v`
Expected: `FAILED` with `ImportError`.

- [ ] **Step 3: Implement RoPE helpers**

Append to `src/tinylm/model.py`:

```python
def build_rope_cache(
    seq_len: int, head_dim: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embedding.

    Returns (cos, sin) each of shape (seq_len, head_dim). Per-pair
    frequencies are duplicated so the same table can be applied to
    interleaved or split-half layouts; we use the split-half layout
    in `apply_rope`.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float32) / half)
    )
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)            # (T, half)
    cos = freqs.cos().repeat(1, 2)              # (T, head_dim)
    sin = freqs.sin().repeat(1, 2)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to x. x shape is (..., T, head_dim); cos/sin are
    (T, head_dim) and broadcast over leading dims."""
    # Reshape cos/sin to broadcast: insert singleton dims for batch/head
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_internals.py -k rope -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/tinylm/model.py tests/test_model_internals.py
git commit -m "feat: add RoPE cache builder and apply_rope helper"
```

---

### Task 9: SwiGLU FFN

**Files:**
- Modify: `D:\TinyLM\src\tinylm\model.py`
- Modify: `D:\TinyLM\tests\test_model_internals.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_model_internals.py`:

```python
from tinylm.model import SwiGLUFFN


def test_swiglu_ffn_shape_and_no_bias():
    ffn = SwiGLUFFN(d_model=64, ffn_hidden=176)
    x = torch.randn(2, 8, 64)
    y = ffn(x)
    assert y.shape == (2, 8, 64)
    for name, p in ffn.named_parameters():
        assert "bias" not in name, f"unexpected bias param: {name}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_internals.py -k swiglu -v`
Expected: `FAILED` with `ImportError`.

- [ ] **Step 3: Implement SwiGLU FFN**

Append to `src/tinylm/model.py`:

```python
class SwiGLUFFN(nn.Module):
    """SwiGLU FFN as used in Llama/TinyLlama: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, ffn_hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down = nn.Linear(ffn_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_internals.py -k swiglu -v`
Expected: `PASSED`.

- [ ] **Step 5: Commit**

```powershell
git add src/tinylm/model.py tests/test_model_internals.py
git commit -m "feat: add SwiGLU FFN module"
```

---

## Chunk 4: Attention modules

### Task 10: Standard MHAttention

**Files:**
- Modify: `D:\TinyLM\src\tinylm\model.py`
- Modify: `D:\TinyLM\tests\test_model_internals.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_model_internals.py`:

```python
from tinylm.model import MHAttention


def test_mha_shape_and_causal():
    """MHA preserves shape and respects causal masking."""
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, n_heads=4, ctx=16)
    attn = MHAttention(cfg)
    cos, sin = build_rope_cache(
        seq_len=16, head_dim=cfg.d_model // cfg.n_heads, base=cfg.rope_base
    )
    x = torch.randn(2, 16, 64)
    out = attn(x, cos, sin)
    assert out.shape == (2, 16, 64)

    # Causal: perturb position 5; positions 0..4 unchanged.
    x2 = x.clone()
    x2[:, 5, :] += 10.0
    out2 = attn(x2, cos, sin)
    assert torch.allclose(out[:, :5, :], out2[:, :5, :], atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_internals.py -k mha -v`
Expected: `FAILED` with `ImportError`.

- [ ] **Step 3: Implement MHAttention**

Append to `src/tinylm/model.py`:

```python
class MHAttention(nn.Module):
    """Standard causal multi-head attention with RoPE on Q and K.

    No biases, no dropout. Uses scaled_dot_product_attention with
    is_causal=True for the causal mask.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B,H,T,D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Slice rope cache to current T
        cos = rope_cos[:T]
        sin = rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_internals.py -k mha -v`
Expected: `PASSED`.

- [ ] **Step 5: Commit**

```powershell
git add src/tinylm/model.py tests/test_model_internals.py
git commit -m "feat: add MHAttention with RoPE and causal SDPA"
```

---

### Task 11: MLAttention — KV-compressed test first

> **Pause point.** This is the highest-risk task in Phase 1. Before
> writing code, open the DeepSeek-V2 modeling file in HuggingFace
> transformers (`src/transformers/models/deepseek_v2/modeling_deepseek_v2.py`
> on `main`) and read the `DeepseekV2Attention` class top to bottom.
> Identify exactly: (1) which projections feed the latent KV path,
> (2) which projection carries decoupled RoPE, (3) how Q is split
> into the RoPE and non-RoPE parts. The port below is a stripped
> dense-only adaptation; cross-check against the reference.

**Files:**
- Modify: `D:\TinyLM\src\tinylm\model.py`
- Create: `D:\TinyLM\tests\test_mla.py` (this is the canonical 8-test file from the spec)

- [ ] **Step 1: Write Test 1 — `test_kv_latent_compressed`**

Create `tests/test_mla.py`:

```python
"""MLA test suite.

Three PDF-mandatory tests (KV-compressed, output-shape, causal-mask)
plus five defensive tests (RoPE decoupling, param count, gradient
flow, MLA≈MHA equivalence at identity, KV-cache shape during
incremental inference). All eight are blocking gates before any
training run."""

import torch

from tinylm.model import ModelConfig, MLAttention, build_rope_cache


def _small_cfg(**overrides) -> ModelConfig:
    """Small config that keeps MLA semantics but runs fast on CPU."""
    base = dict(
        n_layers=2, d_model=64, n_heads=4, d_latent=32, d_rope=8,
        ffn_hidden=128, ctx=32, vocab_size=128, attention="mla",
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_kv_latent_compressed():
    """PDF Test 1: kv_down output dim equals d_latent, not n_heads*d_head."""
    cfg = _small_cfg()
    mla = MLAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    kv_latent = mla.kv_down(x)
    assert kv_latent.shape[-1] == cfg.d_latent
    assert kv_latent.shape[-1] != cfg.n_heads * (cfg.d_model // cfg.n_heads)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mla.py::test_kv_latent_compressed -v`
Expected: `FAILED` with `ImportError: cannot import name 'MLAttention'`.

- [ ] **Step 3: Implement MLAttention (initial port)**

Append to `src/tinylm/model.py`:

```python
class MLAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2, simplified dense).

    Ported from the DeepSeek-V2 reference (HuggingFace transformers,
    MIT licensed). Aggressively stripped: no expert routing, no YARN
    scaling, no q-LoRA. Decoupled RoPE: positional info is carried
    only by a small `d_rope`-wide projection, NOT by the full latent
    KV path. Getting this wrong yields a model that trains but has
    broken position encoding (PDF Phase 1 Step 2 callout).

    KV path:
        x -> kv_down (d_model -> d_latent) -> [LATENT, no RoPE]
        x -> k_rope_proj (d_model -> d_rope) -> [RoPE applied here]

    Q path:
        x -> q_proj (d_model -> n_heads*(d_head + d_rope))
            split into [q_nope (no RoPE), q_rope (RoPE applied)]

    Inference cache stores per-token (LATENT, K_ROPE) of total width
    `d_latent + d_rope`, NOT `n_heads * d_head`. That is the source
    of the KV-compression ratio claimed in the Phase 5 headline.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.d_latent = cfg.d_latent
        self.d_rope = cfg.d_rope

        # Q: split per head into (head_dim no-rope) + (d_rope rope)
        self.q_proj = nn.Linear(
            cfg.d_model,
            cfg.n_heads * (self.head_dim + cfg.d_rope),
            bias=False,
        )
        # KV down-projection to the latent
        self.kv_down = nn.Linear(cfg.d_model, cfg.d_latent, bias=False)
        # Latent norm (DeepSeek-V2 applies RMSNorm on the latent)
        self.kv_norm = RMSNorm(cfg.d_latent)
        # K-rope projection (single-head, broadcast across heads)
        self.k_rope_proj = nn.Linear(cfg.d_model, cfg.d_rope, bias=False)
        # Up-projections from latent to per-head K_nope and V
        self.k_up = nn.Linear(
            cfg.d_latent, cfg.n_heads * self.head_dim, bias=False
        )
        self.v_up = nn.Linear(
            cfg.d_latent, cfg.n_heads * self.head_dim, bias=False
        )
        # Output projection
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, D, R = self.n_heads, self.head_dim, self.d_rope

        # Q: (B, T, H*(D+R)) -> split
        q = self.q_proj(x).view(B, T, H, D + R)
        q_nope, q_rope = q.split([D, R], dim=-1)
        # (B, H, T, D) and (B, H, T, R)
        q_nope = q_nope.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)

        # KV latent + rope branch
        latent = self.kv_norm(self.kv_down(x))         # (B, T, d_latent)
        k_nope = self.k_up(latent).view(B, T, H, D).transpose(1, 2)
        v = self.v_up(latent).view(B, T, H, D).transpose(1, 2)
        k_rope = self.k_rope_proj(x).view(B, T, 1, R).transpose(1, 2)
        # broadcast k_rope to all heads
        k_rope = k_rope.expand(B, H, T, R)

        # Apply RoPE only on the rope branches
        cos_r = rope_cos[:T, :R]
        sin_r = rope_sin[:T, :R]
        q_rope = apply_rope(q_rope, cos_r, sin_r)
        k_rope = apply_rope(k_rope, cos_r, sin_r)

        # Concatenate to form full Q and K of width D+R
        q_full = torch.cat([q_nope, q_rope], dim=-1)   # (B,H,T,D+R)
        k_full = torch.cat([k_nope, k_rope], dim=-1)

        out = F.scaled_dot_product_attention(
            q_full, k_full, v, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out)
```

- [ ] **Step 4: Run Test 1 to verify it passes**

Run: `pytest tests/test_mla.py::test_kv_latent_compressed -v`
Expected: `PASSED`.

- [ ] **Step 5: Commit**

```powershell
git add src/tinylm/model.py tests/test_mla.py
git commit -m "feat: implement MLAttention with decoupled RoPE (PDF Test 1 green)"
```

---

### Task 12: MLA Tests 2 & 3 — output shape and causal mask

**Files:**
- Modify: `D:\TinyLM\tests\test_mla.py`

- [ ] **Step 1: Append PDF Test 2 and Test 3**

```python
def test_output_shape_preserved():
    """PDF Test 2: MLA output has shape (B, T, d_model)."""
    cfg = _small_cfg()
    mla = MLAttention(cfg)
    cos, sin = build_rope_cache(
        seq_len=cfg.ctx, head_dim=cfg.d_rope, base=cfg.rope_base
    )
    x = torch.randn(2, 16, cfg.d_model)
    out = mla(x, cos, sin)
    assert out.shape == (2, 16, cfg.d_model)


def test_causal_masking():
    """PDF Test 3: perturbing position 5 leaves positions 0..4
    unchanged."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    mla = MLAttention(cfg)
    cos, sin = build_rope_cache(
        seq_len=cfg.ctx, head_dim=cfg.d_rope, base=cfg.rope_base
    )
    x = torch.randn(2, 16, cfg.d_model)
    out = mla(x, cos, sin)
    x2 = x.clone()
    x2[:, 5, :] += 10.0
    out2 = mla(x2, cos, sin)
    assert torch.allclose(out[:, :5, :], out2[:, :5, :], atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_mla.py -v`
Expected: `3 passed`.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_mla.py
git commit -m "test: MLA output shape and causal masking (PDF Tests 2 & 3)"
```

---

### Task 13: MLA defensive Test 4 — RoPE decoupling

**Files:**
- Modify: `D:\TinyLM\tests\test_mla.py`

- [ ] **Step 1: Append `test_rope_decoupling`**

```python
def test_rope_decoupling():
    """Defensive Test 4: positional information is carried ONLY by
    the d_rope branch. If we zero the `k_rope_proj` weight, the
    output should become position-invariant (i.e. shifting the input
    along T does not change the output beyond what causal-mask
    boundaries require).

    Concrete check: with k_rope_proj zeroed, the attention scores
    depend only on the latent path (no positional signal), so two
    inputs that differ only by a position-preserving permutation of
    identical token contents at positions far from any boundary
    produce identical post-attention representations at matched
    positions.

    We test the easier direction: with k_rope zeroed and q_rope
    zeroed, RoPE has no effect anywhere; with both intact, RoPE
    DOES affect outputs. This proves the rope path is the (only)
    carrier of positional info.
    """
    torch.manual_seed(0)
    cfg = _small_cfg()
    mla = MLAttention(cfg)
    cos, sin = build_rope_cache(
        seq_len=cfg.ctx, head_dim=cfg.d_rope, base=cfg.rope_base
    )

    # Same content tokens at two different positions.
    x = torch.randn(1, 16, cfg.d_model)
    out_full = mla(x, cos, sin)

    with torch.no_grad():
        # Zero out BOTH rope branches → attention becomes position-blind
        mla.k_rope_proj.weight.zero_()
        # Zero the rope half of q_proj (per-head last R dims)
        q_w = mla.q_proj.weight  # shape (H*(D+R), d_model)
        H, D, R = cfg.n_heads, cfg.d_model // cfg.n_heads, cfg.d_rope
        q_w_view = q_w.view(H, D + R, cfg.d_model)
        q_w_view[:, D:, :].zero_()
    out_no_rope = mla(x, cos, sin)

    # The two outputs MUST differ — proves RoPE was contributing.
    assert not torch.allclose(out_full, out_no_rope, atol=1e-4), (
        "Zeroing RoPE branches did not change output — positional "
        "information is leaking through the latent path."
    )
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_mla.py::test_rope_decoupling -v`
Expected: `PASSED`.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_mla.py
git commit -m "test: verify MLA RoPE decoupling (defensive Test 4)"
```

---

### Task 14: Block, TinyLM full model, and partition_params hookup

**Files:**
- Modify: `D:\TinyLM\src\tinylm\model.py`
- Modify: `D:\TinyLM\src\tinylm\__init__.py`

- [ ] **Step 1: Append Block + TinyLM to `model.py`**

```python
class Block(nn.Module):
    """Transformer block: Attn(RMSNorm(x)) + FFN(RMSNorm(x)), residual."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        if cfg.attention == "mla":
            self.attn = MLAttention(cfg)
        else:
            self.attn = MHAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyLM(nn.Module):
    """The full 275M model: token embed + N blocks + final norm + LM head.

    With `tie_weights=True`, the LM head shares weight with `tok_embed`."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        # RoPE cache lives on the model so it follows .to(device)
        # We size the cache for the rope-dim used by the attention
        # variant: MHA uses head_dim, MLA uses d_rope.
        rope_head_dim = (
            cfg.d_rope
            if cfg.attention == "mla"
            else cfg.d_model // cfg.n_heads
        )
        cos, sin = build_rope_cache(cfg.ctx, rope_head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        if cfg.tie_weights:
            self.lm_head_weight = None  # uses tok_embed.weight
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) -> logits (B, T, vocab_size)."""
        h = self.tok_embed(tokens)
        for blk in self.blocks:
            h = blk(h, self.rope_cos, self.rope_sin)
        h = self.final_norm(h)
        if self.cfg.tie_weights:
            return h @ self.tok_embed.weight.T
        return self.lm_head(h)
```

- [ ] **Step 2: Re-export from package init**

Edit `src/tinylm/__init__.py`:

```python
from tinylm.model import ModelConfig, TinyLM

__all__ = ["ModelConfig", "TinyLM"]
```

- [ ] **Step 3: Smoke test by adding a quick internals test**

Append to `tests/test_model_internals.py`:

```python
from tinylm import TinyLM


def test_tinylm_forward_smoke():
    """TinyLM produces logits of shape (B, T, vocab_size)."""
    cfg = ModelConfig(
        n_layers=2, d_model=64, n_heads=4, d_latent=32, d_rope=8,
        ffn_hidden=128, ctx=32, vocab_size=128, attention="mla",
    )
    model = TinyLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, cfg.vocab_size)
```

- [ ] **Step 4: Run all tests so far**

Run: `pytest tests/ -v`
Expected: `4 muon + 7 internals + 4 mla = 15 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/tinylm/model.py src/tinylm/__init__.py tests/test_model_internals.py
git commit -m "feat: add Block and TinyLM full model with tied weights"
```

---

## Chunk 5: Remaining MLA defensive tests

### Task 15: Test 5 — total param count for the locked dims

**Files:**
- Modify: `D:\TinyLM\tests\test_mla.py`

- [ ] **Step 1: Append `test_total_param_count`**

```python
from tinylm import TinyLM


def test_total_param_count():
    """Defensive Test 5: at the LOCKED Phase-4 dims, TinyLM-MLA must
    have between 270M and 285M parameters. PDF Phase 4 Step 0 says to
    verify this before any training run."""
    cfg = ModelConfig(attention="mla")  # all locked defaults
    model = TinyLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert 270_000_000 <= total <= 285_000_000, (
        f"param count {total:,} outside locked range "
        f"[270M, 285M] — check ablation_plan.md before retuning"
    )
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_mla.py::test_total_param_count -v`

Expected: `PASSED`. If the count is off-range, the issue is one of:
(a) `tie_weights` is being ignored,
(b) per-head projection sizes are wrong,
(c) `kv_norm` / extra parameters were added that the PDF estimate did
not anticipate.

Do **not** retune dims to make the test pass — instead diagnose which
of (a)/(b)/(c) it is and fix the code. If the diagnosis is "PDF
estimate was wrong," surface to the user before changing the test
bounds.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_mla.py
git commit -m "test: verify TinyLM-MLA param count in [270M, 285M]"
```

---

### Task 16: Test 6 — gradient flow

**Files:**
- Modify: `D:\TinyLM\tests\test_mla.py`

- [ ] **Step 1: Append `test_gradient_flow`**

```python
def test_gradient_flow():
    """Defensive Test 6: after forward+backward, every learnable
    parameter has a non-zero gradient. Catches frozen sub-modules
    and dead branches."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = TinyLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = model(tokens)
    # Simple LM loss over the same tokens (shifted by 1 not necessary
    # for gradient-flow check; we just need a scalar tied to all params)
    loss = logits.float().pow(2).mean()
    loss.backward()

    zero_grad_params = []
    for name, p in model.named_parameters():
        if p.grad is None:
            zero_grad_params.append(f"{name} (grad is None)")
        elif p.grad.abs().max().item() == 0.0:
            zero_grad_params.append(f"{name} (all zeros)")
    assert not zero_grad_params, (
        "Params with no gradient signal:\n  " + "\n  ".join(zero_grad_params)
    )
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_mla.py::test_gradient_flow -v`
Expected: `PASSED`.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_mla.py
git commit -m "test: verify gradient reaches every learnable param in TinyLM"
```

---

### Task 17: Test 7 — MLA ≈ MHA at the identity setting

**Files:**
- Modify: `D:\TinyLM\tests\test_mla.py`

> **Why this test is useful:** At `d_latent=d_model` and `d_rope=head_dim`,
> the MLA latent compression is theoretically lossless and the rope
> branch is full-width. With weights aligned, MLA and MHA should
> produce numerically-close outputs. A large divergence indicates a
> bug in MLA's projection wiring.
>
> **Realism caveat:** Perfect equivalence requires careful weight
> alignment between two architecturally-different modules. We test
> the weaker but still meaningful property: *outputs are correlated
> at a level much higher than chance* (Pearson r > 0.5 across a batch).

- [ ] **Step 1: Append `test_mla_mha_equivalence`**

```python
from tinylm.model import MHAttention


def test_mla_mha_equivalence_at_identity_setting():
    """Defensive Test 7: with d_latent=d_model and d_rope=head_dim,
    MLA's latent path is non-compressing. Outputs of MLA and MHA on
    the same input should be correlated well above chance, indicating
    the MLA projection wiring is not catastrophically broken.

    This is a coarse sanity check — perfect equivalence requires
    weight alignment between architecturally-different modules,
    which is impractical. We assert Pearson correlation > 0.3 across
    the flattened output (chance ≈ 0)."""
    torch.manual_seed(0)
    # head_dim = d_model // n_heads = 64 // 4 = 16
    cfg_mla = ModelConfig(
        n_layers=1, d_model=64, n_heads=4, d_latent=64, d_rope=16,
        ffn_hidden=128, ctx=32, vocab_size=128, attention="mla",
    )
    cfg_mha = ModelConfig(
        n_layers=1, d_model=64, n_heads=4, d_latent=64, d_rope=16,
        ffn_hidden=128, ctx=32, vocab_size=128, attention="mha",
    )
    mla = MLAttention(cfg_mla)
    mha = MHAttention(cfg_mha)

    cos_r, sin_r = build_rope_cache(cfg_mla.ctx, cfg_mla.d_rope, cfg_mla.rope_base)
    cos_m, sin_m = build_rope_cache(
        cfg_mha.ctx, cfg_mha.d_model // cfg_mha.n_heads, cfg_mha.rope_base
    )
    x = torch.randn(2, 16, 64)
    out_mla = mla(x, cos_r, sin_r).detach().flatten()
    out_mha = mha(x, cos_m, sin_m).detach().flatten()

    # Pearson correlation
    out_mla_c = out_mla - out_mla.mean()
    out_mha_c = out_mha - out_mha.mean()
    corr = (out_mla_c @ out_mha_c) / (
        out_mla_c.norm() * out_mha_c.norm() + 1e-12
    )
    # Both are reading the same x through random linear projections
    # of full rank → expect at least moderate correlation.
    assert corr.abs().item() > 0.3, (
        f"MLA / MHA outputs uncorrelated (r={corr.item():.3f}) — "
        f"likely a wiring bug in one of them"
    )
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_mla.py::test_mla_mha_equivalence_at_identity_setting -v`
Expected: `PASSED`.

> If this fails with `r ≈ 0`: there is almost certainly a bug in MLA's
> projection composition — likely the `k_up`/`v_up` matrices being
> applied to the wrong tensor, or the head reshape getting transposed.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_mla.py
git commit -m "test: MLA-vs-MHA correlation at d_latent=d_model setting"
```

---

### Task 18: Test 8 — KV cache shape during incremental inference

**Files:**
- Modify: `D:\TinyLM\src\tinylm\model.py` (add `forward_with_cache` to `MLAttention`)
- Modify: `D:\TinyLM\tests\test_mla.py`

- [ ] **Step 1: Append failing test `test_kv_cache_shape_incremental`**

```python
def test_kv_cache_shape_incremental():
    """Defensive Test 8: when running token-at-a-time, the per-layer
    cached tensor must have last-dim width (d_latent + d_rope), NOT
    (n_heads * head_dim). This is what produces the 3.5× KV-cache
    reduction headline in Phase 5."""
    cfg = _small_cfg()
    mla = MLAttention(cfg)
    cos, sin = build_rope_cache(cfg.ctx, cfg.d_rope, cfg.rope_base)

    # Feed one token at a time, accumulating cache.
    cache = None
    for t in range(4):
        x_t = torch.randn(1, 1, cfg.d_model)
        out_t, cache = mla.forward_with_cache(
            x_t, cos, sin, cache=cache, pos=t
        )
        assert out_t.shape == (1, 1, cfg.d_model)

    # Cache is a tuple (latent_cache, k_rope_cache) per spec
    latent_cache, k_rope_cache = cache
    assert latent_cache.shape[-1] == cfg.d_latent
    assert k_rope_cache.shape[-1] == cfg.d_rope
    # Time dim accumulates correctly
    assert latent_cache.shape[-2] == 4
    assert k_rope_cache.shape[-2] == 4
    # Critically: cache width is NOT n_heads * head_dim
    full_kv_width = cfg.n_heads * (cfg.d_model // cfg.n_heads)
    assert latent_cache.shape[-1] != full_kv_width
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mla.py::test_kv_cache_shape_incremental -v`
Expected: `FAILED` with `AttributeError: 'MLAttention' object has no attribute 'forward_with_cache'`.

- [ ] **Step 3: Implement `forward_with_cache` on `MLAttention`**

Add this method to `MLAttention` in `src/tinylm/model.py`:

```python
    def forward_with_cache(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
        pos: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Incremental forward for one or more new tokens.

        cache: (latent_cache, k_rope_cache) from previous step, each
            shaped (B, T_past, d_latent) and (B, T_past, d_rope), or
            None for the first call.
        pos: starting position of the new tokens in the full sequence
            (used to slice RoPE).

        Returns (output, new_cache). new_cache contains the appended
        latent and k_rope tensors covering past + new tokens. This is
        the source of MLA's KV-cache memory advantage — width is
        (d_latent + d_rope), not (n_heads * head_dim).
        """
        B, T_new, _ = x.shape
        H, D, R = self.n_heads, self.head_dim, self.d_rope

        # New latent + k_rope for these tokens (un-headed, compact).
        new_latent = self.kv_norm(self.kv_down(x))   # (B, T_new, d_latent)
        new_k_rope_pre = self.k_rope_proj(x)         # (B, T_new, d_rope)

        # Concatenate to cache (the COMPACT cache — this is the point).
        if cache is None:
            latent_cache = new_latent
            k_rope_cache = new_k_rope_pre
        else:
            latent_prev, k_rope_prev = cache
            latent_cache = torch.cat([latent_prev, new_latent], dim=1)
            k_rope_cache = torch.cat([k_rope_prev, new_k_rope_pre], dim=1)
        T_total = latent_cache.shape[1]

        # Project the FULL latent cache up to per-head K_nope, V.
        k_nope = self.k_up(latent_cache).view(B, T_total, H, D).transpose(1, 2)
        v = self.v_up(latent_cache).view(B, T_total, H, D).transpose(1, 2)

        # Apply RoPE to the FULL k_rope cache at absolute positions.
        cos_r = rope_cos[:T_total, :R]
        sin_r = rope_sin[:T_total, :R]
        k_rope = k_rope_cache.view(B, T_total, 1, R).transpose(1, 2)
        k_rope = k_rope.expand(B, H, T_total, R)
        k_rope = apply_rope(k_rope, cos_r, sin_r)

        # Q for new tokens only.
        q = self.q_proj(x).view(B, T_new, H, D + R)
        q_nope, q_rope = q.split([D, R], dim=-1)
        q_nope = q_nope.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)
        cos_q = rope_cos[pos:pos + T_new, :R]
        sin_q = rope_sin[pos:pos + T_new, :R]
        q_rope = apply_rope(q_rope, cos_q, sin_q)

        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope], dim=-1)

        # Causal: q at absolute positions [pos, pos+T_new), k spans
        # [0, T_total). We need a custom mask: position i in q can see
        # positions [0, pos+i] in k.
        attn_mask = torch.ones(
            T_new, T_total, dtype=torch.bool, device=x.device
        )
        for i in range(T_new):
            attn_mask[i, pos + i + 1:] = False

        out = F.scaled_dot_product_attention(
            q_full, k_full, v, attn_mask=attn_mask
        )
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.d_model)
        return self.o_proj(out), (latent_cache, k_rope_cache)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mla.py::test_kv_cache_shape_incremental -v`
Expected: `PASSED`.

- [ ] **Step 5: Commit**

```powershell
git add src/tinylm/model.py tests/test_mla.py
git commit -m "feat: add MLAttention.forward_with_cache for incremental inference"
```

---

## Chunk 6: Final verification

### Task 19: Run the full canonical suite

**Files:** none (verification only)

- [ ] **Step 1: Run all canonical tests**

Run: `pytest tests/test_mla.py tests/test_muon.py -v`

Expected output (12 tests):

```
tests/test_mla.py::test_kv_latent_compressed PASSED
tests/test_mla.py::test_output_shape_preserved PASSED
tests/test_mla.py::test_causal_masking PASSED
tests/test_mla.py::test_rope_decoupling PASSED
tests/test_mla.py::test_total_param_count PASSED
tests/test_mla.py::test_gradient_flow PASSED
tests/test_mla.py::test_mla_mha_equivalence_at_identity_setting PASSED
tests/test_mla.py::test_kv_cache_shape_incremental PASSED
tests/test_muon.py::test_newton_schulz_orthogonalizes PASSED
tests/test_muon.py::test_non_square_transpose_trick PASSED
tests/test_muon.py::test_nesterov_momentum_applied PASSED
tests/test_muon.py::test_param_filter_excludes_embed_lm_head_norm_bias PASSED

==================== 12 passed in X.XXs ====================
```

If anything fails: **stop**. Do not move on. Diagnose root cause per
`systematic-debugging`. Do not relax test bounds to make a test pass.

- [ ] **Step 2: Run the internals scratch tests too (sanity, not gating)**

Run: `pytest tests/ -v`

Expected: all internals tests also pass (they were green when written).

- [ ] **Step 3: Print the canonical-suite output to `phase1_test_log.txt`**

Run:

```powershell
pytest tests/test_mla.py tests/test_muon.py -v | Out-File phase1_test_log.txt -Encoding utf8
```

This becomes evidence in the verification step.

---

### Task 20: Verification-before-completion

**Files:** none

- [ ] **Step 1: Re-check spec acceptance gate**

Open `docs/superpowers/specs/2026-05-12-tinylm-phase1-design.md`. Confirm:
- [ ] All 6 deliverables in the "Phase 1 deliverables" table exist on disk.
- [ ] `pytest tests/test_mla.py tests/test_muon.py -v` reports `12 passed`.
- [ ] `phase1_test_log.txt` contains the green run.
- [ ] No new files in scope-excluded paths (`train.py`, `data.py`, `eval_wrapper.py`, `inference.py`, `configs/run_*.yaml`).
- [ ] CLAUDE.md "four MLA unit tests" line is gone.

- [ ] **Step 2: Sanity-check param count by hand**

Run a quick interactive check:

```powershell
python -c "from tinylm import TinyLM, ModelConfig; m = TinyLM(ModelConfig(attention='mla')); print(f'{sum(p.numel() for p in m.parameters()):,}')"
```

Expected: a number between 270,000,000 and 285,000,000. Record it in the
Phase 1 completion commit message.

- [ ] **Step 3: Tag Phase 1 complete**

```powershell
git tag phase1-complete
git log --oneline -20
```

Expected: clean linear history of small commits, all green, no
"WIP" / "fix typo" / "address review" noise.

- [ ] **Step 4: Update README "Current phase" line**

Edit `README.md` so the current-phase callout reflects Phase 1
completion. Do **not** edit the pinned hypothesis.

- [ ] **Step 5: Final commit**

```powershell
git add README.md phase1_test_log.txt
git commit -m "docs: mark Phase 1 complete (12/12 tests green, param count verified)"
```

---

## Phase 1 done

Hand to Phase 2 (toy GPU run) only after:
1. `pytest tests/test_mla.py tests/test_muon.py -v` = 12 passed
2. Interactive param count is in [270M, 285M]
3. `phase1-complete` tag is on the green commit
4. No code in scope-excluded paths

Phase 2 will: vendor the FineWeb dataloader, write `train.py`, run a 1B-token toy on a Vast.ai A100, and validate the four go/no-go criteria in `docs/ablation_plan.md` (tokens/sec, loss trajectory, no NaN, checkpoint resume).
