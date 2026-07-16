"""Logit knowledge distillation (v4 track).

Top-k forward-KL distillation from a larger teacher (TinyLlama-1.1B, same Llama-2
vocab) into the 275M student, mixed with the ground-truth cross-entropy:

    L = (1 - alpha) * CE(student, hard_targets)
        + alpha * T^2 * KL_topk(softmax(teacher/T) || softmax(student/T))

Only the teacher's top-k logits per token are distilled — the standard memory-safe
KD approximation. Restricting to k tokens keeps the KL over (N, k) tensors instead of
(N, 32000), so no full-vocab fp32 blow-up (the CE half still goes through
chunked_cross_entropy for the same reason).
"""
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tinylm.losses import chunked_cross_entropy

VOCAB_SIZE = 32000
DEFAULT_TEACHER = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def kd_loss(
    student_logits: torch.Tensor,
    teacher_topk_vals: torch.Tensor,
    teacher_topk_idx: torch.Tensor,
    hard_targets: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
    ignore_index: int = -100,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Mixed top-k forward-KL + cross-entropy distillation loss.

    Args:
        student_logits:    (N, V) student logits, N = B*T.
        teacher_topk_vals: (N, k) teacher logit VALUES at its top-k.
        teacher_topk_idx:  (N, k) teacher top-k token indices into V.
        hard_targets:      (N,) ground-truth ids; ignore_index rows are masked
                           out of both the CE and the KD averages.
        alpha:             KD weight; 0 = pure CE, 1 = pure KD.
        temperature:       softmax temperature T for the soft targets.

    Returns a scalar. Finite (0.0) when every target is masked.
    """
    T = temperature

    # Hard-label CE (no temperature), memory-bounded.
    ce = chunked_cross_entropy(
        student_logits, hard_targets, chunk_size=chunk_size, ignore_index=ignore_index
    )

    # Top-k forward KL at temperature T. Gather the student logits the teacher
    # actually ranked; upcast for a stable softmax.
    student_topk = student_logits.gather(-1, teacher_topk_idx).float()
    teacher_vals = teacher_topk_vals.float()
    log_p_teacher = F.log_softmax(teacher_vals / T, dim=-1)
    p_teacher = log_p_teacher.exp()
    log_q_student = F.log_softmax(student_topk / T, dim=-1)
    per_token_kl = (p_teacher * (log_p_teacher - log_q_student)).sum(dim=-1)  # (N,)

    valid = (hard_targets != ignore_index).to(per_token_kl.dtype)
    kd = (per_token_kl * valid).sum() / valid.sum().clamp(min=1)

    return (1.0 - alpha) * ce + alpha * (T * T) * kd


class Teacher:
    """Frozen teacher LM that emits top-k next-token logits.

    TinyLlama-1.1B shares the Llama-2 tokenizer (vocab 32000), so its logit
    indices align 1:1 with the student's — no vocab remapping needed. Loaded in
    bf16, eval, and only ever called under no_grad.
    """

    def __init__(self, model_name: str, device: str, topk: int):
        from transformers import AutoModelForCausalLM

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            .to(device)
            .eval()
        )
        if self.model.config.vocab_size != VOCAB_SIZE:
            raise ValueError(
                f"Teacher vocab {self.model.config.vocab_size} != student {VOCAB_SIZE}; "
                "KD needs a shared tokenizer (use a Llama-2-vocab teacher)."
            )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.topk = topk

    @torch.no_grad()
    def topk_logits(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T) input ids -> (values, indices) each (B, T, k)."""
        logits = self.model(tokens).logits
        return logits.topk(self.topk, dim=-1)


def make_kd_loss_fn(teacher, alpha: float, temperature: float, topk: int):
    """Build a train()-compatible loss_fn closure that distills from `teacher`.

    `teacher` is any object exposing `topk_logits(tokens) -> (vals, idx)`; the real
    Teacher above, or a stub in tests.
    """

    def kd_fn(model, tokens, cfg):
        inp = tokens[:, :-1]
        student_logits = model(inp).reshape(-1, cfg.vocab_size)
        t_vals, t_idx = teacher.topk_logits(inp)
        t_vals = t_vals.reshape(-1, topk).to(student_logits.device)
        t_idx = t_idx.reshape(-1, topk).to(student_logits.device)
        targets = tokens[:, 1:].reshape(-1)
        return kd_loss(
            student_logits, t_vals, t_idx, targets,
            alpha=alpha, temperature=temperature,
        )

    return kd_fn


@dataclass
class KDConfig:
    teacher_model: str = DEFAULT_TEACHER
    kd_alpha: float = 0.5
    kd_temperature: float = 2.0
    kd_topk: int = 64


def load_kd_config(path: str) -> KDConfig:
    """Read KD hyperparams from the probe YAML (ignores the TrainConfig keys)."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    names = {f.name for f in dataclasses.fields(KDConfig)}
    cfg = KDConfig(**{k: v for k, v in raw.items() if k in names})
    if os.environ.get("TINYLM_TEACHER"):
        cfg.teacher_model = os.environ["TINYLM_TEACHER"]
    return cfg


def run_kd(config_path: str) -> None:
    """v4 KD probe: continued pretraining from Run D with a KD loss.

    Reuses the entire train() loop (Muon+AdamW, cosine, shard loader,
    checkpoint/SIGTERM/rechain) — only the per-step loss is swapped for KD.
    """
    from tinylm.train import apply_env_overrides, load_config, train

    cfg = apply_env_overrides(load_config(config_path))
    kd = load_kd_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"KD: teacher={kd.teacher_model} alpha={kd.kd_alpha} "
          f"T={kd.kd_temperature} top-k={kd.kd_topk}")
    teacher = Teacher(kd.teacher_model, device, kd.kd_topk)
    loss_fn = make_kd_loss_fn(teacher, kd.kd_alpha, kd.kd_temperature, kd.kd_topk)
    train(cfg, loss_fn=loss_fn)


if __name__ == "__main__":
    run_kd(sys.argv[1])
