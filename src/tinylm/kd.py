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

import torch
import torch.nn.functional as F

from tinylm.losses import chunked_cross_entropy


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
