"""Tests for the top-k forward-KL + CE knowledge-distillation loss.

The KD term is checked against an independent F.kl_div reference; the mixture,
temperature scaling, top-k gathering, and masking are pinned structurally.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tinylm.kd import kd_loss  # noqa: E402
from tinylm.losses import chunked_cross_entropy  # noqa: E402


def test_kl_term_is_zero_when_student_matches_teacher():
    """If the student's logits at the teacher's top-k equal the teacher's values,
    the two temperature-softmaxed distributions coincide and KL = 0."""
    V, k = 4, 2
    idx = torch.tensor([[0, 1], [2, 3]])
    vals = torch.tensor([[2.0, 1.0], [0.5, 1.5]])
    student = torch.zeros(2, V)
    student[0, 0], student[0, 1] = 2.0, 1.0     # match teacher on row 0's top-k
    student[1, 2], student[1, 3] = 0.5, 1.5     # match on row 1's top-k
    targets = torch.tensor([0, 2])

    loss = kd_loss(student, vals, idx, targets, alpha=1.0, temperature=2.0)
    assert loss.abs().item() < 1e-6


def test_alpha_zero_is_plain_cross_entropy():
    torch.manual_seed(0)
    N, V, k = 8, 16, 4
    student = torch.randn(N, V)
    idx = torch.randint(0, V, (N, k))
    vals = torch.randn(N, k)
    targets = torch.randint(0, V, (N,))

    got = kd_loss(student, vals, idx, targets, alpha=0.0, temperature=2.0)
    expected = chunked_cross_entropy(student, targets)
    assert torch.allclose(got, expected, atol=1e-6)


def test_alpha_one_matches_kl_div_reference_with_T_squared():
    """alpha=1 ⇒ pure T²·KL(teacher‖student), checked against F.kl_div."""
    T = 2.0
    idx = torch.tensor([[0, 1]])
    vals = torch.tensor([[1.0, 0.0]])
    student = torch.zeros(1, 3)          # top-k logits are [0, 0]
    targets = torch.tensor([0])

    got = kd_loss(student, vals, idx, targets, alpha=1.0, temperature=T)

    p_teacher = F.softmax(vals / T, dim=-1)
    log_q = F.log_softmax(student.gather(-1, idx) / T, dim=-1)
    ref = T * T * F.kl_div(log_q, p_teacher, reduction="batchmean")
    assert torch.allclose(got, ref, atol=1e-6)


def test_kd_ignores_student_logits_outside_teacher_topk():
    """Only the teacher's top-k student logits matter; perturbing the rest
    leaves the KD loss unchanged."""
    idx = torch.tensor([[0, 1]])
    vals = torch.tensor([[1.0, 0.5]])
    targets = torch.tensor([0])
    a = torch.tensor([[0.3, -0.2, 5.0, 9.0]])   # cols 2,3 are outside top-k
    b = a.clone()
    b[0, 2], b[0, 3] = -100.0, 42.0

    la = kd_loss(a, vals, idx, targets, alpha=1.0, temperature=1.0)
    lb = kd_loss(b, vals, idx, targets, alpha=1.0, temperature=1.0)
    assert torch.allclose(la, lb, atol=1e-6)


def test_ce_part_respects_ignore_index():
    torch.manual_seed(1)
    N, V, k = 6, 10, 3
    student = torch.randn(N, V)
    idx = torch.randint(0, V, (N, k))
    vals = torch.randn(N, k)
    targets = torch.randint(0, V, (N,))
    targets[::2] = -100                      # mask half the rows

    got = kd_loss(student, vals, idx, targets, alpha=0.0, temperature=1.0)
    expected = chunked_cross_entropy(student, targets, ignore_index=-100)
    assert torch.allclose(got, expected, atol=1e-6)


def test_fully_masked_batch_is_finite():
    N, V, k = 4, 8, 2
    student = torch.randn(N, V)
    idx = torch.randint(0, V, (N, k))
    vals = torch.randn(N, k)
    targets = torch.full((N,), -100)

    loss = kd_loss(student, vals, idx, targets, alpha=0.5, temperature=2.0)
    assert torch.isfinite(loss).item()
