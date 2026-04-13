"""
models/scrn.py
==============
TADS-X — Team ChipSmiths | DVCon India 2026

Scene Context Re-scoring Network (SCRN)
-----------------------------------------
Refines AGCA scores by letting every candidate object attend to every other
object before final scoring.  This is where inter-object reasoning happens:
a wine glass scores higher when a bottle is also present (reinforcing context),
and a cup scores lower when a wine glass is available (a better option exists).

Math:
    h_i       = [v'_i (256) ‖ agca_score_i (1)]   → (K, 257)   joint feature
    Q         = h · W_Q                             → (K, 64)
    K_mat     = h · W_K                             → (K, 64)
    attn      = softmax( Q · K_mat^T / √64 )        → (K, K)    row-stochastic
    context_i = Σ_j  attn_ij · h_j                 → (K, 257)  context vector
    score_i   = sigmoid( MLP_2([context_i ‖ h_i]) ) → (K,)

⚠ SRS §7.5 equation lists h as (K, 258).  That is a typo: 256 + 1 = 257.
  This implementation uses 257.  Downstream code (pipeline, train) must match.

Training vs. inference scope (FR-07):
    - Inference  : runs over top-K candidates (K ≤ 5), selected by AGCA score.
    - Training   : pipeline passes ALL N proposals so the GT object is never
                   excluded from the loss signal even if it ranks outside top-5.
    SCRN itself is scope-agnostic — it processes whatever K it receives.

References:
    SRS §7.5 (FR-07), Stage 2A, Team ChipSmiths.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from task_definitions import WORKING_DIM as _DIM

# h_dim = WORKING_DIM + 1 (agca_score scalar appended)
_H_DIM = _DIM + 1       # 257


class SCRN(nn.Module):
    """
    Scene Context Re-scoring Network.

    Parameters
    ----------
    dim : int
        Visual embedding dimension (default 256).  Must match TCFG/AGCA output.
    attn_dim : int
        Projection dimension for Q and K in self-attention (default 64).
    mlp_hidden : int
        Hidden units in the final scoring MLP (default 128).

    Inputs
    ------
    v_prime     : Tensor  (K, dim)
        Task-gated visual embeddings from TCFG (same v_prime passed into AGCA).
    agca_scores : Tensor  (K,)
        Preliminary logit scores from AGCA's MLP head (raw, not sigmoid'd).

    Output
    ------
    refined_scores : Tensor  (K,)
        Context-aware final suitability scores in [0, 1] (sigmoid-activated).

    Notes
    -----
    * K ≤ 5 at inference; K = N (all proposals) during training (FR-07).
    * K = 1 edge case: self-attention over one item is the identity transform —
      result equals MLP_2([h ‖ h]) which reduces to a simple MLP over h.
      No special-casing needed; the math degenerates correctly.
    * W_Q, W_K are bias=False (attention projection convention).
    * MLP_2 uses ReLU hidden activation and outputs a scalar per candidate.
    """

    def __init__(
        self,
        dim:        int = _DIM,
        attn_dim:   int = 64,
        mlp_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.dim      = dim
        self.h_dim    = dim + 1         # 257  (v'_i ‖ agca_score_i)
        self.attn_dim = attn_dim
        self.scale    = math.sqrt(attn_dim)

        # Self-attention projections over h (257-D)
        self.W_Q = nn.Linear(self.h_dim, attn_dim, bias=False)
        self.W_K = nn.Linear(self.h_dim, attn_dim, bias=False)

        # MLP_2: [context (257) ‖ h (257)] → 514 → mlp_hidden → 1
        self.mlp2 = nn.Sequential(
            nn.Linear(self.h_dim * 2, mlp_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 1, bias=True),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for proj in (self.W_Q, self.W_K):
            nn.init.xavier_uniform_(proj.weight)
        nn.init.xavier_uniform_(self.mlp2[0].weight)
        nn.init.zeros_(self.mlp2[0].bias)
        nn.init.xavier_uniform_(self.mlp2[2].weight)
        nn.init.zeros_(self.mlp2[2].bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        v_prime:     torch.Tensor,   # (K, dim)
        agca_scores: torch.Tensor,   # (K,)
    ) -> torch.Tensor:
        """
        Returns
        -------
        refined_scores : Tensor  (K,)  in [0, 1]
        """
        # ── shape guards ─────────────────────────────────────────────
        K = v_prime.shape[0]
        assert v_prime.dim() == 2 and v_prime.shape[1] == self.dim, (
            f"SCRN: expected v_prime (K, {self.dim}), got {tuple(v_prime.shape)}"
        )
        assert agca_scores.shape == (K,), (
            f"SCRN: expected agca_scores ({K},), got {tuple(agca_scores.shape)}"
        )

        # ── 1. Build joint feature h ──────────────────────────────────
        # h_i = [v'_i (256) ‖ agca_score_i (1)]  → (K, 257)
        h = torch.cat([v_prime, agca_scores.unsqueeze(1)], dim=1)   # (K, 257)

        # ── 2. Scaled dot-product self-attention ──────────────────────
        Q      = self.W_Q(h)                                  # (K, 64)
        K_mat  = self.W_K(h)                                  # (K, 64)
        # (K, 64) @ (64, K) → (K, K)
        attn_logits = (Q @ K_mat.t()) / self.scale            # (K, K)
        attn        = F.softmax(attn_logits, dim=-1)          # (K, K) row-stochastic

        # ── 3. Context vector via weighted sum over h ─────────────────
        context = attn @ h                                     # (K, 257)

        # ── 4. Final MLP over [context ‖ h] ──────────────────────────
        mlp_in = torch.cat([context, h], dim=1)               # (K, 514)
        logits = self.mlp2(mlp_in).squeeze(1)                 # (K,)

        return torch.sigmoid(logits)                           # (K,)  ∈ [0,1]

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, h_dim={self.h_dim}, "
            f"attn_dim={self.attn_dim}, scale={self.scale:.1f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick unit test  (run: python -m models.scrn)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  SCRN unit test")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 256

    model = SCRN(dim=dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  h_dim     : {model.h_dim}  (expect 257 = 256 + 1)")

    # ── test with K=5 (inference full set) ──────────────────────────
    K = 5
    v_prime     = torch.randn(K, dim)
    agca_scores = torch.randn(K)

    refined = model(v_prime, agca_scores)
    assert refined.shape == (K,), f"Shape: {refined.shape}"
    assert (refined >= 0).all() and (refined <= 1).all(), "Scores outside [0,1]"
    print(f"  K=5 forward: {tuple(refined.shape)}, range [{refined.min():.4f}, {refined.max():.4f}]  ✓")

    # ── test with K=1 (single-candidate edge case, FR-08) ────────────
    refined1 = model(v_prime[:1], agca_scores[:1])
    assert refined1.shape == (1,), f"K=1 shape: {refined1.shape}"
    assert 0 <= refined1.item() <= 1, "K=1 score outside [0,1]"
    print(f"  K=1 edge case: score={refined1.item():.4f}  ✓")

    # ── test with K=3 (fewer than 5, FR-08) ──────────────────────────
    refined3 = model(v_prime[:3], agca_scores[:3])
    assert refined3.shape == (3,)
    print(f"  K=3 (partial): {tuple(refined3.shape)}  ✓")

    # ── context-awareness check ───────────────────────────────────────
    # Give one proposal a very high agca_score; it should influence others
    agca_biased = torch.zeros(K)
    agca_biased[0] = 10.0                      # proposal 0 is strongly preferred
    agca_biased[1:] = -5.0                     # all others are weak
    scores_biased  = model(v_prime, agca_biased)
    scores_uniform = model(v_prime, torch.zeros(K))
    # The high-score proposal in biased case should push its own refined score up
    # (this is a soft check — exact behaviour depends on initialisation)
    print(f"  Context sensitivity: biased[0]={scores_biased[0]:.4f}, "
          f"uniform[0]={scores_uniform[0]:.4f}  "
          f"({'✓ changed' if abs(scores_biased[0] - scores_uniform[0]) > 0.001 else '(random init may be flat)'})")

    # ── training mode: K=N=8 (all proposals, FR-07) ───────────────────
    v_train = torch.randn(8, dim)
    a_train = torch.randn(8)
    refined_train = model(v_train, a_train)
    assert refined_train.shape == (8,)
    print(f"  Training K=8 (all proposals): {tuple(refined_train.shape)}  ✓")

    # ── gradient flow ─────────────────────────────────────────────────
    loss = refined.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad on {name}"
    print(f"  Gradient flow: all params  ✓")

    # ── attention is row-stochastic ───────────────────────────────────
    with torch.no_grad():
        h_test  = torch.cat([v_prime, agca_scores.unsqueeze(1)], dim=1)
        Q_test  = model.W_Q(h_test)
        K_test  = model.W_K(h_test)
        attn    = F.softmax((Q_test @ K_test.t()) / model.scale, dim=-1)
        row_sum = attn.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones(K), atol=1e-5), "Attention not row-stochastic"
    print(f"  Attention row-stochastic: ✓")

    print(f"\n  All SCRN tests passed ✓")
    print("=" * 60)
