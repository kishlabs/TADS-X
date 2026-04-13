"""
models/tcfg.py
==============
TADS-X — Team ChipSmiths | DVCon India 2026

Task-Conditioned Feature Gating (TCFG)
---------------------------------------
Modulates per-proposal visual embeddings with a task-specific gate vector so that
downstream attention (AGCA) operates on a task-focused representation rather than
raw visual features.

Math:
    g(t)  = sigmoid(W_g · t)       W_g ∈ R^{256×256},  g(t) ∈ R^{256}
    v'_i  = v_i ⊙ g(t)            broadcast over N proposals

The gate g(t) is computed ONCE per image (shared across all N proposals):
  - Dimensions where g(t) ≈ 1  → passed through unchanged  (task-relevant)
  - Dimensions where g(t) ≈ 0  → suppressed                (task-irrelevant)

Validation criterion (FR-05):
    After training, mean cosine distance between g(t) vectors across 14 task pairs
    SHALL exceed 0.3 — verifying that the module learns task-specific modulation
    rather than a near-constant gate.

References:
    SRS §7.3 (FR-05), Stage 2A, Team ChipSmiths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from task_definitions import WORKING_DIM  # noqa: F401 — keeps dim constant in one place


# Allow importing WORKING_DIM from task_definitions when available, else fall back
try:
    from task_definitions import WORKING_DIM as _DIM  # type: ignore
except ImportError:
    _DIM = 256


class TCFG(nn.Module):
    """
    Task-Conditioned Feature Gating.

    Parameters
    ----------
    dim : int
        Working embedding dimension (default 256, must match ROI projection output
        and TinyBERT projection output).

    Inputs
    ------
    v_i : Tensor  (N, dim)
        L2-normalised visual embeddings for N proposals, output of the ROI-Align
        + Linear(6272→256) projection in pipeline.py.
    t   : Tensor  (dim,)
        256-D task embedding produced by TaskProjection(TinyBERT CLS),
        already projected but NOT necessarily L2-normalised here (gate handles it).

    Output
    ------
    v_prime : Tensor  (N, dim)
        Task-gated visual embeddings.  Same shape as v_i.

    Notes
    -----
    * g(t) is computed once and broadcast — O(dim²) per image regardless of N.
    * No bias in W_g by design: the sigmoid already has an implicit offset via
      the bias parameter (bias=True is set on the Linear layer).
    * Xavier-uniform initialisation keeps initial gate values centred near 0.5,
      so early gradients flow to both gate-open and gate-closed directions.
    """

    def __init__(self, dim: int = _DIM) -> None:
        super().__init__()
        self.dim = dim
        # W_g ∈ R^{dim×dim} — single linear layer, no activation yet
        self.gate_linear = nn.Linear(dim, dim, bias=True)
        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.zeros_(self.gate_linear.bias)

    def forward(self, v_i: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        v_i : Tensor  (N, dim)
        t   : Tensor  (dim,)  or  (1, dim)

        Returns
        -------
        v_prime : Tensor  (N, dim)
        """
        # --- shape guards ---
        if t.dim() == 2 and t.shape[0] == 1:
            t = t.squeeze(0)                       # (1, 256) → (256,)
        assert t.shape == (self.dim,), (
            f"TCFG: expected t of shape ({self.dim},), got {tuple(t.shape)}"
        )
        assert v_i.dim() == 2 and v_i.shape[1] == self.dim, (
            f"TCFG: expected v_i of shape (N, {self.dim}), got {tuple(v_i.shape)}"
        )

        # --- gate computation (once per image) ---
        g = torch.sigmoid(self.gate_linear(t))     # (dim,)

        # --- element-wise modulation (broadcast over N proposals) ---
        v_prime = v_i * g.unsqueeze(0)             # (N, dim) * (1, dim) → (N, dim)
        return v_prime

    # ------------------------------------------------------------------
    # Diagnostics — called from unit test / ablation logging
    # ------------------------------------------------------------------

    @torch.no_grad()
    def gate_diversity(self, task_embeddings: torch.Tensor) -> float:
        """
        Compute mean pairwise cosine distance between gate vectors for a
        batch of task embeddings.

        Parameters
        ----------
        task_embeddings : Tensor  (T, dim)  — one embedding per task

        Returns
        -------
        float — mean cosine distance; should be > 0.3 after training (FR-05)
        """
        gates = []
        for t in task_embeddings:
            g = torch.sigmoid(self.gate_linear(t))
            gates.append(g)
        gates = torch.stack(gates)                 # (T, dim)
        gates_norm = F.normalize(gates, dim=1)     # L2-normalise
        cos_sim    = gates_norm @ gates_norm.T      # (T, T)

        T = gates.shape[0]
        upper = cos_sim.triu(diagonal=1)
        n_pairs = T * (T - 1) / 2
        mean_cos_sim = upper.sum().item() / n_pairs
        mean_cos_dist = 1.0 - mean_cos_sim
        return mean_cos_dist

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


# ─────────────────────────────────────────────────────────────────────────────
# Quick unit test  (run: python -m models.tcfg)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  TCFG unit test")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 256
    N   = 6      # proposals
    T   = 14     # tasks

    model = TCFG(dim=dim)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- forward pass ---
    v_i = torch.randn(N, dim)
    t   = torch.randn(dim)

    v_prime = model(v_i, t)
    assert v_prime.shape == (N, dim), f"Shape mismatch: {v_prime.shape}"
    print(f"  Forward pass:  v_i {tuple(v_i.shape)} → v_prime {tuple(v_prime.shape)}  ✓")

    # --- gate values should be in (0, 1) ---
    with torch.no_grad():
        g = torch.sigmoid(model.gate_linear(t))
    assert g.min() > 0 and g.max() < 1, "Gate values out of (0,1)"
    print(f"  Gate range:    [{g.min():.4f}, {g.max():.4f}]  ✓")

    # --- test with (1, dim) t shape (pipeline may pass this) ---
    v_prime2 = model(v_i, t.unsqueeze(0))
    assert torch.allclose(v_prime, v_prime2), "Shape-tolerance mismatch"
    print(f"  t shape (1,256) tolerance:  ✓")

    # --- gate diversity on random task embeddings ---
    task_embs = torch.randn(T, dim)
    diversity = model.gate_diversity(task_embs)
    print(f"  Gate diversity (random init): {diversity:.4f}  "
          f"(expect > 0.3 after training)")

    # --- gradient flow ---
    loss = v_prime.sum()
    loss.backward()
    assert model.gate_linear.weight.grad is not None, "No gradient on W_g"
    print(f"  Gradient flow: ✓")

    # --- single-proposal edge case (K=1 from SCRN path) ---
    v_single = torch.randn(1, dim)
    v_out    = model(v_single, t)
    assert v_out.shape == (1, dim)
    print(f"  Single-proposal edge case:  ✓")

    print("\n  All TCFG tests passed ✓")
    print("=" * 60)
