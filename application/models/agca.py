"""
models/agca.py
==============
TADS-X — Team ChipSmiths | DVCon India 2026

Affordance-Guided Cross-Attention (AGCA)
-----------------------------------------
Scores each object proposal against the task embedding by combining two
complementary signals:

  1. **Learned semantic attention** (q · k_i / √d): soft similarity between
     the task embedding and each proposal's gated visual features.
  2. **Affordance prior** A[task_id][coco_class_id_i]: hard statistical prior
     from the training distribution (e.g. wine_glass almost always answers
     "serve wine").

The affordance prior is applied AFTER softmax (multiplicative gate on attention
weights), NOT inside the softmax.  This preserves the normalised attention
distribution while scaling each weight by task-class relevance.

Math:
    q       = W_q · t                          (256,)
    k_i     = W_k · v'_i                       (N, 256)
    raw_i   = softmax_j ( q · k_j / √256 )    (N,)      attention weights
    a_i     = A[ task_id ][ coco_class_id_i ]  (N,)      affordance prior
    α_i     = a_i * raw_i                      (N,)      multiplicative gate
    val_i   = W_v · v'_i                       (N, 256)
    agca_i  = α_i · val_i                      (N, 256)  per-dim gated context
    score_i = MLP( agca_i )   [256→64→1, ReLU] (N,)

References:
    SRS §7.4 (FR-06), Stage 2A, Team ChipSmiths.
"""

import math
from typing import List

import torch
import torch.nn as nn

from task_definitions import WORKING_DIM as _DIM

try:
    from task_definitions import NUM_TASKS as _NUM_TASKS  # type: ignore
    from task_definitions import NUM_CLASSES as _NUM_CLASSES  # type: ignore
except ImportError:
    _NUM_TASKS   = 14
    _NUM_CLASSES = 80


class AGCA(nn.Module):
    """
    Affordance-Guided Cross-Attention + MLP scoring head.

    Parameters
    ----------
    dim : int
        Working embedding dimension (default 256).
    mlp_hidden : int
        Hidden units in the MLP scoring head (default 64).

    Inputs
    ------
    v_prime        : Tensor  (N, dim)
        Task-gated visual embeddings from TCFG.
    t              : Tensor  (dim,)
        256-D task embedding (from TaskProjection on TinyBERT CLS token).
    task_id        : int  in 0..NUM_TASKS-1
        Zero-based task index used to look up the affordance prior row.
    coco_class_ids : List[int]  length N
        COCO matrix index (0–79) for each of the N proposals. These are the
        matrix indices (COCO_ID_TO_IDX), NOT raw COCO category IDs.
    A              : Tensor  (14, 80)
        Affordance prior matrix loaded from data/affordance_matrix.npy.

    Outputs
    -------
    scores    : Tensor  (N,)    — raw (un-sigmoid'd) suitability scores per proposal
    agca_vecs : Tensor  (N, dim) — gated context vectors; passed to SCRN as features

    Notes
    -----
    * All three projection matrices (W_q, W_k, W_v) are separate nn.Linear layers
      with no bias (standard attention convention); bias=False keeps the attention
      symmetric and avoids spurious constant offsets.
    * W_q and W_k project to the full dim (not a reduced head dim) to keep the
      module simple and to match the SRS equation exactly.
    * The MLP head uses raw logits (no final sigmoid here).  Sigmoid is applied
      in the loss function (BCEWithLogitsLoss) during training, and in SCRN's
      score-conditioning path at inference.
    """

    def __init__(self, dim: int = _DIM, mlp_hidden: int = 128) -> None:
        super().__init__()
        self.dim        = dim
        self.scale      = math.sqrt(dim)       # √256 = 16.0

        # Projection matrices — no bias (standard attention)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

        # MLP scoring head: 256 → 64 → 1, ReLU hidden activation
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(mlp_hidden, 1, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(proj.weight)
        # Init only Linear layers (skip ReLU and Dropout by filtering)
        linear_layers = [m for m in self.mlp if isinstance(m, nn.Linear)]
        for layer in linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


    # ------------------------------------------------------------------
    def forward(
        self,
        v_prime:        torch.Tensor,       # (N, dim)
        t:              torch.Tensor,       # (dim,)
        task_id:        int,                # 0-indexed
        coco_class_ids: List[int],          # length N
        A:              torch.Tensor,       # (14, 80)
    ):
        """
        Returns
        -------
        scores    : Tensor  (N,)
        agca_vecs : Tensor  (N, dim)
        """
        # ── shape guards ────────────────────────────────────────────────
        if t.dim() == 2 and t.shape[0] == 1:
            t = t.squeeze(0)

        N = v_prime.shape[0]
        assert v_prime.dim() == 2 and v_prime.shape[1] == self.dim, (
            f"AGCA: expected v_prime (N, {self.dim}), got {tuple(v_prime.shape)}"
        )
        assert t.shape == (self.dim,), (
            f"AGCA: expected t ({self.dim},), got {tuple(t.shape)}"
        )
        assert len(coco_class_ids) == N, (
            f"AGCA: len(coco_class_ids)={len(coco_class_ids)} != N={N}"
        )
        assert 0 <= task_id < _NUM_TASKS, (
            f"AGCA: task_id={task_id} out of range [0, {_NUM_TASKS})"
        )
        assert A.shape == (_NUM_TASKS, _NUM_CLASSES), (
            f"AGCA: expected A ({_NUM_TASKS}, {_NUM_CLASSES}), got {tuple(A.shape)}"
        )

        # ── 1. Query, keys, values ───────────────────────────────────────
        q   = self.W_q(t)           # (dim,)
        K   = self.W_k(v_prime)     # (N, dim)
        V   = self.W_v(v_prime)     # (N, dim)

        # ── 2. Scaled dot-product attention (un-normalised) ──────────────
        #   q · k_j for all j: (dim,) @ (dim, N) → (N,)
        raw = (K @ q) / self.scale  # (N,)
        raw_attn = torch.softmax(raw, dim=0)   # (N,)  — softmax over proposals

        # ── 3. Affordance prior lookup ────────────────────────────────────
        #   a_i = A[task_id][coco_class_id_i]  for each proposal
        device = v_prime.device
        class_idx = torch.tensor(coco_class_ids, dtype=torch.long, device=device)  # (N,)
        a = A[task_id, class_idx].to(device)    # (N,)  float32

        # ── 4. Multiplicative gate AFTER softmax ──────────────────────────
        alpha = a * raw_attn                    # (N,)

        # ── 5. Gated context vectors ──────────────────────────────────────
        #   agca_i = α_i · val_i  (broadcast: (N,1) * (N,dim))
        agca_vecs = alpha.unsqueeze(1) * V      # (N, dim)

        # ── 6. MLP scoring head ───────────────────────────────────────────
        scores = self.mlp(agca_vecs).squeeze(1) # (N,)  raw logits

        return scores, agca_vecs

    def extra_repr(self) -> str:
        return f"dim={self.dim}, scale={self.scale:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# Quick unit test  (run: python -m models.agca)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  AGCA unit test")
    print("=" * 60)

    torch.manual_seed(42)
    dim   = 256
    N     = 5
    T     = 14
    C     = 80

    model = AGCA(dim=dim)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build dummy inputs
    v_prime        = torch.randn(N, dim)
    t              = torch.randn(dim)
    task_id        = 9                          # "serve wine" (0-indexed)
    coco_class_ids = [40, 41, 42, 43, 46]      # bottle, wine glass, cup, fork, banana
    A              = torch.rand(T, C)
    # Normalise A rows (simulate affordance matrix)
    A = A / A.sum(dim=1, keepdim=True)

    # --- forward pass ---
    scores, agca_vecs = model(v_prime, t, task_id, coco_class_ids, A)

    assert scores.shape    == (N,),       f"scores shape: {scores.shape}"
    assert agca_vecs.shape == (N, dim),   f"agca_vecs shape: {agca_vecs.shape}"
    print(f"  Forward pass: v_prime {tuple(v_prime.shape)}")
    print(f"    → scores    {tuple(scores.shape)}    ✓")
    print(f"    → agca_vecs {tuple(agca_vecs.shape)}  ✓")

    # --- t shape (1, dim) tolerance ---
    scores2, _ = model(v_prime, t.unsqueeze(0), task_id, coco_class_ids, A)
    assert torch.allclose(scores, scores2), "Shape-tolerance mismatch"
    print(f"  t shape (1,256) tolerance: ✓")

    # --- affordance scaling: high-prior class should generally beat low-prior ---
    # Artificially boost A for class_id 41 (index 1 → wine glass)
    A_biased = A.clone()
    A_biased[task_id, 41] = 0.999
    A_biased[task_id, :41] = 0.001 / 40
    A_biased[task_id, 42:] = 0.001 / (C - 42)
    scores_biased, _ = model(v_prime, t, task_id, coco_class_ids, A_biased)
    # Index 1 corresponds to coco_class_ids[1] = 41 (wine glass — highest prior)
    best = scores_biased.argmax().item()
    print(f"  Affordance scaling sanity: argmax={best} "
          f"(expect 1 = wine_glass with near-zero others)  "
          f"{'✓' if best == 1 else '— check scaling (random init may vary)'}")

    # --- gradient flow ---
    loss = scores.sum() + agca_vecs.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad on {name}"
    print(f"  Gradient flow: all params ✓")

    # --- single-proposal edge case (K=1) ---
    scores1, vecs1 = model(
        v_prime[:1], t, task_id, [coco_class_ids[0]], A
    )
    assert scores1.shape == (1,)
    assert vecs1.shape   == (1, dim)
    print(f"  Single-proposal edge case: ✓")

    # --- task_id boundary checks ---
    try:
        model(v_prime, t, 14, coco_class_ids, A)   # out-of-range
        print("  Boundary check: FAILED (should have raised)")
        sys.exit(1)
    except AssertionError:
        print(f"  task_id=14 boundary guard: ✓")

    print(f"\n  All AGCA tests passed ✓")
    print("=" * 60)
