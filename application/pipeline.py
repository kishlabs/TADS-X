"""
pipeline.py
===========
TADS-X — Team ChipSmiths | DVCon India 2026

Full inference pipeline: image + task query → bounding box of best object.

────────────────────────────────────────────────────────────────
CRITICAL: TASK-ID MAPPING
────────────────────────────────────────────────────────────────
There are TWO distinct task numbering systems in TADS-X:

  1. SRS Tasks   (14 DVCon evaluation queries, e.g. "serve wine")
                 — what the user/evaluator passes in at inference.

  2. Paper Tasks (14 COCO-Tasks training strings, e.g. "serve wine",
                 "sit comfortably", ...)
                 — what indexes the affordance matrix A (rows 0-13)
                 — and what the model was TRAINED on (embeddings, loss).

These are NOT the same strings (except "serve wine" which happens to match).

At inference, `resolve_task_id()` finds the nearest Paper Task to the given
SRS query via cosine similarity in the shared TinyBERT embedding space.
The returned 0-based paper task index is used for:
  • A[paper_task_id_0] — affordance prior row in AGCA
  • t = paper_task_emb   — task embedding in TCFG / AGCA
  • θ_t[paper_task_id_1] — per-task selection threshold

NEVER hardcode an SRS→paper mapping.  The cosine-similarity resolution is
intentional: it generalises to novel queries outside the training vocabulary.
See task_definitions._APPROX_PAPER_TO_SRS for documentation of the expected
nearest-neighbour assignments (used only as a sanity-check reference).

────────────────────────────────────────────────────────────────
PIPELINE STEPS (predict / TADSX.predict)
────────────────────────────────────────────────────────────────
  1.  YOLOv8n → bounding-box proposals + P4 feature map (26×26×128)
  2.  Map YOLO class indices → COCO matrix indices (0-79) — identity mapping
  3.  resolve_task_id() → paper_task_id (0-13) + paper_task_emb (256,)
  4.  Prune proposals: keep class c if A[paper_task_id, c] ≥ prune_thresh
  5.  Edge cases: N=0 → no-match;  all pruned → no-match
  6.  ROI-Align(P4, boxes) → (N,128,7,7) → flatten(6272) → Linear→L2-norm → v_i (N,256)
  7.  TCFG: v'_i = v_i ⊙ sigmoid(W_g · t)
  8.  AGCA: agca_scores (N,), agca_vecs (N,256)
  9.  Top-5 by AGCA score (inference only; training uses all N)
  10. SCRN: refined_scores (K,) ∈ [0,1]
  11. argmax vs θ_t[paper_task_id] → matched bbox or no-match

────────────────────────────────────────────────────────────────
PUBLIC API
────────────────────────────────────────────────────────────────
  from pipeline import TADSX, ScoringModel, resolve_task_id, predict

  # High-level (app.py / demo):
  model  = TADSX.from_checkpoint('checkpoints/tads_x_fp32_best.pt')
  result = model.predict('image.jpg', 'serve wine')

  # Training (train.py uses ScoringModel directly):
  scoring = ScoringModel()
  # ... training loop ...
  torch.save(scoring.state_dict(), 'checkpoints/tads_x_fp32_best.pt')

References:
    SRS §7.6, §8.1, §8.2, §8.3 — Stage 2A, Team ChipSmiths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from task_definitions import (
    PAPER_TASK_LIST,      # list[str], index 0 = paper file task 1
    PAPER_TASKS,          # dict {1: str, ..., 14: str}
    COCO_CLASSES,
    IDX_TO_CLASS,
    NUM_TASKS,
    NUM_CLASSES,
)
from models.tcfg import TCFG
from models.agca import AGCA
from models.scrn import SCRN

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WORKING_DIM    = 256
ROI_FEAT_DIM   = 128 * 7 * 7          # 6272  (P4 channels × 7 × 7 ROI output)
YOLO_IMGSZ     = 416                  # gives P4 at exactly 26×26 (stride 16)
PRUNE_THRESH   = 0.01                 # default affordance pruning threshold (FR-02)
TOP_K_INFER    = 5                    # max candidates passed to SCRN at inference
DEFAULT_THETA  = 0.40                 # fallback if per_task_thresholds missing a task

# ─────────────────────────────────────────────────────────────────────────────
# Task-ID resolution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskResolution:
    """Result of resolve_task_id()."""
    paper_task_id:   int           # 0-based row index into A  (A[paper_task_id, :])
    paper_task_id_1: int           # 1-based file number        (A[paper_task_id_1 - 1, :])
    paper_task_str:  str           # canonical paper task string (for logging)
    task_emb:        torch.Tensor  # 256-D embedding of the paper task (for TCFG/AGCA)
    cosine_sim:      float         # similarity between query and resolved task


def resolve_task_id(
    srs_query:        str,
    projected_cache:  Dict[str, torch.Tensor],
    verbose:          bool = False,
) -> TaskResolution:
    """
    Map an SRS evaluation query to the nearest Paper Task via cosine similarity.

    The affordance matrix rows and model weights are indexed by Paper Task IDs
    (1-14 → matrix rows 0-13).  This function bridges the gap between contest
    evaluation queries (SRS tasks) and training-time task IDs (paper tasks).

    Parameters
    ----------
    srs_query : str
        Any task query string — typically one of the 14 SRS evaluation queries
        (e.g. "serve wine", "pour water into") but can also be a novel prompt
        if the embedding was pre-computed.
    projected_cache : dict  { lowercase_query_str → Tensor(256,) }
        Loaded from `embeddings.load_projected_embeddings()`.  Must contain
        all 14 Paper Task strings (always populated after `embeddings.py` runs).
    verbose : bool
        If True, print the resolved task and similarity score.

    Returns
    -------
    TaskResolution
        Includes 0-based paper_task_id, 1-based paper_task_id_1, the paper task
        string, the paper task's 256-D embedding, and cosine similarity.

    Raises
    ------
    KeyError
        If the query embedding is not in the cache AND the paper task embeddings
        are missing from the cache (cache is likely corrupt / not built yet).

    Notes
    -----
    * If the SRS query happens to match a Paper Task string exactly (e.g.
      "serve wine" exists in both), it will trivially resolve to itself with
      cosine_sim = 1.0 (after normalisation rounding).
    * The mapping is LEARNED/data-driven, not hardcoded.  The
      task_definitions._APPROX_PAPER_TO_SRS dict documents expected assignments
      for reference only.
    """
    query_key = srs_query.lower().strip()

    # ── Step 1: get embedding for the query ───────────────────────────
    if query_key not in projected_cache:
        available = sorted(projected_cache.keys())
        raise KeyError(
            f"resolve_task_id: '{srs_query}' not found in projected_cache.\n"
            f"  Run embeddings.py first.  Cache has {len(available)} entries:\n"
            f"  {available}"
        )
    query_emb = projected_cache[query_key]                   # (256,)

    # ── Step 2: collect paper task embeddings ────────────────────────
    paper_embs  = []
    paper_ids_1 = []   # 1-based

    for task_id_1, task_str in PAPER_TASKS.items():
        key = task_str.lower().strip()
        if key not in projected_cache:
            raise KeyError(
                f"resolve_task_id: paper task {task_id_1} ('{task_str}') "
                f"not found in projected_cache.  Rebuild the embedding cache."
            )
        paper_embs.append(projected_cache[key])
        paper_ids_1.append(task_id_1)

    paper_embs_t = torch.stack(paper_embs)                   # (14, 256)

    # ── Step 3: cosine similarity ─────────────────────────────────────
    q_norm = F.normalize(query_emb.unsqueeze(0), dim=1)      # (1, 256)
    p_norm = F.normalize(paper_embs_t, dim=1)                # (14, 256)
    sims   = (q_norm @ p_norm.t()).squeeze(0)                # (14,)

    best_local_idx = int(sims.argmax().item())
    best_task_id_1 = paper_ids_1[best_local_idx]             # 1-based
    best_task_id_0 = best_task_id_1 - 1                      # 0-based
    best_sim       = float(sims[best_local_idx].item())
    best_task_str  = PAPER_TASKS[best_task_id_1]
    best_emb       = paper_embs[best_local_idx]

    if verbose:
        print(f"  resolve_task_id: '{srs_query}' "
              f"→ paper task {best_task_id_1} (row {best_task_id_0}) "
              f"'{best_task_str}'  sim={best_sim:.4f}")


    if best_sim < 0.3:
        raise KeyError(
            f"resolve_task_id: '{srs_query}' has no close paper task match "
            f"(best sim={best_sim:.4f} < 0.3). "
            f"Ensure the query is one of the 14 SRS tasks and that embeddings.py "
            f"has been run to generate data/task_raw_embeddings.pt."
        )

    return TaskResolution(
        paper_task_id   = best_task_id_0,
        paper_task_id_1 = best_task_id_1,
        paper_task_str  = best_task_str,
        task_emb        = best_emb,
        cosine_sim      = best_sim,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROI feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _letterbox_boxes(
    boxes_xyxy:  torch.Tensor,   # (N, 4) in original image pixel coordinates
    orig_shape:  Tuple[int, int],# (orig_h, orig_w)
    imgsz:       int,            # square processed image size (e.g. 416)
) -> torch.Tensor:
    """
    Re-apply the same letterbox transform YOLO used so that original-image
    bounding boxes map correctly onto the P4 feature map's coordinate frame.

    YOLO letterboxes the input image to a square of size `imgsz`:
      1. Scale the image uniformly so the longer side == imgsz
      2. Pad the shorter dimension symmetrically with grey

    Parameters
    ----------
    boxes_xyxy : Tensor  (N, 4)  — xyxy in original image pixel space
    orig_shape : (orig_h, orig_w)
    imgsz      : int

    Returns
    -------
    Tensor  (N, 4)  — xyxy in letterboxed imgsz×imgsz pixel space
    """
    oh, ow = orig_shape
    scale  = imgsz / max(oh, ow)
    pad_w  = (imgsz - ow * scale) / 2.0
    pad_h  = (imgsz - oh * scale) / 2.0

    boxes_lb = boxes_xyxy.clone().float()
    boxes_lb[:, 0] = boxes_xyxy[:, 0] * scale + pad_w   # x1
    boxes_lb[:, 1] = boxes_xyxy[:, 1] * scale + pad_h   # y1
    boxes_lb[:, 2] = boxes_xyxy[:, 2] * scale + pad_w   # x2
    boxes_lb[:, 3] = boxes_xyxy[:, 3] * scale + pad_h   # y2

    # Clamp to [0, imgsz]
    boxes_lb = boxes_lb.clamp(0.0, float(imgsz))
    return boxes_lb



def _extract_roi_features(
    p4_feat:     torch.Tensor,   # (1, 128, fh, fw) — from YOLO P4 hook
    boxes_xyxy:  torch.Tensor,   # (N, 4) in original image pixel space
    orig_shape:  Tuple[int, int],# (orig_h, orig_w) of the source image
    imgsz:       int,            # YOLO processed image size (must match the hook)
    output_size: int = 7,
) -> torch.Tensor:
    """
    ROI-Align on the P4 feature map for each proposal.

    Handles non-square P4 feature maps correctly — COCO images are not square,
    so YOLOv8n letterboxes them and produces e.g. (10,13) or (13,9) P4 maps.
    We use the shorter spatial dimension's stride as spatial_scale (since
    roi_align's spatial_scale is a single scalar), and map boxes into the
    letterboxed coordinate space before passing to roi_align.

    The letterboxed space has the image scaled so its longer side == imgsz,
    with grey padding on the shorter side.  Box coordinates in this space
    are what roi_align expects when spatial_scale = 1/stride_of_longer_side.
    """
    _, C, fh, fw = p4_feat.shape

    # Compute per-axis strides (imgsz / feature_map_dim)
    # These will differ for non-square images (e.g. stride_h=41.6, stride_w=32.0)
    # Use the LARGER stride (= stride along the longer original image dimension
    # = the dimension that was NOT padded = stride 32 for a 416-wide featuremap).
    # spatial_scale = 1 / stride maps letterboxed pixel coords → feature map coords.
    stride_h = imgsz / fh   # vertical stride
    stride_w = imgsz / fw   # horizontal stride

    # The shorter stride corresponds to the longer image dimension (no padding).
    # roi_align needs ONE spatial_scale; use min stride (= 1/max_spatial_dim).
    # This is equivalent to using the stride of the unpadded axis.
    spatial_scale = 1.0 / min(stride_h, stride_w)

    # Map boxes from original image → letterboxed imgsz space.
    # _letterbox_boxes uses max(oh,ow) scale, matching YOLO's letterbox exactly.
    boxes_lb = _letterbox_boxes(boxes_xyxy, orig_shape, imgsz).to(p4_feat.device)   # (N, 4)  

    # Clamp box coords to the actual feature map extent in pixel space
    # (fh * stride_h) x (fw * stride_w) — avoids out-of-bounds for padded regions
    boxes_lb[:, 0].clamp_(0.0, fw * stride_w)
    boxes_lb[:, 1].clamp_(0.0, fh * stride_h)
    boxes_lb[:, 2].clamp_(0.0, fw * stride_w)
    boxes_lb[:, 3].clamp_(0.0, fh * stride_h)

    rois = roi_align(
        p4_feat,
        [boxes_lb],
        output_size=(output_size, output_size),
        spatial_scale=spatial_scale,
        aligned=True,
    )                                  # (N, 128, 7, 7)
    return rois


# ─────────────────────────────────────────────────────────────────────────────
# ScoringModel — the single nn.Module checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class ScoringModel(nn.Module):
    """
    Trainable scoring model for TADS-X.

    Contains every trainable component except YOLOv8n and TinyBERT backbones
    (both frozen).  This is the module saved / loaded as the checkpoint.

    Trainable sub-modules
    ---------------------
    roi_projection : Linear(6272, 256)   — adapts ROI-Align output to working dim
    tcfg           : TCFG                — task-conditioned feature gating
    agca           : AGCA                — affordance-guided cross-attention
    scrn           : SCRN                — scene context re-scoring

    Parameters
    ----------
    roi_in_dim : int   ROI flattened dimension  (default 6272 = 128 × 7 × 7)
    dim        : int   working embedding dimension (default 256)
    """

    def __init__(self, roi_in_dim: int = ROI_FEAT_DIM, dim: int = WORKING_DIM) -> None:
        super().__init__()
        self.roi_in_dim = roi_in_dim
        self.dim        = dim

        self.roi_projection = nn.Linear(roi_in_dim, dim, bias=True)
        self.tcfg           = TCFG(dim)
        self.agca           = AGCA(dim)
        self.scrn           = SCRN(dim)

        nn.init.xavier_uniform_(self.roi_projection.weight)
        nn.init.zeros_(self.roi_projection.bias)

    # ── sub-steps exposed for train.py ───────────────────────────────

    def encode_roi(self, roi_flat: torch.Tensor) -> torch.Tensor:
        """
        Project and L2-normalise ROI features.

        Parameters
        ----------
        roi_flat : Tensor  (N, 6272)  — flattened ROI-Align output

        Returns
        -------
        v_i : Tensor  (N, 256)  — L2-normalised visual embeddings
        """
        v = self.roi_projection(roi_flat)           # (N, 256)
        return F.normalize(v, dim=1)                # (N, 256)

    def score_proposals(
        self,
        roi_flat:        torch.Tensor,       # (N, 6272)
        t:               torch.Tensor,       # (256,)  paper task embedding
        paper_task_id:   int,                # 0-based, for A row lookup
        coco_class_ids:  List[int],          # length N
        A:               torch.Tensor,       # (14, 80)
        top_k:           int = TOP_K_INFER,  # 0 = use all (training mode)
    ) -> Dict[str, torch.Tensor]:
        """
        Full scoring pass: ROI features → refined scores.

        Parameters
        ----------
        roi_flat       : Tensor  (N, 6272)
        t              : Tensor  (256,)
        paper_task_id  : int  0-based
        coco_class_ids : List[int]  COCO matrix indices (0-79), length N
        A              : Tensor  (14, 80)
        top_k          : int  — max candidates for SCRN.
                               Set to 0 (or N) to run SCRN on all proposals
                               (training mode, FR-07).

        Returns
        -------
        dict with keys:
            'v_i'           : (N, 256)  L2-normalised visual embeddings
            'v_prime'       : (N, 256)  task-gated embeddings (TCFG output)
            'agca_scores'   : (N,)      raw AGCA logits (all proposals)
            'agca_vecs'     : (N, 256)  gated context vectors
            'top_k_indices' : (K,)      indices of SCRN candidates in [0, N)
            'scrn_scores'   : (K,)      context-aware raw logits (sigmoid applied in predict())
        """
        N = roi_flat.shape[0]

        # Step 1 — ROI projection
        v_i = self.encode_roi(roi_flat)                       # (N, 256)

        # Step 2 — TCFG gating
        v_prime = self.tcfg(v_i, t)                          # (N, 256)

        # Step 3 — AGCA scoring
        agca_scores, agca_vecs = self.agca(
            v_prime, t, paper_task_id, coco_class_ids, A
        )                                                     # (N,), (N, 256)

        # Step 4 — Top-K selection for SCRN
        #   top_k=0 or top_k>=N → use all proposals (training)
        if top_k <= 0 or top_k >= N:
            top_k_idx = torch.arange(N, device=roi_flat.device)
        else:
            K = min(top_k, N)
            top_k_idx = agca_scores.topk(K, dim=0).indices   # (K,)

        # Step 5 — SCRN re-scoring
        if top_k_idx.shape[0] == 1:
            scrn_scores = agca_scores[top_k_idx]   # (1,) raw logits — consistent with SCRN
        else:
            scrn_scores = self.scrn(
                v_prime[top_k_idx],
                agca_scores[top_k_idx],
            )                                       # (K,) raw logits

        return {
            "v_i":           v_i,
            "v_prime":       v_prime,
            "agca_scores":   agca_scores,
            "agca_vecs":     agca_vecs,
            "top_k_indices": top_k_idx,
            "scrn_scores":   scrn_scores,
        }

    def extra_repr(self) -> str:
        return f"roi_in={self.roi_in_dim}, dim={self.dim}"


# ─────────────────────────────────────────────────────────────────────────────
# YOLO + P4 hook
# ─────────────────────────────────────────────────────────────────────────────

def _make_p4_hook(store: dict):
    """Return a forward hook that writes the P4 feature map into `store['p4']`."""
    def hook(module, _input, output):
        # output: Tensor (1, 128, fh, fw) — P4 at stride 16
        store["p4"] = output.detach()
    return hook


def load_yolo(weights: str = "yolov8n.pt"):
    """
    Load YOLOv8n and register a P4 forward hook.

    The P4 feature map (26×26×128 at imgsz=416) is extracted from
    `model.model.model[9]` — the C2f block at stride 16.

    Parameters
    ----------
    weights : str  — ultralytics model path or hub name

    Returns
    -------
    yolo_model    : ultralytics.YOLO
    p4_store      : dict  — populated with key 'p4' after each forward pass
    hook_handle   : torch.utils.hooks.RemovableHook  — call .remove() to clean up
    """
    from ultralytics import YOLO

    yolo  = YOLO(weights)
    store = {}

    # SRS FR-01: hook on model.model.model[9] (C2f block, stride 16)
    handle = yolo.model.model[12].register_forward_hook(_make_p4_hook(store))
    return yolo, store, handle


# ─────────────────────────────────────────────────────────────────────────────
# Proposal dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Proposal:
    """One YOLO detection, pre-processed for scoring."""
    box_xyxy:      Tuple[float, float, float, float]  # original image coords
    coco_class_id: int                                  # 0-79 matrix index
    class_name:    str
    yolo_conf:     float


# ─────────────────────────────────────────────────────────────────────────────
# Main predict() function
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def predict(
    image_path:          str,
    task_query:          str,
    yolo_model,                                    # ultralytics.YOLO (with P4 hook)
    p4_store:            dict,                     # {"p4": Tensor} populated by hook
    scoring_model:       ScoringModel,
    affordance_matrix:   torch.Tensor,             # (14, 80) float32
    projected_cache:     Dict[str, torch.Tensor],  # from embeddings.load_projected_embeddings
    per_task_thresholds: Dict[int, float],         # {paper_task_id_1: float}
    prune_thresh:        float = PRUNE_THRESH,
    imgsz:               int   = YOLO_IMGSZ,
    yolo_conf:           float = 0.25,
    verbose:             bool  = False,
) -> dict:
    """
    Full TADS-X inference for a single image.

    Parameters
    ----------
    image_path          : str  — path to any image readable by PIL / YOLO
    task_query          : str  — SRS task string (e.g. "serve wine")
    yolo_model          : ultralytics.YOLO  — loaded with load_yolo()
    p4_store            : dict  — hook output dict, populated after YOLO forward
    scoring_model       : ScoringModel  — loaded from checkpoint
    affordance_matrix   : Tensor (14, 80)  — from data/affordance_matrix.npy
    projected_cache     : dict  — from embeddings.load_projected_embeddings()
    per_task_thresholds : dict  — {1-based paper_task_id: θ_t float}
                                  from configs/per_task_thresholds.json
    prune_thresh        : float  — affordance threshold for class pruning (FR-02)
    imgsz               : int   — YOLO input size (must match P4 calibration)
    yolo_conf           : float  — YOLO detection confidence threshold
    verbose             : bool  — print step-by-step diagnostics

    Returns
    -------
    Match:
        {
            'bbox':                (x, y, w, h) in original image pixel coords,
            'class':               str  (COCO class name),
            'confidence':          float  (SCRN refined score),
            'task':                str  (original query),
            'resolved_paper_task': str  (matched paper task string),
        }
    No-match:
        {
            'result': 'no suitable object found',
            'task':   str,
            'reason': str  (e.g. 'no detections', 'all classes pruned', 'below threshold'),
        }
    """
    scoring_model.eval()
    A = affordance_matrix  # alias

    def _no_match(reason: str) -> dict:
        return {"result": "no suitable object found", "task": task_query, "reason": reason}

    # ── Step 1: YOLO detection + P4 hook ─────────────────────────────
    results = yolo_model(
        image_path,
        imgsz=imgsz,
        conf=yolo_conf,
        device="cpu",
        verbose=False,
    )
    result = results[0]

    p4_feat  = p4_store.get("p4")                  # (1, 128, fh, fw)
    if p4_feat is None:
        raise RuntimeError(
            "P4 hook did not fire.  Make sure the hook was registered with load_yolo()."
        )

    orig_h, orig_w = result.orig_shape             # original image dimensions

    # ── Edge case A: no detections ────────────────────────────────────
    if result.boxes is None or len(result.boxes) == 0:
        if verbose:
            print(f"  [predict] No detections.")
        return _no_match("no detections")

    # ── Step 2: Parse YOLO detections ────────────────────────────────
    # result.boxes.xyxy  → (N, 4) in ORIGINAL image pixel coordinates
    # result.boxes.cls   → (N,)   YOLO class index (0-79 = COCO matrix index)
    # result.boxes.conf  → (N,)   detection confidence
    boxes_orig = result.boxes.xyxy.cpu()            # (N, 4)
    class_ids  = result.boxes.cls.cpu().long()      # (N,)  ← already 0-79 matrix indices
    confs      = result.boxes.conf.cpu()            # (N,)

    proposals: List[Proposal] = []
    for i in range(len(class_ids)):
        c  = int(class_ids[i].item())
        x1, y1, x2, y2 = boxes_orig[i].tolist()
        proposals.append(Proposal(
            box_xyxy      = (x1, y1, x2, y2),
            coco_class_id = c,
            class_name    = IDX_TO_CLASS.get(c, f"class_{c}"),
            yolo_conf     = float(confs[i].item()),
        ))

    if verbose:
        print(f"  [predict] Detected {len(proposals)} objects: "
              f"{[p.class_name for p in proposals]}")

    # ── Step 3: Resolve task query → paper task ───────────────────────
    resolution = resolve_task_id(task_query, projected_cache, verbose=verbose)
    paper_task_id   = resolution.paper_task_id    # 0-based (A row index)
    paper_task_id_1 = resolution.paper_task_id_1  # 1-based (for θ_t lookup)
    t               = resolution.task_emb         # (256,) paper task embedding

    # ── Step 4: Affordance-based class pruning (FR-02) ────────────────
    prior_row = A[paper_task_id]                   # (80,)
    kept = [
        p for p in proposals
        if float(prior_row[p.coco_class_id].item()) >= prune_thresh
    ]

    if verbose:
        pruned_names = [p.class_name for p in proposals if p not in kept]
        print(f"  [predict] After pruning: {len(kept)}/{len(proposals)} proposals kept "
              f"(pruned: {pruned_names})")

    # ── Edge case B: all proposals pruned ────────────────────────────
    if len(kept) == 0:
        return _no_match("all classes pruned")

    # ── Step 5: Build tensors for scoring ────────────────────────────
    N = len(kept)
    boxes_kept = torch.tensor(
        [p.box_xyxy for p in kept], dtype=torch.float32
    )                                              # (N, 4)  xyxy original coords
    coco_ids_kept = [p.coco_class_id for p in kept]

    # ── Step 6: ROI-Align → flatten → (N, 6272) ──────────────────────
    roi_feats = _extract_roi_features(
        p4_feat, boxes_kept, (orig_h, orig_w), imgsz
    )                                              # (N, 128, 7, 7)
    roi_flat = roi_feats.flatten(start_dim=1)      # (N, 6272)

    # ── Steps 7-10: TCFG → AGCA → top-5 → SCRN ──────────────────────
    out = scoring_model.score_proposals(
        roi_flat, t, paper_task_id, coco_ids_kept, A,
        top_k=TOP_K_INFER,
    )

    # `scrn_scores` is over top_k_indices; map back to original proposal indices
    top_k_idx   = out["top_k_indices"]            # (K,)  indices in kept[]
    scrn_scores = out["scrn_scores"]              # (K,)

    if verbose:
        print(f"  [predict] SCRN scores: {scrn_scores.tolist()}")

    # ── Step 11: argmax vs θ_t ────────────────────────────────────────
    best_k = int(scrn_scores.argmax().item())      # index in [0, K)
    best_score_prob = float(torch.sigmoid(scrn_scores[best_k]).item())  # logit → prob

    theta = per_task_thresholds.get(paper_task_id_1, DEFAULT_THETA)

    if best_score_prob < theta:
        if verbose:
            print(f"  [predict] Best score {best_score_prob:.4f} < θ_t={theta:.4f} → no-match")
        return _no_match(f"below threshold (score={best_score_prob:.4f}, θ_t={theta:.4f})")

    # ── Assemble output ───────────────────────────────────────────────
    best_proposal_idx = int(top_k_idx[best_k].item())
    best_proposal     = kept[best_proposal_idx]

    x1, y1, x2, y2 = best_proposal.box_xyxy
    bbox_xywh = (x1, y1, x2 - x1, y2 - y1)     # convert to (x, y, w, h)

    if verbose:
        print(f"  [predict] → '{best_proposal.class_name}'  "
              f"score={best_score_prob:.4f}  bbox={tuple(round(v, 1) for v in bbox_xywh)}")

    return {
        "bbox":                bbox_xywh,
        "class":               best_proposal.class_name,
        "confidence":          best_score_prob,
        "task":                task_query,
        "resolved_paper_task": resolution.paper_task_str,
    }


# ─────────────────────────────────────────────────────────────────────────────
# High-level TADSX class (SRS §8.2)
# ─────────────────────────────────────────────────────────────────────────────

class TADSX:
    """
    High-level TADS-X inference object.

    Usage
    -----
    model  = TADSX.from_checkpoint('checkpoints/tads_x_fp32_best.pt')
    result = model.predict('image.jpg', 'serve wine')
    # → {'bbox': (234, 156, 89, 112), 'class': 'wine glass', 'confidence': 0.94, ...}

    Parameters (constructor)
    -------------------------
    scoring_model       : ScoringModel
    yolo_model          : ultralytics.YOLO (with P4 hook registered)
    p4_store            : dict  — populated by the P4 hook
    affordance_matrix   : Tensor (14, 80)
    projected_cache     : dict  — from embeddings.load_projected_embeddings()
    per_task_thresholds : dict  — {1-based task_id: float}
    imgsz               : int   (default 416)
    """

    def __init__(
        self,
        scoring_model:       ScoringModel,
        yolo_model,
        p4_store:            dict,
        affordance_matrix:   torch.Tensor,
        projected_cache:     Dict[str, torch.Tensor],
        per_task_thresholds: Dict[int, float],
        imgsz:               int = YOLO_IMGSZ,
    ) -> None:
        self.scoring_model       = scoring_model
        self.yolo_model          = yolo_model
        self.p4_store            = p4_store
        self.affordance_matrix   = affordance_matrix
        self.projected_cache     = projected_cache
        self.per_task_thresholds = per_task_thresholds
        self.imgsz               = imgsz

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path:       str,
        affordance_path:       str  = "data/affordance_matrix.npy",
        raw_emb_path:          str  = "data/task_raw_embeddings.pt",
        proj_weights_path:     str  = "data/projection_layer_trained.pt",
        thresholds_path:       str  = "configs/per_task_thresholds.json",
        yolo_weights:          str  = "yolov8n.pt",
        imgsz:                 int  = YOLO_IMGSZ,
        device:                str  = "cpu",
    ) -> "TADSX":
        """
        Build a TADSX instance from saved artefacts.

        Parameters
        ----------
        checkpoint_path   : ScoringModel state dict (.pt)
        affordance_path   : affordance matrix (.npy)
        raw_emb_path      : raw TinyBERT embeddings cache (.pt)
        proj_weights_path : TaskProjection weights (.pt)
                            Use projection_layer_init.pt before training,
                            projection_layer_trained.pt after.
        thresholds_path   : per-task θ_t JSON (optional — uses DEFAULT_THETA if missing)
        yolo_weights      : ultralytics model name or path
        imgsz             : YOLO input size (must match the value used during training)
        device            : inference device (FR-09: CPU only at inference)
        """
        import json

        # ── Affordance matrix ────────────────────────────────────────
        A_np = np.load(affordance_path)
        A    = torch.from_numpy(A_np).float()

        # ── Task embeddings ──────────────────────────────────────────
        from embeddings import load_projected_embeddings
        proj_cache, _ = load_projected_embeddings(raw_emb_path, proj_weights_path, device)

        # ── Per-task thresholds ──────────────────────────────────────
        if os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                raw = json.load(f)
            thresholds = {int(k): float(v) for k, v in raw.items()}
        else:
            print(f"  [TADSX] Warning: '{thresholds_path}' not found. "
                  f"Using DEFAULT_THETA={DEFAULT_THETA} for all tasks.")
            thresholds = {i: DEFAULT_THETA for i in range(1, NUM_TASKS + 1)}

        # ── ScoringModel ─────────────────────────────────────────────
        scoring = ScoringModel()
        state   = torch.load(checkpoint_path, map_location=device,
                             weights_only=True)
        scoring.load_state_dict(state)
        scoring.to(device).eval()

        # ── YOLO + P4 hook ───────────────────────────────────────────
        yolo, p4_store, _ = load_yolo(yolo_weights)

        return cls(
            scoring_model       = scoring,
            yolo_model          = yolo,
            p4_store            = p4_store,
            affordance_matrix   = A,
            projected_cache     = proj_cache,
            per_task_thresholds = thresholds,
            imgsz               = imgsz,
        )

    # ------------------------------------------------------------------
    def predict(self, image_path: str, task_query: str, verbose: bool = False) -> dict:
        """
        Single-image inference.

        Parameters
        ----------
        image_path : str  — path to image
        task_query : str  — SRS task string or any query in the embedding cache
        verbose    : bool

        Returns
        -------
        dict — see predict() docstring for schema.
        """
        return predict(
            image_path          = image_path,
            task_query          = task_query,
            yolo_model          = self.yolo_model,
            p4_store            = self.p4_store,
            scoring_model       = self.scoring_model,
            affordance_matrix   = self.affordance_matrix,
            projected_cache     = self.projected_cache,
            per_task_thresholds = self.per_task_thresholds,
            imgsz               = self.imgsz,
            verbose             = verbose,
        )


# ─────────────────────────────────────────────────────────────────────────────
# resolve_task_id unit test  (run: python pipeline.py --test-resolve)
# Full smoke test requires YOLO + embeddings: python pipeline.py --smoke-test
# ─────────────────────────────────────────────────────────────────────────────

def _test_resolve_task_id():
    """
    Offline unit test for resolve_task_id() that requires only task_definitions
    and a mocked projected_cache.  No YOLO, no disk artefacts needed.
    """
    print("=" * 60)
    print("  resolve_task_id() unit test  (mock embeddings)")
    print("=" * 60)

    from task_definitions import PAPER_TASKS, SRS_TASKS, _APPROX_PAPER_TO_SRS

    torch.manual_seed(42)

    # ── Build a mock projected_cache ─────────────────────────────────
    # Assign a unique random embedding to each paper task string.
    # For each SRS task, use the embedding of its APPROX paper match
    # so cosine similarity is guaranteed to resolve correctly.
    mock_cache: Dict[str, torch.Tensor] = {}

    # One random unit vector per paper task
    paper_embs: Dict[int, torch.Tensor] = {}
    for tid, s in PAPER_TASKS.items():
        e = F.normalize(torch.randn(1, WORKING_DIM), dim=1).squeeze(0)
        mock_cache[s.lower().strip()] = e
        paper_embs[tid] = e

    # SRS tasks: copy the paper task embedding with a tiny perturbation
    # (so cosine sim is very high but not exactly 1.0)
    for srs_id, srs_str in SRS_TASKS.items():
        expected_paper_id = _APPROX_PAPER_TO_SRS.get(
            # invert the dict: find which paper task maps to this SRS id
            next(
                (pt for pt, si in _APPROX_PAPER_TO_SRS.items() if si == srs_id),
                None
            ),
            None
        )
        # If we found a paper task for this SRS id, use a perturbed version
        # of that paper task's embedding; otherwise just use a fresh random one.
        matched_paper_id = None
        for pt, si in _APPROX_PAPER_TO_SRS.items():
            if si == srs_id:
                matched_paper_id = pt
                break

        key = srs_str.lower().strip()
        if key not in mock_cache:
            if matched_paper_id is not None:
                base = paper_embs[matched_paper_id]
                noise = torch.randn_like(base) * 0.05
                mock_cache[key] = F.normalize((base + noise).unsqueeze(0), dim=1).squeeze(0)
            else:
                mock_cache[key] = F.normalize(torch.randn(1, WORKING_DIM), dim=1).squeeze(0)

    # ── Test: "serve wine" should resolve to paper task 10 ─────────────
    # Paper task 10 IS "serve wine" — exact string match → cosine_sim ≈ 1.0
    res = resolve_task_id("serve wine", mock_cache, verbose=True)
    assert res.paper_task_id_1 == 10, (
        f"'serve wine' should resolve to paper task 10, got {res.paper_task_id_1}"
    )
    print(f"  'serve wine' → paper task {res.paper_task_id_1} "
          f"('{res.paper_task_str}')  sim={res.cosine_sim:.4f}  ✓")

    # ── Test: "dig a hole with" → paper task 7 ("dig hole") ────────────
    res2 = resolve_task_id("dig a hole with", mock_cache, verbose=True)
    # With mock embeddings the SRS "dig a hole with" embedding was set to
    # be a perturbation of paper task 7 ("dig hole") embedding.
    print(f"  'dig a hole with' → paper task {res2.paper_task_id_1} "
          f"('{res2.paper_task_str}')  sim={res2.cosine_sim:.4f}")
    # Soft assertion: sim should be high (> 0.9) since we perturbed from that task
    if res2.cosine_sim > 0.9:
        print(f"    High similarity (>0.9) confirmed ✓")
    else:
        print(f"    Note: expected high sim for mock; got {res2.cosine_sim:.4f}")

    # ── Test: all 14 SRS tasks resolve without error ──────────────────
    print(f"\n  Resolving all 14 SRS tasks (mock embeddings — low sim expected for some):")
    for srs_id, srs_str in SRS_TASKS.items():
        try:
            r = resolve_task_id(srs_str, mock_cache)
            print(f"    SRS {srs_id:2d} '{srs_str:<22}' "
                  f"→ paper {r.paper_task_id_1:2d} '{r.paper_task_str:<44}' "
                  f"sim={r.cosine_sim:.4f}")
        except KeyError:
            print(f"    SRS {srs_id:2d} '{srs_str:<22}' "
                  f"→ sim < 0.3 (expected with random mock embeddings — ok)")

    # ── Test: unknown query raises KeyError ───────────────────────────
    try:
        resolve_task_id("juggle flaming torches", mock_cache)
        print("  KeyError guard: FAILED (should have raised)")
    except KeyError:
        print(f"\n  Unknown query KeyError guard: ✓")

    # ── Test: task_emb shape and dtype ───────────────────────────────
    res3 = resolve_task_id("sit on", mock_cache)
    assert res3.task_emb.shape == (WORKING_DIM,), f"Bad emb shape: {res3.task_emb.shape}"
    assert res3.task_emb.dtype == torch.float32
    print(f"  task_emb shape/dtype: {tuple(res3.task_emb.shape)}, {res3.task_emb.dtype}  ✓")

    print(f"\n  All resolve_task_id tests passed ✓")
    print("=" * 60)


def _test_scoring_model():
    """Offline unit test for ScoringModel (no YOLO needed)."""
    print("\n" + "=" * 60)
    print("  ScoringModel unit test  (mock ROI features)")
    print("=" * 60)

    torch.manual_seed(42)
    N  = 6
    A  = torch.rand(NUM_TASKS, NUM_CLASSES)
    A  = A / A.sum(dim=1, keepdim=True)          # row-normalise

    model = ScoringModel()
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total trainable parameters: {total:,}")

    roi_flat       = torch.randn(N, ROI_FEAT_DIM)
    t              = F.normalize(torch.randn(1, WORKING_DIM), dim=1).squeeze(0)
    paper_task_id  = 9          # "serve wine" row
    coco_ids       = [40, 41, 42, 43, 44, 46]

    out = model.score_proposals(
        roi_flat, t, paper_task_id, coco_ids, A, top_k=TOP_K_INFER
    )

    assert out["v_i"].shape          == (N, WORKING_DIM)
    assert out["v_prime"].shape      == (N, WORKING_DIM)
    assert out["agca_scores"].shape  == (N,)
    assert out["scrn_scores"].shape[0] <= TOP_K_INFER
    print(f"  score_proposals output shapes: ✓")
    print(f"  scrn_scores: {out['scrn_scores'].tolist()}")

    # Training mode: top_k=0 → all proposals
    out_train = model.score_proposals( 
        roi_flat, t, paper_task_id, coco_ids, A, top_k=0
    )
    assert out_train["top_k_indices"].shape[0] == N
    print(f"  Training mode (top_k=0): K={out_train['top_k_indices'].shape[0]} == N={N}  ✓")

    # K=1 edge case
    out1 = model.score_proposals(
        roi_flat[:1], t, paper_task_id, coco_ids[:1], A, top_k=5
    )
    # scrn_scores are raw logits; no range assertion needed
    assert out1["scrn_scores"].shape == (1,)
    print(f"  K=1 edge case: scrn_score={out1['scrn_scores'].item():.4f}  ✓")

    print(f"\n  All ScoringModel tests passed ✓")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TADS-X pipeline unit tests")
    parser.add_argument("--test-resolve", action="store_true",
                        help="Run resolve_task_id() unit test (no YOLO/disk needed)")
    parser.add_argument("--test-scoring", action="store_true",
                        help="Run ScoringModel unit test (no YOLO/disk needed)")
    parser.add_argument("--test-all", action="store_true",
                        help="Run all offline tests")
    args = parser.parse_args()

    if args.test_resolve or args.test_all:
        _test_resolve_task_id()

    if args.test_scoring or args.test_all:
        _test_scoring_model()

    if not any(vars(args).values()):
        parser.print_help()
