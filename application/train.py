"""
train.py
========
TADS-X — Team ChipSmiths | DVCon India 2026

Trains the TADS-X scoring model (ScoringModel + TaskProjection) on
COCO-Tasks train2014 annotations.

WHAT IS TRAINED (SRS §9.1):
  - ScoringModel:
      roi_projection  Linear(6272→256)
      tcfg            TCFG (W_g 256×256)
      agca            AGCA (W_q, W_k, W_v, MLP)
      scrn            SCRN (W_Q, W_K, MLP_2)
  - TaskProjection: Linear(312→256) inside embeddings module

WHAT IS FROZEN:
  - YOLOv8n backbone (inference-only; features cached during dataset build)
  - TinyBERT backbone (raw 312-D embeddings precomputed and cached)

TRAINING PHASES (SRS §9.2):
  Phase 1 — FP32 CrossEntropy only (default: 50 epochs)
  Phase 2 — CrossEntropy + λ·InfoNCE  (default: 20 more epochs, λ=0.1, τ=0.1)

DATASET FORMAT:
  Each sample in the dataset is built from one COCO-Tasks annotation:
    - image_id      (COCO image id)
    - paper_task_id (0-based, row of A)
    - gt_coco_idx   (0-79 COCO matrix index of the correct object)
    - proposals     list of (coco_class_id: int, box_xyxy: tuple, yolo_conf: float)
                    first entry is the GT object; rest are distractors
  ROI features are pre-extracted offline (see --build-cache) to avoid
  running YOLO on every epoch.

OUTPUTS:
  checkpoints/tads_x_fp32_epoch_{N}.pt   — periodic checkpoints
  checkpoints/tads_x_fp32_best.pt        — best val loss checkpoint
  data/projection_layer_trained.pt       — TaskProjection state dict
  configs/per_task_thresholds.json       — θ_t calibrated on val2014

USAGE:
  # Step 1 — pre-extract ROI feature cache (one-time, ~hours on CPU, minutes on GPU)
  python train.py --build-cache \\
      --coco-dir    E:/DVCon/COCO \\
      --tasks-dir   E:/DVCon/COCO/dataset-master/coco-tasks/annotations \\
      --cache-dir   data/roi_cache

  # Step 2 — train
  python train.py \\
      --coco-dir    E:/DVCon/COCO \\
      --tasks-dir   E:/DVCon/COCO/dataset-master/coco-tasks/annotations \\
      --cache-dir   data/roi_cache \\
      --config      configs/train_config.yaml

  # Quick smoke-test (no real data needed)
  python train.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from task_definitions import (
    PAPER_TASKS,
    PAPER_TASK_LIST,
    COCO_ID_TO_IDX,
    IDX_TO_CLASS,
    NUM_TASKS,
    NUM_CLASSES,
)
from embeddings import TaskProjection, load_projected_embeddings, TINYBERT_DIM, WORKING_DIM
from pipeline import (
    ScoringModel,
    ROI_FEAT_DIM,
    TOP_K_INFER,
    DEFAULT_THETA,
    load_yolo,
    _extract_roi_features,
    YOLO_IMGSZ,
)

# ─────────────────────────────────────────────────────────────────────────────
# Default training config  (SRS §7.7 / §9.2)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TRAIN_CONFIG: Dict = {
    "epochs_phase1":        50,
    "epochs_phase2":        20,
    "batch_size":           32,
    "learning_rate":        3e-4,
    "optimizer":            "AdamW",
    "weight_decay":         1e-4,
    "lambda_contrastive":   0.1,
    "tau":                  0.1,
    "device":               "cuda",
    "seed":                 42,
    "val_every_n_epochs":   5,
    "checkpoint_every":     10,
    "checkpoint_dir":       "checkpoints",
    "out_dir":              "data",
    "max_proposals":        8,     # FR-03: up to 8 proposals per image
    "prune_thresh":         0.01,  # same as pipeline PRUNE_THRESH
    "yolo_conf":            0.25,
    "yolo_weights":         "yolov8n.pt",
}

def _compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (float(box_a[2])-float(box_a[0])) * (float(box_a[3])-float(box_a[1]))
    area_b = (float(box_b[2])-float(box_b[0])) * (float(box_b[3])-float(box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def load_config(path: Optional[str]) -> Dict:
    """Load YAML or JSON config; fill missing keys with DEFAULT_TRAIN_CONFIG."""
    cfg = dict(DEFAULT_TRAIN_CONFIG)
    if path is None:
        return cfg
    if not os.path.exists(path):
        print(f"[WARN] Config not found: {path} — using defaults")
        return cfg
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            try:
                import yaml
                loaded = yaml.safe_load(f)
            except ImportError:
                print("[WARN] pyyaml not installed; treating config as JSON")
                f.seek(0)
                loaded = json.load(f)
        else:
            loaded = json.load(f)
    cfg.update(loaded or {})
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Sample dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainSample:
    """
    One training sample — one COCO-Tasks annotation.

    roi_features : Tensor (N, 6272)  — pre-extracted by _build_roi_cache()
    paper_task_id: int  0-based row into A
    gt_index     : int  index inside roi_features/class_ids that is the GT object
    class_ids    : List[int]  COCO matrix indices (0-79), length N
    """
    roi_features:  torch.Tensor        # (N, ROI_FEAT_DIM)
    paper_task_id: int
    gt_index:      int
    class_ids:     List[int]


# ─────────────────────────────────────────────────────────────────────────────
# ROI cache builder
# ─────────────────────────────────────────────────────────────────────────────

def _parse_coco_tasks_annotation(ann_path: str) -> List[dict]:
    """Load preferred-only annotations from a COCO-Tasks JSON file."""
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [a for a in data.get("annotations", [])
            if isinstance(a, dict) and a.get("category_id") == 1]


def _build_roi_cache(
    coco_dir:   str,
    tasks_dir:  str,
    cache_dir:  str,
    split:      str  = "train",     # "train" or "val"
    yolo_weights: str = "yolov8n.pt",
    imgsz:      int  = YOLO_IMGSZ,
    yolo_conf:  float = 0.25,
    max_proposals: int = 8,
    prune_thresh: float = 0.01,
    affordance_matrix: Optional[torch.Tensor] = None,
    device:     str  = "cpu",
    max_images: Optional[int] = None,
) -> None:
    """
    Pre-extract P4 ROI features for all COCO-Tasks annotations and save to disk.

    Each sample is saved as:
        cache_dir/{split}/task_{N}/sample_{image_id}.pt
        → dict with keys: roi_features (N,6272), class_ids list, gt_index int

    This is done ONCE.  train.py loads from cache on subsequent runs.

    Parameters
    ----------
    split : "train" uses task_N_train.json / train2014 images;
            "val"   uses task_N_test.json  / val2014 images
    max_images : if set, cap images per task (useful for debugging)
    """
    import torchvision  # noqa — ensure torchvision available for roi_align

    from pipeline import _make_p4_hook

    ann_suffix = "_train.json" if split == "train" else "_test.json"
    img_subdir = "train2014" if split == "train" else "val2014"
    img_dir = os.path.join(coco_dir, img_subdir, img_subdir)

    print(f"\n{'='*60}")
    print(f"  Building ROI feature cache  [{split}]")
    print(f"  COCO images : {img_dir}")
    print(f"  Tasks dir   : {tasks_dir}")
    print(f"  Cache dir   : {cache_dir}/{split}/")
    print(f"  Device      : {device}")
    print(f"{'='*60}\n")

    # Load YOLO once
    yolo, p4_store, _hook = load_yolo(yolo_weights)

    A = affordance_matrix  # may be None → skip pruning

    total_saved   = 0
    total_skipped = 0

    for file_id in range(1, NUM_TASKS + 1):
        ann_file = os.path.join(tasks_dir, f"task_{file_id}{ann_suffix}")
        if not os.path.exists(ann_file):
            print(f"  [SKIP] task_{file_id}: annotation file not found: {ann_file}")
            continue

        anns = _parse_coco_tasks_annotation(ann_file)
        paper_task_id = file_id - 1  # 0-based

        # Build index: image_id → list of GT annotations
        from collections import defaultdict
        img_to_anns: Dict[int, List[dict]] = defaultdict(list)
        for a in anns:
            img_id = a.get("image_id")
            if img_id is not None:
                img_to_anns[int(img_id)].append(a)

        out_dir_task = os.path.join(cache_dir, split, f"task_{file_id}")
        os.makedirs(out_dir_task, exist_ok=True)

        image_ids = sorted(img_to_anns.keys())
        if max_images:
            image_ids = image_ids[:max_images]

        task_saved = 0
        task_skip  = 0
        for img_id in image_ids:
            out_path = os.path.join(out_dir_task, f"sample_{img_id}.pt")
            if os.path.exists(out_path):
                task_saved += 1
                continue

            # Find image file (COCO naming: COCO_train2014_000000XXXXXX.jpg)
            fname = f"COCO_{img_subdir}_{img_id:012d}.jpg"
            img_path = os.path.join(img_dir, fname)
            if not os.path.exists(img_path):
                task_skip += 1
                continue

            # Collect GT COCO category ids for this image
            gt_coco_cat_ids = set()
            for a in img_to_anns[img_id]:
                cid = a.get("COCO_category_id")
                if cid is not None:
                    try:
                        gt_coco_cat_ids.add(int(cid))
                    except (ValueError, TypeError):
                        pass

            if not gt_coco_cat_ids:
                task_skip += 1
                continue

            # Run YOLO
            try:
                results = yolo(img_path, imgsz=imgsz, conf=yolo_conf,
                               device=device, verbose=False)
            except Exception as e:
                print(f"    [WARN] YOLO failed on {fname}: {e}")
                task_skip += 1
                continue

            result  = results[0]
            p4_feat = p4_store.get("p4")

            if p4_feat is None or result.boxes is None or len(result.boxes) == 0:
                task_skip += 1
                continue

            orig_h, orig_w = result.orig_shape
            boxes_orig = result.boxes.xyxy.cpu()
            class_ids_raw = result.boxes.cls.cpu().long()
            confs = result.boxes.conf.cpu()

            # Affordance pruning (optional if A available)
            if A is not None:
                prior_row = A[paper_task_id]
                keep_mask = [
                    float(prior_row[int(c)].item()) >= prune_thresh
                    for c in class_ids_raw
                ]
            else:
                keep_mask = [True] * len(class_ids_raw)

            # Ensure GT class is kept even if pruned
            kept_indices = []
            for i, keep in enumerate(keep_mask):
                c = int(class_ids_raw[i].item())
                # Check if this detection corresponds to any GT category
                is_gt = False
                for gt_cid in gt_coco_cat_ids:
                    gt_idx = COCO_ID_TO_IDX.get(gt_cid)
                    if gt_idx == c:
                        is_gt = True
                        break
                if keep or is_gt:
                    kept_indices.append(i)

            if not kept_indices:
                task_skip += 1
                continue

            # Limit to max_proposals
            kept_indices = kept_indices[:max_proposals]

            # Find GT index in kept_indices (first match)
            # Collect GT bboxes from annotations (xyxy format)
            gt_bboxes = []
            for a in img_to_anns[img_id]:
                cid = a.get("COCO_category_id")
                bbox = a.get("bbox")   # COCO format: [x, y, w, h]
                if cid is not None and bbox is not None:
                    x, y, w, h = bbox
                    gt_bboxes.append((int(cid), torch.tensor([x, y, x+w, y+h])))

            # IoU-based GT assignment
            gt_index = None
            best_iou = 0.5   # minimum IoU threshold
            for local_i, orig_i in enumerate(kept_indices):
                c = int(class_ids_raw[orig_i].item())
                det_box = boxes_orig[orig_i]   # (4,) xyxy
                for gt_cid, gt_box in gt_bboxes:
                    if COCO_ID_TO_IDX.get(gt_cid) == c:
                        iou = _compute_iou(det_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            gt_index = local_i

            if gt_index is None:
                task_skip += 1
                continue

            # Extract ROI features
            kept_boxes = boxes_orig[kept_indices]   # (K, 4)
            try:
                roi_feats = _extract_roi_features(
                    p4_feat, kept_boxes, (orig_h, orig_w), imgsz
                )                                   # (K, 128, 7, 7)
            except Exception as e:
                print(f"    [WARN] ROI-Align failed on {fname}: {e}")
                task_skip += 1
                continue

            roi_flat = roi_feats.flatten(start_dim=1).cpu()  # (K, 6272)
            class_ids_kept = [int(class_ids_raw[i].item()) for i in kept_indices]

            sample = {
                "roi_features":  roi_flat,
                "class_ids":     class_ids_kept,
                "gt_index":      gt_index,
                "paper_task_id": paper_task_id,
                "image_id":      img_id,
            }
            torch.save(sample, out_path)
            task_saved += 1

        print(f"  File {file_id:2d} ({PAPER_TASKS[file_id]:<44}): "
              f"{task_saved:5d} saved, {task_skip} skipped")
        total_saved   += task_saved
        total_skipped += task_skip

    print(f"\n  Cache built: {total_saved} samples saved, {total_skipped} skipped")
    print(f"  Location: {cache_dir}/{split}/\n")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class COCOTasksDataset(Dataset):
    """
    Loads pre-extracted ROI feature cache from disk.

    One item per COCO-Tasks annotation where the GT object was detected by YOLO.
    Samples are balanced across 14 tasks.

    Parameters
    ----------
    cache_dir : str   path to {split} directory (e.g. data/roi_cache/train)
    max_per_task : int  cap samples per task (set to None for all)
    """

    def __init__(self, cache_dir: str, max_per_task: Optional[int] = None):
        self.samples: List[str] = []   # list of .pt file paths

        for file_id in range(1, NUM_TASKS + 1):
            task_dir = os.path.join(cache_dir, f"task_{file_id}")
            if not os.path.isdir(task_dir):
                continue
            files = sorted(
                os.path.join(task_dir, f)
                for f in os.listdir(task_dir)
                if f.endswith(".pt")
            )
            if max_per_task:
                files = files[:max_per_task]
            self.samples.extend(files)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No .pt sample files found in {cache_dir}. "
                f"Run with --build-cache first."
            )
        print(f"  Dataset: {len(self.samples)} samples from {cache_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.samples[idx], weights_only=True)


def collate_fn(batch: List[dict]) -> List[dict]:
    """
    No padding — proposals per image vary.  Return list of dicts.
    Each dict: roi_features (N,6272), class_ids list, gt_index int, paper_task_id int.
    """
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions  (SRS §9.2)
# ─────────────────────────────────────────────────────────────────────────────

def cross_entropy_loss(scrn_scores: torch.Tensor, gt_index: int) -> torch.Tensor:
    """
    Standard cross-entropy over SCRN scores.

    scrn_scores : (K,)  — raw logits OR probabilities from SCRN
    gt_index    : int   — index of the GT object in [0, K)

    Note: SCRN outputs sigmoid probabilities.  We convert back to logits
    for numerical stability by using BCE with a one-hot target.
    """
    gt_idx = torch.tensor([min(gt_index, scrn_scores.shape[0] - 1)],
                          device=scrn_scores.device)
    return F.cross_entropy(scrn_scores.unsqueeze(0), gt_idx)


def infonce_loss(
    v_prime:       torch.Tensor,   # (N, 256) task-gated visual embeddings
    task_emb:      torch.Tensor,   # (256,) paper task embedding
    gt_index:      int,
    tau:           float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss  (Phase 2, SRS §9.2).

    Pulls the GT object's embedding towards the task embedding (positive pair)
    and pushes distractors away (negative pairs).

    L = -log( exp(sim(v_gt, t) / τ) / Σ_j exp(sim(v_j, t) / τ) )
    """
    # L2-normalise
    v_norm = F.normalize(v_prime, dim=1)        # (N, 256)
    t_norm = F.normalize(task_emb.unsqueeze(0), dim=1)  # (1, 256)

    sims = (v_norm @ t_norm.t()).squeeze(1) / tau  # (N,)

    gt_idx = min(gt_index, v_prime.shape[0] - 1)
    loss = -sims[gt_idx] + torch.logsumexp(sims, dim=0)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Training step  (single sample — variable N proposals)
# ─────────────────────────────────────────────────────────────────────────────

def _train_step(
    sample:         dict,
    scoring_model:  ScoringModel,
    projection:     TaskProjection,
    raw_cache:      dict,           # task_string → Tensor(312,)
    A:              torch.Tensor,   # (14, 80) affordance matrix
    phase:          int,            # 1 or 2
    lambda_nce:     float,
    tau:            float,
    device:         str,
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute loss for one sample.

    Returns: (total_loss, ce_loss_float, nce_loss_float)
    """
    roi_flat       = sample["roi_features"].to(device)   # (N, 6272)
    class_ids      = sample["class_ids"]                 # List[int]
    gt_index       = int(sample["gt_index"])
    paper_task_id  = int(sample["paper_task_id"])        # 0-based

    N = roi_flat.shape[0]
    if N == 0:
        return None, 0.0, 0.0

    # Get paper task embedding via trainable projection
    task_str = PAPER_TASK_LIST[paper_task_id]            # 0-indexed list
    raw_vec  = raw_cache[task_str].to(device)            # (312,)
    with torch.set_grad_enabled(True):
        t = projection(raw_vec)                          # (256,)  gradient flows

    # Score proposals — use top_k=0 during training (all proposals go through SCRN)
    out = scoring_model.score_proposals(
        roi_flat, t,
        paper_task_id, class_ids, A.to(device), top_k=0
    )

    scrn_scores = out["scrn_scores"]         # (N,)
    v_prime     = out["v_prime"]             # (N, 256)

    # Phase 1: CrossEntropy only
    ce = cross_entropy_loss(scrn_scores, gt_index)

    if phase == 2:
        nce  = infonce_loss(v_prime, t, gt_index, tau)
        loss = ce + lambda_nce * nce
        return loss, float(ce.item()), float(nce.item())

    return ce, float(ce.item()), 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-task threshold calibration  (SRS DR-05)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_thresholds(
    scoring_model:    ScoringModel,
    projection:       TaskProjection,
    raw_cache:        dict,
    A:                torch.Tensor,
    val_cache_dir:    str,
    device:           str,
    theta_candidates: Optional[List[float]] = None,
    max_per_task:     int = 200,
) -> Dict[int, float]:
    """
    Calibrate per-task θ_t on val2014 to maximise F1 subject to recall ≥ 0.99.

    Returns dict {paper_task_id_1 (1-indexed): float}.
    Saves to configs/per_task_thresholds.json.
    """
    if theta_candidates is None:
        theta_candidates = [round(0.1 + 0.01 * i, 2) for i in range(80)]  # 0.10 to 0.89

    scoring_model.eval()
    projection.eval()

    thresholds: Dict[int, float] = {}

    for file_id in range(1, NUM_TASKS + 1):
        task_dir = os.path.join(val_cache_dir, f"task_{file_id}")
        if not os.path.isdir(task_dir):
            thresholds[file_id] = DEFAULT_THETA
            continue

        files = sorted(
            os.path.join(task_dir, f)
            for f in os.listdir(task_dir)
            if f.endswith(".pt")
        )[:max_per_task]

        if not files:
            thresholds[file_id] = DEFAULT_THETA
            continue

        paper_task_id = file_id - 1
        task_str = PAPER_TASK_LIST[paper_task_id]
        raw_vec  = raw_cache[task_str].to(device)
        t = projection(raw_vec)

        # Collect scores for each sample
        all_max_scores: List[float] = []
        all_gt_correct: List[bool]  = []

        for fp in files:
            sample    = torch.load(fp, weights_only=True)
            roi_flat  = sample["roi_features"].to(device)
            class_ids = sample["class_ids"]
            gt_index  = int(sample["gt_index"])
            N = roi_flat.shape[0]
            if N == 0:
                continue

            out = scoring_model.score_proposals(
                roi_flat, t, paper_task_id, class_ids, A.to(device), top_k=TOP_K_INFER
            )
            scrn_scores = out["scrn_scores"]            # (K,)
            top_k_idx   = out["top_k_indices"]          # (K,)

            best_k     = int(scrn_scores.argmax().item())
            best_score = float(torch.sigmoid(scrn_scores[best_k]).item())
            pred_idx   = int(top_k_idx[best_k].item())
            correct    = (pred_idx == gt_index)

            all_max_scores.append(best_score)
            all_gt_correct.append(correct)

        if not all_max_scores:
            thresholds[file_id] = DEFAULT_THETA
            continue

        n_total = len(all_max_scores)

        # Find θ that maximises F1 subject to recall ≥ 0.99
        best_theta = DEFAULT_THETA
        best_f1    = -1.0

        for θ in theta_candidates:
            tp = sum(
                1 for score, correct in zip(all_max_scores, all_gt_correct)
                if score >= θ and correct
            )
            fp_count = sum(
                1 for score, correct in zip(all_max_scores, all_gt_correct)
                if score >= θ and not correct
            )
            fn_count = sum(
                1 for score, correct in zip(all_max_scores, all_gt_correct)
                if score < θ and correct
            )

            recall    = tp / (tp + fn_count + 1e-9)
            precision = tp / (tp + fp_count + 1e-9)
            f1        = 2 * precision * recall / (precision + recall + 1e-9)

            if recall >= 0.99 and f1 > best_f1:
                best_f1    = f1
                best_theta = θ

        thresholds[file_id] = best_theta
        print(f"  Task {file_id:2d} ({PAPER_TASKS[file_id]:<44}): "
              f"θ_t={best_theta:.2f}  F1={best_f1:.4f}  n={n_total}")

    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Validation loss pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    scoring_model: ScoringModel,
    projection:    TaskProjection,
    raw_cache:     dict,
    A:             torch.Tensor,
    val_loader:    DataLoader,
    device:        str,
    phase:         int,
    lambda_nce:    float,
    tau:           float,
) -> float:
    """Return mean val loss over all samples."""
    scoring_model.eval()
    projection.eval()

    total_loss = 0.0
    count = 0

    for batch in val_loader:
        for sample in batch:
            paper_task_id = int(sample["paper_task_id"])
            task_str = PAPER_TASK_LIST[paper_task_id]
            raw_vec  = raw_cache[task_str].to(device)
            t = projection(raw_vec)

            roi_flat   = sample["roi_features"].to(device)
            class_ids  = sample["class_ids"]
            gt_index   = int(sample["gt_index"])
            N = roi_flat.shape[0]
            if N == 0:
                continue

            out = scoring_model.score_proposals(
                roi_flat, t, paper_task_id, class_ids, A.to(device), top_k=0
            )
            scrn_scores = out["scrn_scores"]
            v_prime     = out["v_prime"]

            ce = cross_entropy_loss(scrn_scores, gt_index)
            if phase == 2:
                nce = infonce_loss(v_prime, t, gt_index, tau)
                loss = ce + lambda_nce * nce
            else:
                loss = ce

            total_loss += float(loss.item())
            count += 1

    scoring_model.train()
    projection.train()
    return total_loss / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cfg:            Dict,
    train_cache:    str,
    val_cache:      str,
    affordance_path: str = "data/affordance_matrix.npy",
    raw_emb_path:   str  = "data/task_raw_embeddings.pt",
    proj_init_path: str  = "data/projection_layer_init.pt",
) -> None:
    """
    Full training procedure — Phase 1 then Phase 2.

    Parameters
    ----------
    cfg           : training configuration dict
    train_cache   : path to train split ROI cache (data/roi_cache/train)
    val_cache     : path to val split ROI cache   (data/roi_cache/val)
    affordance_path : path to affordance_matrix.npy
    raw_emb_path  : path to task_raw_embeddings.pt
    proj_init_path: path to projection_layer_init.pt (untrained)
    """
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available — falling back to CPU")
        device = "cpu"

    print(f"\n{'='*60}")
    print(f"  TADS-X Training")
    print(f"  Device     : {device}")
    print(f"  Phase 1    : {cfg['epochs_phase1']} epochs  (CrossEntropy)")
    print(f"  Phase 2    : {cfg['epochs_phase2']} epochs  (CE + InfoNCE)")
    print(f"  Batch size : {cfg['batch_size']}")
    print(f"  LR         : {cfg['learning_rate']}")
    print(f"{'='*60}\n")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # ── Load affordance matrix ────────────────────────────────────────────
    A_np = np.load(affordance_path)
    A    = torch.from_numpy(A_np).float()
    print(f"  Affordance matrix loaded: {A.shape}")

    # ── Load raw TinyBERT embeddings (for TaskProjection input) ───────────
    try:
        raw_cache = torch.load(raw_emb_path, weights_only=True)
    except TypeError:
        raw_cache = torch.load(raw_emb_path)
    print(f"  Raw embedding cache: {len(raw_cache)} entries")

    # ── Build models ──────────────────────────────────────────────────────
    scoring_model = ScoringModel().to(device)
    projection    = TaskProjection()
    try:
        proj_sd = torch.load(proj_init_path, weights_only=True)
    except TypeError:
        proj_sd = torch.load(proj_init_path)
    projection.load_state_dict(proj_sd)
    projection.to(device)

    n_scoring = sum(p.numel() for p in scoring_model.parameters() if p.requires_grad)
    n_proj    = sum(p.numel() for p in projection.parameters() if p.requires_grad)
    print(f"  ScoringModel params : {n_scoring:,}")
    print(f"  TaskProjection params: {n_proj:,}")

    # ── Datasets + loaders ───────────────────────────────────────────────
    train_dataset = COCOTasksDataset(train_cache)
    val_dataset   = COCOTasksDataset(val_cache)
    train_loader  = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    all_params = list(scoring_model.parameters()) + list(projection.parameters())
    optimizer  = torch.optim.AdamW(
        all_params,
        lr           = float(cfg["learning_rate"]),
        weight_decay = float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = int(cfg["epochs_phase1"]) + int(cfg["epochs_phase2"]),
        eta_min = 1e-6,
    )

    lambda_nce = float(cfg["lambda_contrastive"])
    tau        = float(cfg["tau"])
    best_val   = float("inf")
    best_ckpt  = os.path.join(cfg["checkpoint_dir"], "tads_x_fp32_best.pt")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1 — CrossEntropy
    # ─────────────────────────────────────────────────────────────────────
    total_epochs = int(cfg["epochs_phase1"]) + int(cfg["epochs_phase2"])

    for epoch in range(1, total_epochs + 1):
        phase = 1 if epoch <= int(cfg["epochs_phase1"]) else 2
        scoring_model.train()
        projection.train()

        epoch_loss  = 0.0
        epoch_ce    = 0.0
        epoch_nce   = 0.0
        n_samples   = 0
        t0 = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            batch_count = 0

            for sample in batch:
                loss, ce_f, nce_f = _train_step(
                    sample, scoring_model, projection, raw_cache,
                    A, phase, lambda_nce, tau, device
                )
                if loss is None:
                    continue
                batch_loss = batch_loss + loss
                epoch_ce  += ce_f
                epoch_nce += nce_f
                batch_count += 1

            if batch_count > 0:
                (batch_loss / batch_count).backward()
                nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                epoch_loss += float(batch_loss.item()) / batch_count
                n_samples  += batch_count

        scheduler.step()

        elapsed = time.time() - t0
        mean_loss = epoch_loss / max(n_samples / cfg["batch_size"], 1)
        mean_ce   = epoch_ce   / max(n_samples, 1)
        mean_nce  = epoch_nce  / max(n_samples, 1)

        print(f"  Epoch {epoch:3d}/{total_epochs} | Phase {phase} | "
              f"Loss={mean_loss:.4f}  CE={mean_ce:.4f}  NCE={mean_nce:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        # ── Validation ─────────────────────────────────────────────────
        if epoch % int(cfg.get("val_every_n_epochs", 5)) == 0 or epoch == total_epochs:
            val_loss = validate(
                scoring_model, projection, raw_cache, A,
                val_loader, device, phase, lambda_nce, tau,
            )
            print(f"          Val loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(scoring_model.state_dict(), best_ckpt)
                print(f"          ✓ New best checkpoint saved → {best_ckpt}")

        # ── Periodic checkpoints ────────────────────────────────────────
        if epoch % int(cfg.get("checkpoint_every", 10)) == 0:
            ckpt = os.path.join(cfg["checkpoint_dir"],
                                f"tads_x_fp32_epoch_{epoch}.pt")
            torch.save(scoring_model.state_dict(), ckpt)

    # ─────────────────────────────────────────────────────────────────────
    # Save trained projection weights
    # ─────────────────────────────────────────────────────────────────────
    proj_trained_path = os.path.join(cfg["out_dir"], "projection_layer_trained.pt")
    torch.save(projection.state_dict(), proj_trained_path)
    print(f"\n  ✓ Trained projection saved → {proj_trained_path}")
    print(f"  ✓ Best checkpoint        → {best_ckpt}")

    # ─────────────────────────────────────────────────────────────────────
    # Calibrate per-task thresholds on val2014
    # ─────────────────────────────────────────────────────────────────────
    print("\n  Calibrating per-task thresholds on val2014...")
    scoring_model.eval()
    projection.eval()

    # Load best weights for calibration
    scoring_best = ScoringModel().to(device)
    try:
        scoring_best.load_state_dict(
            torch.load(best_ckpt, map_location=device, weights_only=True)
        )
    except TypeError:
        scoring_best.load_state_dict(torch.load(best_ckpt, map_location=device))
    scoring_best.eval()

    thresholds = calibrate_thresholds(
        scoring_best, projection, raw_cache, A, val_cache, device,
        max_per_task=200,
    )

    os.makedirs("configs", exist_ok=True)
    thresh_path = "configs/per_task_thresholds.json"
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)
    print(f"  ✓ Per-task thresholds saved → {thresh_path}")

    print(f"\n{'='*60}")
    print("  Training complete.")
    print(f"  Best val loss : {best_val:.4f}")
    print(f"  Checkpoints   : {cfg['checkpoint_dir']}/")
    print(f"  Threshold file: {thresh_path}")
    print(f"\n  Next step:")
    print(f"    python evaluate.py \\")
    print(f"        --coco-dir  <COCO_DIR> \\")
    print(f"        --tasks-dir <COCO_TASKS_DIR> \\")
    print(f"        --checkpoint {best_ckpt}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test (no real data)
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """
    Offline unit test for all training components.
    Requires no real data, no YOLO, no TinyBERT.
    """
    print("=" * 60)
    print("  train.py smoke-test")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cpu"

    N  = 5   # proposals
    A  = torch.rand(NUM_TASKS, NUM_CLASSES)
    A  = (A / A.sum(dim=1, keepdim=True)).float()

    scoring = ScoringModel().to(device)
    proj    = TaskProjection().to(device)

    # Fake raw cache
    raw_cache = {task: torch.randn(TINYBERT_DIM) for task in PAPER_TASK_LIST}

    # Fake sample
    sample = {
        "roi_features":  torch.randn(N, ROI_FEAT_DIM),
        "class_ids":     [40, 41, 42, 43, 44],
        "gt_index":      0,
        "paper_task_id": 9,
        "image_id":      12345,
    }

    print(f"\n  [1] Phase 1 step (CrossEntropy only)...")
    loss1, ce1, nce1 = _train_step(
        sample, scoring, proj, raw_cache, A,
        phase=1, lambda_nce=0.1, tau=0.1, device=device
    )
    assert loss1 is not None, "Loss should not be None"
    print(f"      Loss={float(loss1.item()):.4f}  CE={ce1:.4f}  NCE={nce1:.4f}  ✓")

    print(f"\n  [2] Phase 2 step (CE + InfoNCE)...")
    loss2, ce2, nce2 = _train_step(
        sample, scoring, proj, raw_cache, A,
        phase=2, lambda_nce=0.1, tau=0.1, device=device
    )
    assert loss2 is not None
    assert nce2 > 0, "NCE loss should be positive in Phase 2"
    print(f"      Loss={float(loss2.item()):.4f}  CE={ce2:.4f}  NCE={nce2:.4f}  ✓")

    print(f"\n  [3] Backward pass + gradient check...")
    optimizer = torch.optim.AdamW(
        list(scoring.parameters()) + list(proj.parameters()), lr=3e-4
    )
    optimizer.zero_grad()
    loss2.backward()
    grad_norms = [p.grad.norm().item() for p in scoring.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients flowed into ScoringModel"
    print(f"      {len(grad_norms)} param groups have gradients  ✓")
    optimizer.step()

    print(f"\n  [4] CE loss boundary check...")
    ce_loss = cross_entropy_loss(torch.tensor([0.9, 0.1, 0.05]), 0)
    assert ce_loss.item() < 0.5, f"CE loss too high for easy example: {ce_loss.item()}"
    ce_hard = cross_entropy_loss(torch.tensor([0.05, 0.9, 0.9]), 0)
    assert ce_hard.item() > ce_loss.item(), "Hard example should have higher loss"
    print(f"      Easy CE={ce_loss.item():.4f}  Hard CE={ce_hard.item():.4f}  ✓")

    print(f"\n  [5] InfoNCE loss with GT at index 0...")
    v_prime = F.normalize(torch.randn(5, WORKING_DIM), dim=1)
    task_emb = F.normalize(v_prime[0].unsqueeze(0), dim=1).squeeze(0)  # same direction
    nce = infonce_loss(v_prime, task_emb, gt_index=0, tau=0.1)
    print(f"      NCE (perfect alignment) = {nce.item():.4f}  ✓")

    print(f"\n  All smoke-test assertions passed ✓")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TADS-X train.py — Phase 1 & 2 training for ScoringModel."
    )
    parser.add_argument("--config",        type=str, default=None,
                        help="YAML/JSON config file (default: built-in defaults)")
    parser.add_argument("--coco-dir",      type=str, default=None,
                        help="COCO root dir containing train2014/ and val2014/")
    parser.add_argument("--tasks-dir",     type=str, default=None,
                        help="COCO-Tasks annotation dir (task_N_train.json ...)")
    parser.add_argument("--cache-dir",     type=str, default="data/roi_cache",
                        help="ROI feature cache directory (default: data/roi_cache)")
    parser.add_argument("--affordance",    type=str, default="data/affordance_matrix.npy")
    parser.add_argument("--raw-emb",       type=str, default="data/task_raw_embeddings.pt")
    parser.add_argument("--proj-init",     type=str, default="data/projection_layer_init.pt")
    parser.add_argument("--build-cache",   action="store_true",
                        help="Build ROI feature cache (one-time, then exit)")
    parser.add_argument("--build-val-cache", action="store_true",
                        help="Also build val cache when --build-cache is used")
    parser.add_argument("--max-images",    type=int, default=None,
                        help="Cap images per task during cache build (debug)")
    parser.add_argument("--smoke-test",    action="store_true",
                        help="Run offline smoke-test (no data needed)")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
        return

    cfg = load_config(args.config)

    if args.build_cache:
        if args.coco_dir is None or args.tasks_dir is None:
            parser.error("--build-cache requires --coco-dir and --tasks-dir")

        A = None
        if os.path.exists(args.affordance):
            A = torch.from_numpy(np.load(args.affordance)).float()

        _build_roi_cache(
            coco_dir   = args.coco_dir,
            tasks_dir  = args.tasks_dir,
            cache_dir  = args.cache_dir,
            split      = "train",
            yolo_weights  = cfg.get("yolo_weights", "yolov8n.pt"),
            imgsz         = YOLO_IMGSZ,
            yolo_conf     = cfg.get("yolo_conf", 0.25),
            max_proposals = cfg.get("max_proposals", 8),
            prune_thresh  = cfg.get("prune_thresh", 0.01),
            affordance_matrix = A,
            device= "cuda",
            max_images    = args.max_images,
        )
        if args.build_val_cache:
            _build_roi_cache(
                coco_dir   = args.coco_dir,
                tasks_dir  = args.tasks_dir,
                cache_dir  = args.cache_dir,
                split      = "val",
                yolo_weights  = cfg.get("yolo_weights", "yolov8n.pt"),
                imgsz         = YOLO_IMGSZ,
                yolo_conf     = cfg.get("yolo_conf", 0.25),
                max_proposals = cfg.get("max_proposals", 8),
                prune_thresh  = cfg.get("prune_thresh", 0.01),
                affordance_matrix = A,
                device ="cuda",
                max_images    = args.max_images,
            )
        return

    # ── Normal training ────────────────────────────────────────────────────
    train_cache = os.path.join(args.cache_dir, "train")
    val_cache   = os.path.join(args.cache_dir, "val")

    for path, name in [
        (args.affordance, "affordance_matrix.npy"),
        (args.raw_emb,    "task_raw_embeddings.pt"),
        (args.proj_init,  "projection_layer_init.pt"),
        (train_cache,     "ROI cache (train)"),
        (val_cache,       "ROI cache (val)"),
    ]:
        if not os.path.exists(path):
            print(f"\n[ERROR] Required file/dir not found: {path}  ({name})")
            print(f"  Run --build-cache first, and run embeddings.py to generate embeddings.")
            sys.exit(1)

    train(
        cfg              = cfg,
        train_cache      = train_cache,
        val_cache        = val_cache,
        affordance_path  = args.affordance,
        raw_emb_path     = args.raw_emb,
        proj_init_path   = args.proj_init,
    )


if __name__ == "__main__":
    main()
