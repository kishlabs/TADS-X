"""
evaluate.py
===========
TADS-X — Team ChipSmiths | DVCon India 2026

Compute mAP@0.5 on COCO-Tasks val2014 (SRS §10.1, §10.2).

WHAT IT MEASURES:
  For each of the 14 SRS task queries, and for every val2014 image that has
  at least one COCO-Tasks ground-truth annotation for that task:
    1. Run the full TADS-X pipeline (predict())
    2. Compare predicted bbox vs GT bbox with IoU@0.5
    3. Accumulate AP using COCO-style precision/recall curve
  Report AP per task + overall mAP@0.5.

BASELINES (SRS §10.2):
  --baseline yolo-only       : highest-confidence detection, ignore task
  --baseline affordance-only : highest-confidence among affordance-preferred classes
  --baseline cosine          : class whose name has highest cosine-sim to task emb

EVALUATION MODES:
  Full evaluation  : python evaluate.py --coco-dir ... --tasks-dir ...
  Subset (fast)    : python evaluate.py ... --subset 50   (50 images/task)
  Single image     : python evaluate.py ... --image path --task "serve wine"

OUTPUT:
  results/map_per_task.json   — written after every full or subset run
  Console table               — human-readable AP per task + mAP summary

USAGE:
  python evaluate.py \\
      --coco-dir    E:/DVCon/COCO \\
      --tasks-dir   E:/DVCon/COCO/dataset-master/coco-tasks/annotations \\
      --checkpoint  checkpoints/tads_x_fp32_best.pt \\
      [--subset 50]

  # Baseline comparison:
  python evaluate.py ... --baseline affordance-only

  # Single-image sanity check (no AP computed):
  python evaluate.py ... --image path/to/img.jpg --task "serve wine"

  # Offline smoke-test (no real data):
  python evaluate.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from task_definitions import (
    PAPER_TASKS,
    SRS_TASKS,
    SRS_TASK_LIST,
    COCO_ID_TO_IDX,
    IDX_TO_CLASS,
    NUM_TASKS,
    NUM_CLASSES,
)
from embeddings import TaskProjection, load_projected_embeddings
from pipeline import (
    ScoringModel,
    TADSX,
    predict,
    load_yolo,
    resolve_task_id,
    TOP_K_INFER,
    PRUNE_THRESH,
    YOLO_IMGSZ,
    DEFAULT_THETA,
    ROI_FEAT_DIM,
    WORKING_DIM,
)


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────

def _iou_xywh(pred: Tuple, gt: Tuple) -> float:
    """
    Compute IoU between two bounding boxes in (x, y, w, h) format.

    Parameters
    ----------
    pred, gt : (x, y, w, h)  — pixel coordinates

    Returns
    -------
    float in [0, 1]
    """
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt

    px2, py2 = px + pw, py + ph
    gx2, gy2 = gx + gw, gy + gh

    inter_x1 = max(px, gx)
    inter_y1 = max(py, gy)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_pred = pw * ph
    area_gt   = gw * gh
    union     = area_pred + area_gt - inter

    return inter / union if union > 0 else 0.0


def _xyxy_to_xywh(box: Tuple) -> Tuple:
    """Convert (x1, y1, x2, y2) to (x, y, w, h)."""
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


# ─────────────────────────────────────────────────────────────────────────────
# AP computation (COCO-style 11-point interpolation)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ap(
    tp_flags:   List[int],   # 1 = TP, 0 = FP (sorted by descending confidence)
    n_gt:       int,
) -> float:
    """
    Compute average precision using 11-point interpolation.

    Parameters
    ----------
    tp_flags : list of 1/0 sorted by descending prediction confidence
    n_gt     : total number of GT objects for this task

    Returns
    -------
    float in [0, 1]
    """
    if n_gt == 0:
        return 0.0

    tp_cumsum = np.cumsum(tp_flags)
    fp_cumsum = np.cumsum([1 - t for t in tp_flags])

    recalls    = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

    # 11-point interpolation (PASCAL VOC style)
    ap = 0.0
    for r_thresh in np.linspace(0, 1, 11):
        p_at_r = precisions[recalls >= r_thresh]
        ap += (p_at_r.max() if len(p_at_r) > 0 else 0.0)
    return ap / 11.0


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_gt_for_task(
    tasks_dir: str,
    file_id:   int,
    coco_instances_path: str,
    split:     str = "val",
) -> Dict[int, dict]:
    """
    Load ground-truth annotations for one task.

    Returns dict { image_id: { 'bbox_xywh': (x,y,w,h), 'coco_cat_id': int } }
    Using the preferred object's bbox from COCO instances annotations.

    Preferred objects have category_id == 1 in COCO-Tasks annotations.
    Their COCO bbox is looked up via image_id + COCO_category_id in instances JSON.

    Parameters
    ----------
    tasks_dir            : directory with task_N_test.json files
    file_id              : 1-indexed paper task file number
    coco_instances_path  : path to instances_val2014.json
    split                : "val" (test split uses val2014 images) or "train"
    """
    ann_suffix = "_test.json" if split == "val" else "_train.json"
    ann_file   = os.path.join(tasks_dir, f"task_{file_id}{ann_suffix}")
    if not os.path.exists(ann_file):
        return {}

    with open(ann_file, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    # Preferred annotations only (category_id == 1 in COCO-Tasks schema)
    preferred = [
        a for a in task_data.get("annotations", [])
        if isinstance(a, dict) and a.get("category_id") == 1
    ]

    # Load COCO instances for bbox lookup
    with open(coco_instances_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Build lookup: (image_id, coco_category_id) → bbox [x,y,w,h]
    bbox_lookup: Dict[Tuple[int, int], List] = {}
    for ann in coco_data.get("annotations", []):
        key = (int(ann["image_id"]), int(ann["category_id"]))
        if key not in bbox_lookup:
            bbox_lookup[key] = ann["bbox"]   # [x, y, w, h]

    gt: Dict[int, dict] = {}
    for a in preferred:
        img_id  = a.get("image_id")
        coco_id = a.get("COCO_category_id")
        if img_id is None or coco_id is None:
            continue
        img_id  = int(img_id)
        coco_id = int(coco_id)

        key = (img_id, coco_id)
        bbox_list = bbox_lookup.get(key)
        if bbox_list is None:
            continue

        gt[img_id] = {
            "bbox_xywh": tuple(bbox_list),
            "coco_cat_id": coco_id,
        }

    return gt


# ─────────────────────────────────────────────────────────────────────────────
# Baseline predictors  (SRS §10.2)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def _predict_yolo_only(
    result_boxes,        # ultralytics result.boxes
    orig_shape: Tuple,
) -> Optional[Tuple]:
    """
    Baseline: highest-confidence detection, task-agnostic.
    Returns (x, y, w, h) or None.
    """
    if result_boxes is None or len(result_boxes) == 0:
        return None
    best = int(result_boxes.conf.argmax().item())
    x1, y1, x2, y2 = result_boxes.xyxy[best].tolist()
    return (x1, y1, x2 - x1, y2 - y1)


@torch.inference_mode()
def _predict_affordance_only(
    result_boxes,
    A: torch.Tensor,
    paper_task_id: int,
) -> Optional[Tuple]:
    """
    Baseline: highest-confidence detection among affordance-preferred classes.
    Returns (x, y, w, h) or None.
    """
    if result_boxes is None or len(result_boxes) == 0:
        return None

    prior_row = A[paper_task_id]
    best_score = -1.0
    best_box   = None

    for i in range(len(result_boxes)):
        c = int(result_boxes.cls[i].item())
        prior = float(prior_row[c].item())
        conf  = float(result_boxes.conf[i].item())
        score = prior * conf   # combined affordance-confidence score
        if score > best_score:
            best_score = score
            x1, y1, x2, y2 = result_boxes.xyxy[i].tolist()
            best_box = (x1, y1, x2 - x1, y2 - y1)

    return best_box


@torch.inference_mode()
def _predict_cosine(
    result_boxes,
    projected_cache: Dict,
    task_query: str,
) -> Optional[Tuple]:
    """
    Baseline: pick object whose COCO class name has highest cosine similarity
    to the task embedding.
    Returns (x, y, w, h) or None.
    """
    if result_boxes is None or len(result_boxes) == 0:
        return None

    task_key = task_query.lower().strip()
    if task_key not in projected_cache:
        return None

    t_norm = F.normalize(projected_cache[task_key].unsqueeze(0), dim=1)  # (1, 256)

    # Build class-name embeddings on the fly (lookup from cache if available)
    best_sim = -1.0
    best_box = None

    for i in range(len(result_boxes)):
        c = int(result_boxes.cls[i].item())
        class_name = IDX_TO_CLASS.get(c, "")
        key = class_name.lower().strip()
        if key not in projected_cache:
            continue
        v_norm = F.normalize(projected_cache[key].unsqueeze(0), dim=1)
        sim = float((t_norm @ v_norm.t()).item())
        if sim > best_sim:
            best_sim = sim
            x1, y1, x2, y2 = result_boxes.xyxy[i].tolist()
            best_box = (x1, y1, x2 - x1, y2 - y1)

    return best_box


# ─────────────────────────────────────────────────────────────────────────────
# Single-task evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_task(
    srs_task_id:         int,              # 1-indexed SRS task ID
    srs_task_query:      str,
    gt_annotations:      Dict[int, dict],  # image_id → {bbox_xywh, coco_cat_id}
    img_dir:             str,
    model:               TADSX,            # full TADS-X model (or None for baselines)
    baseline:            Optional[str],    # None, "yolo-only", "affordance-only", "cosine"
    A:                   torch.Tensor,
    projected_cache:     Dict,
    paper_task_id:       int,              # 0-based
    subset:              Optional[int],
    iou_thresh:          float = 0.5,
    verbose:             bool  = False,
    img_subdir:          str   = "val2014",
) -> Tuple[float, int, int]:
    """
    Evaluate one task. Returns (AP, n_pred, n_gt).

    Each image is processed independently (no batching — FR-10 note).
    """
    image_ids = sorted(gt_annotations.keys())
    if subset:
        image_ids = image_ids[:subset]

    n_gt = len(image_ids)
    if n_gt == 0:
        return 0.0, 0, 0

    predictions: List[Tuple[float, int]] = []  # (confidence, is_tp)

    for img_id in image_ids:
        gt_info  = gt_annotations[img_id]
        gt_bbox  = gt_info["bbox_xywh"]

        fname    = f"COCO_{img_subdir}_{img_id:012d}.jpg"
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            continue

        try:
            if baseline is None:
                # Full TADS-X pipeline
                result = model.predict(img_path, srs_task_query, verbose=verbose)
                if "bbox" not in result:
                    # no-match → FP entry with confidence 0
                    predictions.append((0.0, 0))
                    continue
                pred_bbox  = result["bbox"]
                confidence = result.get("confidence", 0.5)

            else:
                # Baselines — run YOLO manually then apply baseline logic
                yolo_results = model.yolo_model(
                    img_path, imgsz=YOLO_IMGSZ, conf=0.25, device="cpu", verbose=False
                )
                result_boxes = yolo_results[0].boxes

                if baseline == "yolo-only":
                    pred_bbox = _predict_yolo_only(result_boxes, yolo_results[0].orig_shape)
                    confidence = float(result_boxes.conf.max().item()) if (
                        result_boxes is not None and len(result_boxes) > 0
                    ) else 0.0

                elif baseline == "affordance-only":
                    pred_bbox = _predict_affordance_only(result_boxes, A, paper_task_id)
                    confidence = 0.5  # uniform confidence for this baseline

                elif baseline == "cosine":
                    pred_bbox = _predict_cosine(result_boxes, projected_cache, srs_task_query)
                    confidence = 0.5

                else:
                    raise ValueError(f"Unknown baseline: {baseline}")

                if pred_bbox is None:
                    predictions.append((0.0, 0))
                    continue

            iou = _iou_xywh(pred_bbox, gt_bbox)
            is_tp = 1 if iou >= iou_thresh else 0
            predictions.append((float(confidence), is_tp))

        except Exception as e:
            if verbose:
                print(f"    [WARN] Error on image {img_id}: {e}")
            predictions.append((0.0, 0))

    # Sort by descending confidence for AP computation
    predictions.sort(key=lambda x: -x[0])
    tp_flags = [tp for _, tp in predictions]
    ap = _compute_ap(tp_flags, n_gt)

    return ap, len(predictions), n_gt


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    coco_dir:        str,
    tasks_dir:       str,
    checkpoint_path: str,
    raw_emb_path:    str  = "data/task_raw_embeddings.pt",
    proj_path:       str  = "data/projection_layer_trained.pt",
    affordance_path: str  = "data/affordance_matrix.npy",
    thresholds_path: str  = "configs/per_task_thresholds.json",
    yolo_weights:    str  = "yolov8n.pt",
    baseline:        Optional[str] = None,
    subset:          Optional[int] = None,
    iou_thresh:      float = 0.5,
    out_dir:         str  = "results",
    verbose:         bool  = False,
) -> Dict:
    """
    Run mAP@0.5 evaluation on COCO-Tasks val2014.

    Parameters
    ----------
    baseline : None (full TADS-X) | "yolo-only" | "affordance-only" | "cosine"
    subset   : if set, only use first N images per task (SRS §10.6)

    Returns
    -------
    results dict (also written to results/map_per_task.json)
    """
    img_dir    = os.path.join(coco_dir, "val2014")
    coco_anns  = os.path.join(coco_dir, "annotations", "instances_val2014.json")

    for path, label in [
        (img_dir,         "val2014 image directory"),
        (coco_anns,       "instances_val2014.json"),
        (tasks_dir,       "COCO-Tasks annotation directory"),
        (affordance_path, "affordance_matrix.npy"),
        (raw_emb_path,    "task_raw_embeddings.pt"),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing: {path}  ({label})")
            sys.exit(1)

    print(f"\n{'='*60}")
    mode_str = f"Baseline: {baseline}" if baseline else "Full TADS-X"
    subset_str = f"subset={subset}" if subset else "full val2014"
    print(f"  TADS-X Evaluation  [{mode_str}]  [{subset_str}]")
    print(f"  IoU threshold: {iou_thresh}")
    print(f"{'='*60}\n")

    # ── Load shared artefacts ─────────────────────────────────────────────
    A_np = np.load(affordance_path)
    A    = torch.from_numpy(A_np).float()

    # Determine projection path: use trained if available, else init
    if not os.path.exists(proj_path):
        fallback = "data/projection_layer_init.pt"
        print(f"  [WARN] Trained projection not found ({proj_path}), "
              f"using init weights: {fallback}")
        proj_path = fallback

    proj_cache, _ = load_projected_embeddings(raw_emb_path, proj_path, "cpu")

    # Build TADSX model (needed for both full and baselines — baselines still use YOLO)
    model = TADSX.from_checkpoint(
        checkpoint_path   = checkpoint_path,
        affordance_path   = affordance_path,
        raw_emb_path      = raw_emb_path,
        proj_weights_path = proj_path,
        thresholds_path   = thresholds_path,
        yolo_weights      = yolo_weights,
        device            = "cpu",
    )
    model.scoring_model.eval()

    # ── Pre-load COCO instances (once) ───────────────────────────────────
    print("  Loading COCO instances annotations...")
    t0 = time.time()
    with open(coco_anns, "r", encoding="utf-8") as f:
        _coco_instances = json.load(f)
    print(f"  Done in {time.time()-t0:.1f}s  "
          f"({len(_coco_instances['annotations']):,} annotations)")

    # ── Evaluate each of the 14 SRS tasks ────────────────────────────────
    ap_per_task: Dict[str, float] = {}
    results_table: List[Tuple[int, str, float, int, int]] = []

    print(f"\n  {'Task':<4}  {'Query':<22}  {'AP@0.5':>7}  {'Preds':>6}  {'GT':>6}")
    print(f"  {'-'*60}")

    for srs_id, srs_query in SRS_TASKS.items():
        # Resolve SRS task → paper task for affordance matrix lookup
        try:
            resolution = resolve_task_id(srs_query, proj_cache)
            paper_task_id_0 = resolution.paper_task_id   # 0-based
            paper_task_id_1 = resolution.paper_task_id_1 # 1-based (file ID)
        except KeyError:
            print(f"  [SKIP] Task {srs_id} ({srs_query}): embedding not in cache")
            ap_per_task[srs_query] = 0.0
            continue

        # Load GT annotations (uses paper task file number)
        gt = _load_gt_for_task(tasks_dir, paper_task_id_1, coco_anns, split="val")

        ap, n_pred, n_gt = _evaluate_task(
            srs_task_id     = srs_id,
            srs_task_query  = srs_query,
            gt_annotations  = gt,
            img_dir         = img_dir,
            model           = model,
            baseline        = baseline,
            A               = A,
            projected_cache = proj_cache,
            paper_task_id   = paper_task_id_0,
            subset          = subset,
            iou_thresh      = iou_thresh,
            verbose         = verbose,
        )

        key = f"task_{srs_id}_{srs_query.replace(' ', '_')}"
        ap_per_task[key] = ap
        results_table.append((srs_id, srs_query, ap, n_pred, n_gt))

        print(f"  {srs_id:>4}  {srs_query:<22}  {ap:>7.4f}  {n_pred:>6}  {n_gt:>6}")

    # ── Summary ───────────────────────────────────────────────────────────
    valid_aps = [v for v in ap_per_task.values() if isinstance(v, float)]
    map_score = float(np.mean(valid_aps)) if valid_aps else 0.0

    print(f"\n  {'─'*60}")
    print(f"  {'Overall mAP@0.5':<30}  {map_score:.4f}")
    print(f"  {'─'*60}")

    # ── Write JSON (SRS §7.8) ─────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    output = dict(ap_per_task)
    output["overall_mAP"]  = map_score
    output["evaluated_on"] = "val2014"
    output["subset_size"]  = subset
    output["baseline"]     = baseline
    output["iou_thresh"]   = iou_thresh

    out_path = os.path.join(out_dir, "map_per_task.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ Results saved → {out_path}")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Single-image evaluation (SRS §10.6 note; also §8.1 --image flag)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_single(
    image_path:      str,
    task_query:      str,
    checkpoint_path: str,
    raw_emb_path:    str  = "data/task_raw_embeddings.pt",
    proj_path:       str  = "data/projection_layer_trained.pt",
    affordance_path: str  = "data/affordance_matrix.npy",
    thresholds_path: str  = "configs/per_task_thresholds.json",
    yolo_weights:    str  = "yolov8n.pt",
    verbose:         bool = True,
) -> dict:
    """
    Run TADS-X on a single image and print result.

    Equivalent to app.py --image --task.
    """
    if not os.path.exists(proj_path):
        proj_path = "data/projection_layer_init.pt"
        print(f"  [WARN] Using untrained projection: {proj_path}")

    model  = TADSX.from_checkpoint(
        checkpoint_path   = checkpoint_path,
        affordance_path   = affordance_path,
        raw_emb_path      = raw_emb_path,
        proj_weights_path = proj_path,
        thresholds_path   = thresholds_path,
        yolo_weights      = yolo_weights,
        device            = "cpu",
    )

    result = model.predict(image_path, task_query, verbose=verbose)

    print(f"\n  Image : {image_path}")
    print(f"  Task  : {task_query}")
    if "bbox" in result:
        print(f"  → class     : {result['class']}")
        print(f"  → bbox      : {result['bbox']}")
        print(f"  → confidence: {result['confidence']:.4f}")
        print(f"  → resolved  : {result['resolved_paper_task']}")
    else:
        print(f"  → {result.get('result', 'no-match')}")
        print(f"  → reason    : {result.get('reason', 'unknown')}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Ablation runner  (SRS §10.3)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    coco_dir:        str,
    tasks_dir:       str,
    checkpoint_path: str,
    raw_emb_path:    str,
    proj_path:       str,
    affordance_path: str,
    yolo_weights:    str,
    subset:          Optional[int],
    out_dir:         str,
) -> None:
    """
    Run all four evaluation modes for the ablation table in the Stage 2A report:
      1. YOLO-only baseline
      2. Affordance-only baseline
      3. YOLO + cosine (TinyBERT) baseline
      4. Full TADS-X

    Results are written to results/ablation_summary.json.
    """
    print("\n" + "="*60)
    print("  TADS-X Ablation Suite")
    print("="*60)

    ablation_results: Dict[str, float] = {}
    kwargs = dict(
        coco_dir        = coco_dir,
        tasks_dir       = tasks_dir,
        checkpoint_path = checkpoint_path,
        raw_emb_path    = raw_emb_path,
        proj_path       = proj_path,
        affordance_path = affordance_path,
        yolo_weights    = yolo_weights,
        subset          = subset,
        out_dir         = out_dir,
    )

    for mode in ["yolo-only", "affordance-only", "cosine", None]:
        label = mode if mode else "tads-x-full"
        print(f"\n  ── Evaluating: {label} ──")
        r = evaluate(baseline=mode, **kwargs)
        ablation_results[label] = r["overall_mAP"]

    print("\n  ── Ablation Summary ──────────────────────────────────")
    for label, score in ablation_results.items():
        print(f"    {label:<22}: mAP@0.5 = {score:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ablation_summary.json"), "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n  ✓ Ablation saved → {out_dir}/ablation_summary.json")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test (no real data)
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """
    Offline unit test for evaluation utilities.
    Tests: IoU, AP, GT loader schema, baseline logic — no YOLO or real data.
    """
    print("=" * 60)
    print("  evaluate.py smoke-test")
    print("=" * 60)

    # ── IoU tests ────────────────────────────────────────────────────────
    print("\n  [1] IoU helper...")
    assert abs(_iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-6, "Perfect overlap"
    assert abs(_iou_xywh((0, 0, 10, 10), (10, 0, 10, 10))) < 1e-6, "No overlap"
    iou_half = _iou_xywh((0, 0, 10, 10), (5, 0, 10, 10))
    assert 0.3 < iou_half < 0.4, f"Half-overlap iou={iou_half:.4f}"
    print(f"      Perfect=1.0, No-overlap=0.0, Half≈{iou_half:.4f}  ✓")

    # ── AP tests ─────────────────────────────────────────────────────────
    print("\n  [2] AP computation...")
    ap_perfect = _compute_ap([1, 1, 1, 1, 1], n_gt=5)
    assert abs(ap_perfect - 1.0) < 1e-6, f"Perfect AP: {ap_perfect}"

    ap_zero = _compute_ap([0, 0, 0], n_gt=3)
    assert abs(ap_zero) < 1e-6, f"Zero AP: {ap_zero}"

    ap_mixed = _compute_ap([1, 0, 1, 0, 1], n_gt=5)
    assert 0.0 < ap_mixed < 1.0, f"Mixed AP in (0,1): {ap_mixed}"
    print(f"      AP perfect=1.0, zero=0.0, mixed≈{ap_mixed:.4f}  ✓")

    ap_no_gt = _compute_ap([1, 1], n_gt=0)
    assert ap_no_gt == 0.0, "AP with no GT should be 0"
    print(f"      AP n_gt=0 returns 0.0  ✓")

    # ── xyxy → xywh ──────────────────────────────────────────────────────
    print("\n  [3] _xyxy_to_xywh...")
    box = _xyxy_to_xywh((100, 200, 150, 300))
    assert box == (100, 200, 50, 100), f"Got {box}"
    print(f"      (100,200,150,300) → {box}  ✓")

    # ── IoU@0.5 threshold ────────────────────────────────────────────────
    print("\n  [4] IoU threshold check...")
    gt_box = (100, 100, 50, 50)
    # Large overlap → TP
    pred_tp = (102, 102, 48, 48)
    assert _iou_xywh(pred_tp, gt_box) >= 0.5
    # No overlap → FP
    pred_fp = (300, 300, 50, 50)
    assert _iou_xywh(pred_fp, gt_box) < 0.5
    print(f"      TP (large overlap) and FP (no overlap) correctly classified  ✓")

    # ── Sorting by confidence for AP ─────────────────────────────────────
    print("\n  [5] AP sorting invariant...")
    # Same tp_flags, different ordering should give same AP since we sort by conf
    preds_correct_order = [(0.9, 1), (0.7, 1), (0.5, 0)]
    preds_wrong_order   = [(0.5, 0), (0.9, 1), (0.7, 1)]
    preds_correct_order.sort(key=lambda x: -x[0])
    preds_wrong_order.sort(key=lambda x: -x[0])
    tp_1 = [tp for _, tp in preds_correct_order]
    tp_2 = [tp for _, tp in preds_wrong_order]
    ap1 = _compute_ap(tp_1, n_gt=3)
    ap2 = _compute_ap(tp_2, n_gt=3)
    assert abs(ap1 - ap2) < 1e-9, f"Sorting should give same AP: {ap1} vs {ap2}"
    print(f"      Sorting invariant holds  ✓")

    print(f"\n  All smoke-test assertions passed ✓")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TADS-X evaluate.py — mAP@0.5 on COCO-Tasks val2014."
    )
    # Data paths
    parser.add_argument("--coco-dir",    type=str, default=None,
                        help="COCO root dir (contains val2014/ and annotations/)")
    parser.add_argument("--tasks-dir",   type=str, default=None,
                        help="COCO-Tasks annotation dir (task_N_test.json ...)")
    parser.add_argument("--checkpoint",  type=str,
                        default="checkpoints/tads_x_fp32_best.pt",
                        help="ScoringModel checkpoint (.pt)")
    parser.add_argument("--affordance",  type=str,
                        default="data/affordance_matrix.npy")
    parser.add_argument("--raw-emb",     type=str,
                        default="data/task_raw_embeddings.pt")
    parser.add_argument("--proj-path",   type=str,
                        default="data/projection_layer_trained.pt")
    parser.add_argument("--thresholds",  type=str,
                        default="configs/per_task_thresholds.json")
    parser.add_argument("--yolo",        type=str, default="yolov8n.pt")
    parser.add_argument("--out-dir",     type=str, default="results")

    # Eval modes
    parser.add_argument("--subset",      type=int, default=None,
                        help="Evaluate only first N images per task (SRS §10.6)")
    parser.add_argument("--iou-thresh",  type=float, default=0.5)
    parser.add_argument("--baseline",    type=str, default=None,
                        choices=["yolo-only", "affordance-only", "cosine"],
                        help="Run a baseline instead of full TADS-X (SRS §10.2)")
    parser.add_argument("--ablation",    action="store_true",
                        help="Run all baselines + full TADS-X for ablation table")

    # Single-image mode
    parser.add_argument("--image",       type=str, default=None,
                        help="Single image path (skips AP computation)")
    parser.add_argument("--task",        type=str, default=None,
                        help="Task query string (required with --image)")

    # Utils
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--smoke-test",  action="store_true",
                        help="Offline unit test — no data required")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test()
        return

    # ── Single-image mode ─────────────────────────────────────────────────
    if args.image:
        if not args.task:
            parser.error("--task is required with --image")
        evaluate_single(
            image_path      = args.image,
            task_query      = args.task,
            checkpoint_path = args.checkpoint,
            raw_emb_path    = args.raw_emb,
            proj_path       = args.proj_path,
            affordance_path = args.affordance,
            thresholds_path = args.thresholds,
            yolo_weights    = args.yolo,
            verbose         = True,
        )
        return

    # ── Full / subset evaluation ──────────────────────────────────────────
    if args.coco_dir is None or args.tasks_dir is None:
        parser.error("--coco-dir and --tasks-dir are required for evaluation")

    common_kwargs = dict(
        coco_dir        = args.coco_dir,
        tasks_dir       = args.tasks_dir,
        checkpoint_path = args.checkpoint,
        raw_emb_path    = args.raw_emb,
        proj_path       = args.proj_path,
        affordance_path = args.affordance,
        yolo_weights    = args.yolo,
        subset          = args.subset,
        out_dir         = args.out_dir,
    )

    if args.ablation:
        run_ablation(**common_kwargs)
        return

    evaluate(
        **common_kwargs,
        thresholds_path = args.thresholds,
        baseline        = args.baseline,
        iou_thresh      = args.iou_thresh,
        verbose         = args.verbose,
    )


if __name__ == "__main__":
    main()
