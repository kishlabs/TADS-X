

1. **14K samples** — that's not the full COCO dataset size. That's the number of *COCO-Tasks annotated training samples* that survived the ROI cache build — i.e., images where YOLO detected something AND the ground-truth object was captured. The full COCO train2014 (83K) is irrelevant here; only the subset that COCO-Tasks annotated for 14 tasks matters, and only those where YOLO actually fired. So 14,066 is the real effective training set size, and it may be correct or may indicate missed GT objects — we need to verify.

2. **27 embeddings** — confirmed: 27 is correct because one Paper Task string and its SRS Task string are identical (they overlap), so only 27 unique strings exist, not 28. This is NOT a bug.

Now let me write the precise Antigravity prompt.Good. Now I have everything I need. Let me write the precise, accurate prompt.Now I have the complete and accurate picture. Here is the Antigravity prompt:

---

## Prompt for Antigravity

---

You are working on **TADS-X** (Task-Aware Dual-Stream Detection with Affordance Gating), a task-aware object detection system built in Python/PyTorch for the DVCon India 2026 Design Contest Stage 2A. The codebase is located at `E:\DVCon\application\`. The full architecture and requirements are defined in `TADS_X_SRS_.md`.

---

### System Overview

TADS-X answers: *"Which object in this image should I use for a given task?"* — not just *"What objects are in this image?"*

The pipeline:
1. **YOLOv8n** — detects all objects, extracts P4 feature map (26×26×128) via a forward hook on `model.model.model[9]`
2. **Class Pruning** — removes proposals where `A[task_id][class] < θ_t` using a prebuilt 14×80 affordance matrix
3. **ROI-Align** — extracts 7×7×128 features per proposal from the P4 map
4. **ROI Projection** — Linear(6272→256), L2-normalised → visual embedding `v_i`
5. **TinyBERT** (frozen) — encodes task query → 312-D; trainable Linear(312→256) → task embedding `t`
6. **TCFG** (`models/tcfg.py`) — `v'_i = v_i ⊙ sigmoid(Dropout(W_g · t))`
7. **AGCA** (`models/agca.py`) — affordance-guided cross-attention → `agca_scores (N,)` + `agca_vecs (N, 256)`
8. **SCRN** (`models/scrn.py`) — self-attention over top-5 candidates → `refined_scores (K,)` as raw logits
9. **Final selection** — `sigmoid(refined_scores)` vs per-task threshold `θ_t`

**14 tasks** (task IDs 1–14): serve wine, pour water into, cut something with, hit something with, dig a hole with, scoop something with, pound something with, cool something in, sit on, lie on, carry things in, read, check the time with, look through.

**Target metric:** mAP@0.5 ≥ 0.60 on COCO-Tasks val2014. Minimum acceptable: 0.55.

**Datasets on disk:**
- `E:\DVCon\COCO\train2014\` — 82,783 images
- `E:\DVCon\COCO\val2014\` — 40,504 images
- `E:\DVCon\COCO\dataset-master\coco-tasks\annotations\` — `task_N_train.json` and `task_N_test.json` for N=1..14

**Training uses a pre-built ROI feature cache** at `data/roi_cache/train/` and `data/roi_cache/val/`. The cache contains 14,066 training samples and 3,323 validation samples. This number is small relative to the full COCO dataset because COCO-Tasks only annotates a subset of COCO images for each of the 14 tasks — these are the images where a relevant task-object annotation exists. The cache only stores samples where YOLOv8n detected at least one object.

**What was already fixed** (do not revert these):
- `train.py` loss accumulation bug fixed — `epoch_loss` correctly averages over batches
- `models/tcfg.py` — `nn.Dropout(p=0.3)` added after `gate_linear`, before sigmoid
- `train.py` optimizer — separate AdamW parameter groups: `scoring_model` weight_decay=5e-3, `projection` weight_decay=1e-2
- `configs/train_config.yaml` — `learning_rate: 5e-5`

**What training produced** (Phase 1 complete, 50 epochs, CrossEntropy only):
- Best checkpoint: `checkpoints/tads_x_fp32_best.pt` — saved at epoch 20
- Val Top-1 at epoch 20: **0.6970** (best)
- Val Top-1 at epoch 50: 0.6961 — flat since epoch 20, confirming the model's best generalization was at epoch 20
- Train loss at epoch 50: 0.153 — strong overfitting, val loss rose from 0.77 → 1.13 while train loss fell to 0.15
- Phase 2 (InfoNCE, 20 epochs) was **NOT run** — training was stopped at epoch 50

**Task embedding cache:** `data/task_raw_embeddings.pt` has 27 entries. This is correct — one Paper Task string and its corresponding SRS Task string are identical, so there are only 27 unique query strings across 14 Paper Tasks + 14 SRS Tasks. This is not a bug.

**Per-task thresholds:** `configs/per_task_thresholds.json` does **NOT exist yet** — threshold calibration was never run.

---

### Root Cause Analysis — Why mAP@0.5 target was not met

**Root Cause 1 — Val Top-1 ≠ mAP@0.5 (metric mismatch — most important)**

Val Top-1 accuracy (0.697) measures only whether the correct object *class* was ranked first. mAP@0.5 additionally requires that the predicted bounding box has IoU ≥ 0.5 with the ground-truth bbox. The model was trained purely on ranking (CrossEntropy over class scores) with no bbox quality signal. A correct class prediction with a loose or shifted bbox still scores 0 in mAP. This is the most structurally significant gap — 0.697 Top-1 does not translate linearly to mAP.

**Root Cause 2 — Per-task thresholds θ_t were never calibrated**

`configs/per_task_thresholds.json` does not exist. The inference pipeline has no calibrated thresholds, meaning either a hardcoded default is being used or threshold application is broken. A miscalibrated θ_t on any task can collapse AP on that task to near zero by either suppressing all correct predictions (recall = 0) or accepting everything (precision = 0). Calibrating θ_t correctly is the highest-leverage zero-training-cost fix.

**Root Cause 3 — Phase 2 InfoNCE training never ran**

The training was designed in two phases. Phase 1 (CrossEntropy, 50 epochs) establishes basic ranking. Phase 2 (CrossEntropy + λ=0.1 × InfoNCE, 20 epochs) explicitly pulls correct object visual embeddings toward the task embedding and pushes distractors away — this is what makes `θ_t` thresholds meaningful and what drives TCFG gate diversity (FR-05: cosine distance > 0.3 across tasks). Without Phase 2, the embedding space is not task-discriminative enough for threshold-based selection to work reliably.

**Root Cause 4 — Overfitting: train/val gap of 0.98 at epoch 50**

Val loss rose monotonically from 0.77 (epoch 10) to 1.13 (epoch 50) while train loss fell to 0.15. The gap of 0.98 far exceeds the target of < 0.05. Only TCFG has dropout (0.3); AGCA MLP (256→64→1) and SCRN MLP_2 have no dropout. The ROI feature cache means the model sees identical tensors every epoch with no augmentation — this is equivalent to training on a fixed lookup table, which memorizes training samples rather than generalizing.

**Root Cause 5 — ROI cache may have missed GT objects**

The cache was built by running YOLOv8n over training images and capturing detections. If YOLO failed to detect the ground-truth object (missed detection, low confidence, occlusion), that sample was stored without a valid positive — or excluded entirely. The model may have been trained on samples where the correct answer was never in the candidate set. This silently reduces effective supervision quality without being visible in training logs.

---

### Required Fixes — Implement All of These

**Fix 1 — Run Phase 2 from the epoch 20 checkpoint (not epoch 50)**

The epoch 50 weights are overfit. Load the best checkpoint before starting Phase 2:

```python
# In train.py, before the Phase 2 training loop:
ckpt = torch.load('checkpoints/tads_x_fp32_best.pt', map_location=device)
scoring_model.load_state_dict(ckpt['scoring_model'])
projection.load_state_dict(ckpt['projection'])
```

Then run Phase 2: 20 epochs, loss = `F.cross_entropy(scores, gt_index) + 0.1 × InfoNCE(v_prime_gt, t, v_prime_negatives)`, τ=0.1. Save the best Phase 2 checkpoint as `checkpoints/tads_x_fp32_phase2_best.pt`. Keep the epoch 20 Phase 1 checkpoint untouched as fallback.

**Fix 2 — Run threshold calibration immediately after Phase 2**

Implement and run the calibration algorithm from SRS DR-05:

```
For each task t in 1..14:
    For each threshold candidate θ in [0.1, 0.9] step 0.01:
        For each val image annotated for task t:
            run predict(image, task_t) → get score
        compute recall = correct / GT_count
        compute precision = correct / predicted_count
        compute F1 = 2 * P * R / (P + R)
    θ_t[t] = argmax F1 subject to recall >= 0.99
```

Save to `configs/per_task_thresholds.json` with keys "1" through "14". This must exist before `evaluate.py` is run.

**Fix 3 — Add dropout to AGCA and SCRN MLPs**

Currently only TCFG has dropout. Add `nn.Dropout(p=0.2)` between every linear layer in:
- `models/agca.py`: inside the MLP scoring head (between Linear(256→64) and Linear(64→1))
- `models/scrn.py`: inside MLP_2 (between its linear layers)

These are the modules doing the final scoring — they are the most likely to memorize training patterns.

**Fix 4 — Add feature-level augmentation to the ROI cache Dataset**

The ROI cache is fixed on disk. Every epoch the model sees the same tensors. Add augmentation in the `__getitem__` method of the dataset class, applied only during training:

```python
def augment_roi_features(self, v_i: torch.Tensor) -> torch.Tensor:
    # Small Gaussian noise
    v_i = v_i + torch.randn_like(v_i) * 0.015
    # Random feature dropout — zero 15% of dimensions
    mask = (torch.rand_like(v_i) > 0.15).float()
    v_i = v_i * mask
    return v_i
```

Apply this only when `self.split == 'train'`. This is the most direct regularization against the fixed-cache overfitting.

**Fix 5 — Verify GT object presence in the ROI cache**

Add a diagnostic check to measure what percentage of training samples in the cache actually contain the ground-truth object in their candidate set. Run this before retraining:

```python
# Pseudocode diagnostic
gt_in_candidates = 0
total = 0
for sample in train_cache:
    if sample['gt_index'] is not None and sample['gt_index'] >= 0:
        gt_in_candidates += 1
    total += 1
print(f"GT in candidates: {gt_in_candidates}/{total} = {gt_in_candidates/total:.3f}")
```

If this ratio is below 0.90, the cache build logic needs to be revisited — specifically, ensure the YOLO confidence threshold during cache building is low enough (≤ 0.15) to capture GT objects that YOLO is uncertain about.

**Fix 6 — Run subset evaluation before full evaluation**

After Phase 2 + threshold calibration, run:
```bash
python evaluate.py --coco-dir E:\DVCon\COCO --tasks-dir E:\DVCon\COCO\dataset-master\coco-tasks\annotations --subset 50 --checkpoint checkpoints/tads_x_fp32_phase2_best.pt
```

This gives a per-task AP estimate in ~20 minutes. If any task shows AP = 0.00, investigate that task specifically — it almost certainly has a threshold calibration problem or a class pruning issue. Fix task-level issues before running the full overnight evaluation.

---

### Constraints — Do Not Change These

- YOLOv8n backbone stays frozen and stays as the nano variant (CPU inference speed requirement)
- TinyBERT backbone stays frozen
- All inference must run on CPU (`torch.inference_mode()`, no `.cuda()` calls at inference)
- Seed fixed at 42 everywhere
- Output schema must remain: `{"bbox": [x,y,w,h], "class": str, "confidence": float}` or `{"result": "no suitable object found", "task": str}`
- The 14 task strings and their IDs must not change — they are fixed by the contest
- `data/task_raw_embeddings.pt` with 27 entries is correct — do not regenerate unless embeddings.py itself has changed
- The affordance matrix at `data/affordance_matrix.npy` (shape 14×80) is already correctly built — do not rebuild it

---

### Expected Outcome After All Fixes

- Phase 2 InfoNCE should push Val Top-1 from 0.697 toward 0.71–0.73
- Calibrated θ_t should prevent false no-match and false match cases that silently destroy AP per task
- Dropout in AGCA/SCRN + ROI feature augmentation should reduce the train/val gap significantly
- Subset mAP@0.5 (50 images/task) should be in the 0.55–0.65 range if everything is working correctly
- Final full val2014 mAP@0.5 target: ≥ 0.60; minimum acceptable: ≥ 0.55