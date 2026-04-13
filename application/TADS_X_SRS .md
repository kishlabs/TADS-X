# Software Requirements Specification (SRS)
## TADS-X: Task-Aware Dual-Stream Detection with Affordance Gating
### Team ChipSmiths | DVCon India 2026 Design Contest — Stage 2A

> **Revision Note:** This SRS covers Stage 2A (software, CPU inference, functional correctness) only.
> Hardware accelerator design (systolic array, VEGA integration) is deferred to Stage 3 and
> documented separately in Appendix A.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [System Architecture](#3-system-architecture)
4. [Functional Requirements](#4-functional-requirements)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [Data Requirements](#6-data-requirements)
7. [Module Specifications](#7-module-specifications)
8. [Interface Requirements](#8-interface-requirements)
9. [Training Requirements](#9-training-requirements)
10. [Evaluation Requirements](#10-evaluation-requirements)
11. [Project Structure](#11-project-structure)
12. [Implementation Roadmap](#12-implementation-roadmap)
13. [Appendix A — Hardware Addendum (Stage 3 Reference)](#13-appendix-a--hardware-addendum-stage-3-reference)

---

## 1. Introduction

### 1.1 Purpose

This SRS defines the complete software requirements for TADS-X — a task-aware object
detection application built for the DVCon India 2026 Design Contest (Stage 2A). It serves
as the single source of truth for what the system must do, how it must behave, and how
each module must be built.

### 1.2 Problem Statement

Standard object detectors answer: **"What objects are in this image?"**

TADS-X must answer: **"Which object in this image should I use for a given task?"**

**Example:**
- Image contains: wine glass, cup, bottle, person
- Task: `"serve wine"` → correct answer: **wine glass**
- Task: `"carry things in"` → correct answer: **bottle**
- Task: `"pour water into"` → correct answer: **cup**

Same image. Different correct answer depending on the task. This is task-aware object detection.

### 1.3 Why Task-Aware Detection is Non-Trivial

Standard object detectors treat all detected objects equally — they have no concept of
goal or context. Task-aware detection is harder for three reasons:

1. **Suitability depends on alternatives.** A cup is the best answer for "serve wine"
   only if no wine glass is present. The correct answer changes based on what else
   is in the scene.

2. **Suitability depends on affordance, not just appearance.** A baseball bat and a
   hammer look very different, but both could answer "hit something with". The system
   must learn which object *functions* correctly for a task, not just which object
   *looks* like a typical answer.

3. **COCO images are dense.** COCO val2014 images contain on average **7.7 object
   instances** per image across 80 classes. Selecting the single correct object from
   many candidates requires a ranking mechanism, not just detection.

These three properties make task-aware detection a fundamentally different — and harder
— problem than standard object detection.

Why this matters for the contest: The DVCon evaluation queries (SRS §2.2) are
semantically related but not identical to the training task strings (Paper Tasks).
A system that memorises training strings will fail on novel phrasing. TADS-X
generalises by resolving queries via cosine similarity in embedding space rather
than exact string matching — this is the core architectural decision that makes
the system robust beyond its training vocabulary.

### 1.4 Scope

The application must:
- Accept any COCO-format image and one of 14 predefined task queries
- Run the full detection and scoring pipeline on **CPU only** at inference time
- Output the single most task-appropriate detected object with its bounding box and confidence
- Achieve **mAP@0.5 > 0.60** on the COCO-Tasks val2014 benchmark

### 1.5 Definitions

| Term | Meaning |
|------|---------|
| **COCO-Tasks** | Dataset of 40K images annotated for 14 everyday tasks (Sawatzky et al., 2019) |
| **mAP@0.5** | Mean Average Precision at IoU threshold 0.5 — main evaluation metric |
| **IoU** | Intersection over Union — measures overlap between predicted and ground-truth bbox |
| **P4 feature map** | Intermediate feature map from YOLOv8n backbone (26×26×128) |
| **ROI-Align** | Region of Interest alignment — extracts fixed-size features for each proposal |
| **Affordance** | The functional relationship between an object class and a task |
| **TCFG** | Task-Conditioned Feature Gating — modulates object features based on task |
| **AGCA** | Affordance-Guided Cross-Attention — scores objects against the task |
| **SCRN** | Scene Context Re-scoring Network — refines scores using inter-object relationships |
| **QACT** | Quantization-Aware Contrastive Training — preserves accuracy under INT8 |
| **θ_t** | Per-task selection threshold — calibrated per task on val2014 |
| **Paper Tasks** | The 14 COCO-Tasks training task strings (Sawatzky et al. 2019) — distinct from SRS evaluation queries; defined in `task_definitions.py` |
| **SRS Tasks** | The 14 DVCon contest evaluation queries (§2.2); resolved to the nearest Paper Task via cosine similarity at inference |

---

## 2. Overall Description

### 2.1 What the Application Does

```
INPUT:  Image (COCO val2014) + Task Query (e.g., "serve wine")
           │
           ▼
    ┌─────────────────────────────────┐
    │         TADS-X Pipeline         │
    │                                 │
    │  1. Detect all objects (YOLO)   │
    │  2. Prune irrelevant classes    │
    │  3. Extract object features     │
    │  4. Encode the task (TinyBERT)  │
    │  5. Gate features by task       │
    │  6. Score objects vs task       │
    │  7. Re-score with scene context │
    │  8. Select best object          │
    └─────────────────────────────────┘
           │
           ▼
OUTPUT: { bbox: (x,y,w,h), class: "wine_glass", confidence: 0.94 }
        OR { result: "no suitable object found", task: "serve wine" }
```

### 2.2 The 14 Tasks

| Task ID | Query String |
|---------|-------------|
| 1 | serve wine |
| 2 | pour water into |
| 3 | cut something with |
| 4 | hit something with |
| 5 | dig a hole with |
| 6 | scoop something with |
| 7 | pound something with |
| 8 | cool something in |
| 9 | sit on |
| 10 | lie on |
| 11 | carry things in |
| 12 | read |
| 13 | check the time with |
| 14 | look through |

### 2.3 Conceptual Scoring Model

The final suitability score for each candidate object is a function of three factors:

```
score_i = f( AGCA_score_i,  SCRN_context_i,  detection_conf_i )
```

- **AGCA score** — how well this object's features align with the task embedding,
  weighted by affordance prior knowledge
- **SCRN context** — how this object's score changes given what other objects are
  present in the scene (inter-object relationships via self-attention)
- **Detection confidence** — how certain YOLOv8n is that this object exists

The SCRN module is critical: it models relationships between the top-5 candidates so
that, for example, a wine glass scores higher when a bottle is also present (reinforcing
"serve wine" context), and a cup scores lower when a wine glass is available (because
a better option exists). In practice, the combination is learned end-to-end by the
MLP scoring head.

### 2.4 User Interaction

For Stage 2A, the application is used via:
- A **command-line interface (CLI)** for evaluation and single-image prediction
- A **simple Python API** (`predict(image_path, task_query)`) for demo purposes

---

## 3. System Architecture

### 3.1 Full Pipeline Flow (CPU Software — Stage 2A)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TADS-X INFERENCE PIPELINE                      │
│                    (All operations: PyTorch CPU)                     │
│                                                                     │
│  ┌─────────┐   ┌───────────────┐   ┌──────────────────────────┐    │
│  │  Image  │──▶│   YOLOv8n     │──▶│  P4 Feature Map          │    │
│  └─────────┘   │  (CPU, ~1s)   │   │  (26×26×128, float32)    │    │
│                └──────┬────────┘   └─────────────┬────────────┘    │
│                       │                          │                  │
│  ┌─────────┐   ┌──────▼────────┐   ┌─────────────▼────────────┐    │
│  │  Task   │   │ Class Pruning │   │  ROI-Align + Projection   │    │
│  │  Query  │   │ using A[t,:]  │   │  v_i ∈ R^256 (L2-normed)  │    │
│  └────┬────┘   └───────────────┘   └─────────────┬────────────┘    │
│       │                                          │                  │
│       ▼                                          │                  │
│  ┌──────────┐                        ┌───────────▼────────────┐    │
│  │ TinyBERT │                        │      TCFG Module        │    │
│  │  Cache   │──── t (256-D) ────────▶│  v'_i = v_i ⊙ g(t)     │    │
│  └──────────┘                        │  g(t) = sigmoid(W_g·t)  │    │
│  (dict lookup                        └───────────┬────────────┘    │
│   O(1) at inference)                             │                  │
│                                      ┌───────────▼────────────┐    │
│                                      │      AGCA Module        │    │
│                                      │  Affordance-guided      │    │
│                                      │  cross-attention        │    │
│                                      │  → score per object     │    │
│                                      └───────────┬────────────┘    │
│                                                  │                  │
│                                      ┌───────────▼────────────┐    │
│                                      │      SCRN Module        │    │
│                                      │  Self-attention over    │    │
│                                      │  top-5: each object     │    │
│                                      │  attends to all others  │    │
│                                      │  → refined scores       │    │
│                                      └───────────┬────────────┘    │
│                                                  │                  │
│                                      ┌───────────▼────────────┐    │
│                                      │    Final Selection      │    │
│                                      │  argmax vs θ_t          │    │
│                                      └───────────┬────────────┘    │
│                                                  ▼                  │
│                          { bbox, class, confidence } or no-match    │
└─────────────────────────────────────────────────────────────────────┘
```

Why this architecture: Each module addresses a specific failure mode of naive detection:
  - YOLOv8n alone cannot rank candidates by task relevance (no task input)
  - Class pruning via affordance matrix eliminates clearly irrelevant classes early,
    reducing SCRN's search space by ~50% and preventing noise from dominating scores
  - TCFG gates visual features per-task so AGCA attention operates on a focused
    representation, not raw visual features where task-irrelevant dimensions dominate
  - AGCA combines learned attention with a statistical prior (affordance matrix) so
    rare-but-correct objects (e.g. a spade for "dig a hole") are not suppressed
    purely because they appear infrequently in the training distribution
  - SCRN models inter-object context so the system can reason: "a cup scores lower
    when a wine glass is also present" — impossible without multi-candidate reasoning

### 3.2 Component Responsibilities

| Component | Responsibility | Runs On |
|-----------|---------------|---------|
| YOLOv8n | Detect all objects, extract P4 feature map | CPU (torch) |
| Affordance Matrix A | Prior: which classes suit which tasks | Precomputed (numpy) |
| Class Pruner | Remove task-irrelevant proposals | CPU (numpy) |
| ROI-Align | Extract 7×7×128 region per proposal | CPU (torchvision) |
| Projection Layer | Project 6272-D → 256-D embedding v_i | CPU (nn.Linear) |
| TinyBERT + Cache | Encode task text → 256-D embedding t | CPU (transformers) |
| TCFG | Modulate v_i using task embedding t | CPU (torch) |
| AGCA | Score each object against the task | CPU (torch) |
| SCRN | Refine scores using inter-object context | CPU (torch) |
| Final Selector | Pick best object or report no-match | CPU (Python) |

---

## 4. Functional Requirements

### FR-01: Object Detection
- The system SHALL use `ultralytics.YOLO('yolov8n.pt')` — the **nano** variant of YOLOv8,
  with default confidence threshold 0.25, running entirely on CPU
- The system SHALL extract the P4 intermediate feature map (26×26×128)
- The system SHALL produce bounding boxes, `coco_class_id` (0–79 matrix index), and
  confidence scores for all detected objects
- The system SHALL support all 80 COCO object classes

### FR-02: Class Pruning
- The system SHALL load the affordance prior matrix A (14×80) from disk
- The system SHALL suppress classes where `A[task_id][class] < θ_t` before NMS
- The system SHALL reduce average proposals from ~18 to ≤ 8 per image
- The system SHALL retain the ground-truth object in ≥ 99% of val2014 images after pruning

### FR-03: Visual Feature Extraction
- The system SHALL apply ROI-Align on the P4 feature map to extract 7×7×128 features per proposal
- The P4 feature map is extracted by registering a forward hook on YOLOv8n's backbone
  at the layer that produces the 26×26×128 feature map. In Ultralytics YOLOv8, this
  corresponds to `model.model.model[9]` (the C2f block at stride 16). Example:
  ```python
  features = {}
  def hook(module, input, output):
      features['p4'] = output
  model.model.model[9].register_forward_hook(hook)
  ```
- The system SHALL project the flattened 6272-D ROI features to 256-D L2-normalised
  vectors (v_i) via a trained linear layer
- The system SHALL process up to N=8 proposals per image


 ### FR-04: Task Embedding
 - The system SHALL precompute and cache task embeddings at startup using TinyBERT-4.
 - TinyBERT produces a 312-D CLS token; a **trainable linear layer (312→256)** projects
   it to the working dimension — this layer is trained jointly with TCFG/AGCA/SCRN.
 - The system SHALL cache embeddings for **all task query strings used by the pipeline**:
   **14 Paper Task strings + 14 SRS Task strings**.
 - Raw TinyBERT embeddings (312-D) SHALL be stored in `data/task_raw_embeddings.pt`,
   and SHALL be reprojected at load time using the current projection weights.
 - For novel task queries outside the predefined task strings, TinyBERT SHALL compute
   embeddings on-the-fly (slower fallback path); **this fallback is disabled during contest
   evaluation** to ensure deterministic, reproducible behavior.

   Why cosine similarity resolution (not a hardcoded lookup table): A hardcoded
SRS→Paper mapping would break if the contest changes evaluation queries slightly
or adds novel tasks. Cosine similarity in TinyBERT embedding space generalises
to any semantically related query — "pour liquid into" would correctly resolve
to "water plant" or "pour sugar" even though it was never seen at training time.
This is the key generalisation mechanism of TADS-X.

### FR-05: Task-Conditioned Feature Gating (TCFG)
- The system SHALL compute `g(t) = sigmoid(W_g · t)` where W_g ∈ R^(256×256)
- The system SHALL compute `v'_i = v_i ⊙ g(t)` (element-wise, broadcast across N proposals)
- The gate vector g(t) SHALL be computed once per image (shared across all proposals)
- After training, mean cosine distance between g(t) vectors across 14 task pairs SHALL
  exceed 0.3 (validates task-specific modulation)

### FR-06: Affordance-Guided Cross-Attention (AGCA)
- The system SHALL compute `q = W_q · t` and `k_i = W_k · v'_i`
- The system SHALL look up `A[task_id][coco_class_id_i]` as the affordance prior for
  each proposal, where `coco_class_id_i` ∈ 0–79 is the matrix index of the detected class
- The system SHALL compute: `α_i = A[task_id][coco_class_id_i] · softmax_j(q · k_j / √256)`
- The system SHALL compute: `val_i = W_v · v'_i`, `agca_i = α_i · val_i`
- The system SHALL score each proposal: `agca_score_i = MLP(agca_i)` (256→64→1, ReLU)

### FR-07: Scene Context Re-scoring (SCRN)
- The system SHALL select top-5 candidates by AGCA score **at inference time only**
- **During training:** SCRN loss is computed over ALL N proposals (not just top-5),
  ensuring the ground-truth object is never accidentally excluded from the loss signal
  if it falls outside the top-5 AGCA scores. Top-5 selection is an inference-time
  optimisation only.
- The system SHALL form: h_i = [v'_i (256) || agca_score_i (1)] → shape (K, 257)
- The system SHALL apply scaled dot-product self-attention over all K candidates —
  each candidate attends to every other, so that co-occurring objects influence
  each other's final scores
- The system SHALL output: `score_i = sigmoid(MLP_2([context_i || h_i]))`
- Top-5 at inference SHALL retain ground-truth object in ≥ 99.5% of val2014 images

### FR-08: Final Selection and Threshold Calibration
- The system SHALL output `argmax_i(score_i)` if `max(score_i) ≥ θ_t[task_id]`
- **θ_t calibration:** Per-task thresholds are calibrated on val2014 to maximise F1
  while maintaining recall ≥ 0.99 (ground-truth object is almost never suppressed)
- If no candidate exceeds θ_t, output `"no suitable object found"`
- Output bbox SHALL be in original image pixel coordinates (x, y, w, h)
- **Edge case handling:**
  - If YOLOv8n detects **N=0 objects** → immediately return no-match without entering scoring
  - If class pruning **removes all proposals** → return no-match with reason `"all classes pruned"`
  - If fewer than 5 proposals survive pruning → SCRN runs on K < 5 candidates (no padding needed;
    attention operates on whatever K candidates are available, K ≥ 1)
  - If exactly **1 proposal** survives → skip SCRN (self-attention over 1 item is trivial);
    use AGCA score directly for final selection

### FR-09: CPU-Only Inference
- All inference SHALL run on CPU (torch, torchvision)
- No `.cuda()` or `.to('cuda')` calls during inference
- All forward passes SHALL use `torch.inference_mode()`
- GPU MAY be used during training only

### FR-10: Batch Evaluation
- The system SHALL provide an evaluation script for the full COCO-Tasks val2014 split
- The system SHALL compute mAP@0.5 per task and report overall mean
- The evaluation script SHALL support `--subset N` for quick iteration on N images per task
- **Note:** Evaluation loops over single images (no batching). This is intentional —
  CPU inference is the target and batching adds complexity without Stage 2A benefit

---

## 5. Non-Functional Requirements

### NFR-01: Accuracy
- **Stage 2A target:** mAP@0.5 ≥ 0.60 on COCO-Tasks val2014 in **FP32 only**
- **Stage 2A scope:** All modules run in FP32 (float32). INT8 quantisation is **not
  part of Stage 2A** — it is deferred to Stage 3 when FPGA deployment begins
- **INT8 target (Stage 3):** mAP@0.5 within 3% of FP32 baseline (via QACT using
  PyTorch's `torch.quantization` or `brevitas` library — to be decided in Stage 3)
- **Minimum acceptable for Stage 2A submission:** mAP@0.5 ≥ 0.55

### NFR-02: Performance (tracked but not graded in Stage 2A)
- CPU inference per image: target < 2 seconds
- 500-image subset evaluation: should complete in < 20 minutes

### NFR-03: Correctness
- Must produce qualitatively correct results for all 14 tasks on representative images
- No-match cases must be handled correctly (e.g., "dig a hole" when no shovel-class object detected)

### NFR-04: Reproducibility
- All random seeds fixed at 42
- Same checkpoint + same input = identical output

### NFR-05: Code Quality
- Every module has a docstring with input/output tensor shapes
- README.md provides complete setup and run instructions

---

## 6. Data Requirements

### DR-01: COCO 2014 Dataset

| File | Purpose | Required For |
|------|---------|-------------|
| `val2014/` (41K images, 6GB) | Evaluation images | Evaluation |
| `train2014/` (83K images, 13GB) | Training images | Training |
| `instances_val2014.json` | COCO object annotations | Evaluation |
| `instances_train2014.json` | COCO object annotations | Training |

### DR-02: COCO-Tasks Annotations

| File | Purpose |
|------|---------|
| `task_1_train.json` ... `task_14_train.json` | Training ground truth per task |
| `task_1_test.json` ... `task_14_test.json` | Evaluation ground truth per task |

> **Note:** COCO-Tasks uses the COCO 2014 val split as its test set.

> **Key field:** Each annotation uses `COCO_category_id` (not `category_id`) for the
> COCO object class. The affordance builder must read `COCO_category_id` of the
> **correct answer object only** — not all objects in the image.

### DR-03: Affordance Matrix Construction

- **Shape:** (14, 80) — 14 tasks × 80 COCO classes
- **Source:** Only the ground-truth correct object per annotation (`COCO_category_id`)
- **Normalisation:** `A[t] = (count[t] + ε) / (sum(count[t]) + ε×80)`, ε=1e-6
  (epsilon prevents any class from having exactly zero probability)
- **Expected result:** Task 1 top class = wine glass, Task 12 top class = book
- **Saved as:** `data/affordance_matrix.npy`

Why epsilon smoothing (DR-03): Without epsilon, any COCO class that never appears
as a preferred object for a task gets A[task][class] = 0. This hard zero would
permanently suppress that class via affordance gating even if it is genuinely
relevant. Epsilon = 1e-6 ensures every class has a non-zero prior while keeping
the dominant classes' relative weights unchanged.

Why preferred-only annotations (category_id == 1): COCO-Tasks annotates objects
at three levels — preferred, acceptable, and not suitable. Training on all three
would teach the model to score acceptable objects nearly as high as preferred ones,
reducing precision. Using preferred-only gives clean positive supervision with
distractors providing implicit negative signal via the cross-entropy ranking loss.

### DR-04: Task Embeddings Cache
- 14 tasks with multiple query aliases (Paper + SRS wording)
- 28 raw embeddings (14 Paper Task strings + 14 SRS Task strings), each 312-D float32
- Computed once via frozen TinyBERT-4 [CLS] token; saved permanently as `data/task_raw_embeddings.pt`
- At pipeline startup, `load_projected_embeddings()` applies the current Linear(312→256)
  projection to produce 256-D working embeddings in memory — no disk re-write needed after training
- Initial projection weights: `data/projection_layer_init.pt` (saved by `embeddings.py`)
- Trained projection weights: `data/projection_layer_trained.pt` (saved by `train.py`)  

### DR-05: Per-Task Thresholds

- 14 float values, one per task (θ_t for task IDs 1–14)
- **Calibration algorithm:**
  ```
  For each task t in 1..14:
      For each threshold candidate θ in [0.1, 0.9] step 0.01:
          predictions = [predict(img, t) for img in val2014 if img has task_t annotation]
          recall    = count(correct predictions) / count(all GT objects)
          precision = count(correct predictions) / count(all predictions)
          F1        = 2 * precision * recall / (precision + recall)
      θ_t[t] = argmax_θ F1  subject to recall >= 0.99
  ```
- **Stored as:** `configs/per_task_thresholds.json`
- **Format:**
```json
{
  "1": 0.42,
  "2": 0.38,
  "3": 0.51,
  "...": "...",
  "14": 0.45
}
```
- Loaded at inference startup; passed into `predict()` as `per_task_thresholds: Dict[int, float]`

---

## 7. Module Specifications

### 7.1 `scripts/build_affordance.py`

**Purpose:** One-time script — builds the 14×80 affordance prior matrix.

**Critical:** Use only the `COCO_category_id` of the correct answer object per
annotation. Do not count all objects in the image.

**Key functions:**
```python
def build_affordance_matrix(tasks_dir: str) -> np.ndarray:
    # Returns A of shape (14, 80)

def get_relevant_classes(task_id: int, A: np.ndarray, threshold: float) -> List[int]:
    # Returns COCO matrix indices where A[task_id][idx] >= threshold
```

---

### 7.2 `embeddings.py`

**Purpose:** Generate and cache 256-D task embeddings.

**Model:** `huawei-noah/TinyBERT_General_4L_312D` (backbone frozen)

**Projection:** Trainable `nn.Linear(312, 256)` on top of [CLS] token — trained
jointly with TCFG/AGCA/SCRN. This is the only trainable part of the text branch.

**Key functions:**
```python
def compute_raw_embeddings(tokenizer, model, device="cpu") -> Dict[str, Tensor]:
    # Returns dict: query_string → raw 312-D CLS tensor (CPU)
    # Encodes all PAPER_TASKS + SRS_TASKS strings (28 unique queries)

def load_projected_embeddings(raw_path, proj_path, device="cpu") -> Tuple[Dict, TaskProjection]:
    # Loads raw cache + projection, returns projected 256-D cache + projection module
    # Cache keys are lowercase-stripped for robust lookup
    # Call with projection_layer_init.pt before training, projection_layer_trained.pt after

def get_embedding(task_query, cache, tokenizer=None, model=None,
                  projection=None, device="cpu", allow_novel=False) -> Tensor:
    # Exact match only (normalised lowercase key lookup)
    # allow_novel=False during contest evaluation (SRS FR-04)
    # Raises KeyError for unknown queries unless allow_novel=True
```

---

### 7.3 `models/tcfg.py`

**Purpose:** Task-Conditioned Feature Gating.

**Why it works:** A visual embedding v_i captures everything about an object — shape,
colour, texture, context. Not all dimensions are equally relevant to every task. For
"serve wine", the elongated stem and transparent bowl dimensions matter; for "sit on",
they don't. TCFG learns a per-task gate vector g(t) that suppresses task-irrelevant
dimensions and amplifies task-relevant ones before any scoring occurs. This makes
downstream attention (AGCA) operate on a cleaner, task-focused representation.

```
g(t)  = sigmoid(W_g · t)     W_g ∈ R^(256×256)
v'_i  = v_i ⊙ g(t)           broadcast across N proposals
```

```python
class TCFG(nn.Module):
    """
    Inputs:
        v_i : Tensor (N, 256) — visual embeddings for N proposals
        t   : Tensor (256,)   — task embedding
    Output:
        v_prime : Tensor (N, 256) — task-gated visual embeddings
    Note: g(t) computed once, shared across all N proposals.
    """
```

---

### 7.4 `models/agca.py`

**Purpose:** Affordance-Guided Cross-Attention + MLP scoring.

**Why it works:** Two complementary signals are combined here. The attention mechanism
(q·k_i) measures learned semantic similarity between the task embedding and each
object's gated features — capturing fine-grained alignment. The affordance prior
A[task_id][coco_class_id_i] injects structural prior knowledge from the training
distribution (e.g., wine glasses almost always answer "serve wine"). Multiplying these
two signals means an object must score well on both dimensions: it must be semantically
relevant AND belong to a class that the training data associates with the task. Neither
signal alone is sufficient.

```
q       = W_q · t                                    (256,)
k_i     = W_k · v'_i                                 (N, 256)
raw_i   = softmax_j(q · k_j / √256)                 (N,)     attention weight
a_i     = A[task_id][coco_class_id_i]                (N,)     affordance prior
α_i     = a_i * raw_i                                (N,)     multiplicative gate
          ← affordance is applied AFTER softmax, not inside it.
          This preserves the normalised attention distribution while
          scaling each weight by task-class relevance prior.
val_i   = W_v · v'_i                                 (N, 256)
agca_i  = α_i · val_i                                (N, 256)
score_i = MLP(agca_i)    [256→64→1, ReLU]            (N,)
```
Why multiplicative gate after softmax (not inside): Applying the affordance prior
inside the softmax would distort the attention distribution non-linearly and make
the prior interact with all other candidates' scores. Applying it after softmax
as a multiplicative gate preserves the learned attention distribution while scaling
each candidate's weight by task-class relevance — the two signals remain
interpretable and separable.

```python
class AGCA(nn.Module):
    """
    Inputs:
        v_prime        : Tensor (N, 256)
        t              : Tensor (256,)
        task_id        : int             — 0–13
        coco_class_ids : List[int]       — matrix index (0–79) per proposal
        A              : Tensor (14, 80)
    Outputs:
        scores    : Tensor (N,)     — preliminary suitability scores
        agca_vecs : Tensor (N, 256) — gated context vectors (input to SCRN)
    """
```

---

### 7.5 `models/scrn.py`

**Purpose:** Scene Context Re-scoring via self-attention over top-5 candidates.

**Why it works:** AGCA scores each object independently — it has no awareness of what
other objects are present. SCRN fixes this by letting all top-5 candidates attend to
each other before final scoring. If both a wine glass and a cup are present for "serve
wine", the wine glass's context vector will encode the presence of the cup (a weaker
alternative), reinforcing its own score. The cup's context vector encodes the presence
of the wine glass (a better alternative), suppressing its own score. This inter-object
reasoning is the key mechanism that makes TADS-X prefer the most task-appropriate
object when multiple valid candidates exist.

```
h_i       = [v'_i (256) || agca_score_i (1)]      (K, 257)
Q         = h · W_Q                                (K, 64)
K_mat     = h · W_K                                (K, 64)
attn      = softmax(Q · K_mat^T / √64)             (K, K)
context_i = Σ_j attn_ij · h_j                      (K, 257)
score_i = MLP_2([context_i ‖ h_i]) → (K,)   raw logits

Note: sigmoid is NOT applied inside SCRN. It is applied externally:
  - At inference (predict()): sigmoid(score_i) for threshold comparison and confidence output
  - At training (train.py): F.cross_entropy(scores, gt_index) which applies log-softmax internally
  This keeps logits consistent throughout the scoring pipeline (FR-07).
```

```python
class SCRN(nn.Module):
    """
    Inputs:
        v_prime     : Tensor (K, 256) — task-gated features, K ≤ 5
        agca_scores : Tensor (K,)     — preliminary scores from AGCA
    Output:
        refined_scores : Tensor (K,) — context-aware final scores [0, 1]
    """
```

---

### 7.6 `pipeline.py`

**Purpose:** Connect all modules into a single callable inference function.

```python
def predict(image_path, task_query, yolo_model, scoring_model,
            affordance_matrix, task_embeddings, per_task_thresholds) -> dict:
    """
    Full TADS-X inference. All ops on CPU.
    Returns:
        Match:    { 'bbox':(x,y,w,h), 'class':str, 'confidence':float }
        No-match: { 'result':'no suitable object found', 'task':str }

    Steps:
        1. YOLOv8n → proposals + P4 feature map
        2. Map detected YOLO class indices to coco_class_id (0–79 matrix index)
        3. Prune using A[task_id, :] and θ_t from per_task_thresholds
        4. ROI-Align → flatten → project → L2-normalise → v_i (N, 256)
        5. Look up task embedding t (O(1))
        6. TCFG: v'_i = v_i ⊙ sigmoid(W_g · t)
        7. AGCA: agca_scores (N,) + agca_vecs (N, 256)
        8. Top-5 by agca_score
        9. SCRN: refined_scores (K,) via self-attention
        10. argmax vs θ_t[task_id] → output or no-match
    """
```

---

### 7.7 `train.py`

**Purpose:** Train ROI projection + TinyBERT projection + TCFG + AGCA + SCRN jointly.
YOLOv8n backbone and TinyBERT backbone are frozen.

**Loss:**
```
Phase 1:  L = CrossEntropy(scores, correct_object_index)
Phase 2:  L = CrossEntropy + λ · InfoNCE    (λ=0.1, τ=0.1)
```

**Config:**
```python
TRAIN_CONFIG = {
    'epochs': 50, 'batch_size': 32, 'learning_rate': 3e-4,
    'optimizer': 'AdamW', 'weight_decay': 1e-4,
    'lambda_contrastive': 0.1, 'tau': 0.1,
    'device': 'cuda', 'seed': 42,
}
```

---

### 7.8 `evaluate.py`

**Purpose:** Compute mAP@0.5 on COCO-Tasks val2014.

**Console output:**
```
Task  1  (serve wine          ):  AP = 0.xx
...
Task 14  (look through        ):  AP = 0.xx
─────────────────────────────────────────────
Overall mAP@0.5:                      0.xx
```

**JSON output:** The script SHALL write `results/map_per_task.json` after every run:
```json
{
  "task_1_serve_wine":           0.xx,
  "task_2_pour_water_into":      0.xx,
  "...":                         "...",
  "task_14_look_through":        0.xx,
  "overall_mAP":                 0.xx,
  "evaluated_on":                "val2014",
  "subset_size":                 null
}
```
This JSON is used as the primary evidence of results in the Stage 2A report.

---

## 8. Interface Requirements

### 8.1 Command-Line Interface

```bash
# Single image prediction
python app.py --image path/to/image.jpg --task "serve wine"

# Full evaluation
python evaluate.py --coco-dir E:/DVCon/COCO --tasks-dir E:/DVCon/COCO/coco-tasks

# Subset evaluation (faster iteration)
python evaluate.py --coco-dir E:/DVCon/COCO --tasks-dir E:/DVCon/COCO/coco-tasks --subset 500

# Train
python train.py --config configs/train_config.yaml
```

### 8.2 Python API

```python
from pipeline import TADSX
model = TADSX(checkpoint='checkpoints/tads_x_best.pt')
result = model.predict('image.jpg', 'serve wine')
# {'bbox': (234, 156, 89, 112), 'class': 'wine glass', 'confidence': 0.94}
```

### 8.3 Output Schema

```json
{ "bbox": [x, y, w, h], "class": "wine glass", "confidence": 0.94 }
{ "result": "no suitable object found", "task": "serve wine" }
```

---

## 9. Training Requirements

### 9.1 What Gets Trained vs Frozen

| Module | Status | Reason |
|--------|--------|--------|
| YOLOv8n backbone | **Frozen** | Pre-trained COCO weights sufficient |
| TinyBERT backbone | **Frozen** | Used only as feature extractor |
| TinyBERT projection (312→256) | **Trained** | Adapts language space to working dimension |
| ROI Projection (6272→256) | **Trained** | Learns task-relevant visual representation |
| TCFG (W_g) | **Trained** | Learns task-specific feature modulation |
| AGCA (W_q, W_k, W_v, MLP) | **Trained** | Learns task-object affinity scoring |
| SCRN (W_Q, W_K, MLP_2) | **Trained** | Learns inter-object context reasoning |

### 9.2 Training Procedure

```
Phase 1 — FP32 baseline (50 epochs):
  Loss = CrossEntropy only
  AdamW, lr=3e-4, weight_decay=1e-4

Phase 2 — QACT (20 additional epochs):
  Insert fake-quantisation nodes (QAT)
  Loss = CrossEntropy + 0.1 × InfoNCE
  Target: INT8 mAP within 3% of FP32
```
Why two phases: Phase 1 (CrossEntropy only) establishes basic ranking ability —
the model learns which object class to select. Phase 2 adds InfoNCE contrastive
loss which pulls the GT object's visual embedding toward the task embedding and
pushes distractors away. This explicitly trains the embedding space to be
task-discriminative, which is required for the θ_t threshold to be meaningful
and for TCFG gate diversity (FR-05) to exceed 0.3.

Why F.cross_entropy on logits (not BCE on probabilities): CrossEntropy treats
the K candidates as a mutually exclusive set — exactly one is correct. BCE treats
each candidate independently, which allows the model to assign high probability
to multiple candidates simultaneously. For ranking tasks, CrossEntropy provides
stronger gradient signal toward the correct candidate.
---

## 10. Evaluation Requirements

### 10.1 Primary Metric
- **mAP@0.5** on COCO-Tasks val2014 (all 14 tasks)

### 10.2 Baseline Comparison Strategy

Before evaluating TADS-X, we establish two simple baselines to confirm that our
modules provide genuine improvement over naive approaches:

| Baseline | Description | Expected mAP@0.5 |
|----------|-------------|-----------------|
| **YOLO-only** | Run YOLOv8n, pick highest-confidence detection regardless of task | ~0.20–0.30 (random task alignment) |
| **Affordance-only** | Pick the highest-confidence object whose class has highest A[task][class] — no learned scoring | ~0.35–0.45 |
| **YOLO + cosine (TinyBERT)** | Pick object whose class name has highest cosine similarity to task embedding | ~0.40–0.50 |
| **TADS-X (ours)** | Full pipeline: TCFG + AGCA + SCRN | Target ≥ 0.60 |

These baselines are cheap to implement (no training needed for the first three) and
provide a clear story: each component adds measurable value over the previous baseline.

### 10.3 Ablation Experiments (for report)

| Experiment | What It Measures | Expected Outcome |
|-----------|-----------------|-----------------|
| Without TCFG | TCFG contribution | ~3–5% mAP drop; task-agnostic features reduce ranking precision |
| Without SCRN | Scene context contribution | ~2–3% mAP drop; mainly affects tasks with multiple valid object classes |
| FP32 vs INT8 PTQ | Quantisation degradation | ~10–12% mAP drop without QACT |
| FP32 vs INT8 QACT | QACT effectiveness (Stage 3) | < 3% mAP drop with QACT (target) |
| With / without class pruning | Pruning contribution | Minimal mAP change; 50%+ reduction in proposals |

### 10.4 Qualitative Verification (for demo video)

For each of the 14 tasks, one screenshot showing:
- Input image with bounding box overlay
- Task query, predicted class, confidence score
- Match or no-match result

### 10.5 Failure Case Analysis

Known failure modes and how the system handles them:

| Failure Mode | Example | Expected Behaviour | Mitigation |
|---|---|---|---|
| **No relevant class detected** | "dig a hole" — image has no shovel, spade, or fork | Return no-match correctly | Class pruning + θ_t threshold |
| **Ambiguous task** | "cut something with" — both knife and scissors present | Return whichever scores higher via AGCA+SCRN | SCRN inter-object context |
| **Visually similar objects** | Wine glass vs champagne flute — both class 46 in COCO | Both map to same COCO class; system returns one with higher detection confidence | Inherent COCO annotation limitation |
| **Multiple equally valid objects** | Two wine glasses in one image | Return the one with the highest AGCA score (likely the more prominent one) | Acceptable; ground truth is also one object |
| **GT object missed by YOLO** | Wine glass is very small or occluded | YOLO miss → no-match or wrong answer | Improve YOLOv8n confidence threshold or fine-tune |
| **All candidates pruned** | Task 5 "dig hole" — image has only food items | Return no-match with reason "all classes pruned" | Handled in FR-08 edge cases |
| **Task query not in 14** | `--allow-novel-tasks` disabled | Novel task queries outside the 14 SRS strings are not supported at inference time. The pipeline will raise a KeyError if the query embedding is missing from the cache or if cosine similarity to all paper tasks is below 0.3. Run embeddings.py to regenerate the cache if new queries are needed. | FR-04 fallback flag |

These failure modes are documented so the demo video can explicitly demonstrate
correct no-match behaviour on cases 1 and 6.

### 10.6 Validation Plan

Before running the full val2014 evaluation (which processes ~41K images and can take
hours on CPU), we follow a two-stage validation strategy:

**Stage A — Subset validation (fast iteration):**
- Run on 50 images per task = 700 images total
- Command: `python evaluate.py --subset 50`
- Purpose: sanity-check that the pipeline produces correct outputs and that θ_t
  thresholds are reasonably calibrated
- Expected runtime: ~15–30 minutes on CPU

**Stage B — Full evaluation (overnight):**
- Run on all annotated val2014 images for each task
- Command: `python evaluate.py`
- Purpose: produce final mAP@0.5 numbers for the Stage 2A report
- Output: `results/map_per_task.json`

This two-stage approach ensures we catch implementation errors early without waiting
for a full overnight run.

---

## 11. Project Structure

```
E:\DVCon\application\
│
├── data\
│   ├── affordance_matrix.npy         ✓ Built
│   ├── affordance_matrix_raw.npy     ✓ Built
│   ├── affordance_matrix_info.json   ✓ Built
│   ├── task_raw_embeddings.pt        ← Build via embeddings.py (one-time)
│   ├── projection_layer_init.pt      ← Saved by embeddings.py (untrained)
│   └── projection_layer_trained.pt   ← Saved by train.py (after training)
├── models\
│   ├── __init__.py
│   ├── tcfg.py
│   ├── agca.py
│   └── scrn.py
│
├── checkpoints\
│   ├── tads_x_fp32_best.pt
│   └── tads_x_int8_qact.pt
│
├── configs\
│   ├── train_config.yaml
│   └── per_task_thresholds.json    ← θ_t for all 14 tasks (calibrated post-training)
│
├── results\
│   └── map_per_task.json           ← Written by evaluate.py after each run
│
├── task_definitions.py     ← Single source of truth: PAPER_TASKS, SRS_TASKS, COCO constants
├── build_affordance.py
├── embeddings.py
├── pipeline.py
...
├── embeddings.py
├── pipeline.py
├── train.py
├── evaluate.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 12. Implementation Roadmap

### Phase 1: Foundation — ✓ COMPLETE
- [x] Download COCO 2014 val + annotations
- [x] Download COCO-Tasks annotations
- [x] Set up virtual environment + install libraries
- [x] Create task_definitions.py (PAPER_TASKS, SRS_TASKS, COCO constants)
- [x] Build affordance matrix with correct preferred-only logic
- [x] Build task embedding scripts (embeddings.py)
- [ ] Download TinyBERT locally and run embeddings.py to generate task_raw_embeddings.pt
- [ ] Verify YOLOv8n on a sample val image

### Phase 2: Core Modules
- [ ] `models/tcfg.py` + unit test
- [ ] `models/agca.py` + unit test
- [ ] `models/scrn.py` + unit test
- [ ] ROI-Align + projection in `pipeline.py`

### Phase 3: Full Pipeline
- [ ] Connect all modules in `pipeline.py`
- [ ] Test `predict()` on 5 sample images across tasks
- [ ] Verify output format

### Phase 4: Training
- [ ] `train.py` Phase 1 (FP32, CrossEntropy)
- [ ] Train on COCO-Tasks train split (GPU)
- [ ] Validate mAP@0.5 > 0.55

### Phase 5: Evaluation & Submission
- [ ] Phase 2 QACT training
- [ ] Full `evaluate.py` → mAP per task
- [ ] Qualitative tests on all 14 tasks
- [ ] Demo video
- [ ] README + code cleanup

---

## 13. Appendix A — Hardware Addendum (Stage 3 Reference)

> This section is **not part of Stage 2A**. It is retained for reference only.
> Full hardware specification will be developed for Stage 3.

The Stage 1 proposal (TADS-X v5) describes an FPGA-VEGA co-design where:
- YOLOv8n backbone runs on a 16-lane INT8 systolic array on Kintex-7 FPGA fabric
- TCFG/AGCA matrix operations run in GEMM mode on the same time-multiplexed array
- VEGA RISC-V processor handles non-linear activations (softmax, sigmoid) via FPU
- YOLO weights are DMA-streamed from DDR3 (exceed on-chip BRAM capacity)
- Target: ~47 ms end-to-end on Genesys-2 / Kintex-7, < 4W power

All hardware details (BRAM map, DMA controller, AXI4 interface, RTL, Vivado synthesis)
are deferred to Stage 3.

### A.1 Fixed Tensor Shapes for Hardware Mapping

All hardware blocks must support the following fixed tensor shapes (Stage 3 target):

| Signal              | Shape          | Dtype  | Notes                        |
|---------------------|----------------|--------|------------------------------|
| YOLO P4 feature map | (1, 128, 26, 26) | INT8 | imgsz=416, stride-16 output  |
| ROI-Align output    | (N, 128, 7, 7) | INT8   | N ≤ 8 proposals per image    |
| ROI projection      | (N, 256)       | INT8   | Linear(6272→256)             |
| Task embedding t    | (256,)         | INT8   | from TaskProjection output   |
| TCFG gate g(t)      | (256,)         | INT8   | sigmoid(W_g · t)             |
| AGCA output         | (N, 256)       | INT8   | gated context vectors        |
| SCRN logits         | (K,)           | INT32  | K ≤ 5; sigmoid at CPU output |

### A.2 Quantization Calibration Procedure

1. Train FP32 model to convergence (Phase 1 + Phase 2, SRS §9.2)
2. Collect calibration data: 500 images × 14 tasks = 7000 forward passes
3. Per-layer activation ranges recorded using min/max observers
4. Apply symmetric INT8 PTQ (post-training quantization) per linear layer
5. Measure mAP@0.5 degradation: target < 3% with QACT, < 10% without
6. Export to ONNX with fixed shapes above for RTL simulation

### A.3 Acceptance Criteria Per Block (Stage 3)

| Block          | Acceptance Criterion                                      |
|----------------|-----------------------------------------------------------|
| YOLOv8n        | INT8 mAP within 2% of FP32 on COCO val2014               |
| ROI projection | Output L2 norm within 5% of FP32 reference               |
| TCFG           | Gate vector cosine distance > 0.3 across 14 tasks (FR-05)|
| AGCA           | Top-1 class ranking matches FP32 in ≥ 95% of test cases  |
| SCRN           | Final argmax matches FP32 in ≥ 95% of test cases         |
| End-to-end     | mAP@0.5 ≥ 0.57 (< 3% drop from FP32 ≥ 0.60 target)     |

---

*TADS-X SRS v2.3 — Team ChipSmiths — DVCon India 2026*

*v2.0: ChatGPT + DeepSeek review incorporated*
*v2.1: Notation unified, DR-05 added, evaluate JSON, YOLOv8n pinned, fallback scoped, validation plan*
*v2.2: Why-it-works explanations (TCFG/AGCA/SCRN), baseline comparison, failure analysis,*
*ablation expected outcomes, AGCA multiplicative clarification, SCRN training scope,*
*θ_t calibration pseudocode, INT8 deferred to Stage 3, FR-08 edge cases, P4 hook snippet,*
*--allow-novel-tasks flag, evaluation single-image note*
*v2.3: task_definitions.py introduced as single source of truth; PAPER_TASKS vs SRS_TASKS*
*architecture clarified; DR-04 updated to reflect raw-cache + projection lifecycle;*
*project structure updated; Phase 1 roadmap marked complete*