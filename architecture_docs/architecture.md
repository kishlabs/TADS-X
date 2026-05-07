# TADS-X Architecture Documentation

## Overview
**TADS-X** (Task-Aware Object Detection System - X) is designed to find the most suitable object in an image for a specific user-defined task (e.g., "serve wine", "sit comfortably"). Instead of just detecting what objects are present, it evaluates which detected object best affords the requested task action.

The architecture comprises two frozen backbones (for vision and language feature extraction) and a trainable scoring model that dynamically evaluates object proposals.

---

## 1. Core Architecture

### Backbones (Frozen)
*   **Vision (YOLOv8n)**: Proposes bounding boxes and extracts a deep feature map. Specifically, a forward hook is registered on the P4 layer (C2f block, stride 16) to extract a `26×26×128` feature map.
*   **Language (TinyBERT)**: Encodes text task queries (e.g., "pour water into") into raw 312-D embeddings.

### Trainable Scoring Model
The scoring model (`ScoringModel` in `pipeline.py`) takes the extracted features and evaluates each proposal:

1.  **TaskProjection (`embeddings.py`)**: Projects the raw 312-D TinyBERT task embedding into a 256-D working dimension (`t`).
2.  **ROI Extraction & Projection**: Uses ROI-Align on the YOLO P4 feature map for each bounding box to extract a `128×7×7` feature. This is flattened (6272-D) and projected linearly to a 256-D visual embedding (`v_i`).
3.  **TCFG - Task-Conditioned Feature Gating (`models/tcfg.py`)**: 
    *   Computes a task-specific gate: `g(t) = sigmoid(Dropout(W_g · t))`
    *   Modulates the visual embeddings element-wise: `v'_i = v_i ⊙ g(t)`
    *   This forces downstream layers to focus only on task-relevant visual features.
4.  **AGCA - Affordance-Guided Cross-Attention (`models/agca.py`)**:
    *   Combines learned semantic attention with a hard statistical affordance prior.
    *   Looks up the training distribution prior for the task-class pair: `a_i = A[task_id][coco_class_id]`.
    *   Multiplies the attention weights by this affordance gate to yield context vectors and raw logits.
5.  **SCRN - Scene Context Re-scoring (`models/scrn.py`)**:
    *   Takes the top-K proposals from AGCA and re-evaluates them using self-attention.
    *   This allows the model to adjust scores based on other objects in the scene (e.g., selecting a cup over a bowl if a pitcher is also present).

---

## 2. File Structure & Responsibilities

### Root Directory
*   `README.md`: Top-level documentation.
*   `TADS_X_SRS.md`: System Requirements Specification detailing all functional requirements and evaluation protocols.
*   `yolov8n.pt`: Pre-trained YOLOv8n weights.

### `/application/` Directory
*   **`pipeline.py`**: The central inference orchestrator. Contains the `predict()` function which ties YOLO, ROI extraction, `resolve_task_id`, and the scoring model together to output the final matched bounding box.
*   **`train.py`**: Training loop supporting a two-phase training strategy:
    *   Phase 1: CrossEntropy loss.
    *   Phase 2: CrossEntropy + InfoNCE contrastive loss.
    *   Also handles per-task threshold calibration (`θ_t`).
*   **`evaluate.py`**: Evaluates the model against the COCO-Tasks validation set using a 101-point F1/Recall/Precision metric.
*   **`build_affordance.py`**: Pre-computes the affordance matrix `A` (stored as `data/affordance_matrix.npy`) from the training data distribution.
*   **`embeddings.py`**: Uses TinyBERT to generate and cache task embeddings.
*   **`task_definitions.py`**: Maps paper task IDs to strings, and defines COCO-to-matrix ID mappings.
*   **`draw_bbox.py`**: Utility script to visualize inference results on images.

### `/application/models/` Directory
*   `tcfg.py`: Implementation of Task-Conditioned Feature Gating.
*   `agca.py`: Implementation of Affordance-Guided Cross-Attention.
*   `scrn.py`: Implementation of Scene Context Re-scoring Network.

### `/application/configs/` Directory
*   `train_config.yaml`: Contains hyperparameter settings for training (e.g., learning rate, weight decay, epochs, batch size).

### Generated Data Directories
*   `/application/data/`: Stores caches like `roi_cache`, `affordance_matrix.npy`, and trained projection layers.
*   `/application/checkpoints/`: Stores `.pt` model state dicts from training.

---

## 3. Inference Flow

1.  **Input**: Image + Text Query (e.g., "serve wine").
2.  **Detection**: YOLOv8n proposes bounding boxes.
3.  **Task Resolution**: Maps the text query to the nearest 256-D task embedding via cosine similarity.
4.  **Pruning**: Drops bounding boxes whose COCO class affordance prior is below `PRUNE_THRESH` (e.g., 0.01).
5.  **Scoring**: Remaining proposals pass through ROI-Align -> TCFG -> AGCA -> SCRN.
6.  **Thresholding**: The top SCRN score is converted to a probability and checked against the task-calibrated threshold (`θ_t`).
7.  **Output**: Returns the bounding box if the threshold is met, else returns "no-match".
