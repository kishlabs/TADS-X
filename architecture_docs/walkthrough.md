# TADS-X Comprehensive Walkthrough

This document serves as a complete, step-by-step guide to setting up, training, running inference, and evaluating the **TADS-X** (Task-Aware Object Detection System) model.

---

## 1. Prerequisites and Data Setup

Before training or running the model, ensure that the COCO-Tasks dataset is available. You will need:
*   **Images**: The MS COCO `train2014` and `val2014` image directories.
*   **Annotations**: The COCO-Tasks annotation JSON files (`task_N_train.json` and `task_N_test.json`).

### Step 1.1: Generate the Affordance Matrix
The system relies on a statistical affordance prior to prune irrelevant objects early in the pipeline.
Run the affordance script to calculate the co-occurrence of object classes and tasks:
```bash
python build_affordance.py
```
*   **Output**: Generates `data/affordance_matrix.npy` (a `14×80` float matrix).

### Step 1.2: Generate Task Embeddings
We use a frozen TinyBERT model to encode the text descriptions of tasks into 312-D vectors.
```bash
python embeddings.py
```
*   **Output**: Generates `data/task_raw_embeddings.pt` and a randomly initialized `data/projection_layer_init.pt`.

---

## 2. Training the Model

Training involves two phases: CrossEntropy (classification) followed by a contrastive InfoNCE loss phase. To speed up training drastically, we first pre-extract the YOLO features so we don't have to run YOLOv8 on every image per epoch.

### Step 2.1: Build ROI Feature Cache (One-Time Setup)
This step runs YOLOv8n over the dataset, extracts bounding boxes, performs ROI-Align on the P4 feature map, and saves the resulting tensors to disk.
```bash
cd E:\DVCon\application
python train.py --build-cache --build-val-cache --coco-dir E:\DVCon\COCO --tasks-dir E:\DVCon\COCO\dataset-master\coco-tasks\annotations
```
*   **Time estimate**: A few minutes on a GPU, or a few hours on CPU.
*   **Output**: Populates the `data/roi_cache/train/` and `data/roi_cache/val/` directories.

### Step 2.2: Execute the Training Loop
With the features cached, initiate the training process. The hyperparameters (learning rate, epochs, weight decay) are governed by `configs/train_config.yaml`.
```bash
cd E:\DVCon\application
python train.py --coco-dir E:\DVCon\COCO --tasks-dir E:\DVCon\COCO\dataset-master\coco-tasks\annotations --config configs\train_config.yaml
```

**What happens during training?**
1.  **Phase 1 (Epochs 1-50):** Trains the scoring modules (TCFG, AGCA, SCRN) and the task projection layer using CrossEntropy Loss.
2.  **Phase 2 (Epochs 51-70):** Incorporates InfoNCE contrastive loss to pull task-relevant visual embeddings closer to their corresponding task vector.
3.  **Threshold Calibration:** At the very end of training, the script runs over the `val2014` set to calibrate the optimal confidence threshold (`θ_t`) for each of the 14 tasks to maximize F1-score.

**Training Outputs / Checkpoints:**
*   `checkpoints/tads_x_fp32_epoch_{N}.pt`: Periodic backups.
*   `checkpoints/tads_x_fp32_best.pt`: The model weights that achieved the highest Top-1 Validation Accuracy.
*   `data/projection_layer_trained.pt`: The trained task-projection weights.
*   `configs/per_task_thresholds.json`: The calibrated `θ_t` thresholds.

---

## 3. Running Inference (How to Use)

The TADS-X model does not output a standard list of bounding boxes. Instead, you provide an image and a **Task Query string**, and the model outputs the *single most suitable object* for that task.

### Giving Task Input
You interact with the model using natural language queries (e.g., "serve wine", "sit comfortably", "pour water into"). 
The `resolve_task_id()` function in the pipeline calculates the cosine similarity between your input string's TinyBERT embedding and the 14 known paper tasks, mapping your prompt to the closest semantic cluster.

### Python API Usage
Inference is handled via the `TADSX` class in `pipeline.py`.

```python
from application.pipeline import TADSX

# 1. Load the model from the best checkpoint
# It will automatically load the affordance matrix, embeddings, and thresholds.
model = TADSX.from_checkpoint(
    checkpoint_path="application/checkpoints/tads_x_fp32_best.pt",
    yolo_weights="yolov8n.pt",
    device="cpu"  # Inference is generally fast enough on CPU
)

# 2. Run prediction
image_path = "path/to/your/test_image.jpg"
task_query = "serve wine"

result = model.predict(
    image_path=image_path,
    task_query=task_query,
    verbose=True
)

# 3. View Results
print(result)
```

**Expected Result Output (Match):**
```json
{
    "bbox": (234, 156, 89, 112), 
    "class": "wine glass", 
    "confidence": 0.9423,
    "task": "serve wine",
    "resolved_paper_task": "serve wine"
}
```

**Expected Result Output (No Match):**
If no object passes the affordance prune step, or if the highest confidence score is below the calibrated threshold (`θ_t`), it returns:
```json
{
    "result": "no suitable object found",
    "task": "cut paper",
    "reason": "below threshold (score=0.2301, θ_t=0.4500)"
}
```

---

## 4. Evaluation and Visualization

### Evaluating on the Validation Set
To generate rigorous metrics against the COCO-Tasks validation set, run:
```bash
python evaluate.py \
    --coco-dir <PATH_TO_COCO_ROOT> \
    --tasks-dir <PATH_TO_COCO_TASKS_ANNOTATIONS> \
    --checkpoint checkpoints/tads_x_fp32_best.pt
```
This script computes the 101-point interpolated F1 score, Recall, and Precision (the standard COCO-Tasks benchmark metrics).

### Visualizing Results
You can visualize the model's decision on a specific image using the draw script:
```bash
python draw_bbox.py \
    --image <PATH_TO_IMAGE> \
    --task "serve wine" \
    --checkpoint checkpoints/tads_x_fp32_best.pt
```
This will output a new image with the winning bounding box drawn in green, displaying the matched class and confidence score.
