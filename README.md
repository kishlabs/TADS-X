<div align="center">
  <h1>🎯 TADS-X: Task-Aware Dual-Stream Detection</h1>
  <p><strong>A Next-Generation Object Detector with Affordance Gating</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
  [![YOLOv8](https://img.shields.io/badge/YOLO-v8n-yellow.svg)](https://ultralytics.com/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

<br />

## 📖 About The Repository

**TADS-X** (Task-Aware Dual-Stream Detection with Affordance Gating) is the official software submission by **Team ChipSmiths** for the **DVCon India 2026 Design Contest (Stage 2A)**. 

Unlike conventional object detectors that simply identify what items are present in a scene, TADS-X is designed with a **goal-oriented understanding**. It processes both a visual scene and a linguistic task prompt, filtering out irrelevant objects to answer one critical question: 

> *"Which single object in this image should I use for a specific user-provided task?"*

If presented with an image of a **cup**, a **wine glass**, and a **bottle**, TADS-X ranks and predicts the correct object based on tasks like `"serve wine"`, `"pour water into"`, or `"carry things in"`.

---

## ✨ Features & Pipeline

TADS-X employs a dual-stream architecture (Vision + Language) to combine world knowledge with spatial awareness:

1. **Vision Stream (YOLOv8n)**: Extremely fast CPU-based detection that isolates all spatial boundary boxes and computes intermediate `P4` feature maps.
2. **Language Stream (TinyBERT)**: Encodes natural language task inputs into high-dimensional intent embeddings.
3. **Task-Conditioned Feature Gating (TCFG)**: An attention mechanism that suppresses visual properties unrelated to the textual prompt and highlights critical affordance traits.
4. **Affordance-Guided Cross-Attention (AGCA)**: A scoring threshold utilizing the relationship between the textual task constraints and object traits.
5. **Scene Context Re-scoring (SCRN)**: Analyzes the top 5 predicted candidates against *each other* to ensure that a dominant object (e.g., wine glass) correctly suppresses a valid but weaker alternative (e.g., a paper cup) based on spatial availability.

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/kishlabs/TADS-X.git
   cd TADS-X
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment (`.venv`).
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install torch torchvision ultralytics transformers numpy scipy
   ```

3. **COCO Dataset Preparation**
   The MS COCO framework is used for evaluation and validation, but due to file sizes (>20GB) it is **excluded from Git**. You must manually download and arrange it locally in the repository root:
   ```text
   📦 COCO
    ┣ 📂 train2014/                     (COCO 2014 Training Images)
    ┣ 📂 val2014/                       (COCO 2014 Validation Images)
    ┣ 📂 annotations_trainval2014/      (Standard COCO JSON annotations)
    ┗ 📂 dataset-master/                (COCO-Tasks Annotations task_1 to task_14)
   ```

---

## 🚀 Usage & Available Options

The command-line tools can execute single predictions, batch testing, or complete model training loops. All inference processing strictly restricts operations to **CPU** as per DVCon 2026 Stage 2A parameters.

### 1. Single Image Inference
Predict the most suitable object for a given task within a single test image.
```bash
python application/pipeline.py --image "path/to/test_image.jpg" --task "serve wine"
```
**Output Example:**
```json
{
  "bbox": [234, 156, 89, 112],
  "class": "wine glass",
  "confidence": 0.942
}
```

### 2. Validation & Evaluation Benchmark
Evaluates the PyTorch checkpoint across the entire metric threshold of all 14 designated tasks globally.
```bash
python application/evaluate.py \
    --coco-dir "./COCO" \
    --tasks-dir "./COCO/dataset-master"
```

### 3. Fast Validation (Subset Mode)
For iterative debugging or performance optimization, you can rapidly test against a random subset.
```bash
python application/evaluate.py \
    --coco-dir "./COCO" \
    --tasks-dir "./COCO/dataset-master" \
    --subset 500
```

### 4. Neural Training
Train the TCFG, AGCA, and SCRN neural heads over epochs defined within your yaml config.
```bash
python application/train.py --config "configs/train_config.yaml"
```

---

## 📁 Repository Structure

```text
📦 TADS-X
 ┣ 📂 application/
 ┃  ┣ 📂 models/           # Core Neural Architectures (SCRN, AGCA, TCFG)
 ┃  ┣ 📂 data/             # Persistent matrix structures & affordance lookup arrays
 ┃  ┣ 📜 pipeline.py       # Core prediction aggregation framework (Inference End-Point)
 ┃  ┣ 📜 embeddings.py     # TinyBERT logic handles
 ┃  ┣ 📜 train.py          # Unified model weighting script
 ┃  ┣ 📜 TADS_X_SRS .md    # The exhaustive Stage 2A Specification Guide (System Requirements)
 ┃  ┗ 📜 yolov8n.pt        # Base pre-trained visual detection weights
 ┣ 📂 Reference Docs/      # DVCon Problem Statements, constraints, and research papers
 ┣ 📂 tads_x/              # Local module and matrix data mirror
 ┣ 📜 README.md            # Repository overview
 ┗ 📜 .gitignore           # Overarching git cache, python configs, & dataset omission specs
```

---

## 📊 Performance Goals

By running iterative scene context refinement on FP32 environments, the application attempts to surpass normal object detection constraints and is actively targeted to achieve an `mAP@0.5 > 0.60` on the **COCO-Tasks val2014 benchmark**.

*Refer to the full 40+ page documentation trace within `application/TADS_X_SRS .md` for rigorous matrix ablation charts, algorithm math, and problem specifications.*
