# TADS-X: Task-Aware Dual-Stream Detection with Affordance Gating

## Overview
Standard object detectors answer: **"What objects are in this image?"**

**TADS-X** answers: **"Which object in this image should I use for a given task?"**

Instead of treating all detected objects equally, TADS-X implements a task-aware approach. It determines the functional relationship between an object and a goal, filtering and ranking detected items based on their relevance to a provided task (e.g., `"serve wine"`, `"sit on"`, `"dig a hole with"`).

This project represents Team ChipSmiths' software submission for the **DVCon India 2026 Design Contest — Stage 2A**. 

## Repository Structure

*   `application/`: Contains the core Python scripts, modules, and SRS for the inference pipeline.
    *   `pipeline.py`: Main integration joining all modules.
    *   `TADS_X_SRS .md`: Complete Software Requirements Specifications covering the architecture, requirements, and theoretical background of the system.
    *   `build_affordance.py`: Helper script to construct the affordance prior matrix.
    *   `embeddings.py`: TinyBERT text embedding cache generator.
*   `tads_x/`: Auxiliary data folder.
*   `Reference Docs/`: Reference papers (e.g., COCO-Tasks) and DVCon contest guidelines/Q&A docs.

### The Dataset (`COCO/` Folder)

> [!NOTE]
> The `COCO` subdirectory contains large dataset files (approx. 20GB) and is **explicitly ignored via `.gitignore`** to prevent bloating the remote repository. 

To recreate the environment, you must manually reconstruct the `COCO` directory at the project root with the following subdirectories:

*   `train2014/`: COCO 2014 Training images (83K images).
*   `val2014/`: COCO 2014 Evaluation images (41K images).
*   `annotations_trainval2014/`: Standard COCO `instances_train2014.json` / `instances_val2014.json`. 
*   `dataset-master/`: COCO-Tasks specific annotation files (e.g., `task_1_train.json` ... `task_14_test.json`).

## Architecture & How It Works

The inference pipeline currently executes natively on CPU using the following core components:

1.  **Detection Base:** Extracts bounding boxes and P4 feature maps via `YOLOv8n`.
2.  **Task Encoding:** Caches and retrieves task embeddings using `TinyBERT`.
3.  **Task-Conditioned Feature Gating (TCFG):** Modulates visual dimensions based on task relevance.
4.  **Affordance-Guided Cross-Attention (AGCA):** Combines system prior knowledge with text-vision alignment scoring.
5.  **Scene Context Re-scoring (SCRN):** Models dynamic inter-object relationships using self-attention across the top candidate boxes.

For comprehensive details on methodology and evaluation benchmarks, consult the full `application/TADS_X_SRS .md` document.
