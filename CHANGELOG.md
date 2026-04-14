# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.0.0] — 2026-04-14

### Added
- **TADS-X core pipeline** (`pipeline.py`): full inference from image + task query to bounding-box selection.
- **YOLOv8n vision stream**: fast CPU-based object detection generating P4 feature maps.
- **TinyBERT language stream** (`embeddings.py`): encodes natural-language task queries into 312-D embeddings projected to 256-D.
- **Task-Conditioned Feature Gating (TCFG)** (`models/tcfg.py`): element-wise gating of visual features by task embedding.
- **Affordance-Guided Cross-Attention (AGCA)** (`models/agca.py`): scores object candidates against affordance priors.
- **Scene Context Re-scoring Network (SCRN)** (`models/scrn.py`): pairwise re-ranking of top-K candidates.
- **Training script** (`train.py`): two-phase FP32 training (Cross-Entropy + InfoNCE contrastive loss).
- **Evaluation harness** (`evaluate.py`): mAP@0.5 across all 14 COCO-Tasks evaluation queries with baseline comparisons.
- **Affordance matrix** pre-computed from COCO-Tasks annotations (14 tasks × 80 COCO classes).
- **Per-task calibrated thresholds** (`configs/per_task_thresholds.json`).
- **ROI feature cache builder** (`--build-cache` flag in `train.py`) for offline pre-extraction.
- **Task definitions** (`task_definitions.py`): single source of truth for paper tasks, SRS tasks, and COCO class mappings.
- **`build_affordance.py`** and **`check_mapping.py`** utility scripts.
- Comprehensive System Requirements Specification (`application/TADS_X_SRS .md`, 40+ pages).

[Unreleased]: https://github.com/kishlabs/TADS-X/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/kishlabs/TADS-X/releases/tag/v1.0.0
