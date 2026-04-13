"""
build_affordance.py
====================
TADS-X — Team ChipSmiths | DVCon India 2026

Builds the 14×80 affordance prior matrix A from COCO-Tasks annotations.

Matrix rows are indexed by PAPER task file number (1-indexed):
  A[0] = file task 1 = "step on something to reach top of a shelf"
  A[9] = file task 10 = "serve wine"
  ...

Per SRS DR-03:
  - Count ONLY preferred objects (category_id == 1)
  - Normalise per task row with epsilon smoothing

Output:
  data/affordance_matrix.npy      — shape (14, 80), float32
  data/affordance_matrix_raw.npy  — raw counts before normalisation
  data/affordance_matrix_info.json
  data/affordance_matrix_readable.txt

Usage:
  python build_affordance.py --tasks-dir E:/DVCon/COCO/dataset-master/coco-tasks/annotations
"""

import argparse
import json
import os
import sys
import numpy as np

from task_definitions import (
    PAPER_TASKS,
    COCO_CLASSES,
    COCO_ID_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    NUM_TASKS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Annotation loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_annotation_file(tasks_dir: str, file_id: int) -> str | None:
    """Return path to task_{file_id}_train.json or None if not found."""
    for name in [f"task_{file_id}_train.json", f"task{file_id}_train.json"]:
        p = os.path.join(tasks_dir, name)
        if os.path.exists(p):
            return p
    return None


def _load_annotations(path: str) -> list:
    """Load and strictly validate a COCO-Tasks annotation JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at root, got {type(data).__name__}")
    if "annotations" not in data:
        raise ValueError(f"{path}: missing required 'annotations' key")
    annotations = data["annotations"]
    if not isinstance(annotations, list):
        raise ValueError(f"{path}: 'annotations' must be a list, got {type(annotations).__name__}")
    return annotations


# ─────────────────────────────────────────────────────────────────────────────
# Matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def build_from_annotations(tasks_dir: str) -> np.ndarray:
    """
    Build raw count matrix from COCO-Tasks annotation files.

    Fails fast if any required annotation file is missing.

    Returns
    -------
    counts : np.ndarray, shape (14, 80), dtype float32
    """
    # Check all 14 files exist BEFORE starting (fail fast)
    missing = []
    for file_id in range(1, NUM_TASKS + 1):
        if _find_annotation_file(tasks_dir, file_id) is None:
            missing.append(f"task_{file_id}_train.json")
    if missing:
        raise FileNotFoundError(
            f"Missing annotation files in '{tasks_dir}':\n"
            + "\n".join(f"  {m}" for m in missing)
        )

    counts = np.zeros((NUM_TASKS, NUM_CLASSES), dtype=np.float32)

    for file_id in range(1, NUM_TASKS + 1):
        path = _find_annotation_file(tasks_dir, file_id)
        annotations = _load_annotations(path)

        n_preferred = 0
        n_skipped   = 0

        for ann_idx, ann in enumerate(annotations):
            if not isinstance(ann, dict):
                print(f"  [WARN] File {file_id}, annotation {ann_idx}: not a dict, skipping")
                n_skipped += 1
                continue

            if ann.get("category_id") != 1:
                continue

            coco_cat_id = ann.get("COCO_category_id")
            if coco_cat_id is None:
                n_skipped += 1
                continue

            try:
                coco_cat_int = int(coco_cat_id)
            except (ValueError, TypeError):
                print(f"  [WARN] File {file_id}, annotation {ann_idx}: "
                      f"invalid COCO_category_id '{coco_cat_id}', skipping")
                n_skipped += 1
                continue

            idx_class = COCO_ID_TO_IDX.get(coco_cat_int)
            if idx_class is None:
                n_skipped += 1
                continue

            counts[file_id - 1, idx_class] += 1
            n_preferred += 1
        task_name = PAPER_TASKS[file_id]
        print(f"  File {file_id:2d} ({task_name:<44}): "
              f"{n_preferred:5d} preferred, {n_skipped} skipped")

    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise(counts: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Row-normalise counts with epsilon smoothing (SRS DR-03).

    A[t, c] = (counts[t, c] + eps) / (sum(counts[t]) + eps * 80)

    Guarantees no zero entries and rows sum to ~1.0.
    """
    numerator   = counts + epsilon
    denominator = counts.sum(axis=1, keepdims=True) + epsilon * NUM_CLASSES
    return (numerator / denominator).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_relevant_classes(file_task_id: int, A: np.ndarray, threshold: float = 0.01):
    """
    Return COCO matrix indices where A[file_task_id-1][idx] >= threshold.

    Used in pipeline class pruning. file_task_id is 1-indexed paper file number.
    """
    row = A[file_task_id - 1]
    return list(np.where(row >= threshold)[0])


def print_top_classes(A: np.ndarray, top_k: int = 5) -> dict:
    info = {}
    print("\n── Top classes per task ─────────────────────────────────────────────────────")
    for file_id in range(1, NUM_TASKS + 1):
        row         = A[file_id - 1]
        top_indices = np.argsort(row)[::-1][:top_k]
        top         = [(IDX_TO_CLASS[i], float(row[i])) for i in top_indices]
        info[str(file_id)] = top
        task_name   = PAPER_TASKS[file_id]
        top_str     = ", ".join(f"{name}={score:.4f}" for name, score in top)
        print(f"  File {file_id:2d} ({task_name:<44}): {top_str}")
    print("─────────────────────────────────────────────────────────────────────────────")
    return info


def validate_matrix(A: np.ndarray) -> bool:
    """
    Structural sanity checks. Does NOT hardcode per-row class expectations
    (those are dataset-dependent and brittle).
    """
    errors = []

    if A.shape != (NUM_TASKS, NUM_CLASSES):
        errors.append(f"Wrong shape: {A.shape}, expected ({NUM_TASKS}, {NUM_CLASSES})")

    if (A == 0).any():
        errors.append("Matrix contains zero entries — epsilon smoothing failed")

    row_sums = A.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        errors.append(
            f"Rows do not sum to 1: min={row_sums.min():.6f}, max={row_sums.max():.6f}"
        )

    # Check no row is uniform (all ~1/80) — would indicate annotation load failure
    expected_uniform = 1.0 / NUM_CLASSES
    uniform_rows = [
        i + 1 for i in range(NUM_TASKS)
        if A[i].max() < expected_uniform * 3   # max < 3× uniform = suspicious
    ]
    if uniform_rows:
        errors.append(
            f"Rows appear near-uniform (possible missing preferred annotations): "
            f"file tasks {uniform_rows}"
        )

    if errors:
        print("\n[VALIDATION ERRORS]")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("\n[VALIDATION] ✓ All checks passed")

    return len(errors) == 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI


def main():
    parser = argparse.ArgumentParser(
        description="Build TADS-X 14×80 affordance prior matrix."
    )
    parser.add_argument(
        "--tasks-dir", type=str, required=True,
        help="Directory containing task_N_train.json files."
    )
    parser.add_argument(
        "--out-dir", type=str, default="data",
        help="Output directory (default: data/)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-6,
        help="Epsilon for row normalisation (default: 1e-6)"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TADS-X Affordance Matrix Builder")
    print(f"  Tasks dir : {args.tasks_dir}")
    print(f"  Output    : {args.out_dir}/")
    print(f"  Epsilon   : {args.epsilon}")
    print(f"{'='*60}\n")

    # Step 1: Build raw counts
    print("Step 1: Counting preferred objects per annotation file...")
    try:
        counts = build_from_annotations(args.tasks_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Normalise
    print("\nStep 2: Normalising...")
    A = normalise(counts, epsilon=args.epsilon)
    print(f"  Row sum range : [{A.sum(axis=1).min():.8f}, {A.sum(axis=1).max():.8f}]")
    print(f"  Value range   : [{A.min():.8f}, {A.max():.8f}]")
    non_tiny = (A > args.epsilon * 100).sum()
    print(f"  Non-trivial entries: {non_tiny} / {NUM_TASKS * NUM_CLASSES}")

    # Step 3: Print top classes
    info = print_top_classes(A, top_k=5)

    # Step 4: Validate
    print("\nStep 3: Validating...")
    valid = validate_matrix(A)

    # Step 5: Save
    print("\nStep 4: Saving...")
    out_matrix = os.path.join(args.out_dir, "affordance_matrix.npy")
    out_raw    = os.path.join(args.out_dir, "affordance_matrix_raw.npy")
    out_info   = os.path.join(args.out_dir, "affordance_matrix_info.json")
    out_txt    = os.path.join(args.out_dir, "affordance_matrix_readable.txt")

    np.save(out_matrix, A)
    np.save(out_raw, counts)

    full_info = {
        "shape"             : [NUM_TASKS, NUM_CLASSES],
        "epsilon"           : args.epsilon,
        "row_indexing"      : "paper file number (1-indexed), NOT SRS task ID",
        "paper_tasks"       : {str(k): v for k, v in PAPER_TASKS.items()},
        "top_5_per_file"    : info,
        "validation_passed" : valid,
    }
    with open(out_info, "w", encoding="utf-8") as f:
        json.dump(full_info, f, indent=2)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("TADS-X Affordance Matrix  [14 paper tasks x 80 COCO classes]\n")
        f.write("Row index = paper file number - 1  (row 0 = task_1_train.json)\n")
        f.write("=" * 70 + "\n\n")
        for file_id in range(1, NUM_TASKS + 1):
            f.write(f"File {file_id:2d}: {PAPER_TASKS[file_id]}\n")
            row   = A[file_id - 1]
            above = [
                (COCO_CLASSES[i], float(row[i]))
                for i in np.where(row > args.epsilon * 100)[0]
            ]
            above.sort(key=lambda x: -x[1])
            for cls, val in above:
                bar = "#" * int(val * 300)
                f.write(f"  {cls:<22} {val:.6f}  {bar}\n")
            f.write("\n")

    print(f"  ✓ {out_matrix}  (shape {A.shape}, dtype {A.dtype})")
    print(f"  ✓ {out_raw}")
    print(f"  ✓ {out_info}")
    print(f"  ✓ {out_txt}")

    print(f"\n{'='*60}")
    print("  Load in pipeline:")
    print("    A = np.load('data/affordance_matrix.npy')")
    print("    # A[file_task_id - 1, coco_matrix_idx] -> float32 score")
    print("    # file_task_id resolved from SRS query via cosine similarity")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
